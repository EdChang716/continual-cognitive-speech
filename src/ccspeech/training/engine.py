"""Controlled two-domain experiments with a reusable supervised source stage."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib.metadata
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from ccspeech.evaluation.metrics import continual_summary, paired_bootstrap_change
from ccspeech.methods.memory import ReservoirMemory
from ccspeech.models import build_model, configure_updates
from .config import base_config, canonical_hash, semantic_config, validate_config
from .data import load_records, make_collator
from .state import (atomic_json, exclusive_run, read_checkpoint, rebind_memory,
                    seed_everything)
from .suites import run_name


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprint(config: dict, splits=('train', 'validation')) -> dict:
    """Hash exact prepared manifests, including text and segment provenance."""
    root = Path(config['data']['root']).resolve()
    files = []
    if config['phase'] == 'text':
        for domain in config['domains']:
            files.extend(root / config['data']['text_dir'] / domain / f'{split}.parquet' for split in splits)
    else:
        files.extend(root / config['data']['speech_dir'] / f'{split}.csv' for split in splits)
        for manifest in files:
            with manifest.open(newline='') as source:
                for row in csv.DictReader(source):
                    audio = (root / row['audio_path']).resolve()
                    if not audio.is_relative_to(root) or _hash_file(audio) != row['sha256']:
                        raise ValueError(f'Speech window integrity failure: {row["segment_id"]}')
    hashes = {str(path.relative_to(root)): _hash_file(path) for path in files}
    if config['phase'] == 'speech' and 'test' in splits:
        truth = root / config['data']['test_labels']
        hashes[str(truth.relative_to(root))] = _hash_file(truth)
    return hashes


def _limited(records: list[dict], limit: int | None, seed: int, speech: bool) -> list[dict]:
    if limit is None:
        return records
    if type(limit) is not int or limit < 1:
        raise ValueError('Smoke limits must be positive integers')
    field = 'participant_id' if speech else 'example_id'
    labels, groups = {}, {}
    for row in records:
        key = row[field]
        groups.setdefault(key, []).append(row)
        labels[key] = row.get('label', -1)
    by_label = {}
    for key in sorted(groups):
        by_label.setdefault(labels[key], []).append(key)
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < limit and any(by_label.values()):
        for label in sorted(by_label):
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].pop())
    chosen = set(selected)
    # All overlapping windows and recordings of a selected person stay together.
    return [row for row in records if row[field] in chosen]


def load_split(config: dict, split: str, *, allow_test_labels=False) -> dict[str, list[dict]]:
    runtime = config.get('runtime', {})
    limit = runtime.get('train_limit' if split == 'train' else 'evaluation_limit')
    return {domain: _limited(load_records(config['data'], split, domain, allow_test_labels=allow_test_labels),
                             limit, config['seed'], config['phase'] == 'speech')
            for domain in config['domains']}


def audit_split_separation(train: dict, validation: dict, speech: bool) -> None:
    training = [row for rows in train.values() for row in rows]
    checking = [row for rows in validation.values() for row in rows]
    if {r['example_id'] for r in training} & {r['example_id'] for r in checking}:
        raise ValueError('Training/validation example leakage')
    if speech and {r['participant_id'] for r in training} & {r['participant_id'] for r in checking}:
        raise ValueError('Training/validation participant leakage')


def compact_report(report: dict) -> dict:
    return {'selection_score': report['selection_score'],
            'domains': {domain: {key: value for key, value in item.items()
                               if key not in {'records', 'probabilities', 'participant_records', 'participant_probabilities'}}
                        for domain, item in report['domains'].items()}}


def _save_predictions(directory: Path, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for domain, result in report['domains'].items():
        variants = [('examples', result['records'], result['probabilities'])]
        if 'participant_records' in result:
            variants.append(('participants', result['participant_records'], result['participant_probabilities']))
        for name, records, probabilities in variants:
            rows = []
            for record, probability in zip(records, probabilities):
                row = {key: record[key] for key in ('example_id', 'segment_id', 'participant_id', 'recording_id',
                                                    'domain', 'language', 'split', 'label') if key in record}
                row.update({f'p_{index}': float(value) for index, value in enumerate(probability)})
                rows.append(row)
            path = directory / f'{domain}_{name}.csv'
            fields = sorted(set().union(*(row.keys() for row in rows)))
            with path.open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def _device(config: dict) -> torch.device:
    requested = config.get('runtime', {}).get('device', 'cuda')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Use a GPU machine, or an explicit CPU pilot config.')
    if requested == 'mps':
        raise ValueError('Use cuda for full runs or cpu for tested smoke runs; MPS is not validated')
    device = torch.device(requested)
    if config['training'].get('precision') == 'bfloat16' and device.type != 'cuda':
        raise ValueError('bfloat16 is supported only on CUDA; use float32 on CPU')
    if device.type == 'cuda' and config['training'].get('precision') == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support bfloat16')
    return device


def _initial_models(config: dict, device: torch.device):
    seed_everything(config['seed'])
    torch.set_num_threads(config.get('runtime', {}).get('num_threads', 4))
    model = build_model(config['model'], 2 if config['phase'] == 'speech' else 5)
    retrieval = deepcopy(model)
    retrieval.requires_grad_(False)
    retrieval.eval()
    return model.to(device), retrieval.to(device), make_collator(config['model'], config['training']['max_length'])


def _restore(model, memory, path: Path, data_root: str) -> dict:
    checkpoint = read_checkpoint(path)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    memory.load_state_dict(checkpoint['memory_state'])
    rebind_memory(memory, Path(data_root))
    return checkpoint


def _environment() -> dict:
    result = {'versions': {name: importlib.metadata.version(name) for name in
                          ('torch', 'transformers', 'numpy', 'pyarrow', 'scikit-learn', 'soundfile')},
              'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda}
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name()
    try:
        result['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        result['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        result['git_commit'] = None
    return result


def _source_config(config: dict) -> dict:
    value = base_config(config['phase'])
    for key in ('model', 'data', 'seed', 'domains', 'runtime'):
        value[key] = deepcopy(config[key])
    value['method'] = 'sequential'
    value['training'] = deepcopy(config['training'])
    value['training']['target_epochs'] = value['training']['source_epochs']
    value['evaluation'] = deepcopy(config['evaluation'])
    value['memory']['capacity'] = config['memory']['capacity']
    value['updates'] = {'policy': 'full'}
    value['timescales'] = {'head_interval': 1, 'encoder_interval': 1}
    return value


def _memory_cost(memory, phase: str) -> dict:
    result = memory.statistics()
    paths = {entry.record.get('_audio_file') for entry in memory.entries} - {None}
    result['referenced_audio_bytes'] = sum(Path(path).stat().st_size for path in paths)
    result['participant_coverage'] = len({entry.record['participant_id'] for entry in memory.entries if 'participant_id' in entry.record})
    result['domain_counts'] = {domain: sum(entry.record.get('domain') == domain for entry in memory.entries)
                               for domain in sorted({entry.record.get('domain', '') for entry in memory.entries})}
    result['unit'] = 'audio_segment' if phase == 'speech' else 'text_review'
    return result


def run_experiment(config: dict, *, output_root: Path = Path('runs'), resume: bool = False) -> Path:
    from .loop import train_stage
    from .prediction import evaluate_domains
    validate_config(config)
    device = _device(config)
    root = output_root.resolve()
    output = root / run_name(config)
    started = time.perf_counter()
    with exclusive_run(output):
        fingerprint = data_fingerprint(config)
        if (output / 'result.json').is_file():
            import json
            old = json.loads((output / 'result.json').read_text())
            if old['data_fingerprint'] != fingerprint or old['config_hash'] != canonical_hash(semantic_config(config)):
                raise ValueError('Prepared data changed since this completed run')
            print(f'Already complete: {output}', flush=True)
            return output
        if (output / 'run.json').exists() and not resume:
            raise FileExistsError(f'Incomplete run exists: {output}; use --resume')
        train, validation = load_split(config, 'train'), load_split(config, 'validation')
        audit_split_separation(train, validation, config['phase'] == 'speech')
        model, retrieval, collator = _initial_models(config, device)
        metadata = {'schema_version': 1, 'config': config, 'config_hash': canonical_hash(semantic_config(config)),
                    'data_fingerprint': fingerprint, 'model': model.metadata(), 'environment': _environment(),
                    'records': {split: {domain: len(rows) for domain, rows in values.items()}
                                for split, values in [('train', train), ('validation', validation)]}}
        if (output / 'run.json').exists():
            import json
            old = json.loads((output / 'run.json').read_text())
            if old['data_fingerprint'] != fingerprint or old['config_hash'] != metadata['config_hash']:
                raise ValueError('Resume data/config fingerprint differs')
        atomic_json(output / 'run.json', metadata)
        method = config['method']
        if method in {'joint', 'separate'}:
            result = _run_reference(config, model, retrieval, collator, train, validation, output, resume)
        else:
            source_config = _source_config(config)
            source_key = canonical_hash({'config': semantic_config(source_config), 'data': fingerprint,
                                         'model': model.metadata()})[:20]
            source_dir = root / '_source' / source_key
            memory = ReservoirMemory(config['memory']['capacity'], config['seed'])
            source = config['domains'][0]
            configure_updates(model, {'policy': 'full'})
            with exclusive_run(source_dir):
                if not (source_dir / 'complete.json').exists():
                    print(f'Source warmup: {source} ({source_key})', flush=True)
                    summary = train_stage(model, retrieval, collator, train[source], memory, source_config,
                                          source_dir, stage='source', epochs=config['training']['source_epochs'],
                                          method='sequential', resume=(source_dir / 'last.pt').exists(),
                                          validate_fn=lambda m, b: compact_report(evaluate_domains(
                                              m, retrieval, collator, {source: validation[source]}, b,
                                              source_config, method='sequential')))
                    atomic_json(source_dir / 'complete.json', {'best_checkpoint': 'best.pt',
                                'best_score': summary['best_score'], 'data_fingerprint': fingerprint,
                                'training_stats': summary['training_stats']})
                _restore(model, memory, source_dir / 'best.pt', config['data']['root'])
            # All adaptation policies branch from the SAME already-trained head.
            seed_everything(config['seed'] + 1_000_000)
            update_stats = configure_updates(model, config['updates'])
            seed_everything(config['seed'] + 1_000_000)
            before = evaluate_domains(model, retrieval, collator, validation, memory, config)
            _save_predictions(output / 'validation' / 'source', before)
            atomic_json(output / 'source.json', {'cache_path': str(source_dir.relative_to(root)),
                                                'validation': compact_report(before), 'updates': update_stats})
            target = config['domains'][1]
            target_config = deepcopy(config)
            if method == 'sequential':
                # The common warmup exists for controlled initialization. A
                # no-replay adaptation does not retain its memory or key model.
                memory = ReservoirMemory(0, config['seed'])
                target_config['memory']['capacity'] = 0
                retrieval = None
            print(f'Adaptation: {source} -> {target}, {method}, {config["updates"]["policy"]}', flush=True)
            stage = train_stage(model, retrieval, collator, train[target], memory, target_config, output / 'target',
                                stage='target', epochs=config['training']['target_epochs'], method=method,
                                resume=resume and (output / 'target' / 'last.pt').exists(),
                                validate_fn=lambda m, b: compact_report(evaluate_domains(
                                    m, retrieval, collator, validation, b, config)))
            _restore(model, memory, Path(stage['best_checkpoint']), config['data']['root'])
            after = evaluate_domains(model, retrieval, collator, validation, memory, config)
            _save_predictions(output / 'validation' / 'final', after)
            scores = [[r['domains'][d]['metrics']['accuracy'] for d in config['domains']] for r in (before, after)]
            result = {'source_cache': str(source_dir.relative_to(root)), 'checkpoint': 'target/best.pt',
                      'validation': {'source': compact_report(before), 'final': compact_report(after)},
                      'continual': continual_summary(scores, config['domains']), 'updates': update_stats,
                      'training_stats': stage['training_stats'], 'memory': _memory_cost(memory, config['phase'])}
            import json
            result['source_training_stats'] = json.loads((source_dir / 'complete.json').read_text()).get('training_stats')
        result.update(schema_version=1, config_hash=metadata['config_hash'], data_fingerprint=fingerprint,
                      config=config, smoke=bool(config.get('runtime', {}).get('smoke')),
                      wall_seconds=time.perf_counter()-started)
        atomic_json(output / 'result.json', result)
        print(f'Completed validation run: {output}', flush=True)
    return output


def _run_reference(config, model, retrieval, collator, train, validation, output, resume):
    from .loop import train_stage
    from .prediction import evaluate_domains
    device = next(model.parameters()).device
    is_joint = config['method'] == 'joint'
    groups = {'joint': [row for domain in config['domains'] for row in train[domain]]} if is_joint else train
    reports, checkpoints, costs = {}, {}, {}
    for index, (name, records) in enumerate(groups.items()):
        if index:
            # Same pretrained weights and head seed for independent language models.
            model, retrieval, collator = _initial_models(config, device)
        configure_updates(model, {'policy': 'full'})
        retrieval = None
        memory = ReservoirMemory(0, config['seed'])
        local_config = deepcopy(config)
        local_config['memory']['capacity'] = 0
        selected_validation = validation if is_joint else {name: validation[name]}
        stage = train_stage(model, retrieval, collator, records, memory, local_config, output / name,
                            stage=name, epochs=config['training']['target_epochs'], method='sequential',
                            resume=resume and (output / name / 'last.pt').exists(),
                            validate_fn=lambda m, b: compact_report(evaluate_domains(
                                m, retrieval, collator, selected_validation, b, config, method='sequential')))
        _restore(model, memory, Path(stage['best_checkpoint']), config['data']['root'])
        result = evaluate_domains(model, retrieval, collator, selected_validation, memory, config, method='sequential')
        _save_predictions(output / 'validation' / name, result)
        reports.update(result['domains'])
        checkpoints[name] = str(Path(stage['best_checkpoint']).relative_to(output))
        costs[name] = stage['training_stats']
    combined = {'selection_score': float(np.mean([r['metrics']['accuracy'] for r in reports.values()])), 'domains': reports}
    return {'checkpoints': checkpoints, 'validation': {'final': compact_report(combined)},
            'training_stats': costs, 'reference': config['method'],
            'reference_budget_note': 'Joint sees the union for target_epochs; separate trains each domain independently. Reference costs are reported, not assumed equal.'}


def final_test(run_directory: Path, *, data_root: str | None = None, device: str | None = None,
               allow_smoke: bool = False) -> Path:
    """Explicit, separate test access. Never updates checkpoints or selects settings."""
    import json
    from .prediction import evaluate_domains
    output = run_directory.resolve()
    with exclusive_run(output):
        result = json.loads((output / 'result.json').read_text())
        config = result['config']
        if result.get('smoke') and not allow_smoke:
            raise ValueError('Smoke runs are truncated; final-test requires a full experiment')
        if (output / 'test.json').exists():
            raise FileExistsError('Final-test report already exists; retain this immutable evaluation record')
        if data_root is not None:
            config['data']['root'] = data_root
        if device is not None:
            config['runtime']['device'] = device
        if data_fingerprint(config) != result['data_fingerprint']:
            raise ValueError('Training data differs from the completed experiment')
        test = load_split(config, 'test', allow_test_labels=True)
        train, validation = load_split(config, 'train'), load_split(config, 'validation')
        audit_split_separation(train, test, config['phase'] == 'speech')
        audit_split_separation(validation, test, config['phase'] == 'speech')
        model, retrieval, collator = _initial_models(config, _device(config))
        reports = {}
        if config['method'] in {'joint', 'separate'}:
            for name, relative in result['checkpoints'].items():
                configure_updates(model, {'policy': 'full'})
                memory = ReservoirMemory(0, config['seed'])
                _restore(model, memory, output / relative, config['data']['root'])
                subset = test if name == 'joint' else {name: test[name]}
                report = evaluate_domains(model, retrieval, collator, subset, memory, config,
                                          method='sequential', include_intervals=True)
                reports.update(report['domains'])
            final = {'selection_score': float(np.mean([r['metrics']['accuracy'] for r in reports.values()])), 'domains': reports}
            report = {'final': compact_report(final)}
            _save_predictions(output / 'test' / 'final', final)
        else:
            memory = ReservoirMemory(config['memory']['capacity'], config['seed'])
            # Resolve relative to the runs root without storing workstation paths.
            source_reference = json.loads((output / 'source.json').read_text())['cache_path']
            # run_name always has phase/order/name (three components).
            source_dir = output.parents[2] / source_reference
            _restore(model, memory, source_dir / 'best.pt', config['data']['root'])
            seed_everything(config['seed'] + 1_000_000)
            configure_updates(model, config['updates'])
            before = evaluate_domains(model, retrieval, collator, test, memory, config, include_intervals=True)
            if config['method'] == 'sequential':
                memory = ReservoirMemory(0, config['seed'])
                retrieval = None
            _restore(model, memory, output / result['checkpoint'], config['data']['root'])
            after = evaluate_domains(model, retrieval, collator, test, memory, config, include_intervals=True)
            scores = [[r['domains'][d]['metrics']['accuracy'] for d in config['domains']] for r in (before, after)]
            report = {'source': compact_report(before), 'final': compact_report(after),
                      'continual': continual_summary(scores, config['domains'])}
            if config['phase'] == 'speech':
                source = config['domains'][0]
                old, new = before['domains'][source], after['domains'][source]
                if old['participant_records'] != new['participant_records']:
                    raise ValueError('Paired retention units differ')
                report['paired_source_change'] = paired_bootstrap_change(
                    [r['label'] for r in old['participant_records']], old['participant_probabilities'],
                    new['participant_probabilities'], n_bootstrap=config['evaluation']['bootstrap_samples'],
                    confidence=config['evaluation']['confidence'], seed=config['seed'])
            _save_predictions(output / 'test' / 'source', before)
            _save_predictions(output / 'test' / 'final', after)
        report.update(schema_version=1, config_hash=result['config_hash'], smoke=result.get('smoke', False),
                      test_data_fingerprint=data_fingerprint(config, splits=('test',)),
                      note='Explicit final-test evaluation; no parameter selection or checkpoint updates.')
        atomic_json(output / 'test.json', report)
    return output / 'test.json'
