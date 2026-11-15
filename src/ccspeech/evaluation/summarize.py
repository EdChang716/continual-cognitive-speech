"""Export comparable per-run results and separate repeated-seed summaries."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as output:
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['status']
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, output: Path, split: str = 'validation') -> dict:
    from ccspeech.training.config import canonical_hash, semantic_config
    if split not in {'validation', 'test'}:
        raise ValueError('Unknown summary split')
    rows, groups = [], defaultdict(list)
    for path in sorted(root.rglob('result.json')):
        result = json.loads(path.read_text())
        config = result['config']
        if split == 'test':
            if not (path.parent / 'test.json').exists():
                continue
            report = json.loads((path.parent / 'test.json').read_text())
            final = report['final']
            continual = report.get('continual', {})
        else:
            final = result['validation']['final']
            continual = result.get('continual', {})
        settings = semantic_config(config)
        settings.pop('seed')
        group = canonical_hash(settings)[:12]
        row = {'run': str(path.parent), 'group': group, 'split': split, 'phase': config['phase'],
               'method': config['method'], 'order': '->'.join(config['domains']), 'seed': config['seed'],
               'updates': config['updates']['policy'], 'placement': config['updates'].get('placement', 'all'),
               'memory_capacity': result.get('memory', {}).get('capacity', 0 if config['method'] in {'joint', 'separate'} else config['memory']['capacity']),
               'smoke': result.get('smoke', False),
               'head_interval': config['timescales']['head_interval'], 'encoder_interval': config['timescales']['encoder_interval'],
               'domain_average_accuracy': final['selection_score'], 'wall_seconds': result['wall_seconds'],
               'forgetting': continual.get('mean_forgetting')}
        for domain, report in final['domains'].items():
            for key in ('accuracy', 'macro_f1', 'balanced_accuracy', 'auroc', 'n'):
                row[f'{domain}_{key}'] = report['metrics'][key]
            row[f'{domain}_inference_seconds'] = report['cost'].get('prediction_wall_seconds')
        rows.append(row)
        groups[group].append(row)
    _write(output, rows)
    aggregated = []
    metadata_keys = {'phase', 'method', 'order', 'updates', 'placement', 'memory_capacity', 'smoke', 'head_interval', 'encoder_interval'}
    for group, values in groups.items():
        if len({row['seed'] for row in values}) != len(values):
            raise ValueError('Duplicate seeds in one scientific configuration; avoid combining copied run directories')
        entry = {'group': group, 'seeds': len(values), 'seed_ids': ','.join(str(row['seed']) for row in values)}
        entry.update({key: values[0][key] for key in metadata_keys})
        for key in values[0]:
            if key in metadata_keys | {'seed', 'run', 'group', 'split'}:
                continue
            numbers = [row[key] for row in values if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)]
            if numbers:
                entry[f'{key}_mean'] = statistics.mean(numbers)
                entry[f'{key}_std'] = statistics.stdev(numbers) if len(numbers) > 1 else None
                entry[f'{key}_valid_seeds'] = len(numbers)
        aggregated.append(entry)
    by_seed = output.with_name(output.stem + '_seeds.csv')
    _write(by_seed, aggregated)
    return {'runs': len(rows), 'groups': len(groups), 'per_run': str(output), 'seed_summary': str(by_seed),
            'note': 'Smoke rows are explicitly flagged. Seed SD is not a participant bootstrap interval.'}
