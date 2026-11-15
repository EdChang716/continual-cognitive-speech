"""Scientific run identity, explicit planning, and repeated-seed summaries."""
from copy import deepcopy
import csv
import json

import pytest

from ccspeech.cli import _indices, main
from ccspeech.evaluation.summarize import summarize
from ccspeech.training.config import base_config, canonical_hash, semantic_config, validate_config
from ccspeech.training.suites import expand_suite, run_name


@pytest.mark.parametrize('phase', ['text', 'speech'])
def test_all_suite_comparisons_are_unique_and_fixed_label_domains(phase):
    configurations = expand_suite({'schema_version': 1, 'phase': phase,
                                  'families': ['core', 'modules', 'placement', 'timescales']})
    fingerprints = [canonical_hash(semantic_config(config)) for config in configurations]
    assert len(fingerprints) == len(set(fingerprints))
    assert {c['method'] for c in configurations} == {'sequential', 'er', 'derpp', 'sparse_replay', 'mbpa', 'mbpa_random', 'joint', 'separate'}
    assert {c['seed'] for c in configurations} == {42, 43, 44}
    assert len({tuple(c['domains']) for c in configurations}) == 2
    assert {c['memory']['capacity'] for c in configurations} == ({32, 128} if phase == 'speech' else {256, 1024})
    for config in configurations:
        validate_config(config)
    placement = [c for c in configurations if c['experiment_family'] == 'placement']
    assert {c['updates']['placement'] for c in placement} == {'early', 'late'}
    assert len({c['updates']['num_layers'] for c in placement}) == 1


def test_portable_identity_but_scientific_changes_create_new_runs():
    base = base_config('text')
    moved = deepcopy(base)
    moved['data']['root'] = '/another/machine/data'
    moved['runtime']['device'] = 'cpu'
    moved['model']['cache_dir'] = '/new/cache'
    assert run_name(base) == run_name(moved)
    moved['memory']['capacity'] = 3
    assert run_name(base) != run_name(moved)
    base['runtime']['max_steps_per_stage'] = 2
    with pytest.raises(ValueError, match='smoke'):
        validate_config(base)


def test_plan_cli_writes_configs_without_training(tmp_path, capsys):
    spec = tmp_path / 'suite.json'
    spec.write_text(json.dumps({'schema_version': 1, 'phase': 'speech', 'families': ['core'], 'seeds': [42]}))
    main(['plan', '--suite', str(spec), '--output', str(tmp_path / 'plan')])
    response = json.loads(capsys.readouterr().out)
    assert response['count'] == 26
    assert not list(tmp_path.rglob('*.pt'))
    plan = json.loads((tmp_path / 'plan/plan.json').read_text())
    assert len(plan['runs']) == 26
    assert _indices('0,2,4-6', 26) == [0, 2, 4, 5, 6]
    with pytest.raises(ValueError):
        _indices('26', 26)


def test_seed_summary_keeps_methods_and_orders_separate(tmp_path):
    for seed, score in [(42, .5), (43, .75)]:
        config = base_config('text')
        config['seed'] = seed
        directory = tmp_path / f'run{seed}'
        directory.mkdir()
        metrics = {'accuracy': score, 'macro_f1': score, 'balanced_accuracy': score, 'auroc': None, 'n': 10}
        final = {'selection_score': score, 'domains': {domain: {'metrics': metrics,
                 'cost': {'prediction_wall_seconds': 2.}} for domain in config['domains']}}
        (directory / 'result.json').write_text(json.dumps({'config': config, 'validation': {'final': final},
                 'wall_seconds': 5., 'memory': {'capacity': 1024}, 'smoke': False}))
    output = tmp_path / 'summary.csv'
    result = summarize(tmp_path, output)
    assert result['runs'] == 2 and result['groups'] == 1
    with output.open() as source:
        rows = list(csv.DictReader(source))
    assert rows[0]['amazon_inference_seconds'] == '2.0'
    with (tmp_path / 'summary_seeds.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert float(rows[0]['domain_average_accuracy_mean']) == .625
    assert rows[0]['seeds'] == '2'
    assert 'amazon_auroc_mean' not in rows[0]
