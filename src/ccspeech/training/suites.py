"""Small, inspectable experiment families; planning never launches training."""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import json

from .config import base_config, canonical_hash, merge_config, semantic_config, validate_config

FAMILIES = {"core", "modules", "placement", "timescales"}


def run_name(config: dict) -> str:
    fingerprint = canonical_hash(semantic_config(config))[:12]
    order = '-to-'.join(config['domains'])
    return f"{config['phase']}/{order}/{config['method']}-{config['updates']['policy']}-s{config['seed']}-{fingerprint}"


def expand_suite(spec: dict) -> list[dict]:
    if spec.get('schema_version') != 1 or spec.get('phase') not in {'text', 'speech'}:
        raise ValueError('Expected suite schema_version=1 and phase=text/speech')
    phase = spec['phase']
    base = merge_config(base_config(phase), spec.get('overrides', {}))
    domains = base['domains']
    orders = spec.get('orders', [domains, list(reversed(domains))])
    seeds = spec.get('seeds', [42, 43, 44])
    budgets = spec.get('memory_budgets', [32, 128] if phase == 'speech' else [256, 1024])
    families = spec.get('families', ['core'])
    if not families or set(families) - FAMILIES or not seeds or not budgets:
        raise ValueError('Empty or unknown suite settings')
    largest = max(budgets)
    variants = []
    for family in families:
        if family == 'core':
            for method in ('sequential', 'er', 'derpp', 'sparse_replay', 'mbpa', 'mbpa_random', 'joint', 'separate'):
                capacities = budgets if method in {'er', 'derpp', 'sparse_replay', 'mbpa', 'mbpa_random'} else [largest]
                for capacity in capacities:
                    variants.append((family, {'method': method, 'updates': {'policy': 'full'}, 'memory': {'capacity': capacity}}))
        elif family == 'modules':
            for method, policy in product(('sequential', 'derpp', 'mbpa'),
                                          ('classifier_only', 'encoder_lora_only', 'lora_classifier')):
                variants.append((family, {'method': method, 'updates': {'policy': policy}, 'memory': {'capacity': largest}}))
        elif family == 'placement':
            for method, placement in product(('derpp', 'mbpa'), ('early', 'late')):
                variants.append((family, {'method': method, 'memory': {'capacity': largest},
                    'updates': {'policy': 'lora_classifier', 'placement': placement, 'num_layers': 6 if phase == 'speech' else 3}}))
        else:
            for method, intervals in product(('derpp', 'mbpa'), ((1, 1), (1, 4), (4, 4))):
                variants.append((family, {'method': method, 'memory': {'capacity': largest},
                    'updates': {'policy': 'lora_classifier'},
                    'timescales': {'head_interval': intervals[0], 'encoder_interval': intervals[1]}}))
    unique = {}
    for (family, variant), order, seed in product(variants, orders, seeds):
        config = merge_config(base, variant)
        config['domains'], config['seed'], config['experiment_family'] = list(order), seed, family
        validate_config(config)
        unique.setdefault(canonical_hash(semantic_config(config)), config)
    return list(unique.values())


def load_suite(path: Path) -> list[dict]:
    return expand_suite(json.loads(path.read_text()))


def write_plan(configs: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, config in enumerate(configs):
        filename = f'{index:04d}.json'
        (output / filename).write_text(json.dumps(config, indent=2) + '\n')
        entries.append({'index': index, 'config': filename, 'run': run_name(config),
                        'family': config.get('experiment_family'), 'phase': config['phase'],
                        'method': config['method'], 'seed': config['seed'], 'domains': config['domains']})
    manifest = {'schema_version': 1, 'count': len(entries), 'runs': entries,
                'notice': 'Planning only. No model downloads, GPU provisioning, or training has started.'}
    (output / 'plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
