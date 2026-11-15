"""Thin command-line entry points; planning does not import model dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _configuration(args):
    from .training.config import read_config
    config = read_config(args.config)
    if getattr(args, 'device', None):
        config['runtime']['device'] = args.device
    if getattr(args, 'data_root', None):
        config['data']['root'] = str(args.data_root)
    return config


def _indices(value: str, size: int) -> list[int]:
    selected = set()
    for item in value.split(','):
        if '-' in item:
            first, last = map(int, item.split('-'))
            selected.update(range(first, last+1))
        else:
            selected.add(int(item))
    if not selected or min(selected) < 0 or max(selected) >= size:
        raise ValueError('Run indices are outside this plan')
    return sorted(selected)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Reproducible text/speech continual learning')
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan', help='Expand a suite into explicit configs, without launching jobs')
    plan.add_argument('--suite', required=True, type=Path)
    plan.add_argument('--output', required=True, type=Path)
    run = commands.add_parser('run', help='Train/select on validation; never access test labels')
    run.add_argument('--config', required=True, type=Path)
    run.add_argument('--output-root', type=Path, default=Path('runs'))
    run.add_argument('--data-root', type=Path)
    run.add_argument('--device')
    run.add_argument('--resume', action='store_true')
    batch = commands.add_parser('run-plan', help='Sequentially run selected plan entries')
    batch.add_argument('--plan', type=Path, required=True)
    selection = batch.add_mutually_exclusive_group(required=True)
    selection.add_argument('--indices', help='For example: 0,2,5-8')
    selection.add_argument('--all', action='store_true')
    batch.add_argument('--output-root', type=Path, default=Path('runs'))
    batch.add_argument('--data-root', type=Path)
    batch.add_argument('--device')
    batch.add_argument('--resume', action='store_true')
    test = commands.add_parser('final-test', help='Explicit one-time held-out scoring after settings are fixed')
    test.add_argument('--run-dir', type=Path, required=True)
    test.add_argument('--data-root', type=Path)
    test.add_argument('--device')
    inspect = commands.add_parser('preflight', help='Verify prepared data and config, without downloading models')
    inspect.add_argument('--config', type=Path, required=True)
    inspect.add_argument('--data-root', type=Path)
    inspect.add_argument('--device')
    inspect.add_argument('--load-model', action='store_true', help='Also download/load the configured pretrained model')
    summary = commands.add_parser('summarize', help='Export validation or final-test results to CSV')
    summary.add_argument('--runs-root', type=Path, default=Path('runs'))
    summary.add_argument('--output', type=Path, default=Path('runs/summary.csv'))
    summary.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        from .training.suites import load_suite, write_plan
        result = write_plan(load_suite(args.suite), args.output)
        print(json.dumps({'count': result['count'], 'plan': str(args.output / 'plan.json'), 'notice': result['notice']}, indent=2))
    elif args.command == 'run':
        from .training.engine import run_experiment
        print(run_experiment(_configuration(args), output_root=args.output_root, resume=args.resume))
    elif args.command == 'run-plan':
        from .training.engine import run_experiment
        from .training.config import read_config
        plan_data = json.loads(args.plan.read_text())
        entries = plan_data['runs']
        if plan_data.get('schema_version') != 1 or plan_data['count'] != len(entries):
            raise ValueError('Invalid plan')
        selected = range(len(entries)) if args.all else _indices(args.indices, len(entries))
        for index in selected:
            relative = Path(entries[index]['config'])
            if relative.name != str(relative):
                raise ValueError('Plan configs must be basenames in the plan directory')
            config = read_config(args.plan.parent / relative)
            if args.device:
                config['runtime']['device'] = args.device
            if args.data_root:
                config['data']['root'] = str(args.data_root)
            print(f'Plan run {index+1}/{len(entries)}', flush=True)
            run_experiment(config, output_root=args.output_root, resume=args.resume)
    elif args.command == 'final-test':
        from .training.engine import final_test
        print(final_test(args.run_dir, data_root=str(args.data_root) if args.data_root else None, device=args.device))
    elif args.command == 'preflight':
        from .training.engine import _device, _initial_models, audit_split_separation, data_fingerprint, load_split
        from .models import configure_updates
        config = _configuration(args)
        fingerprint = data_fingerprint(config)
        train, validation = load_split(config, 'train'), load_split(config, 'validation')
        audit_split_separation(train, validation, config['phase'] == 'speech')
        result = {'data_fingerprint': fingerprint, 'records': {split: {d: len(rows) for d, rows in domains.items()}
                  for split, domains in [('train', train), ('validation', validation)]},
                  'test_labels_read': False, 'smoke': config.get('runtime', {}).get('smoke', False)}
        if args.load_model:
            model, _, _ = _initial_models(config, _device(config))
            result['model'] = model.metadata()
            result['updates'] = configure_updates(model, config['updates'])
        print(json.dumps(result, indent=2))
    else:
        from .evaluation.summarize import summarize
        print(json.dumps(summarize(args.runs_root, args.output, args.split), indent=2))


if __name__ == '__main__':
    main()
