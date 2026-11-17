"""Drive download selection must match the requested portable release."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def download_workspace(tmp_path):
    repo = tmp_path / 'project with spaces'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'configs/data').mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / 'scripts/download_from_drive.sh'
    shutil.copyfile(original, repo / 'scripts/download_from_drive.sh')
    for version in (2, 3):
        manifest = {'bundles': [
            {'dataset': 'text', 'file': 'text_data_v1.zip'},
            {'dataset': 'taukadial', 'file': 'taukadial_data_v1.zip'},
            {'dataset': f'speech_v{version}', 'file': f'speech_data_v{version}.zip'},
        ]}
        (repo / f'configs/data/release_v{version}.json').write_text(json.dumps(manifest))
    fake_bin = tmp_path / 'bin'; fake_bin.mkdir()
    fake = fake_bin / 'rclone'
    fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
filters = Path(args[args.index('--include-from') + 1]).read_text().splitlines()
Path(os.environ['DOWNLOAD_TEST_CAPTURE']).write_text(json.dumps({'args': args, 'filters': filters}))
''')
    fake.chmod(0o700)
    capture = tmp_path / 'download.json'
    env = {**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH'],
           'DOWNLOAD_TEST_CAPTURE': str(capture)}
    return repo, env, capture


@pytest.mark.parametrize(('arguments', 'version', 'expected'), [
    (['authorized'], 3, {'text_data_v1.zip', 'speech_data_v3.zip'}),
    (['authorized', '3', 'speech_v3'], 3, {'speech_data_v3.zip'}),
    (['authorized', '2', 'speech_v2'], 2, {'speech_data_v2.zip'}),
])
def test_only_requested_release_bundles_are_downloaded(download_workspace, arguments, version, expected):
    repo, env, capture = download_workspace
    result = subprocess.run(['bash', str(repo / 'scripts/download_from_drive.sh'), *arguments],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    call = json.loads(capture.read_text())
    assert call['args'][0:2] == ['copy', 'authorized:']
    assert call['args'][2] == str(repo / f'artifacts/data_v{version}')
    assert set(call['filters']) == {'/' + name for name in expected} | {
        f'/SHA256SUMS_v{version}', f'/README_DATA_v{version}.md'}
    assert '1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp' in call['args']


def test_unavailable_dataset_fails_before_contacting_drive(download_workspace):
    repo, env, capture = download_workspace
    result = subprocess.run(['bash', str(repo / 'scripts/download_from_drive.sh'),
                             'authorized', '2', 'speech_v3'], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not capture.exists()
