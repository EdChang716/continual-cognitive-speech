"""Synthetic transcript coverage, provenance, and held-out-label isolation."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from ccspeech.data.speech import match_word_timestamps, window_bounds
from ccspeech.data.transcript_audit import (compare_text, normalize_units, run_audit,
                                          summarize_records, window_word_indices)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as source:
        writer = csv.DictWriter(source, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def audit_inputs(tmp_path):
    root = tmp_path / 'data'
    references, parents, resolutions = [], [], []
    orig_by_split, window_by_split = {}, {}
    definitions = [('en-a', 'en', 'train', 41, 'hello uh hello there', True),
                   ('en-b', 'en', 'validation', 20, 'hello world', False),
                   ('zh-c', 'zh', 'test', 10, '學習', True)]
    words = {'en-a': [{'word': 'hello', 'start': 1., 'end': 2.}, {'word': ' uh', 'start': 2., 'end': 3.},
                      {'word': ' hello', 'start': 28., 'end': 29.}, {'word': ' there', 'start': 38., 'end': 39.}],
             'en-b': [{'word': 'hello', 'start': 1., 'end': 2.}, {'word': ' world', 'start': 3., 'end': 4.}],
             'zh-c': [{'word': '學習', 'start': 1., 'end': 2.}]}
    for rid, language, split, duration, text, verified in definitions:
        pid = rid + '-person'
        original_hash = hashlib.sha256((rid+'original').encode()).hexdigest()
        normalized_hash = hashlib.sha256((rid+'normalized').encode()).hexdigest()
        candidate_hash = hashlib.sha256((rid+'candidate').encode()).hexdigest()
        original = {'recording_id': rid, 'participant_id': pid, 'language': language, 'split': split, 'sha256': original_hash}
        if split != 'test':
            original['label'] = 0
        orig_by_split[split] = [original]
        human_path = tmp_path / 'human' / f'{rid}.txt'
        human_path.parent.mkdir(exist_ok=True)
        human_path.write_text(text)
        reference = {'recording_id': rid, 'participant_id': pid, 'language': language,
                     'source_dataset': 'Chou' if language == 'zh' else 'Delaware',
                     'original_wav_sha256': original_hash, 'normalized_wav_sha256': normalized_hash,
                     'reference_participant': text, 'reference_scope_review_complete': verified,
                     'source_transcript_path': str(human_path), 'source_transcript_sha256': _digest(human_path)}
        references.append(reference)
        if language == 'en':
            resolutions.append({'recording_id': rid, 'reference': text, 'scope_review_complete': verified})
        parent = {'recording_id': rid, 'participant_id': pid, 'language': language,
                  'original_wav_sha256': original_hash, 'sha256': candidate_hash}
        parents.append(parent)
        source = {**parent, 'normalized_original_sha256': normalized_hash, 'sample_rate': 16000,
                  'duration_seconds': duration, 'num_samples': duration * 16000,
                  'words': words[rid], 'quality_flags': ['speaker_purity_not_manually_verified']}
        _json(root / 'sources/speech_v2/records' / f'{rid}.json', source)
        _json(tmp_path / 'raw' / f'{rid}.json', {'recording_id': rid, 'language': language,
              'source_sha256': normalized_hash, 'reference_blind': True, 'text': text})
        rows = []
        for index, (a, b) in enumerate(window_bounds(duration*16000)[0]):
            start, end = a/16000, b/16000
            row = {'recording_id': rid, 'participant_id': pid, 'language': language, 'split': split,
                   'segment_id': f'{rid}-{index}', 'start_seconds': start, 'end_seconds': end,
                   'source_sha256': candidate_hash, 'original_wav_sha256': original_hash,
                   'matched_transcript': match_word_timestamps(words[rid], start, end, duration)['matched_transcript']}
            if split != 'test':
                row['label'] = 0
            rows.append(row)
        window_by_split[split] = rows
    split_hashes = {}
    orig_fields = ['recording_id', 'participant_id', 'language', 'split', 'sha256']
    window_fields = ['recording_id', 'participant_id', 'language', 'split', 'segment_id', 'start_seconds',
                     'end_seconds', 'source_sha256', 'original_wav_sha256', 'matched_transcript']
    for split in ('train', 'validation', 'test'):
        labels = [] if split == 'test' else ['label']
        path = root / 'processed/taukadial' / f'{split}.csv'
        _csv(path, orig_by_split[split], orig_fields+labels)
        split_hashes[split] = _digest(path)
        _csv(root / 'processed/speech_v2' / f'{split}.csv', window_by_split[split], window_fields+labels)
    manifest, resolution_path = tmp_path / 'references.jsonl', tmp_path / 'resolutions.jsonl'
    _jsonl(manifest, references)
    _jsonl(resolution_path, resolutions)
    _jsonl(root / 'sources/speech_v2/manifest.jsonl', parents)
    model = {'repo': 'synthetic-whisper', 'revision': 'test-revision'}
    _json(root / 'sources/speech_v2/provenance.json', {'original_reference_manifest_sha256': _digest(manifest),
          'split_manifest_sha256': split_hashes, 'asr_runs': [{'settings': {'model': model}}]})
    _json(tmp_path / 'raw/run.json', {'model': model, 'reference_blind': True, 'word_timestamps': True})
    return {'data_root': root, 'reference_manifest': manifest, 'resolutions': resolution_path,
            'raw_asr_dir': tmp_path / 'raw', 'output': root / 'reports/transcript_comparison_v1'}


def test_normalization_preserves_repetitions_fillers_and_unicode():
    assert normalize_units("<I I> [/] &-um don't ɹɛd &=laughs", 'en') == ['i', 'i', 'um', "don't", 'ɹɛd']
    assert normalize_units('學習，學習！', 'zh') == list('学习学习')
    counts, _ = compare_text('學習學習', '学习', 'zh')
    assert counts['D'] == 2 and counts['S'] == counts['I'] == 0


def test_overlap_is_union_of_indices_not_word_strings():
    words = [{'word': ' yes', 'start': 28., 'end': 29.}, {'word': ' yes', 'start': 30., 'end': 31.}]
    assert window_word_indices(words, [(0., 30.), (27., 57.)], 60.) == [0, 1]


def test_full_audit_counts_tail_and_whole_omissions_without_reading_test_labels(audit_inputs, monkeypatch):
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert 'locked' not in path.parts, 'Read a locked diagnosis file'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    report = run_audit(**audit_inputs)
    assert report['recordings'] == 3 and report['whole_recordings_omitted_by_window_policy'] == 1
    assert report['verified_reference_recordings'] == 2 and report['unverified_reference_recordings'] == 1
    assert report['provenance']['human_source_files']['verified_original_files'] == 3
    details = {row['recording_id']: row for row in map(json.loads, (audit_inputs['output'] / 'differences.jsonl').read_text().splitlines())}
    assert details['en-a']['counts']['window']['D'] == 1
    assert details['en-a']['counts']['candidate']['D'] == 0
    assert details['zh-c']['window_asr'] == '' and details['zh-c']['counts']['window']['D'] == 2
    assert details['en-a']['counts']['raw']['normalized_reference_sha256'] == details['en-a']['counts']['window']['normalized_reference_sha256']
    assert len([group for group in report['groups'] if group['split'] == 'all']) == 3
    assert str(audit_inputs['data_root'].parent) not in (audit_inputs['output'] / 'summary.json').read_text()
    assert run_audit(**audit_inputs) == report


def test_25_unverified_english_remain_separate_summary_population():
    counts, _ = compare_text('hello there', 'hello', 'en')
    rows = [{'recording_id': str(i), 'participant_id': str(i), 'language': 'en', 'split': 'train',
             'reference_scope_verified': i < 221, 'counts': {key: counts for key in ('raw', 'candidate', 'window')}}
            for i in range(246)]
    groups = [row for row in summarize_records(rows) if row['split'] == 'all']
    assert {row['reference_scope_verified']: row['recordings'] for row in groups} == {True: 221, False: 25}


@pytest.mark.parametrize('failure', ['reference_hash', 'participant_identity', 'original_hash', 'raw_hash', 'window_text', 'raw_model', 'human_source'])
def test_provenance_or_content_mismatch_rejected(audit_inputs, failure):
    root = audit_inputs['data_root']
    if failure == 'reference_hash':
        with audit_inputs['reference_manifest'].open('a') as f:
            f.write('\n')
    elif failure in {'participant_identity', 'original_hash'}:
        path = root / 'sources/speech_v2/records/en-a.json'
        value = json.loads(path.read_text())
        value['participant_id' if failure == 'participant_identity' else 'original_wav_sha256'] = 'wrong'
        _json(path, value)
    elif failure == 'raw_hash':
        path = audit_inputs['raw_asr_dir'] / 'en-a.json';value = json.loads(path.read_text())
        value['source_sha256'] = 'wrong';_json(path, value)
    elif failure == 'raw_model':
        path = audit_inputs['raw_asr_dir'] / 'run.json';value = json.loads(path.read_text())
        value['model']['revision'] = 'different';_json(path, value)
    elif failure == 'human_source':
        (root.parent / 'human/en-a.txt').write_text('changed reference')
    else:
        path = root / 'processed/speech_v2/train.csv'
        path.write_text(path.read_text().replace('hello uh hello', 'edited'))
    with pytest.raises(ValueError):
        run_audit(**audit_inputs)
    assert not audit_inputs['output'].exists()


def test_test_manifest_label_column_rejected_before_rows(audit_inputs):
    path = audit_inputs['data_root'] / 'processed/speech_v2/test.csv'
    text = path.read_text().rstrip('\n') + ',label\n'
    path.write_text(text)
    with pytest.raises(ValueError, match='diagnoses'):
        run_audit(**audit_inputs)


def test_missing_human_source_retains_identical_pinned_reference(audit_inputs):
    path = audit_inputs['data_root'].parent / 'human/en-a.txt'
    path.unlink()
    report = run_audit(**audit_inputs)
    assert report['provenance']['human_source_files']['archived_reference_fallback_files'] == 1
    assert report['recordings'] == 3
