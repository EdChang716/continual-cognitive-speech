"""Same-scope reference agreement, piecewise clocks, and provenance checks."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from ccspeech.data.source_cut_audit import mapped_words, retention, run_audit, select_window


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_piecewise_word_mapping_does_not_invent_contiguous_source_span():
    mapping = [{'source_start': 0, 'source_end': 2, 'clean_start': 0, 'clean_end': 2},
               {'source_start': 5, 'source_end': 8, 'clean_start': 2, 'clean_end': 5}]
    words = mapped_words([{'start': 1.5, 'end': 2.5, 'word': '甲'},
                          {'start': 3, 'end': 4, 'word': '甲'}], mapping)
    assert words[0]['source_midpoint'] == 5
    assert words[0]['crosses_concatenation_seam']
    assert [(p['original_start_seconds'], p['original_end_seconds']) for p in words[0]['source_pieces']] == [(1.5, 2), (5, 5.5)]
    assert not words[1]['crosses_concatenation_seam']
    assert len(select_window(words, 4, 8, 8)) == 2  # repeated real words survive


def test_audio_coverage_unions_overlapping_windows():
    report = retention([(0, 30), (27, 57)], [(0, 20), (25, 50)])
    assert report['reference_audio_seconds'] == 57
    assert report['retained_audio_seconds'] == 45
    assert report['omitted_original_intervals'] == [(20, 25), (50, 57)]
    assert retention([], [])['retained_audio_fraction'] is None


@pytest.fixture
def inputs(tmp_path):
    root, old, asr, final = tmp_path/'data', tmp_path/'legacy', tmp_path/'asr/raw', tmp_path/'final_tasks'
    source_csv = tmp_path/'final_windows.csv'
    word = lambda text, a: {'word': text, 'start': a, 'end': a+.5}
    model = {'repo': 'test-whisper', 'revision': 'pinned'}
    settings = {'model': model, 'reference_blind': True, 'word_timestamps': True, 'segment_seconds': 30, 'overlap_seconds': 3}
    _json(asr/'run.json', settings)
    parents, cross, splits, final_splits, split_hashes = [], [], {}, {}, {}
    for rid, split, duration, full in [('zh-a', 'train', 40, '甲乙丙丁'), ('zh-short', 'test', 10, '甲乙')]:
        original_hash, normalized_hash, candidate_hash = rid+'original', rid+'normalized', rid+'candidate'
        pid = rid+'person'
        words = [word('甲', 1), word('乙', 21), word('丙', 28), word('丁', 35)] if rid == 'zh-a' else [word('甲', 1), word('乙', 3)]
        candidate_words = [word('甲', 1), word('丙', 23), word('丁', 30)] if rid == 'zh-a' else words
        mapping = ([{'source_start': 0, 'source_end': 20, 'clean_start': 0, 'clean_end': 20},
                    {'source_start': 25, 'source_end': 40, 'clean_start': 20, 'clean_end': 35}] if rid == 'zh-a'
                   else [{'source_start': 0, 'source_end': 10, 'clean_start': 0, 'clean_end': 10}])
        parent = {'recording_id': rid, 'participant_id': pid, 'language': 'zh', 'sha256': candidate_hash,
                  'original_wav_sha256': original_hash}
        parents.append(parent)
        _json(asr/f'{rid}.json', {'recording_id': rid, 'language': 'zh', 'source_sha256': normalized_hash,
                                'reference_blind': True, 'text': full, 'words': words})
        candidate_path = asr.parent/'candidate'/f'{rid}.json'
        _json(candidate_path, {'recording_id': rid, 'language': 'zh', 'source_sha256': candidate_hash,
                              'reference_blind': True, 'words': candidate_words})
        _json(root/'sources/speech_v2/records'/f'{rid}.json', {**parent, 'sample_rate': 16000,
              'num_samples': (35 if rid == 'zh-a' else 10)*16000, 'normalized_original_num_samples': duration*16000,
              'normalized_original_sha256': normalized_hash, 'candidate_pcm_matches_normalized_original_intervals': True,
              'source_asr_sha256': _sha(candidate_path), 'words': candidate_words, 'time_map': mapping})
        splits[split] = [{'recording_id': rid, 'participant_id': pid, 'language': 'zh', 'split': split,
                          'official_split': split, 'sha256': original_hash}]
        cross.append({'tauk_recording': rid, 'tauk_participant_id': pid, 'source_dataset': 'Chou', 'source_recording': rid})
        final_splits[split] = [{'dataset': 'Chou', 'audio_path': f'Chou/{rid}.wav', 'transcript': full}]
    for split in ('train', 'validation', 'test'):
        path = root/'processed/taukadial'/f'{split}.csv'
        _csv(path, splits.get(split, []), ['recording_id', 'participant_id', 'language', 'split', 'official_split', 'sha256'])
        split_hashes[split] = _sha(path)
        _csv(final/f'{split}.csv', final_splits.get(split, []), ['dataset', 'audio_path', 'transcript'])
    _jsonl(root/'sources/speech_v2/manifest.jsonl', parents)
    _json(root/'sources/speech_v2/provenance.json', {'split_manifest_sha256': split_hashes,
          'candidate_pcm_verified_against_normalized_original': True,
          'asr_runs': [{'source_run': 'candidate', 'settings': settings}]})
    links, inv, checks, current = [], [], [], []
    for index, seconds, text in [(0, 30, '甲乙丙'), (1, 13, '丙丁')]:
        sid = f'zh-a_seg{index:03d}'
        links.append({'segment_id': sid, 'tauk_recording': 'zh-a', 'tauk_split': 'train', 'source_recording': 'zh-a',
                      'source_segment_index': str(index), 'transcript_csv': str(source_csv),
                      'matched_transcript': text, 'whisper_transcription': text, 'match_method': 'whisper_llm_v6'})
        inv.append({'segment_id': sid, 'recording_id': 'zh-a', 'old_csv': str(source_csv), 'sample_count_matches': True,
                    'actual_samples': seconds*16000, 'declared_samples': str(seconds*16000), 'rate': 16000})
        checks.append({'segment_id': sid, 'direct_27s_hop_correlation': .999, 'duration_difference_seconds': 0})
        current.append({'dataset': 'Chou', 'segment_id': sid, 'recording_id': 'zh-a', 'segment_index': str(index),
                        'transcript': '甲乙丙丁', 'matched_transcript': text, 'whisper_transcription': text, 'match_method': 'whisper_llm_v6'})
    _csv(source_csv, current, list(current[0]))
    _csv(old/'chou_existing_segment_links.csv', links, list(links[0]))
    _csv(old/'recording_crosswalk_complete.csv', cross, list(cross[0]))
    _json(old/'chou_segment_inventory.json', inv)
    _json(old/'chou_segment_waveform_checks.json', checks)
    return dict(data_root=root, legacy_audit_root=old, raw_asr_dir=asr, final_task_splits=final,
                output=root/'reports/source_cut_reference_v1')


def test_primary_all_recordings_secondary_same_scope_and_no_locked_labels(inputs, monkeypatch):
    original_open = Path.open
    def guard(path, *args, **kwargs):
        assert 'locked' not in path.parts
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    summary = run_audit(**inputs)
    assert summary['final_full_reference_recordings'] == 2
    assert summary['recordings_with_prior_windows'] == 1
    assert summary['prior_windows'] == 2
    assert summary['unique_audio_coverage']['unique_prior_window_audio']['reference_audio_seconds'] == 40
    train = [r for r in summary['primary_full_recording_final_reference'] if r['split'] == 'train']
    assert {r['condition']: r['CER'] for r in train} == {'current_raw': 0, 'current_candidate': .25}
    secondary = [r for r in summary['secondary_overlapping_window_reference_agreement'] if r['split'] == 'all']
    assert all(r['reference_units'] == 5 for r in secondary)
    assert run_audit(**inputs) == summary


@pytest.mark.parametrize('failure', ['empty_reference', 'raw_hash', 'candidate_hash', 'changed_window', 'inconsistent_full_text'])
def test_reference_and_provenance_fail_before_output(inputs, failure):
    if failure in {'raw_hash', 'candidate_hash'}:
        path = inputs['raw_asr_dir']/('zh-a.json') if failure == 'raw_hash' else inputs['raw_asr_dir'].parent/'candidate/zh-a.json'
        value = json.loads(path.read_text()); value['source_sha256'] = 'wrong'; _json(path, value)
    elif failure == 'empty_reference':
        path = inputs['final_task_splits']/'test.csv'
        _csv(path, [{'dataset': 'Chou', 'audio_path': 'Chou/zh-short.wav', 'transcript': ''}], ['dataset', 'audio_path', 'transcript'])
    else:
        path = inputs['data_root'].parent/'final_windows.csv'
        with path.open() as source: rows = list(csv.DictReader(source))
        rows[0]['matched_transcript' if failure == 'changed_window' else 'transcript'] = '改動'
        _csv(path, rows, list(rows[0]))
    with pytest.raises(ValueError): run_audit(**inputs)
    assert not inputs['output'].exists()
