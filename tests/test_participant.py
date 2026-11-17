"""Frozen participant recovery, versioned PCM, and fresh-ASR finalization."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ccspeech.data.participant import (VOICE_THRESHOLD, build, finalize, make_time_map,
                                      recovery_intervals)
from ccspeech.data.asr import MODEL_REPO, MODEL_FILES
from ccspeech.data.speech import ASR_REVISION, json_bytes, read_pcm, verify_candidate_pcm


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def _jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words():
    return [{'word': text, 'start': start, 'end': start+.3} for text, start in zip('甲乙丙丁戊己', [1, 3, 4, 5, 8, 9])]


def _role():
    return {'voice_calibration': {'threshold': VOICE_THRESHOLD, 'split_used': 'train_only', 'test_used_for_threshold': False},
            'embedding_windows': [{'start': a, 'end': b, 'role': 'PAR', 'gallery_top3_similarity': .7}
                                  for a, b in [(1, 6), (2, 7)]],
            'speech_intervals': [{'start': 0, 'end': 2, 'role': 'PAR'}, {'start': 2, 'end': 7, 'role': 'unknown'},
                                 {'start': 7, 'end': 10, 'role': 'PAR'}]}


def test_long_anchor_and_unanimous_acoustics_are_both_required():
    kept = [(0, 2), (7, 10)]
    updated, approvals = recovery_intervals(_words(), '甲乙丙丁戊己', _role(), kept, 10)
    assert len(approvals) == 3
    assert approvals[0]['source_start'] == pytest.approx(2.85)
    assert approvals[0]['source_end'] == pytest.approx(3.45)
    assert updated[0] == (0, 2) and updated[-1] == (7, 10)
    assert recovery_intervals(_words(), '甲乙丙丁戊', _role(), kept, 10) == (kept, [])
    role = _role(); role['embedding_windows'].append({'start': 3.9, 'end': 4.4, 'role': 'unknown', 'gallery_top3_similarity': .1})
    _, approvals = recovery_intervals(_words(), '甲乙丙丁戊己', role, kept, 10)
    assert [r['raw_word_index'] for r in approvals] == [1, 3]
    role = _role(); role['embedding_windows'] = role['embedding_windows'][:1]
    assert recovery_intervals(_words(), '甲乙丙丁戊己', role, kept, 10) == (kept, [])


def test_frozen_threshold_and_support_boundaries():
    role = _role(); role['voice_calibration']['threshold'] = .3
    with pytest.raises(ValueError, match='Frozen'): recovery_intervals(_words(), '甲乙丙丁戊己', role, [(0, 2)], 10)
    role = _role(); role['embedding_windows'][1]['start'] = 3.1
    _, approvals = recovery_intervals(_words(), '甲乙丙丁戊己', role, [(0, 2), (7, 10)], 10)
    assert 1 not in [r['raw_word_index'] for r in approvals]  # whole word, not just midpoint


@pytest.fixture
def inputs(tmp_path):
    root, legacy, normalized = tmp_path/'data', tmp_path/'legacy', tmp_path/'normalized'
    prior = root/'sources/speech_v2'
    parents, references, resolutions, split_hashes = [], [], [], {}
    for index, language, split in [(1, 'zh', 'train'), (2, 'en', 'validation'), (3, 'en', 'test')]:
        rid, pid = f'taukdial-{index:03d}-1', f'taukdial-{index:03d}'
        samples = np.arange(160000, dtype=np.int16)
        normalized.mkdir(exist_ok=True)
        sf.write(normalized/f'{rid}.wav', samples, 16000, subtype='PCM_16')
        intervals = [(0, 2), (7, 10)] if language == 'zh' else [(0, 10)]
        mapping = make_time_map(intervals)
        candidate = np.concatenate([samples[round(a*16000):round(b*16000)] for a, b in intervals])
        audio = prior/'audio'/f'{rid}.wav'; audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio, candidate, 16000, subtype='PCM_16')
        original_hash = rid+'original'
        parent = {'recording_id': rid, 'participant_id': pid, 'language': language, 'audio_path': f'audio/{rid}.wav',
                  'sha256': _sha(audio), 'original_wav_sha256': original_hash}
        item = {**parent, 'sample_rate': 16000, 'channels': 1, 'num_samples': len(candidate), 'duration_seconds': len(candidate)/16000,
                'normalized_original_num_samples': len(samples), 'normalized_original_sha256': _sha(normalized/f'{rid}.wav'),
                'candidate_pcm_matches_normalized_original_intervals': True, 'words': [{'word': '甲', 'start': 1, 'end': 1.3}],
                'time_map': mapping, 'source_asr_sha256': 'priorASR', 'source_sidecar_sha256': 'priorSidecar',
                'extraction_condition': 'old_PAR', 'quality_flags': [], 'task_scope_verified': index != 3}
        _json(prior/'records'/f'{rid}.json', item)
        reference = {**parent, 'normalized_wav_sha256': item['normalized_original_sha256'], 'reference_participant': '甲乙丙丁戊己'}
        parents.append(parent); references.append(reference)
        if language == 'zh':
            role = {**_role(), 'recording_id': rid, 'participant_id': pid, 'features_path': str(legacy/'features')}
            _json(legacy/'outputs/taukadial_asr/chinese_speaker_roles_calibrated'/f'{rid}.json', role)
            _json(legacy/'features'/f'{rid}.json', {'recording_id': rid, 'audio_sha256': item['normalized_original_sha256']})
            _json(legacy/'outputs/taukadial_asr/raw_whisper_v1'/f'{rid}.json', {'recording_id': rid, 'language': 'zh',
                  'source_sha256': item['normalized_original_sha256'], 'reference_blind': True, 'words': _words()})
        else:
            resolutions.append({'recording_id': rid, 'scope_review_complete': index==2, 'extraction_timing_complete': index==2})
        row = {'recording_id': rid, 'participant_id': pid, 'language': language, 'split': split, 'sha256': original_hash}
        path = root/'processed/taukadial'/f'{split}.csv'
        _csv(path, [row], list(row)); split_hashes[split] = _sha(path)
    ref_path = legacy/'data/taukadial_asr/reference_manifest.jsonl'
    _jsonl(ref_path, references)
    _jsonl(legacy/'data/taukadial_asr/reference_resolution.jsonl', resolutions)
    _json(legacy/'data/taukadial_asr/voice_calibration_train.json', _role()['voice_calibration'])
    _jsonl(prior/'manifest.jsonl', parents)
    _json(prior/'provenance.json', {'original_reference_manifest_sha256': _sha(ref_path),
          'split_manifest_sha256': split_hashes, 'asr_runs': [], 'source_recordings': 3})
    return {'data_root': root, 'legacy_root': legacy, 'normalized_originals': normalized, 'output': root/'sources/speech_v3'}


def test_build_preserves_all_audio_and_quarantines_unresolved_english(inputs, monkeypatch):
    original_open = Path.open
    def guard(path, *args, **kwargs):
        assert 'locked' not in path.parts
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    result = build(**inputs)
    assert result['recordings'] == 3 and result['changed_audio_recordings'] == 1
    assert result['eligibility'] == {'participant_candidate': 2, 'quarantined': 1}
    target = inputs['output']
    pending = [json.loads(line) for line in (target/'asr_pending.jsonl').read_text().splitlines()]
    assert len(pending) == 1 and pending[0]['audio_path'] == 'audio/taukdial-001-1.wav'
    changed = json.loads((target/'staged_records/taukdial-001-1.json').read_text())
    assert 'words' not in changed and changed['added_source_seconds'] == pytest.approx(1.8)
    verify_candidate_pcm(read_pcm(target/changed['audio_path']), read_pcm(inputs['normalized_originals']/'taukdial-001-1.wav'), changed['time_map'])
    assert changed['review_intervals']  # remaining unknown speech was not promoted
    assert changed['unknown_speech_seconds'] == pytest.approx(3.2)
    assert build(**inputs) == result


@pytest.mark.parametrize('recognized_word', ['獨立辨識', '   '])
def test_finalize_requires_actual_matching_redecode_and_preserves_reuse(inputs, recognized_word):
    build(**inputs)
    target = inputs['output']; asr = target/'asr'
    run = {'asr_protocol_version': 1, 'model': {'repo': MODEL_REPO, 'revision': ASR_REVISION},
           'model_files_sha256': MODEL_FILES, 'temperature': 0, 'condition_on_previous_text': False,
           'initial_prompt': None, 'word_timestamps': True, 'task': 'transcribe', 'random_seed': 0,
           'segment_seconds': 30, 'overlap_seconds': 3, 'overlap_policy': 'midpoint ownership with 1.5s margins',
           'reference_blind': True, 'local_files_only': True}
    _json(asr/'run.json', run)
    item = json.loads((target/'staged_records/taukdial-001-1.json').read_text())
    result = {'recording_id': item['recording_id'], 'language': 'zh', 'source_sha256': item['sha256'],
              'reference_blind': True, 'run_settings_sha256': _sha(asr/'run.json'),
              'words': [{'word': recognized_word, 'start': .5, 'end': 1}]}
    _json(asr/'taukdial-001-1.json', {**result, 'source_sha256': 'wrong'})
    with pytest.raises(ValueError, match='incompatible'): finalize(inputs['data_root'], target, asr)
    _json(asr/'taukdial-001-1.json', result)
    summary = finalize(inputs['data_root'], target, asr)
    assert summary['redecoded'] == 1
    final = json.loads((target/'records/taukdial-001-1.json').read_text())
    assert final['words'] == result['words']
    assert ('empty_asr_transcript' in final['quality_flags']) == (not recognized_word.strip())
    reused = json.loads((target/'records/taukdial-002-1.json').read_text())
    assert reused['words'] == [{'word': '甲', 'start': 1, 'end': 1.3}]
    assert reused['source_asr_sha256'] == 'priorASR'
    assert finalize(inputs['data_root'], target, asr) == summary
    _json(asr/'run.json', {**run, 'model': {'repo': 'other', 'revision': 'wrong'}})
    with pytest.raises(ValueError, match='run settings'): finalize(inputs['data_root'], target, asr)


def test_normalized_hash_mismatch_fails_before_changed_audio_is_written(inputs):
    path = inputs['normalized_originals']/'taukdial-001-1.wav'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='original hash'): build(**inputs)
    assert not (inputs['output']/'audio/taukdial-001-1.wav').exists()


@pytest.mark.parametrize('failure', ['staged_words', 'staged_asr_hash', 'prior_record', 'previous_audio_hash'])
def test_finalize_rechecks_unchanged_asr_reuse_against_pinned_v2(inputs, failure):
    # Disable recovery through insufficient acoustic support, so no new ASR is
    # needed and every source must be an exact reuse of the pinned v2 inputs.
    role_path = inputs['legacy_root']/'outputs/taukadial_asr/chinese_speaker_roles_calibrated/taukdial-001-1.json'
    role = json.loads(role_path.read_text()); role['embedding_windows'] = role['embedding_windows'][:1]
    _json(role_path, role)
    assert build(**inputs)['changed_audio_recordings'] == 0
    path = inputs['output']/'staged_records/taukdial-001-1.json'
    if failure == 'prior_record':
        path = inputs['data_root']/'sources/speech_v2/records/taukdial-001-1.json'
    value = json.loads(path.read_text())
    if failure in {'staged_words', 'prior_record'}:
        value['words'][0]['word'] = 'altered'
    elif failure == 'staged_asr_hash':
        value['source_asr_sha256'] = 'altered'
    else:
        value['previous_source_sha256'] = 'altered'
    _json(path, value)
    with pytest.raises(ValueError, match='changed after build|reuse differs'):
        finalize(inputs['data_root'], inputs['output'], inputs['output']/'unused_asr')
