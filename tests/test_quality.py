"""Release QC cannot mix references, leak text, drop repeats, or score quarantine."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ccspeech.data.asr import MODEL_FILES
from ccspeech.data.quality import BASE_SETTINGS, run_audit
from ccspeech.data.speech import json_bytes
from ccspeech.data.speech_v3 import prepare_windows_v3


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))


def write_csv(path, values, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path):
    root, legacy, refs, raw = (tmp_path / name for name in ('data', 'legacy', 'final_tasks', 'raw_asr'))
    new_settings = {**BASE_SETTINGS, 'asr_protocol_version': 1, 'model_files_sha256': MODEL_FILES,
                    'task': 'transcribe', 'random_seed': 0, 'local_files_only': True,
                    'overlap_policy': 'midpoint ownership with 1.5s margins'}
    run_hash = hashlib.sha256(json_bytes(new_settings)).hexdigest()
    write_json(raw / 'run.json', BASE_SETTINGS)
    parents = {'speech_v2': [], 'speech_v3': []}
    splits, cross, resolutions, split_hashes = {}, [], [], {}
    for index, (rid, language, split, quarantined) in enumerate([
        ('taukdial-001-1', 'zh', 'train', False), ('taukdial-002-1', 'en', 'validation', True), ('taukdial-003-1', 'en', 'test', False),
    ], 1):
        pid, original_hash = rid.rsplit('-', 1)[0], hashlib.sha256((rid + 'original').encode()).hexdigest()
        normalized_hash = hashlib.sha256((rid + 'normalized').encode()).hexdigest()
        seconds = 41 if language == 'zh' else 4
        samples = np.random.default_rng(index).integers(-1000, 1000, seconds * 16000, dtype=np.int16)
        v2_words = ([{'word': w, 'start': t, 'end': t + .5} for w, t in [('甲', 1), ('乙', 28), ('丙', 39)]]
                    if language == 'zh' else [{'word': ' hello', 'start': .5, 'end': 1.}, {'word': ' world', 'start': 1.5, 'end': 2.}])
        v3_words = ([v2_words[0], {'word': '甲', 'start': 2, 'end': 2.5}, *v2_words[1:]] if language == 'zh' else v2_words)
        prior = None
        for version in ('speech_v2', 'speech_v3'):
            base = root / 'sources' / version
            audio = base / 'audio' / f'{rid}.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            data = samples.copy()
            if version == 'speech_v3' and language == 'zh':
                data[0] += 1
            sf.write(audio, data, 16000, subtype='PCM_16')
            item = {'recording_id': rid, 'participant_id': pid, 'language': language,
                    'audio_path': f'audio/{rid}.wav', 'sha256': sha(audio), 'original_wav_sha256': original_hash,
                    'normalized_original_sha256': normalized_hash, 'normalized_original_num_samples': len(data),
                    'sample_rate': 16000, 'channels': 1, 'num_samples': len(data), 'duration_seconds': seconds,
                    'candidate_pcm_matches_normalized_original_intervals': True,
                    'source_asr_sha256': hashlib.sha256((rid + 'old-asr').encode()).hexdigest(),
                    'words': v2_words if version == 'speech_v2' else v3_words,
                    'time_map': [{'source_start': 0., 'source_end': seconds, 'clean_start': 0., 'clean_end': seconds}],
                    'quality_flags': ['speaker_purity_not_manually_verified'], 'extraction_condition': 'fixture'}
            if version == 'speech_v3':
                item.update(previous_source_sha256=prior['sha256'],
                            eligibility='quarantined' if quarantined else 'participant_candidate',
                            exclusion_reasons=['scope_unresolved'] if quarantined else [], review_intervals=[],
                            asr_state='locally_redecoded_changed_audio' if language == 'zh' else 'reused_identical_audio')
                if language == 'zh':
                    item.update(source_asr_run_sha256=run_hash,
                                source_asr_sha256=hashlib.sha256((rid + 'new-asr').encode()).hexdigest())
            parent = {key: item[key] for key in ('recording_id', 'participant_id', 'language', 'audio_path', 'sha256', 'original_wav_sha256')}
            if version == 'speech_v3':
                parent['eligibility'] = item['eligibility']
            parents[version].append(parent)
            write_json(base / 'records' / f'{rid}.json', item)
            prior = item
        original = {'recording_id': rid, 'participant_id': pid, 'language': language, 'split': split,
                    'official_split': 'test' if split == 'test' else 'train', 'sha256': original_hash}
        if split != 'test':
            original['label'] = 0
        splits[split] = [original]
        cross.append({'tauk_recording': rid, 'tauk_participant_id': pid,
                      'source_dataset': 'Chou' if language == 'zh' else 'Delaware', 'source_recording': rid,
                      'source_offset_seconds': '0', 'full_waveform_corr': '.9999'})
        if language == 'en':
            resolutions.append({'recording_id': rid, 'scope_review_complete': not quarantined,
                                'extraction_timing_complete': not quarantined,
                                'reference': 'PRIVATE UNRESOLVED REFERENCE' if quarantined else 'hello world'})
        if not quarantined:
            words = v3_words if language == 'zh' else v2_words
            write_json(raw / f'{rid}.json', {'recording_id': rid, 'language': language,
                       'source_sha256': normalized_hash, 'reference_blind': True, 'words': words,
                       'text': ''.join(w['word'] for w in words)})
    for split in ('train', 'validation', 'test'):
        path = root / 'processed/taukadial' / f'{split}.csv'
        write_csv(path, splits[split], list(splits[split][0]))
        split_hashes[split] = sha(path)
        write_csv(refs / f'{split}.csv', [{'dataset': 'Chou', 'audio_path': 'Chou/taukdial-001-1.wav',
                                        'transcript': '甲甲乙丙'}] if split == 'train' else [],
                  ['dataset', 'audio_path', 'transcript'])
    for version in ('speech_v2', 'speech_v3'):
        base = root / 'sources' / version
        write_jsonl(base / 'manifest.jsonl', parents[version])
        provenance = {'split_manifest_sha256': split_hashes, 'asr_runs': [{'settings': BASE_SETTINGS}]}
        if version == 'speech_v3':
            provenance.update(parent_manifest_sha256=sha(root / 'sources/speech_v2/manifest.jsonl'),
                              local_redecode_settings=new_settings, local_redecode_run_sha256=run_hash,
                              eligibility_counts={'participant_candidate': 2, 'quarantined': 1})
        write_json(base / 'provenance.json', provenance)
    write_csv(legacy / 'recording_crosswalk_complete.csv', cross, list(cross[0]))
    resolution_path = legacy / 'reference_resolution.jsonl'
    write_jsonl(resolution_path, resolutions)
    write_json(root / 'reports/speech_v2/file_hashes.json', {
        p.relative_to(root).as_posix(): sha(p) for p in (root / 'sources/speech_v2').rglob('*') if p.is_file()})
    prepare_windows_v3(root)
    return dict(data_root=root, legacy_audit_root=legacy, final_task_splits=refs,
                english_resolutions=resolution_path, raw_asr_dir=raw, output=root / 'reports/speech_v3/quality')


def refresh_release_hash(release, path):
    root = release['data_root']
    hashes_path = root / 'reports/speech_v3/file_hashes.json'
    hashes = json.loads(hashes_path.read_text())
    hashes[path.relative_to(root).as_posix()] = sha(path)
    write_json(hashes_path, hashes)


def test_paired_qc_keeps_repeats_and_quarantines_bad_scope_without_text_export(release):
    summary = run_audit(**release)
    assert summary['recordings'] == 3
    assert summary['scored_recordings'] == 2
    assert summary['unscored_quarantined_recordings'] == 1
    zh = {r['condition']: r for r in summary['groups'] if r['language'] == 'zh' and r['split'] == 'train'}
    assert zh['v2_complete']['D'] == 1
    assert zh['v3_complete']['reference_units'] == 4
    assert zh['v3_complete']['error_rate'] == zh['v3_window_union']['error_rate'] == 0
    with (release['output'] / 'review_queue.csv').open() as stream:
        review = list(csv.DictReader(stream))
    assert {r['split'] for r in review} == {'train', 'validation'}
    assert {r['recording_id'] for r in review} == {'taukdial-001-1', 'taukdial-002-1'}
    for path in release['output'].iterdir():
        text = path.read_text()
        assert 'PRIVATE UNRESOLVED REFERENCE' not in text
        assert 'hello world' not in text and '甲甲乙丙' not in text
    assert run_audit(**release) == summary


@pytest.mark.parametrize('fault', ['original_identity', 'window_text', 'window_omission', 'promote_quarantine', 'model_settings', 'source_corruption'])
def test_invalid_release_is_rejected_before_quality_outputs(release, fault):
    root = release['data_root']
    if fault in {'original_identity', 'promote_quarantine'}:
        rid = 'taukdial-001-1' if fault == 'original_identity' else 'taukdial-002-1'
        path = root / 'sources/speech_v3/records' / f'{rid}.json'
        item = json.loads(path.read_text())
        item['original_wav_sha256' if fault == 'original_identity' else 'eligibility'] = 'wrong' if fault == 'original_identity' else 'participant_candidate'
        write_json(path, item)
        refresh_release_hash(release, path)
    elif fault in {'window_text', 'window_omission'}:
        path = root / 'processed/speech_v3/train.csv'
        with path.open() as stream:
            reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
        if fault == 'window_text':
            rows[0]['matched_transcript'] = 'invented text'
        else:
            rows.pop()
        write_csv(path, rows, fields)
        refresh_release_hash(release, path)
    elif fault == 'model_settings':
        path = release['raw_asr_dir'] / 'run.json'
        settings = json.loads(path.read_text()); settings['initial_prompt'] = 'reference prompt'
        write_json(path, settings)
    else:
        path = root / 'sources/speech_v3/audio/taukdial-001-1.wav'
        content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
    with pytest.raises(ValueError):
        run_audit(**release)
    assert not release['output'].exists()
