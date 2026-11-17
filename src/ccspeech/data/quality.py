"""Sanitized, paired release QC against fixed source references.

WER/CER mix recognition and retained-input coverage. They do not measure speaker
purity. Test results are descriptive; only train/validation enter the review
queue. No transcript text, diagnosis, or source-machine path is exported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

from .asr import MODEL_FILES, MODEL_REPO, require_local
from .speech import (ASR_REVISION, SAMPLE_RATE, json_bytes, load_jsonl,
                     match_word_timestamps, stable_write, validate_time_map,
                     validate_words, window_bounds)
from .taukadial import safe_member_path, sha256_file
from .transcript_audit import (_indexed, _read_csv, _recording_splits,
                               compare_text, normalize_units, window_word_indices)

CONDITIONS = ('raw', 'v2_complete', 'v3_complete', 'v3_window_union')
BASE_SETTINGS = {
    'model': {'repo': MODEL_REPO, 'revision': ASR_REVISION},
    'temperature': 0, 'condition_on_previous_text': False, 'initial_prompt': None,
    'word_timestamps': True, 'segment_seconds': 30, 'overlap_seconds': 3,
    'reference_blind': True,
}


def _json(path: Path):
    require_local(path)
    return json.loads(path.read_text())


def _settings(settings: dict, new: bool = False) -> None:
    required = dict(BASE_SETTINGS)
    if new:
        required.update(asr_protocol_version=1, model_files_sha256=MODEL_FILES,
                        task='transcribe', random_seed=0, local_files_only=True,
                        overlap_policy='midpoint ownership with 1.5s margins')
    if any(key not in settings or settings[key] != value for key, value in required.items()):
        raise ValueError('ASR settings differ from the pinned reference-blind protocol')


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['recording_id']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    stable_write(path, stream.getvalue().encode())


def _final_chou(directory: Path, hashes: dict) -> dict[str, str]:
    refs = {}
    for split in ('train', 'validation', 'test'):
        path = directory / f'{split}.csv'
        require_local(path)
        hashes[f'final_task_splits/{split}.csv'] = sha256_file(path)
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['dataset'] != 'Chou':
                    continue
                key = Path(row['audio_path']).stem
                if key in refs:
                    raise ValueError('Duplicate final Chou task reference')
                if not normalize_units(row['transcript'], 'zh'):
                    raise ValueError('Empty final Chou task reference')
                refs[key] = row['transcript']
    return refs


def _source(root: Path, version: str, rid: str, parent: dict, known: dict,
            hashes: dict, release_hashes: dict | None = None) -> dict:
    import soundfile as sf

    base = root / 'sources' / version
    path = base / 'records' / f'{rid}.json'
    item = _json(path)
    digest = sha256_file(path)
    hashes[f'sources/{version}/records/{rid}.json'] = digest
    if release_hashes is not None and release_hashes.get(path.relative_to(root).as_posix()) != digest:
        raise ValueError('Source record differs from the exported release hashes')
    for record in (item, parent):
        if (any(record.get(key) != known[key] for key in ('recording_id', 'participant_id', 'language'))
                or record.get('original_wav_sha256') != known['sha256']):
            raise ValueError('Source original identity/hash binding mismatch')
    if item['sha256'] != parent['sha256'] or item['audio_path'] != parent['audio_path']:
        raise ValueError('Source manifest/audio binding mismatch')
    audio = safe_member_path(item['audio_path'], base)
    require_local(audio)
    if sha256_file(audio) != item['sha256']:
        raise ValueError('Source audio hash mismatch')
    info = sf.info(audio)
    if (info.frames != item['num_samples'] or info.samplerate != SAMPLE_RATE or info.channels != 1
            or abs(item['duration_seconds'] - info.frames / SAMPLE_RATE) > 1 / SAMPLE_RATE):
        raise ValueError('Source audio duration/format mismatch')
    if item.get('candidate_pcm_matches_normalized_original_intervals') is not True:
        raise ValueError('Source PCM-to-original verification is missing')
    validate_words(item['words'], info.frames / SAMPLE_RATE)
    validate_time_map(item['time_map'], info.frames, item['normalized_original_num_samples'])
    return item


def _groups(scored: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in scored:
        for split in ('all', row['split']):
            grouped[row['language'], split, row['condition']].append(row)
    result = []
    for (language, split, condition), rows in sorted(grouped.items()):
        counts = {key: sum(row[key] for row in rows)
                  for key in ('reference_units', 'hypothesis_units', 'S', 'D', 'I')}
        result.append({'language': language, 'split': split, 'condition': condition,
                       'recordings': len(rows), 'participants': len({row['participant_id'] for row in rows}),
                       'metric': 'CER' if language == 'zh' else 'WER', **counts,
                       'error_rate': (counts['S'] + counts['D'] + counts['I']) / counts['reference_units'],
                       'test_descriptive_only': split in {'all', 'test'}})
    return result


def run_audit(data_root: Path, legacy_audit_root: Path, final_task_splits: Path,
              english_resolutions: Path, raw_asr_dir: Path, output: Path) -> dict:
    root, destination = data_root.resolve(), output.resolve()
    if not destination.is_relative_to(root / 'reports'):
        raise ValueError('Quality output must remain inside data/reports')
    known, hashes = _recording_splits(root)
    split_hashes = {Path(key).stem: value for key, value in hashes.items()}
    cross_path = legacy_audit_root / 'recording_crosswalk_complete.csv'
    require_local(cross_path)
    with cross_path.open(newline='') as stream:
        cross = _indexed(list(csv.DictReader(stream)), 'tauk_recording', 'source crosswalk')
    require_local(english_resolutions)
    resolved = _indexed(load_jsonl(english_resolutions), 'recording_id', 'English resolutions')
    if set(cross) != set(known) or set(resolved) != {rid for rid, row in known.items() if row['language'] == 'en'}:
        raise ValueError('Reference crosswalk/resolution coverage differs from original recordings')
    hashes.update(source_crosswalk=sha256_file(cross_path), english_resolutions=sha256_file(english_resolutions))
    references = _final_chou(final_task_splits, hashes)
    parents, provenance = {}, {}
    for version in ('speech_v2', 'speech_v3'):
        base = root / 'sources' / version
        parents[version] = _indexed(load_jsonl(base / 'manifest.jsonl'), 'recording_id', version)
        provenance[version] = _json(base / 'provenance.json')
        if set(parents[version]) != set(known) or provenance[version]['split_manifest_sha256'] != split_hashes:
            raise ValueError('Versioned source identity/split coverage differs')
        if not provenance[version].get('asr_runs'):
            raise ValueError('Missing source ASR provenance')
        for run in provenance[version]['asr_runs']:
            _settings(run['settings'])
        for filename in ('manifest.jsonl', 'provenance.json'):
            hashes[f'sources/{version}/{filename}'] = sha256_file(base / filename)
    p3 = provenance['speech_v3']
    if p3['parent_manifest_sha256'] != hashes['sources/speech_v2/manifest.jsonl']:
        raise ValueError('V3 is not bound to the current v2 parent manifest')
    redecode = p3.get('local_redecode_settings')
    if redecode is not None:
        _settings(redecode, new=True)
        if hashlib.sha256(json_bytes(redecode)).hexdigest() != p3['local_redecode_run_sha256']:
            raise ValueError('V3 local ASR settings digest mismatch')
    raw_run = _json(raw_asr_dir / 'run.json')
    _settings(raw_run)
    hashes['raw_asr/run.json'] = sha256_file(raw_asr_dir / 'run.json')
    release_path = root / 'reports/speech_v3/file_hashes.json'
    release_hashes = _json(release_path)
    hashes['release_file_hashes'] = sha256_file(release_path)
    prior_release_path = root / 'reports/speech_v2/file_hashes.json'
    prior_release_hashes = _json(prior_release_path)
    hashes['v2_release_file_hashes'] = sha256_file(prior_release_path)
    windows, segment_ids = defaultdict(list), set()
    for split in ('train', 'validation', 'test'):
        path = root / 'processed/speech_v3' / f'{split}.csv'
        hashes[f'processed/speech_v3/{split}.csv'] = sha256_file(path)
        if release_hashes.get(path.relative_to(root).as_posix()) != hashes[f'processed/speech_v3/{split}.csv']:
            raise ValueError('Window CSV differs from the exported release hashes')
        for row in _read_csv(path, split):
            rid = row['recording_id']
            if rid not in known or any(row[key] != known[rid][key] for key in ('participant_id', 'language', 'split')):
                raise ValueError('Window identity/split mismatch')
            if row['segment_id'] in segment_ids:
                raise ValueError('Duplicate v3 segment identity')
            segment_ids.add(row['segment_id'])
            windows[rid].append(row)
    quarantine_path = root / 'processed/speech_v3/quarantined_recordings.jsonl'
    quarantine = _indexed(load_jsonl(quarantine_path), 'recording_id', 'quarantine')
    hashes['quarantined_recordings'] = sha256_file(quarantine_path)
    expected_quarantine = {rid for rid, row in resolved.items()
                           if row.get('scope_review_complete') is not True or row.get('extraction_timing_complete') is not True}
    if set(quarantine) != expected_quarantine or set(windows) != set(known) - expected_quarantine:
        raise ValueError('Primary/quarantine membership differs from verified scope policy')
    rows, review, pairs, eligibility_counts = [], [], [], Counter()
    for rid, original in sorted(known.items()):
        match = cross[rid]
        language, split = original['language'], original['split']
        if (match['tauk_participant_id'] != original['participant_id']
                or match['source_dataset'] != {'zh': 'Chou', 'en': 'Delaware'}[language]):
            raise ValueError('Reference source participant/language mismatch')
        v2 = _source(root, 'speech_v2', rid, parents['speech_v2'][rid], original, hashes, prior_release_hashes)
        v3 = _source(root, 'speech_v3', rid, parents['speech_v3'][rid], original, hashes, release_hashes)
        if (v3['normalized_original_sha256'] != v2['normalized_original_sha256']
                or v3['normalized_original_num_samples'] != v2['normalized_original_num_samples']
                or v3['previous_source_sha256'] != v2['sha256']):
            raise ValueError('V2/v3 original or parent audio mismatch')
        changed = v3['sha256'] != v2['sha256']
        if changed:
            if (v3.get('asr_state') != 'locally_redecoded_changed_audio' or redecode is None
                    or v3.get('source_asr_run_sha256') != p3['local_redecode_run_sha256']):
                raise ValueError('Changed audio lacks new pinned ASR provenance')
        elif v3['words'] != v2['words'] or v3['source_asr_sha256'] != v2['source_asr_sha256']:
            raise ValueError('Unchanged audio must reuse identical ASR words and provenance')
        expected_eligibility = 'quarantined' if rid in expected_quarantine else 'participant_candidate'
        if (v3['eligibility'] != expected_eligibility or parents['speech_v3'][rid]['eligibility'] != expected_eligibility
                or bool(v3['exclusion_reasons']) != (expected_eligibility == 'quarantined')):
            raise ValueError('Source eligibility differs from the verified primary policy')
        eligibility_counts[expected_eligibility] += 1
        base = {'recording_id': rid, 'participant_id': original['participant_id'], 'language': language,
                'split': split, 'eligibility': expected_eligibility, 'audio_changed_v3': changed,
                'original_wav_sha256': original['sha256'], 'v2_audio_sha256': v2['sha256'],
                'v3_audio_sha256': v3['sha256'], 'test_descriptive_only': split == 'test',
                'quality_flags': ';'.join(sorted(set(v3.get('quality_flags', []))))}
        if rid in expected_quarantine:
            q = quarantine[rid]
            if (any(q[key] != base[key] for key in ('participant_id', 'language', 'split', 'eligibility'))
                    or q['sha256'] != v3['sha256'] or q['words'] != v3['words'] or q['time_map'] != v3['time_map']):
                raise ValueError('Quarantine artifact differs from its source record')
            rows.append({**base, 'scored': False, 'condition': 'unscored',
                         'reference_scope': 'unresolved_English_scope_or_timing'})
            if split != 'test':
                review.append({**base, 'review_reason': 'quarantined_scope_or_timing', 'paired_error_delta': None})
            continue
        duration = v3['num_samples'] / SAMPLE_RATE
        current = sorted(windows[rid], key=lambda row: float(row['start_seconds']))
        bounds = [(float(row['start_seconds']), float(row['end_seconds'])) for row in current]
        expected, dropped = window_bounds(v3['num_samples'], min_tail_seconds=0)
        if dropped or [(round(a * SAMPLE_RATE), round(b * SAMPLE_RATE)) for a, b in bounds] != expected:
            raise ValueError('V3 windows do not follow the complete 30/27-second policy')
        for row, (a, b) in zip(current, bounds):
            if (row['source_sha256'] != v3['sha256'] or row['original_wav_sha256'] != original['sha256']
                    or row['eligibility'] != expected_eligibility):
                raise ValueError('Window source/hash/eligibility mismatch')
            if row['matched_transcript'] != match_word_timestamps(v3['words'], a, b, duration)['matched_transcript']:
                raise ValueError('Window transcript differs from actual source ASR words')
            audio = safe_member_path(row['audio_path'], root)
            require_local(audio)
            if sha256_file(audio) != row['sha256'] or release_hashes.get(row['audio_path']) != row['sha256']:
                raise ValueError('Window audio differs from release hashes')
        selected = window_word_indices(v3['words'], bounds, duration)
        if selected != list(range(len(v3['words']))):
            raise ValueError('V3 retained-window word union omits source ASR words')
        if language == 'zh':
            if (match['source_recording'] not in references or float(match['source_offset_seconds']) != 0
                    or float(match['full_waveform_corr']) < .98):
                raise ValueError('Final Chou reference lacks full original-recording identity evidence')
            reference = references[match['source_recording']]
            scope = 'user_final_Chou_full_task_text'
        else:
            reference = resolved[rid]['reference']
            scope = 'verified_Delaware_CHAT_clip_scope'
        raw_path = raw_asr_dir / f'{rid}.json'
        raw = _json(raw_path)
        if (raw['recording_id'] != rid or raw['language'] != language or raw.get('reference_blind') is not True
                or raw['source_sha256'] != v2['normalized_original_sha256']):
            raise ValueError('Raw ASR identity/original hash mismatch')
        validate_words(raw['words'], v2['normalized_original_num_samples'] / SAMPLE_RATE)
        if raw['text'] != ''.join(word['word'] for word in raw['words']):
            raise ValueError('Raw ASR text differs from its timed words')
        hashes[f'raw_asr/{rid}.json'] = sha256_file(raw_path)
        hypotheses = {'raw': raw['text'], 'v2_complete': ''.join(w['word'] for w in v2['words']),
                      'v3_complete': ''.join(w['word'] for w in v3['words']),
                      'v3_window_union': ''.join(v3['words'][index]['word'] for index in selected)}
        metrics = {}
        for condition, hypothesis in hypotheses.items():
            counts, _ = compare_text(reference, hypothesis, language)
            metrics[condition] = counts
            rows.append({**base, 'scored': True, 'condition': condition, 'reference_scope': scope,
                         'retained_windows': len(bounds), 'complete_asr_word_count': len(v3['words']),
                         'window_union_word_count': len(selected), 'empty_normalized_hypothesis': not normalize_units(hypothesis, language),
                         **counts})
        before, after = metrics['v2_complete'], metrics['v3_complete']
        delta = after['error_rate'] - before['error_rate']
        pairs.append({'language': language, 'split': split, 'delta': delta, 'changed': changed})
        if split != 'test':
            reason = 'higher_reference_error' if delta > 0 else 'paired_descriptive_review'
            review.append({**base, 'review_reason': reason, 'paired_error_delta': delta,
                           'v2_error_rate': before['error_rate'], 'v3_error_rate': after['error_rate'],
                           'delta_S': after['S'] - before['S'], 'delta_D': after['D'] - before['D'],
                           'delta_I': after['I'] - before['I'], 'reference_units': after['reference_units']})
    if dict(eligibility_counts) != p3['eligibility_counts']:
        raise ValueError('Source provenance eligibility counts differ from audited membership')
    scored = [row for row in rows if row['scored']]
    summary = {
        'schema_version': 1, 'recordings': len(known), 'scored_recordings': len(pairs),
        'unscored_quarantined_recordings': len(expected_quarantine), 'eligibility_counts': dict(eligibility_counts),
        'conditions': list(CONDITIONS), 'groups': _groups(scored),
        'paired_v3_vs_v2': {split: {'recordings': len(group), 'audio_changed': sum(r['changed'] for r in group),
                                  'lower_error': sum(r['delta'] < 0 for r in group),
                                  'equal_error': sum(r['delta'] == 0 for r in group),
                                  'higher_error': sum(r['delta'] > 0 for r in group)}
                            for split in ('train', 'validation', 'test')
                            for group in [[r for r in pairs if r['split'] == split]]},
        'review_queue_records': len(review), 'input_sha256': hashes,
        'checks': {'same_reference_all_conditions': True, 'original_identity_hash_binding': True,
                   'source_and_window_hashes_verified': True, 'csv_matches_timed_source_words': True,
                   'v3_window_union_equals_complete_word_indices': True, 'pinned_ASR_provenance': True,
                   'quarantine_excluded_from_scoring_and_primary_windows': True,
                   'review_queue_excludes_test': True, 'diagnostic_threshold_selection': False,
                   'diagnoses_used_for_quality_comparison': False,
                   'human_reference_text_exported': False, 'ASR_hypothesis_text_exported': False},
        'notes': ['Scores include recognition and retained-input differences; they are not speaker-purity measurements.',
                  'Raw audio can contain interviewer speech, so raw-versus-PAR-reference errors are not speaker-matched ASR WER.',
                  'Chou uses the user final full-task text; English uses verified CHAT clip-scope text, not a whole final task.',
                  'Reference-assisted extraction means these scores are not independent validation of speaker identity.',
                  'Test and all-split summaries are descriptive only; no threshold or extraction policy is selected here.',
                  'Real repeated words retain separate source indices; overlapping windows do not double-count them.'],
    }
    review.sort(key=lambda row: (row['paired_error_delta'] is not None,
                                -(row['paired_error_delta'] or 0), row['recording_id']))
    destination.mkdir(parents=True, exist_ok=True)
    _csv(destination / 'per_recording.csv', rows)
    _csv(destination / 'review_queue.csv', review)
    stable_write(destination / 'summary.json', json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--legacy-audit-root', type=Path, required=True)
    parser.add_argument('--final-task-splits', type=Path, required=True)
    parser.add_argument('--english-resolutions', type=Path, required=True)
    parser.add_argument('--raw-asr-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/reports/speech_v3/quality'))
    args = parser.parse_args()
    summary = run_audit(**vars(args))
    print(json.dumps({key: summary[key] for key in ('recordings', 'scored_recordings', 'unscored_quarantined_recordings', 'paired_v3_vs_v2')}, indent=2))


if __name__ == '__main__':
    main()
