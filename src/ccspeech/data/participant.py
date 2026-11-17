"""Versioned, conservative repair of participant candidates from TAUKADIAL PCM.

Frozen recovery needs both a long source-reference anchor and unanimous local
acoustic support. Unsupported unknown speech stays unresolved. Changed audio requires
new reference-blind ASR before the standalone source package is finalized.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .asr import MODEL_FILES, MODEL_REPO
from .audio_identity import availability
from .speech import (ASR_REVISION, SAMPLE_RATE, json_bytes, load_jsonl, read_pcm, stable_write,
                     validate_time_map, validate_words, verify_candidate_pcm)
from .source_cut_audit import retention, union_intervals
from .taukadial import safe_member_path, sha256_file
from .transcript_audit import _indexed, _recording_splits, normalize_units

VOICE_THRESHOLD = 0.46845851570367814
MIN_ANCHOR_CHARACTERS = 6
PADDING_SECONDS = .15
POLICY = 'long_reference_anchor_and_unanimous_acoustic_word_recovery_v1'


def reference_anchors(reference: str, words: list[dict]) -> list[tuple[float, float]]:
    chars = [(c, w['start'], w['end']) for w in words for c in normalize_units(w['word'], 'zh')]
    blocks = SequenceMatcher(None, normalize_units(reference, 'zh'), [c[0] for c in chars], autojunk=False).get_matching_blocks()
    return union_intervals([(chars[b.b][1], chars[b.b+b.size-1][2]) for b in blocks if b.size >= MIN_ANCHOR_CHARACTERS])


def recovery_intervals(words: list[dict], reference: str, role: dict,
                       kept: list[tuple[float, float]], duration: float) -> tuple[list[tuple[float, float]], list[dict]]:
    """Add supported whole-word intervals only; never replace existing PAR audio."""
    calibration = role['voice_calibration']
    if (calibration.get('split_used') != 'train_only' or calibration.get('test_used_for_threshold') is not False
            or abs(calibration['threshold']-VOICE_THRESHOLD) > 1e-12):
        raise ValueError('Frozen acoustic calibration differs from the approved policy')
    anchors, windows = reference_anchors(reference, words), role['embedding_windows']
    boundaries = sorted({v for w in windows for v in (w['start'], w['end'])})
    supported = []
    for left, right in zip(boundaries, boundaries[1:]):
        active = [w for w in windows if w['start'] < right and w['end'] > left]
        if len(active) >= 2 and all(w['role'] == 'PAR' and w.get('gallery_top3_similarity') is not None
                                    and w['gallery_top3_similarity'] >= VOICE_THRESHOLD for w in active):
            supported.append((left, right))
    supported = union_intervals(supported)
    approved = []
    for index, word in enumerate(words):
        a, b = float(word['start']), float(word['end'])
        if b <= a or retention([(a, b)], kept)['omitted_audio_seconds'] <= 1/SAMPLE_RATE:
            continue
        anchor = next(((x, y) for x, y in anchors if x <= a and b <= y), None)
        if anchor is None:
            continue
        overlapping = [w for w in windows if w['start'] < b and w['end'] > a]
        if len(overlapping) < 2 or not all(w['role'] == 'PAR' and w.get('gallery_top3_similarity') is not None
                                         and w['gallery_top3_similarity'] >= VOICE_THRESHOLD for w in overlapping):
            continue
        # Every instant of the complete word needs at least two agreeing
        # windows; padding stays inside that continuous support and its anchor.
        support = next(((x, y) for x, y in supported if x <= a and b <= y), None)
        if support is None:
            continue
        support_left, support_right = support
        left, right = max(0., a-PADDING_SECONDS, anchor[0], support_left), min(duration, b+PADDING_SECONDS, anchor[1], support_right)
        approved.append({'raw_word_index': index, 'source_start': left, 'source_end': right,
                         'anchor_start': anchor[0], 'anchor_end': anchor[1], 'supporting_windows': len(overlapping),
                         'minimum_gallery_similarity': min(w['gallery_top3_similarity'] for w in overlapping)})
    return union_intervals(kept+[(r['source_start'], r['source_end']) for r in approved]), approved


def make_time_map(intervals: list[tuple[float, float]]) -> list[dict]:
    cursor, result = 0, []
    # Quantize once to the real 16 kHz clock before constructing the map.
    for a, b in union_intervals([(round(a*SAMPLE_RATE), round(b*SAMPLE_RATE)) for a, b in intervals]):
        if b <= a:
            continue
        result.append({'source_start': a/SAMPLE_RATE, 'source_end': b/SAMPLE_RATE,
                       'clean_start': cursor/SAMPLE_RATE, 'clean_end': (cursor+b-a)/SAMPLE_RATE})
        cursor += b-a
    return result


def _wav_bytes(samples: np.ndarray) -> bytes:
    stream = io.BytesIO()
    sf.write(stream, samples, SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return stream.getvalue()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    stable_write(path, ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows).encode())


def _local_json(path: Path) -> dict:
    if availability(path) != 'materialized':
        raise ValueError(f'Evidence file must be materialized before use: {path}')
    return json.loads(path.read_text())


def _source_root(data_root: Path, output: Path) -> tuple[Path, Path]:
    root, target = data_root.resolve(), output.resolve()
    if not target.is_relative_to(root/'sources') or target == root/'sources/speech_v2':
        raise ValueError('Choose a new version directory under data/sources')
    return root, target


def build(data_root: Path, legacy_root: Path, normalized_originals: Path, output: Path) -> dict:
    root, target = _source_root(data_root, output)
    prior = root/'sources/speech_v2'
    known, split_hashes = _recording_splits(root)
    parents = _indexed(load_jsonl(prior/'manifest.jsonl'), 'recording_id', 'v2 parents')
    provenance_path = prior/'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    reference_path = legacy_root/'data/taukadial_asr/reference_manifest.jsonl'
    if sha256_file(reference_path) != provenance['original_reference_manifest_sha256']:
        raise ValueError('Legacy source reference differs from pinned v2 provenance')
    references = _indexed(load_jsonl(reference_path), 'recording_id', 'legacy references')
    resolution_path = legacy_root/'data/taukadial_asr/reference_resolution.jsonl'
    resolutions = _indexed(load_jsonl(resolution_path), 'recording_id', 'English resolutions')
    calibration_path = legacy_root/'data/taukadial_asr/voice_calibration_train.json'
    calibration = _local_json(calibration_path)
    if set(known) != set(parents) or set(known) != set(references):
        raise ValueError('Source manifests must preserve all project recording identities')
    expected_splits = {Path(k).stem: v for k, v in split_hashes.items()}
    if provenance['split_manifest_sha256'] != expected_splits:
        raise ValueError('v2 provenance does not match the participant splits')
    hashes = {**split_hashes, 'v2_manifest': sha256_file(prior/'manifest.jsonl'),
              'v2_provenance': sha256_file(provenance_path), 'legacy_references': sha256_file(reference_path),
              'english_resolutions': sha256_file(resolution_path), 'frozen_voice_calibration': sha256_file(calibration_path)}
    staged, pending, decisions = [], [], []
    for rid in sorted(known):
        original, parent = known[rid], parents[rid]
        item_path = prior/'records'/f'{rid}.json'; item = _local_json(item_path)
        reference = references[rid]
        if (any(r['recording_id'] != rid or r['participant_id'] != original['participant_id']
                or r['language'] != original['language'] or r['original_wav_sha256'] != original['sha256']
                for r in (item, parent, reference)) or item['sha256'] != parent['sha256']
                or reference['normalized_wav_sha256'] != item['normalized_original_sha256']
                or item.get('candidate_pcm_matches_normalized_original_intervals') is not True):
            raise ValueError('v2 candidate/reference/original identity or hash differs')
        audio_path = safe_member_path(item['audio_path'], prior)
        if availability(audio_path) != 'materialized' or sha256_file(audio_path) != item['sha256']:
            raise ValueError('Portable candidate audio is unavailable or changed')
        candidate = read_pcm(audio_path)
        if len(candidate) != item['num_samples']:
            raise ValueError('Portable candidate sample count differs')
        validate_time_map(item['time_map'], len(candidate), item['normalized_original_num_samples'])
        validate_words(item['words'], len(candidate)/SAMPLE_RATE)
        duration = item['normalized_original_num_samples']/SAMPLE_RATE
        kept = [(p['source_start'], p['source_end']) for p in item['time_map']]
        new = dict(item)
        new.update(eligibility='participant_candidate', exclusion_reasons=[], review_intervals=[],
                   previous_source_sha256=item['sha256'], previous_source_asr_sha256=item['source_asr_sha256'],
                   source_policy=POLICY, manual_speaker_qa_complete=False, split=original['split'])
        approved, added, changed = [], 0., False
        record_hashes = {'v2_record': sha256_file(item_path)}
        if original['language'] == 'en':
            resolution = resolutions[rid]
            eligible = resolution.get('scope_review_complete') is True and resolution.get('extraction_timing_complete') is True
            if not eligible:
                new['eligibility'] = 'quarantined'
                new['exclusion_reasons'] = ['english_reference_scope_or_extraction_timing_unresolved']
                new['review_intervals'] = [{'source_start': 0., 'source_end': duration,
                                            'reason': 'unresolved_english_scope_or_speaker_boundaries'}]
        elif original['language'] == 'zh':
            role_path = legacy_root/'outputs/taukadial_asr/chinese_speaker_roles_calibrated'/f'{rid}.json'
            raw_path = legacy_root/'outputs/taukadial_asr/raw_whisper_v1'/f'{rid}.json'
            role, raw = _local_json(role_path), _local_json(raw_path)
            if role['voice_calibration'] != calibration:
                raise ValueError('Speaker role evidence uses a different calibration artifact')
            feature_path = Path(role['features_path'])/f'{rid}.json'
            features = _local_json(feature_path)
            if (role['recording_id'] != rid or role['participant_id'] != original['participant_id']
                    or raw['recording_id'] != rid or raw['language'] != 'zh' or raw.get('reference_blind') is not True
                    or raw['source_sha256'] != item['normalized_original_sha256']
                    or features['recording_id'] != rid or features['audio_sha256'] != item['normalized_original_sha256']):
                raise ValueError('Speaker/raw-ASR evidence has a different original identity')
            validate_words(raw['words'], duration)
            updated, approved = recovery_intervals(raw['words'], reference['reference_participant'], role, kept, duration)
            mapping = make_time_map(updated)
            added = sum(round((p['source_end']-p['source_start'])*SAMPLE_RATE) for p in mapping)-len(candidate)
            changed = added > 0
            if added < 0:
                raise ValueError('Recovery policy must never remove existing candidate samples')
            if changed:
                normalized_path = normalized_originals/f'{rid}.wav'
                if availability(normalized_path) != 'materialized':
                    raise ValueError(f'Normalized TAUKADIAL original unavailable without cloud hydration: {rid}')
                if sha256_file(normalized_path) != item['normalized_original_sha256']:
                    raise ValueError('Normalized TAUKADIAL original hash differs')
                normalized = read_pcm(normalized_path)
                if len(normalized) != item['normalized_original_num_samples']:
                    raise ValueError('Normalized original sample count differs')
                verify_candidate_pcm(candidate, normalized, item['time_map'])
                candidate = np.concatenate([normalized[round(p['source_start']*SAMPLE_RATE):round(p['source_end']*SAMPLE_RATE)] for p in mapping])
                verify_candidate_pcm(candidate, normalized, mapping)
                new.update(time_map=mapping, num_samples=len(candidate), duration_seconds=len(candidate)/SAMPLE_RATE,
                           extraction_condition='reference_assisted_acoustic_PAR_supported_recovery')
            retained = [(p['source_start'], p['source_end']) for p in new['time_map']]
            unknown = [(s['start'], s['end']) for s in role['speech_intervals'] if s['role'] == 'unknown']
            unresolved = retention(unknown, retained)['omitted_original_intervals']
            new['review_intervals'] = [{'source_start': a, 'source_end': b, 'reason': 'unresolved_speaker_role_not_promoted'} for a, b in unresolved]
            new['previous_unknown_speech_seconds'] = item.get('unknown_speech_seconds')
            new['unknown_speech_seconds'] = sum(b-a for a, b in unresolved)
            interviewer = [(s['start'], s['end']) for s in role['speech_intervals'] if s['role'] == 'INV']
            new['excluded_interviewer_intervals'] = [
                {'source_start': a, 'source_end': b, 'reason': 'identified_interviewer_excluded'}
                for a, b in retention(interviewer, retained)['omitted_original_intervals']]
            record_hashes.update(raw_asr=sha256_file(raw_path), speaker_roles=sha256_file(role_path), speaker_features=sha256_file(feature_path))
        else:
            raise ValueError('Unexpected language')
        relative_audio = f'audio/{rid}.wav'
        content = _wav_bytes(candidate) if changed else audio_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        new.update(audio_path=relative_audio, sha256=digest, added_source_seconds=added/SAMPLE_RATE,
                   recovery_word_count=len(approved), candidate_pcm_matches_normalized_original_intervals=True,
                   asr_state='pending_local_redecode' if changed else 'reused_identical_audio')
        new['quality_flags'] = sorted(set(new.get('quality_flags', [])+['speaker_purity_not_manually_verified']+
                                         (['supported_recovery_requires_local_redecode'] if changed else [])+
                                         (['quarantined_scope_or_timing'] if new['eligibility']=='quarantined' else [])))
        decision = {'recording_id': rid, 'policy': POLICY, 'evidence_sha256': record_hashes,
                    'previous_source_sha256': item['sha256'], 'source_sha256': digest,
                    'recovered_word_intervals': approved, 'added_source_seconds': added/SAMPLE_RATE,
                    'time_map': new['time_map'], 'review_intervals': new['review_intervals']}
        sidecar = json_bytes(decision)
        new['previous_source_sidecar_sha256'] = item['source_sidecar_sha256']
        new['source_sidecar_sha256'] = hashlib.sha256(sidecar).hexdigest()
        if changed:
            new.pop('words'); new.pop('source_asr_sha256')
            pending.append({k: new[k] for k in ('recording_id', 'participant_id', 'language', 'split', 'audio_path', 'sha256')})
        stable_write(target/relative_audio, content)
        stable_write(target/'sidecars'/f'{rid}.json', sidecar)
        stable_write(target/'staged_records'/f'{rid}.json', json_bytes(new))
        staged.append({k: new[k] for k in ('recording_id', 'participant_id', 'language', 'audio_path', 'sha256', 'original_wav_sha256', 'eligibility')})
        hashes[f'evidence/{rid}'] = record_hashes
        decisions.append(decision)
    _write_jsonl(target/'staged_manifest.jsonl', staged)
    _write_jsonl(target/'asr_pending.jsonl', pending)
    summary = {'schema_version': 3, 'policy': POLICY, 'recordings': len(staged), 'changed_audio_recordings': len(pending),
               'exact_audio_asr_reuse_recordings': len(staged)-len(pending), 'added_source_seconds': sum(r['added_source_seconds'] for r in decisions),
               'eligibility': dict(Counter(r['eligibility'] for r in staged)), 'input_sha256': hashes,
               'split_manifest_sha256': expected_splits, 'legacy_source_provenance_sha256': sha256_file(provenance_path),
               'frozen_rule': {'minimum_anchor_characters': MIN_ANCHOR_CHARACTERS, 'minimum_acoustic_windows': 2,
                               'all_overlapping_windows_must_support_PAR': True, 'voice_threshold': VOICE_THRESHOLD,
                               'maximum_padding_seconds': PADDING_SECONDS, 'unsupported_unknown_promoted_to_PAR': False,
                               'previous_coarse_unknown_recovery': 'Only complete words satisfying the frozen reference and acoustic support checks may be recovered.'},
               'manual_speaker_purity_verified': False, 'locked_diagnoses_read': False,
               'final_source_cut_text_used_for_recovery': False, 'speaker_extraction_reference_assisted': True,
               'original_audio_source': 'TAUKADIAL only; no Chou/Delaware replacement audio',
               'selection': 'Frozen existing official-training calibration; no policy selection from test CER'}
    stable_write(target/'build_summary.json', json_bytes(summary))
    return summary


def finalize(data_root: Path, output: Path, asr_results: Path) -> dict:
    root, target = _source_root(data_root, output)
    build_report = json.loads((target/'build_summary.json').read_text())
    parents = _indexed(load_jsonl(target/'staged_manifest.jsonl'), 'recording_id', 'staged manifest')
    pending = _indexed(load_jsonl(target/'asr_pending.jsonl'), 'recording_id', 'pending ASR')
    known, split_hashes = _recording_splits(root)
    if set(parents) != set(known) or build_report['split_manifest_sha256'] != {Path(k).stem: v for k, v in split_hashes.items()}:
        raise ValueError('Staged package no longer matches project recording splits')
    run_hash = sha256_file(asr_results/'run.json') if pending else None
    run = json.loads((asr_results/'run.json').read_text()) if pending else None
    if pending:
        required = {'asr_protocol_version': 1, 'model': {'repo': MODEL_REPO, 'revision': ASR_REVISION},
                    'model_files_sha256': MODEL_FILES, 'temperature': 0, 'condition_on_previous_text': False,
                    'initial_prompt': None, 'word_timestamps': True, 'task': 'transcribe', 'random_seed': 0,
                    'segment_seconds': 30, 'overlap_seconds': 3, 'overlap_policy': 'midpoint ownership with 1.5s margins',
                    'reference_blind': True, 'local_files_only': True}
        if any(key not in run or run[key] != value for key, value in required.items()):
            raise ValueError('New ASR run settings differ from the pinned reference-blind protocol')
    completed = []
    for rid, parent in sorted(parents.items()):
        item = json.loads((target/'staged_records'/f'{rid}.json').read_text())
        prior_path = root/'sources/speech_v2/records'/f'{rid}.json'
        prior_digest = build_report['input_sha256'][f'evidence/{rid}']['v2_record']
        if sha256_file(prior_path) != prior_digest:
            raise ValueError('Original v2 source record changed after build')
        previous = json.loads(prior_path.read_text())
        if item['sha256'] != parent['sha256'] or item['original_wav_sha256'] != known[rid]['sha256']:
            raise ValueError('Staged source identity changed')
        audio = safe_member_path(item['audio_path'], target)
        if sha256_file(audio) != item['sha256']:
            raise ValueError('Staged source audio changed')
        if rid in pending:
            path = asr_results/f'{rid}.json'; result = json.loads(path.read_text())
            if (result['recording_id'] != rid or result['language'] != item['language']
                    or result['source_sha256'] != item['sha256'] or result.get('reference_blind') is not True
                    or result.get('run_settings_sha256') != run_hash):
                raise ValueError('New ASR has incompatible audio identity or decoding provenance')
            item.update(words=result['words'], source_asr_sha256=sha256_file(path), asr_state='locally_redecoded_changed_audio',
                        source_asr_run_sha256=run_hash)
            item['quality_flags'] = [flag for flag in item['quality_flags'] if flag != 'supported_recovery_requires_local_redecode']
            if not any(word['word'].strip() for word in result['words']):
                item['quality_flags'].append('empty_asr_transcript')
        elif (item['sha256'] != item['previous_source_sha256'] or item['sha256'] != previous['sha256']
              or item['words'] != previous['words'] or item['source_asr_sha256'] != previous['source_asr_sha256']):
            raise ValueError('Unchanged audio/ASR reuse differs from the original pinned v2 record')
        validate_words(item['words'], item['num_samples']/SAMPLE_RATE)
        validate_time_map(item['time_map'], item['num_samples'], item['normalized_original_num_samples'])
        if item['eligibility'] not in {'participant_candidate', 'quarantined'} or bool(item['exclusion_reasons']) != (item['eligibility']=='quarantined'):
            raise ValueError('Eligibility and exclusion reasons disagree')
        stable_write(target/'records'/f'{rid}.json', json_bytes(item))
        completed.append(parent)
    provenance = json.loads((root/'sources/speech_v2/provenance.json').read_text())
    if sha256_file(root/'sources/speech_v2/provenance.json') != build_report['legacy_source_provenance_sha256']:
        raise ValueError('Source provenance changed after build')
    provenance.update(schema_version=3, parent_manifest_sha256=sha256_file(root/'sources/speech_v2/manifest.jsonl'),
                      source_recordings=len(completed), source_policy=POLICY, build_summary_sha256=sha256_file(target/'build_summary.json'),
                      eligibility_counts=dict(Counter(r['eligibility'] for r in completed)),
                      local_redecode_run_sha256=run_hash, local_redecode_settings=run,
                      manual_speaker_purity_verified=False, candidate_pcm_verified_against_normalized_original=True,
                      speaker_preprocessing='Existing PAR candidates plus frozen lexical/acoustic supported recovery; unresolved English scopes quarantined',
                      asr_reuse='Unchanged audio retains exact pinned v2 words; changed audio is decoded afresh without reference prompts')
    _write_jsonl(target/'manifest.jsonl', completed)
    stable_write(target/'provenance.json', json_bytes(provenance))
    return {'recordings': len(completed), 'redecoded': len(pending), 'eligibility': provenance['eligibility_counts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'finalize'])
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('data/sources/speech_v3'))
    parser.add_argument('--legacy-root', type=Path)
    parser.add_argument('--normalized-originals', type=Path)
    parser.add_argument('--asr-results', type=Path)
    args = parser.parse_args()
    if args.stage == 'build':
        if args.legacy_root is None or args.normalized_originals is None:
            parser.error('build requires --legacy-root and --normalized-originals')
        result = build(args.data_root, args.legacy_root, args.normalized_originals, args.output)
    else:
        if args.asr_results is None:
            parser.error('finalize requires --asr-results')
        result = finalize(args.data_root, args.output, args.asr_results)
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
