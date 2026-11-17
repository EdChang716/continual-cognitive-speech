"""Compare private TAUKADIAL ASR artifacts with identity-bound source references.

This audit never changes audio or transcripts, never selects model settings, and
never reads locked diagnoses. Its errors combine recognition and retained-source
coverage; they do not measure speaker purity or provide segment-level gold text.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import unicodedata

from .speech import (SAMPLE_RATE, json_bytes, load_jsonl, match_word_timestamps,
                     stable_write, validate_words, window_bounds)
from .taukadial import sha256_file

CONDITIONS = ('raw', 'candidate', 'window')


def clean_chat(text: str) -> str:
    """Match the earlier audit's CHAT cleanup without correcting spoken words."""
    text = re.sub(r'\x15[^\x15]*\x15', ' ', text)
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'&=[^\s]+', ' ', text)
    text = re.sub(r'&\*[A-Za-z0-9]+:[^\s]+', ' ', text)
    text = re.sub(r'&\+[^\s]+', ' ', text)
    text = text.replace('&-', '')
    text = re.sub(r'\b0[^\s]+', ' ', text)
    text = re.sub(r'\b(?:xxx|yyy|www)\b', ' ', text)
    text = re.sub(r'@[^\s]+', ' ', text)
    text = text.replace('<', ' ').replace('>', ' ')
    text = re.sub(r'\([^)]*\)', ' ', text)
    return ' '.join(text.split())


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC
    return OpenCC('t2s')


def normalize_units(text: str, language: str) -> list[str]:
    text = unicodedata.normalize('NFKC', clean_chat(text)).lower().replace('’', "'")
    if language == 'zh':
        return [character for character in _converter().convert(text) if character.isalnum()]
    if language != 'en':
        raise ValueError('Expected en/zh reference language')
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)


def compare_text(reference: str, hypothesis: str, language: str) -> tuple[dict, list[dict]]:
    from rapidfuzz.distance import Levenshtein
    ref, hyp = normalize_units(reference, language), normalize_units(hypothesis, language)
    if not ref:
        raise ValueError('Cannot score an empty source reference')
    operations = Levenshtein.editops(ref, hyp)
    counts = {key: sum(op.tag == tag for op in operations)
              for key, tag in (('S', 'replace'), ('D', 'delete'), ('I', 'insert'))}
    counts.update(reference_units=len(ref), hypothesis_units=len(hyp),
                  metric='CER' if language == 'zh' else 'WER', error_rate=len(operations) / len(ref),
                  normalized_reference_sha256=hashlib.sha256('\0'.join(ref).encode()).hexdigest())
    alignment = [{'tag': op.tag, 'reference_start': op.src_start, 'reference_end': op.src_end,
                  'hypothesis_start': op.dest_start, 'hypothesis_end': op.dest_end,
                  'reference_tokens': ref[op.src_start:op.src_end],
                  'hypothesis_tokens': hyp[op.dest_start:op.dest_end]}
                 for op in operations.as_opcodes()]
    return counts, alignment


def window_word_indices(words: list[dict], windows: list[tuple[float, float]], duration: float) -> list[int]:
    """Take a union of source WORD INDICES, preserving genuine repeated words."""
    validate_words(words, duration)
    selected = []
    for index, word in enumerate(words):
        midpoint = (float(word['start']) + float(word['end'])) / 2
        if any(start <= midpoint < end or
               (abs(end-duration) <= 1/SAMPLE_RATE and midpoint == end)
               for start, end in windows):
            selected.append(index)
    return selected


def _indexed(rows: list[dict], key: str, name: str) -> dict:
    result = {}
    for row in rows:
        identifier = row[key]
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f'Duplicate or invalid identity in {name}')
        # IDs become file basenames; never accept path fragments supplied by data.
        if Path(identifier).name != identifier or identifier in {'.', '..'} or '\\' in identifier:
            raise ValueError(f'Unsafe identity in {name}')
        result[identifier] = row
    return result


def _read_csv(path: Path, split: str) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        if split == 'test' and 'label' in (reader.fieldnames or []):
            raise ValueError('Test manifest contains diagnoses; refusing to read rows')
        return [{key: value for key, value in row.items() if key != 'label'} for row in reader]


def _recording_splits(root: Path) -> tuple[dict, dict]:
    records, owners, hashes = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        path = root / 'processed/taukadial' / f'{split}.csv'
        hashes[f'processed/taukadial/{split}.csv'] = sha256_file(path)
        for row in _read_csv(path, split):
            rid, pid = row['recording_id'], row['participant_id']
            if row['split'] != split or rid in records or (pid in owners and owners[pid] != split):
                raise ValueError('Recording/participant split mismatch')
            records[rid] = row
            owners[pid] = split
    return _indexed(list(records.values()), 'recording_id', 'recording splits'), hashes


def _window_splits(root: Path, recordings: dict) -> tuple[dict, dict]:
    grouped, hashes, segment_ids = defaultdict(list), {}, set()
    for split in ('train', 'validation', 'test'):
        path = root / 'processed/speech_v2' / f'{split}.csv'
        hashes[f'processed/speech_v2/{split}.csv'] = sha256_file(path)
        for row in _read_csv(path, split):
            rid = row['recording_id']
            if rid not in recordings:
                raise ValueError('Window has no original recording identity')
            original = recordings[rid]
            if (row['split'] != split or original['split'] != split or
                    row['participant_id'] != original['participant_id'] or row['language'] != original['language']):
                raise ValueError('Window identity or split mismatch')
            if not row['segment_id'] or row['segment_id'] in segment_ids:
                raise ValueError('Duplicate or empty window identity')
            segment_ids.add(row['segment_id'])
            grouped[rid].append(row)
    return grouped, hashes


def summarize_records(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        for split in ('all', row['split']):
            grouped[row['language'], split, row['reference_scope_verified']].append(row)
    summaries = []
    for (language, split, verified), rows in sorted(grouped.items()):
        group = {'language': language, 'split': split, 'reference_scope_verified': verified,
                 'recordings': len(rows), 'participants': len({r['participant_id'] for r in rows}),
                 'reference_units': sum(r['counts']['raw']['reference_units'] for r in rows), 'conditions': {}}
        for condition in CONDITIONS:
            counts = {key: sum(row['counts'][condition][key] for row in rows)
                      for key in ('S', 'D', 'I', 'reference_units', 'hypothesis_units')}
            counts.update(metric='CER' if language == 'zh' else 'WER',
                          error_rate=(counts['S'] + counts['D'] + counts['I']) / counts['reference_units'])
            group['conditions'][condition] = counts
        summaries.append(group)
    return summaries


def _verify_available_source_files(references: dict) -> dict:
    """Check materialized human-source files, without blocking on cloud stubs.

    The pinned archived reference manifest remains the comparison input even
    when a prior workstation source has moved or become an iCloud placeholder.
    Such files are explicitly unverified here, never reported as freshly read.
    """
    files = {}
    for row in references.values():
        path, digest = row['source_transcript_path'], row['source_transcript_sha256']
        if path in files and files[path] != digest:
            raise ValueError('Conflicting source transcript hashes')
        files[path] = digest
    statuses = []
    for name, expected in sorted(files.items()):
        path = Path(name)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            status = 'missing_original_using_pinned_archive'
        else:
            # SF_DATALESS on macOS denotes a File Provider placeholder. Opening
            # it can synchronously wait for network hydration indefinitely.
            dataless = sys.platform == 'darwin' and bool(getattr(metadata, 'st_flags', 0) & 0x40000000)
            if dataless:
                status = 'unmaterialized_original_using_pinned_archive'
            else:
                if sha256_file(path) != expected:
                    raise ValueError('Available human-source transcript hash differs from archived reference provenance')
                status = 'original_file_hash_verified'
        statuses.append({'expected_sha256': expected, 'status': status})
    return {'files': len(files), 'verified_original_files': sum(row['status'] == 'original_file_hash_verified' for row in statuses),
            'archived_reference_fallback_files': sum(row['status'] != 'original_file_hash_verified' for row in statuses),
            'file_checks': statuses,
            'comparison_text': 'Pinned archived reference manifest; original-file availability does not alter the reference words.'}


def run_audit(data_root: Path, reference_manifest: Path, resolutions: Path,
              raw_asr_dir: Path, output: Path) -> dict:
    root = data_root.resolve()
    source_root = root / 'sources/speech_v2'
    provenance_path = source_root / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    reference_digest = sha256_file(reference_manifest)
    if reference_digest != provenance['original_reference_manifest_sha256']:
        raise ValueError('Reference manifest hash differs from prepared speech provenance')
    references = _indexed(load_jsonl(reference_manifest), 'recording_id', 'reference manifest')
    human_sources = _verify_available_source_files(references)
    resolved = _indexed(load_jsonl(resolutions), 'recording_id', 'reference resolutions')
    parents = _indexed(load_jsonl(source_root / 'manifest.jsonl'), 'recording_id', 'portable sources')
    recordings, split_hashes = _recording_splits(root)
    if set(references) != set(parents) or set(parents) != set(recordings):
        raise ValueError('Reference/source/recording coverage differs; no guessed matches allowed')
    english = {rid for rid, row in references.items() if row['language'] == 'en'}
    if set(resolved) != english:
        raise ValueError('English resolution coverage differs from matched English recordings')
    if provenance.get('split_manifest_sha256') != {Path(key).stem: value for key, value in split_hashes.items()}:
        raise ValueError('Recording split hashes differ from prepared source provenance')
    windows, window_hashes = _window_splits(root, recordings)
    input_hashes = {**split_hashes, **window_hashes,
                    'source_provenance': sha256_file(provenance_path),
                    'portable_source_manifest': sha256_file(source_root / 'manifest.jsonl'),
                    'reference_manifest': reference_digest, 'reference_resolutions': sha256_file(resolutions)}
    raw_run_path = raw_asr_dir / 'run.json'
    raw_run_verified = False
    if raw_run_path.exists():
        raw_run = json.loads(raw_run_path.read_text())
        expected_models = [run['settings']['model'] for run in provenance['asr_runs']]
        if (not expected_models or any(raw_run.get('model') != model for model in expected_models)
                or raw_run.get('reference_blind') is not True or raw_run.get('word_timestamps') is not True):
            raise ValueError('Raw ASR run model or reference-blind timestamp settings differ')
        input_hashes['raw_asr/run.json'] = sha256_file(raw_run_path)
        raw_run_verified = True
    details, csv_rows = [], []
    for rid in sorted(recordings):
        reference_row, parent, known = references[rid], parents[rid], recordings[rid]
        source_path = source_root / 'records' / f'{rid}.json'
        source = json.loads(source_path.read_text())
        for row in (reference_row, parent, source):
            if (row['recording_id'] != rid or row['participant_id'] != known['participant_id'] or
                    row['language'] != known['language']):
                raise ValueError('Source/reference participant or language mismatch')
        original_hash = known['sha256']
        if any(row['original_wav_sha256'] != original_hash for row in (reference_row, parent, source)):
            raise ValueError('Source/reference original audio hash mismatch')
        if (source['sha256'] != parent['sha256'] or
                source['normalized_original_sha256'] != reference_row['normalized_wav_sha256']):
            raise ValueError('Candidate or normalized original audio hash mismatch')
        if source.get('sample_rate') != SAMPLE_RATE:
            raise ValueError('Unexpected candidate sample rate')
        duration = source['num_samples'] / SAMPLE_RATE
        if abs(duration - source['duration_seconds']) > 1 / SAMPLE_RATE:
            raise ValueError('Candidate duration differs from sample count')
        words = source['words']
        validate_words(words, duration)
        current_windows = sorted(windows.get(rid, []), key=lambda r: float(r['start_seconds']))
        bounds = [(float(row['start_seconds']), float(row['end_seconds'])) for row in current_windows]
        expected_samples, _ = window_bounds(source['num_samples'])
        if [(round(start*SAMPLE_RATE), round(end*SAMPLE_RATE)) for start, end in bounds] != expected_samples:
            raise ValueError('Retained windows differ from the prepared 30/27/15-second policy')
        for row, (start, end) in zip(current_windows, bounds):
            if row['source_sha256'] != source['sha256'] or row['original_wav_sha256'] != original_hash:
                raise ValueError('Window audio source hashes differ')
            expected = match_word_timestamps(words, start, end, duration)['matched_transcript']
            if row['matched_transcript'] != expected:
                raise ValueError('Window transcript differs from timestamp-selected source words')
        raw_path = raw_asr_dir / f'{rid}.json'
        raw = json.loads(raw_path.read_text())
        if (raw['recording_id'] != rid or raw['language'] != known['language'] or
                raw['source_sha256'] != source['normalized_original_sha256'] or raw.get('reference_blind') is not True):
            raise ValueError('Raw ASR identity, original source hash, or decoding provenance mismatch')
        language = known['language']
        if language == 'zh':
            if reference_row['source_dataset'] != 'Chou':
                raise ValueError('Mandarin reference must match the verified Chou source')
            reference = reference_row['reference_participant']
            verified = reference_row.get('reference_scope_review_complete') is True
            scope = 'full_recording_source_participant_text_without_word_timestamps'
        elif language == 'en':
            if reference_row['source_dataset'] != 'Delaware':
                raise ValueError('English reference must match the verified Delaware source')
            resolution = resolved[rid]
            reference = resolution['reference']
            verified = resolution.get('scope_review_complete') is True
            scope = resolution.get('reference_scope', 'source_clip_participant_text')
        else:
            raise ValueError('Unknown source language')
        covered = window_word_indices(words, bounds, duration)
        hypotheses = {'raw': raw['text'], 'candidate': ''.join(word['word'] for word in words).strip(),
                      'window': ''.join(words[index]['word'] for index in covered).strip()}
        if any(not isinstance(value, str) for value in (reference, *hypotheses.values())):
            raise ValueError('Reference and ASR texts must be strings')
        counts, alignments = {}, {}
        for condition, hypothesis in hypotheses.items():
            counts[condition], alignments[condition] = compare_text(reference, hypothesis, language)
            csv_rows.append({'recording_id': rid, 'participant_id': known['participant_id'], 'language': language,
                             'split': known['split'], 'reference_scope_verified': verified,
                             'source_dataset': reference_row['source_dataset'], 'condition': condition,
                             'retained_windows': len(bounds), **counts[condition]})
        flags = list(source.get('quality_flags', []))
        if not verified:
            flags.append('reference_scope_unverified')
        if not bounds:
            flags.append('whole_recording_omitted_by_minimum_tail_policy')
        details.append({'recording_id': rid, 'participant_id': known['participant_id'], 'language': language,
                        'split': known['split'], 'reference_scope_verified': verified,
                        'source_dataset': reference_row['source_dataset'], 'reference_scope': scope,
                        'reference': reference, 'raw_asr': hypotheses['raw'], 'candidate_asr': hypotheses['candidate'],
                        'window_asr': hypotheses['window'], 'counts': counts, 'alignments': alignments,
                        'quality_flags': sorted(set(flags)), 'source_word_count': len(words),
                        'window_covered_word_count': len(covered), 'retained_windows': len(bounds)})
        input_hashes[f'source_records/{rid}.json'] = sha256_file(source_path)
        input_hashes[f'raw_asr/{rid}.json'] = sha256_file(raw_path)
    summary = {'schema_version': 1, 'recordings': len(details),
               'participants': len({row['participant_id'] for row in details}), 'conditions': list(CONDITIONS),
               'groups': summarize_records(details),
               'whole_recordings_omitted_by_window_policy': sum(row['retained_windows'] == 0 for row in details),
               'verified_reference_recordings': sum(row['reference_scope_verified'] for row in details),
               'unverified_reference_recordings': sum(not row['reference_scope_verified'] for row in details),
               'checks': {'reference_manifest_hash': True, 'source_identity_and_audio_hash_binding': True,
                          'raw_asr_source_hash_binding': True, 'all_csv_transcripts_match_source_words': True,
                          'raw_asr_run_metadata_verified': raw_run_verified,
                          'overlap_deduplicated_by_source_word_index': True, 'same_reference_all_conditions': True,
                          'locked_diagnoses_read': False, 'audio_or_transcripts_modified': False},
               'normalization': 'Earlier audit convention: CHAT cleanup, NFKC/lowercase; English Unicode words/apostrophes; Mandarin OpenCC t2s then alphanumeric characters; repetitions and filled pauses retained.',
               'provenance': {'input_sha256': input_hashes, 'human_source_files': human_sources},
               'notes': ['WER/CER include ASR errors and input coverage differences; lower is better and values may exceed 1.',
                         'Raw audio includes other speakers; raw-versus-participant reference is not speaker-matched ASR WER.',
                         'Candidate selection was reference-assisted; these scores are not independent speaker-purity validation.',
                         'Window condition is the source-word-index union of retained windows, not concatenated overlapping text.',
                         'All conditions use complete identical per-recording references, including text omitted by windowing.',
                         'Chou references have no word times; this audit does not establish per-window human gold transcripts.',
                         'English unverified reference scopes are reported separately, never silently trimmed to ASR.',
                         'English reference scope was previously resolved using alignment and ASR-assisted task checks; verified means prior pipeline checks, not new manual gold annotation.',
                         'Test split is included only for transcript/data auditing; no diagnosis, model tuning, or preprocessing selection is performed.']}
    output = output.resolve()
    if not output.is_relative_to(root / 'reports'):
        raise ValueError('Private transcript audit output must stay under data/reports')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    stable_write(output / 'per_recording.csv', buffer.getvalue().encode())
    stable_write(output / 'differences.jsonl', ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in details).encode())
    stable_write(output / 'summary.json', json_bytes(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--resolutions', type=Path, required=True)
    parser.add_argument('--raw-asr-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/reports/transcript_comparison_v1'))
    args = parser.parse_args()
    report = run_audit(args.data_root, args.reference_manifest, args.resolutions, args.raw_asr_dir, args.output)
    print(json.dumps({key: report[key] for key in ('recordings', 'participants', 'whole_recordings_omitted_by_window_policy',
                                                 'verified_reference_recordings', 'unverified_reference_recordings', 'groups')}, indent=2))


if __name__ == '__main__':
    main()
