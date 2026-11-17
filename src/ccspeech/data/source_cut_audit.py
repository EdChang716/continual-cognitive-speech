"""Compare frozen TAUKADIAL ASR with final Chou source text and prior windows.

Full-recording final source text is primary. Window scores are agreement with
Whisper/LLM-selected reference spans, not independent ASR or speaker accuracy.
No audio is replaced, decoded, or sent to an external service.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import re

from .audio_identity import availability
from .speech import (SAMPLE_RATE, json_bytes, load_jsonl, original_time_pieces,
                     stable_write, validate_time_map, validate_words)
from .taukadial import sha256_file
from .transcript_audit import _indexed, _recording_splits, compare_text


def union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def retention(windows: list[tuple[float, float]], kept: list[tuple[float, float]]) -> dict:
    """Unique original-time coverage: overlapping windows never count twice."""
    windows, kept = union_intervals(windows), union_intervals(kept)
    omitted = []
    for left, right in windows:
        cursor = left
        for a, b in kept:
            if b <= cursor or a >= right:
                continue
            if a > cursor:
                omitted.append((cursor, min(a, right)))
            cursor = max(cursor, min(b, right))
        if cursor < right:
            omitted.append((cursor, right))
    total = sum(b-a for a, b in windows)
    missing = sum(b-a for a, b in omitted)
    return {'reference_audio_seconds': total, 'retained_audio_seconds': total-missing,
            'omitted_audio_seconds': missing, 'retained_audio_fraction': (total-missing)/total if total else None,
            'omitted_original_intervals': omitted}


def mapped_words(words: list[dict], mapping: list[dict]) -> list[dict]:
    """Preserve separate source pieces for words crossing a concatenation seam."""
    result = []
    for index, word in enumerate(words):
        start, end = float(word['start']), float(word['end'])
        middle = (start+end)/2
        interval = next((p for p in mapping if p['clean_start'] <= middle < p['clean_end']), None)
        if interval is None and mapping and middle == mapping[-1]['clean_end']:
            interval = mapping[-1]
        if interval is None:
            raise ValueError('Candidate word midpoint is outside its original-time map')
        pieces = original_time_pieces(mapping, start, end)
        discontinuous = any(b['original_start_seconds']-a['original_end_seconds'] > 1/SAMPLE_RATE
                            for a, b in zip(pieces, pieces[1:]))
        result.append({'word_index': index, 'word': word['word'], 'source_midpoint':
                       interval['source_start']+middle-interval['clean_start'],
                       'source_pieces': pieces, 'crosses_time_map_boundary': len(pieces)>1,
                       'crosses_concatenation_seam': discontinuous})
    return result


def select_window(words: list[dict], start: float, end: float, duration: float) -> list[dict]:
    return [w for w in words if start <= w['source_midpoint'] < end or
            (abs(end-duration) <= 1/SAMPLE_RATE and w['source_midpoint'] == end)]


def _csv(path: Path) -> list[dict]:
    with path.open(newline='') as source:
        return [{k: v for k, v in r.items() if k not in {'label', 'label_id', 'subgroup'}}
                for r in csv.DictReader(source)]


def _table(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
    writer.writeheader(); writer.writerows(rows)
    stable_write(path, buffer.getvalue().encode())


def _groups(rows: list[dict]) -> list[dict]:
    result = []
    for split in ('all', 'train', 'validation', 'test'):
        subset = [r for r in rows if split == 'all' or r['split'] == split]
        for condition in sorted({r['condition'] for r in subset}):
            chosen = [r for r in subset if r['condition'] == condition]
            counts = {k: sum(r[k] for r in chosen) for k in ('S', 'D', 'I', 'reference_units')}
            result.append({'split': split, 'condition': condition, 'items': len(chosen),
                           'recordings': len({r['recording_id'] for r in chosen}), **counts,
                           'CER': (counts['S']+counts['D']+counts['I'])/counts['reference_units']})
    return result


def _candidate_asr(record: dict, provenance: dict, raw_dir: Path, hashes: dict) -> str:
    available = []
    for run in provenance['asr_runs']:
        name = run['source_run']
        if Path(name).name != name:
            raise ValueError('Unsafe source ASR run name')
        path = raw_dir.parent/name/f"{record['recording_id']}.json"
        if availability(path) != 'materialized':
            continue
        available.append(path)
        digest = sha256_file(path)
        if digest == record['source_asr_sha256']:
            old = json.loads(path.read_text())
            if (old['recording_id'] != record['recording_id'] or old['language'] != 'zh'
                    or old['source_sha256'] != record['sha256'] or old.get('reference_blind') is not True
                    or old['words'] != record['words']):
                raise ValueError('Candidate ASR differs from the portable record')
            hashes[f"candidate_asr/{record['recording_id']}.json"] = digest
            return 'original_candidate_asr_hash_and_words_verified'
    if available:
        raise ValueError('Candidate ASR file hash differs from portable provenance')
    return 'original_candidate_asr_unavailable_using_pinned_portable_record'


def run_audit(data_root: Path, legacy_audit_root: Path, raw_asr_dir: Path, output: Path,
              final_task_splits: Path) -> dict:
    root, destination = data_root.resolve(), output.resolve()
    if not destination.is_relative_to(root/'reports'):
        raise ValueError('Private audit must remain under data/reports')
    files = {name: legacy_audit_root/name for name in ('chou_existing_segment_links.csv',
             'chou_segment_inventory.json', 'chou_segment_waveform_checks.json', 'recording_crosswalk_complete.csv')}
    links = _indexed(_csv(files['chou_existing_segment_links.csv']), 'segment_id', 'old links')
    inventory = _indexed(json.loads(files['chou_segment_inventory.json'].read_text()), 'segment_id', 'old inventory')
    checks = _indexed(json.loads(files['chou_segment_waveform_checks.json'].read_text()), 'segment_id', 'prior waveform checks')
    crosswalk = _indexed(_csv(files['recording_crosswalk_complete.csv']), 'tauk_recording', 'crosswalk')
    if set(links) != set(inventory) or set(links) != set(checks):
        raise ValueError('Prior segment inventories have inconsistent coverage')
    records, hashes = _recording_splits(root)
    hashes.update({name: sha256_file(path) for name, path in files.items()})
    portable = root/'sources/speech_v2'
    parents = _indexed(load_jsonl(portable/'manifest.jsonl'), 'recording_id', 'portable manifest')
    provenance = json.loads((portable/'provenance.json').read_text())
    if provenance.get('split_manifest_sha256') != {Path(k).stem: v for k, v in hashes.items() if k.startswith('processed/')}:
        raise ValueError('Portable provenance differs from recording splits')
    run_path = raw_asr_dir/'run.json'; raw_run = json.loads(run_path.read_text())
    expected_runs = [r['settings'] for r in provenance['asr_runs']]
    if (not expected_runs or any(raw_run.get(k) != r.get(k) for r in expected_runs for k in
            ('model', 'reference_blind', 'word_timestamps', 'segment_seconds', 'overlap_seconds'))
            or raw_run.get('reference_blind') is not True or raw_run.get('word_timestamps') is not True
            or provenance.get('candidate_pcm_verified_against_normalized_original') is not True):
        raise ValueError('ASR or original-audio provenance is inconsistent')
    hashes.update(portable_manifest=sha256_file(portable/'manifest.jsonl'),
                  portable_provenance=sha256_file(portable/'provenance.json'), raw_asr_run=sha256_file(run_path))
    current, source_status = {}, {}
    for name in sorted({r['transcript_csv'] for r in links.values()}):
        path = Path(name); state = availability(path)
        source_status[name] = state
        if state == 'materialized':
            hashes[f'final_source_csv/{path.name}'] = sha256_file(path)
            for row in _csv(path):
                if row['dataset'] != 'Chou':
                    continue
                sid = row['segment_id']
                if sid in current:
                    raise ValueError('Duplicate final source segment')
                current[sid] = row
    grouped = defaultdict(list)
    for sid, link in links.items():
        rid = link['tauk_recording']; known = records[rid]; inv = inventory[sid]; proof = checks[sid]
        index = int(link['source_segment_index']); suffix = re.search(r'_seg(\d+)$', sid)
        if (known['language'] != 'zh' or crosswalk[rid]['source_dataset'] != 'Chou'
                or crosswalk[rid]['source_recording'] != link['source_recording']
                or crosswalk[rid]['tauk_participant_id'] != known['participant_id']
                or link['tauk_split'] != known['official_split'] or index < 0 or suffix is None or int(suffix[1]) != index
                or inv['recording_id'] != link['source_recording'] or inv['old_csv'] != link['transcript_csv']
                or not inv['sample_count_matches'] or int(inv['actual_samples']) != int(inv['declared_samples'])
                or int(inv['rate']) != SAMPLE_RATE or inv['actual_samples'] <= 0
                or proof['direct_27s_hop_correlation'] < .98 or abs(proof['duration_difference_seconds']) > 1/SAMPLE_RATE):
            raise ValueError('Prior window identity, offset, or waveform proof is inconsistent')
        if source_status[link['transcript_csv']] == 'materialized':
            final = current.get(sid)
            if final is None or any(final[k] != link[k] for k in ('matched_transcript', 'whisper_transcription', 'match_method')):
                raise ValueError('Final source segment text differs from pinned archived reference')
            if final['recording_id'] != link['source_recording'] or int(final['segment_index']) != index:
                raise ValueError('Final source segment identity differs')
        grouped[rid].append((link, index*27., index*27.+inv['actual_samples']/inv['rate']))
    task_texts = {}
    for split in ('train', 'validation', 'test'):
        path = final_task_splits/f'{split}.csv'
        if availability(path) != 'materialized':
            raise ValueError('Final full-task reference CSV is unavailable; no silent text substitution')
        hashes[f'final_task_splits/{split}.csv'] = sha256_file(path)
        for row in _csv(path):
            if row['dataset'] != 'Chou':
                continue
            source_id = Path(row['audio_path']).stem
            if source_id in task_texts:
                raise ValueError('Duplicate final full-task source identity')
            task_texts[source_id] = row['transcript']
    full_references = {}
    for rid, known in records.items():
        if known['language'] == 'zh':
            match = crosswalk[rid]
            if match['source_dataset'] != 'Chou' or match['tauk_participant_id'] != known['participant_id']:
                raise ValueError('Full-task crosswalk participant or language differs')
            source_id = Path(match['source_recording']).stem
            if source_id not in task_texts:
                raise ValueError('Final full-task reference does not cover every Chou recording')
            full_references[rid] = task_texts[source_id]
            grouped.setdefault(rid, [])
    full_rows, segment_rows, recording_details, window_details = [], [], [], []
    for rid, windows in sorted(grouped.items()):
        known, parent = records[rid], parents[rid]
        path = portable/'records'/f'{rid}.json'; item = json.loads(path.read_text())
        raw_path = raw_asr_dir/f'{rid}.json'; raw = json.loads(raw_path.read_text())
        if (any(r['participant_id'] != known['participant_id'] or r['language'] != 'zh' or r['recording_id'] != rid
                or r['original_wav_sha256'] != known['sha256'] for r in (item, parent))
                or item['sha256'] != parent['sha256'] or item['sample_rate'] != SAMPLE_RATE
                or item.get('candidate_pcm_matches_normalized_original_intervals') is not True
                or raw['recording_id'] != rid or raw['language'] != 'zh' or raw.get('reference_blind') is not True
                or raw['source_sha256'] != item['normalized_original_sha256']):
            raise ValueError('TAUKADIAL original/candidate/ASR identity or hash differs')
        duration = item['normalized_original_num_samples']/SAMPLE_RATE
        validate_words(raw['words'], duration)
        validate_words(item['words'], item['num_samples']/SAMPLE_RATE)
        validate_time_map(item['time_map'], item['num_samples'], item['normalized_original_num_samples'])
        candidate_status = _candidate_asr(item, provenance, raw_asr_dir, hashes)
        hashes[f'portable_records/{rid}.json'], hashes[f'raw_asr/{rid}.json'] = sha256_file(path), sha256_file(raw_path)
        candidate = mapped_words(item['words'], item['time_map'])
        raw_words = [{**w, 'source_midpoint': (w['start']+w['end'])/2} for w in raw['words']]
        kept = [(p['source_start'], p['source_end']) for p in item['time_map']]
        identity = {'recording_id': rid, 'participant_id': known['participant_id'], 'split': known['split']}
        final_texts = {current[w[0]['segment_id']]['transcript'] for w in windows if w[0]['segment_id'] in current}
        if len(final_texts) > 1:
            raise ValueError('Final full-recording transcripts are inconsistent across source segments')
        if final_texts and final_texts != {full_references[rid]}:
            raise ValueError('Final window full transcript differs from the final task reference')
        detail = {**identity, 'candidate_asr_provenance': candidate_status,
                  'full_recording_audio': retention([(0, duration)], kept),
                  'unique_prior_window_audio': retention([(a, b) for _, a, b in windows], kept),
                  'candidate_word_seam_crossings': sum(w['crosses_concatenation_seam'] for w in candidate),
                  'final_full_reference_available': True, 'prior_window_count': len(windows)}
        reference = full_references[rid]
        for condition, text in [('current_raw', raw['text']), ('current_candidate', ''.join(w['word'] for w in item['words']))]:
            counts, _ = compare_text(reference, text, 'zh')
            full_rows.append({**identity, 'condition': condition, **counts})
        for link, start, end in windows:
            if end > duration+1/SAMPLE_RATE:
                raise ValueError('Prior source window exceeds TAUKADIAL original duration')
            selected_raw = select_window(raw_words, start, end, duration)
            selected_candidate = select_window(candidate, start, end, duration)
            coverage = retention([(start, end)], kept)
            wd = {**identity, 'segment_id': link['segment_id'], 'original_start': start, 'original_end': end,
                  'source_csv_status': source_status[link['transcript_csv']], 'match_method': link['match_method'], **coverage,
                  'raw_boundary_crossing_words': sum(w['start'] < start or w['end'] > end for w in selected_raw),
                  'candidate_seam_crossing_words': sum(w['crosses_concatenation_seam'] for w in selected_candidate),
                  'candidate_boundary_crossing_words': sum(any(p['original_start_seconds'] < start or p['original_end_seconds'] > end
                                                          for p in w['source_pieces']) for w in selected_candidate)}
            window_details.append(wd)
            hypotheses = {'prior_whisper': link['whisper_transcription'],
                          'current_raw_projected': ''.join(w['word'] for w in selected_raw),
                          'current_candidate_projected': ''.join(w['word'] for w in selected_candidate)}
            for condition, text in hypotheses.items():
                counts, _ = compare_text(link['matched_transcript'], text, 'zh')
                segment_rows.append({**identity, 'segment_id': link['segment_id'], 'condition': condition, **counts,
                                     **{k: v for k, v in wd.items() if k.endswith('_words') or k == 'retained_audio_fraction'}})
        recording_details.append(detail)
    summary = {'schema_version': 1, 'language': 'zh', 'recordings_with_prior_windows': sum(bool(w) for w in grouped.values()),
               'prior_windows': len(links), 'current_chou_recordings': sum(r['language']=='zh' for r in records.values()),
               'final_full_reference_recordings': sum(r['final_full_reference_available'] for r in recording_details),
               'source_csv_availability': dict(Counter(source_status.values())),
               'candidate_asr_provenance': dict(Counter(r['candidate_asr_provenance'] for r in recording_details)),
               'primary_full_recording_final_reference': _groups(full_rows),
               'secondary_overlapping_window_reference_agreement': _groups(segment_rows),
               'unique_audio_coverage': {key: {metric: sum(r[key][metric] for r in recording_details)
                   for metric in ('reference_audio_seconds', 'retained_audio_seconds', 'omitted_audio_seconds')}
                   for key in ('full_recording_audio', 'unique_prior_window_audio')},
               'input_sha256': hashes, 'locked_diagnoses_read': False, 'new_asr_or_external_calls': False,
               'project_audio_source': 'TAUKADIAL originals only; final source audio/text used for validation only',
               'notes': ['Primary text is one final full-task split_merge transcript per recording, checked against all final window CSV copies when available; overlapping spans are never concatenated.',
                         'Candidate-vs-full-reference CER combines recognition and removed content; it does not establish speaker purity.',
                         'Secondary reference spans were selected using prior Whisper/LLM output. These are agreement scores, not independent ASR accuracy.',
                         'Prior Whisper is documented as medium; archived rows do not pin its weights/decoding run. Current ASR uses the pinned model in provenance.',
                         'Current window hypotheses are timestamp projections of frozen ASR, not new decoding of those prior windows.',
                         'Window error totals count overlapping reference words more than once; audio coverage is unioned per recording.',
                         'Waveform correspondence of old windows is prior cached evidence, not a fresh final-cut audio or speaker-label audit.',
                         'Source references assisted existing extraction; full-recording comparisons are not independent speaker-label validation.']}
    _table(destination/'full_recording_reference_agreement.csv', full_rows)
    _table(destination/'window_reference_agreement.csv', segment_rows)
    stable_write(destination/'recording_details.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in recording_details).encode())
    stable_write(destination/'window_details.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in window_details).encode())
    stable_write(destination/'summary.json', json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--legacy-audit-root', type=Path, required=True)
    parser.add_argument('--raw-asr-dir', type=Path, required=True)
    parser.add_argument('--final-task-splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/reports/source_cut_reference_v1'))
    args = parser.parse_args()
    result = run_audit(args.data_root, args.legacy_audit_root, args.raw_asr_dir, args.output, args.final_task_splits)
    print(json.dumps({k: v for k, v in result.items() if k not in {'input_sha256', 'notes'}}, indent=2))


if __name__ == '__main__':
    main()
