"""Fresh original-WAV identity/clock checks, separate from prior cached evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly

from .speech import json_bytes, stable_write
from .taukadial import sha256_file

RATE = 2000


def availability(path: Path) -> str:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return 'missing_original'
    if not path.is_file():
        return 'not_regular_file'
    if sys.platform == 'darwin' and getattr(metadata, 'st_flags', 0) & 0x40000000:
        return 'cloud_placeholder_not_read'
    return 'materialized'


def read_original(path: Path, expected_pcm: str, expected_file: str | None = None) -> tuple[dict, np.ndarray]:
    """Use the legacy inventory's exact decoded-PCM hashing convention."""
    if availability(path) != 'materialized':
        raise ValueError('Unavailable original must not be opened or hydrated')
    if expected_file is not None and sha256_file(path) != expected_file:
        raise ValueError('Current TAUKADIAL original file SHA256 mismatch')
    rate, samples = wavfile.read(path)
    digest = hashlib.sha256(str((rate, samples.shape, samples.dtype)).encode() + samples.tobytes()).hexdigest()
    if digest != expected_pcm:
        raise ValueError('Original decoded PCM differs from the archived inventory')
    if samples.ndim not in {1, 2} or not len(samples):
        raise ValueError('Expected nonempty mono/stereo WAV samples')
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    mono = samples.astype(np.float64)
    if samples.dtype == np.uint8:
        mono -= 128
    if channels > 1:
        mono = mono.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError('Nonfinite original waveform')
    divisor = gcd(int(rate), RATE)
    reduced = resample_poly(mono, RATE//divisor, int(rate)//divisor)
    return {'sample_rate': int(rate), 'channels': channels, 'frames': len(samples),
            'duration_seconds': len(samples)/rate, 'pcm_sha256': digest}, reduced


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a, b = first-first.mean(), second-second.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a, b)/denominator, -1, 1)) if denominator > 0 else None


def _ncc(longer: np.ndarray, query: np.ndarray) -> tuple[float | None, int]:
    n = len(query)
    if len(longer) < n or n == 0:
        return None, 0
    centered = query-query.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 0:
        return None, 0
    sums = np.concatenate(([0.], np.cumsum(longer)))
    squares = np.concatenate(([0.], np.cumsum(longer*longer)))
    variance = np.maximum(squares[n:]-squares[:-n]-(sums[n:]-sums[:-n])**2/n, 0)
    denominator = np.sqrt(variance*energy)
    scores = np.full(len(variance), -np.inf)
    valid = variance > max(float(variance.max())*1e-10, 1e-18)
    numerator = correlate(longer, centered, mode='valid', method='fft')
    scores[valid] = numerator[valid]/denominator[valid]
    scores[np.abs(scores)>1.000001] = -np.inf
    index = int(np.argmax(scores))
    return (float(np.clip(scores[index], -1, 1)), index) if np.isfinite(scores[index]) else (None, 0)


def anchor_starts(samples: np.ndarray, count: int = 5) -> list[int]:
    """Select one energetic one-second window from each nonoverlapping time bin."""
    if len(samples) < count*RATE:
        return []
    threshold = max(float(np.var(samples))*1e-6, 1e-18)
    selected = []
    edges = np.linspace(0, len(samples), count+1).astype(int)
    for left, right in zip(edges[:-1], edges[1:]):
        options = np.linspace(left, right-RATE, 11).astype(int)
        energy = [float(np.var(samples[start:start+RATE])) for start in options]
        best = int(np.argmax(energy))
        if energy[best] > threshold:
            selected.append(int(options[best]))
    return selected


def check_constant_offset(tauk: np.ndarray, source: np.ndarray, offset: float) -> dict:
    """Check a proposed source_time - TAUK_time shift; never infer CHAT time."""
    shift = round(offset*RATE)
    x0, y0 = max(0, -shift), max(0, shift)
    length = min(len(tauk)-x0, len(source)-y0)
    if length <= 0:
        raise ValueError('Proposed offset has no waveform overlap')
    first, second = tauk[x0:x0+length], source[y0:y0+length]
    corr = _correlation(first, second)
    xc, yc = first-first.mean(), second-second.mean()
    xenergy, yenergy = float(np.dot(xc, xc)), float(np.dot(yc, yc))
    gain = float(np.dot(xc, yc)/xenergy) if xenergy > 0 else None
    bias = float(second.mean()-gain*first.mean()) if gain is not None else None
    residual = float(np.linalg.norm(second-(gain*first+bias))/np.sqrt(yenergy)) if gain is not None and yenergy > 0 else None
    anchors = []
    for start in anchor_starts(tauk):
        predicted = start+shift
        lo, hi = max(0, predicted-400), min(len(source), predicted+RATE+400)
        score, local = _ncc(source[lo:hi], tauk[start:start+RATE])
        anchors.append({'tauk_seconds': start/RATE, 'correlation': score,
                        'offset_seconds': (lo+local-start)/RATE if score is not None else None})
    offsets = [a['offset_seconds'] for a in anchors if a['offset_seconds'] is not None]
    spread = max(offsets)-min(offsets) if offsets else None
    complete = length == len(tauk)
    accepted = (complete and corr is not None and corr >= .98 and len(anchors) == 5 and len(offsets) == 5
                and min(a['correlation'] for a in anchors) >= .98 and spread <= .002)
    return {'proposed_offset_seconds': offset, 'full_2khz_overlap_correlation': corr,
            'tauk_coverage_fraction': length/len(tauk), 'affine_source_from_tauk_gain': gain,
            'affine_source_from_tauk_dc': bias, 'normalized_affine_residual': residual,
            'anchor_count': len(anchors), 'anchor_min_correlation': min((a['correlation'] for a in anchors if a['correlation'] is not None), default=None),
            'anchor_offset_spread_seconds': spread, 'constant_offset_supported': accepted, 'anchors': anchors}


def _read_csv(path: Path, test: bool = False) -> list[dict]:
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        if test and 'label' in (reader.fieldnames or []):
            raise ValueError('Test manifest must not expose diagnoses')
        return [{k:v for k,v in row.items() if k != 'label'} for row in reader]


def _safe_record_id(value: str) -> str:
    if not value or Path(value).name != value or value in {'.', '..'} or '\\' in value:
        raise ValueError('Unsafe recording identity')
    return value


def run_audit(data_root: Path, legacy_audit_root: Path, chat_resolutions: Path, output: Path) -> dict:
    root = data_root.resolve()
    destination = output.resolve()
    if not destination.is_relative_to(root/'reports'):
        raise ValueError('Private audit must remain under data/reports')
    cross_path, inventory_path = legacy_audit_root/'recording_crosswalk_complete.csv', legacy_audit_root/'inventory.json'
    cross = _read_csv(cross_path)
    inventory = json.loads(inventory_path.read_text())
    by_path = {row['location']:row for row in inventory}
    tauk_inventory = {Path(row['location']).stem:row for row in inventory if row['group'].startswith('TAUK_')}
    resolved_rows = [json.loads(line) for line in chat_resolutions.read_text().splitlines() if line.strip()]
    resolved = {row['recording_id']:row for row in resolved_rows}
    if len(resolved) != len(resolved_rows):
        raise ValueError('Duplicate CHAT resolution')
    recordings, participants, hashes = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        path = root/'processed/taukadial'/f'{split}.csv'
        hashes[f'processed/taukadial/{split}.csv'] = sha256_file(path)
        for row in _read_csv(path, split == 'test'):
            rid, pid = _safe_record_id(row['recording_id']), row['participant_id']
            if rid in recordings or row['split'] != split or (pid in participants and participants[pid] != split):
                raise ValueError('Recording/participant split mismatch')
            recordings[rid], participants[pid] = row, split
    ids = [_safe_record_id(row['tauk_recording']) for row in cross]
    if len(set(ids)) != len(ids) or set(ids) != set(recordings):
        raise ValueError('Crosswalk coverage differs from current recordings')
    if set(resolved) != {rid for rid,r in recordings.items() if r['language']=='en'}:
        raise ValueError('CHAT resolution coverage differs from English recordings')
    task_path = legacy_audit_root.parent/'taukadial_asr/task_source_verification.json'
    task_checks = {r['recording_id']:r for r in json.loads(task_path.read_text())} if task_path.exists() else {}
    hashes.update(crosswalk=sha256_file(cross_path), inventory=sha256_file(inventory_path), chat_resolutions=sha256_file(chat_resolutions))
    if task_path.exists(): hashes['processed_task_anchors'] = sha256_file(task_path)
    rows, detail_rows = [], []
    previous_source, cached_source = None, None
    for index, match in enumerate(sorted(cross, key=lambda r:r['source_path']), 1):
        rid = match['tauk_recording']; known = recordings[rid]
        expected_language = {'Chou':'zh', 'Delaware':'en'}.get(match['source_dataset'])
        if (expected_language != known['language'] or match['tauk_participant_id'] != known['participant_id']
                or match['tauk_split'] != known['official_split']):
            raise ValueError('Crosswalk source/participant/split mismatch')
        original = (root/known['audio_path']).resolve()
        if not original.is_relative_to(root): raise ValueError('Original audio path escapes data root')
        source = Path(match['source_path'])
        source_inventory, ti = by_path[match['source_path']], tauk_inventory[rid]
        if ti['group'] != f'TAUK_{known["official_split"]}': raise ValueError('Inventory official split differs')
        if source != previous_source:
            previous_source, cached_source = source, None
        source_state, tauk_state = availability(source), availability(original)
        row = {'recording_id':rid, 'participant_id':known['participant_id'], 'language':known['language'], 'split':known['split'],
               'source_dataset':match['source_dataset'], 'source_path':str(source),
               'source_availability':source_state, 'tauk_availability':tauk_state,
               'prior_identity_evidence':match['identity_evidence'], 'prior_offset_seconds':match['source_offset_seconds'],
               'prior_full_resampled_correlation':match['full_waveform_corr'],
               'prior_decoded_pcm_exact_equal':source_inventory['pcm_sha256']==ti['pcm_sha256'],
               'fresh_status':'unavailable_originals' if source_state != 'materialized' or tauk_state != 'materialized' else 'both_originals_hash_verified'}
        detail = {'recording_id':rid}
        prior_anchors = []
        if not match['source_offset_seconds']:
            candidates = task_checks.get(rid, {}).get('candidates', [])
            selected = next((c for c in candidates if c['processed_reference']==str(source)), None)
            prior_anchors = selected['anchors'] if selected else []
            offsets = [a['reference_start']-a['tauk_start'] for a in prior_anchors]
            row.update(prior_anchor_count=len(prior_anchors), prior_anchor_offset_spread_seconds=max(offsets)-min(offsets) if offsets else None)
            detail['prior_processed_task_anchors'] = prior_anchors
        if row['fresh_status'] == 'both_originals_hash_verified':
            if cached_source is None: cached_source = read_original(source, source_inventory['pcm_sha256'])
            source_meta, source_samples = cached_source
            tauk_meta, tauk_samples = read_original(original, ti['pcm_sha256'], known['sha256'])
            row.update({f'source_{key}':value for key,value in source_meta.items()})
            row.update({f'tauk_{key}':value for key,value in tauk_meta.items()})
            row['decoded_pcm_exact_equal'] = source_meta['pcm_sha256']==tauk_meta['pcm_sha256']
            if match['source_offset_seconds']:
                check = check_constant_offset(tauk_samples, source_samples, float(match['source_offset_seconds']))
                detail['fresh_constant_offset'] = check
                row.update({key:value for key,value in check.items() if key!='anchors'})
            else:
                checks=[]
                for anchor in prior_anchors:
                    a,b,n=round(anchor['tauk_start']*RATE),round(anchor['reference_start']*RATE),round(anchor['seconds']*RATE)
                    if a+n>len(tauk_samples) or b+n>len(source_samples): raise ValueError('Prior anchor is outside original audio')
                    checks.append(_correlation(tauk_samples[a:a+n],source_samples[b:b+n]))
                row['fresh_processed_anchor_count']=len(checks)
                row['fresh_processed_anchor_min_correlation']=min((v for v in checks if v is not None),default=None)
                row['constant_offset_supported']=False
                detail['fresh_processed_anchor_correlations']=checks
        if known['language']=='en':
            clock=resolved[rid]
            row.update(chat_offset_seconds=clock.get('source_offset_seconds'),chat_offset_method=clock.get('offset_method'),
                       chat_clock_validated=clock.get('clock_validated'),chat_scope_complete=clock.get('scope_review_complete'),
                       chat_timing_complete=clock.get('extraction_timing_complete'),
                       chat_clock_differs_from_waveform=bool(clock.get('CHAT_media_clock_differs_from_waveform_clock')))
        rows.append(row);detail_rows.append(detail)
        if index%50==0: print(f'Checked availability/identity {index}/{len(cross)}',flush=True)
    summary={'schema_version':1,'recordings':len(rows),'participants':len(participants),
             'fresh_status':dict(Counter(r['fresh_status'] for r in rows)),
             'source_availability':dict(Counter(r['source_availability'] for r in rows)),
             'tauk_availability':dict(Counter(r['tauk_availability'] for r in rows)),
             'fresh_constant_offset_supported':sum(r.get('constant_offset_supported') is True for r in rows),
             'fresh_decoded_pcm_exact_equal':sum(r.get('decoded_pcm_exact_equal') is True for r in rows),
             'prior_decoded_pcm_exact_equal':sum(r['prior_decoded_pcm_exact_equal'] for r in rows),
             'prior_evidence':dict(Counter(r['prior_identity_evidence'] for r in rows)),
             'chat_scope_complete':sum(r.get('chat_scope_complete') is True for r in rows),
             'chat_clock_differs_from_waveform':sum(r.get('chat_clock_differs_from_waveform') is True for r in rows),
             'input_sha256':hashes,'locked_diagnoses_read':False,'originals_modified':False,
             'thresholds':{'analysis_rate_hz':RATE,'anchor_seconds':1,'anchors':5,'local_search_radius_seconds':.2,
                           'minimum_correlation':.98,'maximum_anchor_offset_spread_seconds':.002},
             'notes':['Correlation is measured after mono conversion and resampling to 2kHz; it is not full-band bit identity.',
                      'NCC accepts affine gain/DC differences; normalized residual and decoded PCM equality are separate fields.',
                      'Unavailable originals are not read or silently replaced by cached waveforms; prior evidence remains separately labeled.',
                      'Constant shift requires complete TAUK coverage and five distributed non-silent anchors; thresholds are diagnostics, not manual ground truth.',
                      'Blank prior offsets identify edited-source matches; no global shift is assigned to those recordings.',
                      'CHAT/media clock is separate from raw-audio correspondence; prior clock validation is not new manual speaker annotation.']}
    summary['by_dataset']={ds:{'recordings':len(rr),'fresh_both_originals':sum(r['fresh_status']=='both_originals_hash_verified' for r in rr),
                               'constant_offset_supported':sum(r.get('constant_offset_supported') is True for r in rr)}
                           for ds in ('Chou','Delaware') if (rr:=[r for r in rows if r['source_dataset']==ds])}
    for dataset, group in summary['by_dataset'].items():
        offsets=[float(r['prior_offset_seconds']) for r in rows if r['source_dataset']==dataset and r['prior_offset_seconds']!='']
        group['prior_known_offsets']={'count':len(offsets),'min_seconds':min(offsets) if offsets else None,
                                     'max_seconds':max(offsets) if offsets else None,
                                     'median_seconds':statistics.median(offsets) if offsets else None}
    rows.sort(key=lambda r:r['recording_id'])
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
    writer.writeheader();writer.writerows(rows)
    stable_write(destination/'per_recording.csv',buffer.getvalue().encode())
    stable_write(destination/'details.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in detail_rows).encode())
    stable_write(destination/'summary.json',json_bytes(summary))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--legacy-audit-root',type=Path,required=True)
    parser.add_argument('--chat-resolutions',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('data/reports/audio_identity_v1'))
    args=parser.parse_args()
    result=run_audit(args.data_root,args.legacy_audit_root,args.chat_resolutions,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {'input_sha256','notes'}},indent=2))


if __name__=='__main__':main()
