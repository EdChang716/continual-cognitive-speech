"""Clock checks must distinguish a fixed crop from edits, gain, and silence."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from ccspeech.data.audio_identity import (RATE, availability, check_constant_offset,
                                         read_original, run_audit)


def pcm_hash(rate, samples):
    return hashlib.sha256(str((rate, samples.shape, samples.dtype)).encode()+samples.tobytes()).hexdigest()


def test_constant_leading_padding_and_affine_gain_are_supported():
    rng=np.random.default_rng(4)
    tauk=rng.normal(size=RATE*12)
    source=np.r_[np.zeros(RATE*2),2.5*tauk+7,np.zeros(RATE)]
    report=check_constant_offset(tauk,source,2.)
    assert report['constant_offset_supported']
    assert report['tauk_coverage_fraction']==1
    assert report['anchor_offset_spread_seconds']==0
    assert report['affine_source_from_tauk_gain']==pytest.approx(2.5)
    assert report['affine_source_from_tauk_dc']==pytest.approx(7)
    assert report['normalized_affine_residual']<1e-12


def test_source_subclip_is_supported_but_edited_middle_is_not():
    rng=np.random.default_rng(5)
    source=rng.normal(size=RATE*20)
    tauk=source[RATE*3:RATE*15].copy()
    assert check_constant_offset(tauk,source,3.)['constant_offset_supported']
    edited=np.r_[tauk[:RATE*6],tauk[RATE*7:],rng.normal(size=RATE*2)]
    report=check_constant_offset(tauk,edited,0.)
    assert report['tauk_coverage_fraction']==1
    assert not report['constant_offset_supported']
    assert report['anchor_min_correlation']<.98


def test_silent_anchors_and_partial_coverage_never_validate():
    silence=np.zeros(RATE*12)
    report=check_constant_offset(silence,silence,0.)
    assert not report['constant_offset_supported']
    assert report['anchor_count']==0
    assert report['full_2khz_overlap_correlation'] is None
    rng=np.random.default_rng(1)
    speech=rng.normal(size=RATE*12)
    partial=check_constant_offset(speech,speech[:-RATE],0.)
    assert partial['full_2khz_overlap_correlation']==pytest.approx(1)
    assert partial['tauk_coverage_fraction']<1
    assert not partial['constant_offset_supported']


def test_pcm_hash_and_current_original_file_hash_are_required(tmp_path):
    samples=np.arange(16000,dtype=np.int16)
    path=tmp_path/'source.wav';wavfile.write(path,16000,samples)
    digest=pcm_hash(16000,samples)
    metadata, reduced=read_original(path,digest,hashlib.sha256(path.read_bytes()).hexdigest())
    assert metadata['frames']==16000 and len(reduced)==RATE
    with pytest.raises(ValueError,match='PCM'):
        read_original(path,'invalid')
    with pytest.raises(ValueError,match='file SHA256'):
        read_original(path,digest,'invalid')


def test_cloud_placeholder_is_not_read(tmp_path,monkeypatch):
    path=tmp_path/'placeholder.wav'
    path.write_bytes(b'not to be opened')
    import ccspeech.data.audio_identity as module
    monkeypatch.setattr(module.sys,'platform','darwin')
    original_stat=Path.stat
    def stat(value,*args,**kwargs):
        if value==path:return SimpleNamespace(st_flags=0x40000000)
        return original_stat(value,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(Path,'is_file',lambda value:True if value==path else False)
    monkeypatch.setattr(module.wavfile,'read',lambda *args,**kwargs:pytest.fail('placeholder opened'))
    assert availability(path)=='cloud_placeholder_not_read'
    with pytest.raises(ValueError,match='Unavailable'):
        read_original(path,'unused')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ['recording_id','participant_id','official_split','split','audio_path','language','sha256']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def test_audit_reads_original_pair_and_keeps_chat_clock_separate(tmp_path,monkeypatch):
    data=tmp_path/'data';legacy=tmp_path/'old/overlap_audit';legacy.mkdir(parents=True)
    rng=np.random.default_rng(10)
    samples=rng.integers(-2000,2000,size=16000*8,dtype=np.int16)
    source_samples=np.r_[np.zeros(16000*2,dtype=np.int16),samples,np.zeros(16000,dtype=np.int16)]
    original=data/'raw/recording.wav';original.parent.mkdir(parents=True);wavfile.write(original,16000,samples)
    source=tmp_path/'source.wav';wavfile.write(source,16000,source_samples)
    row={'recording_id':'synthetic-1','participant_id':'synthetic','official_split':'train','split':'train',
         'audio_path':'raw/recording.wav','language':'en','sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
    for split in ('train','validation','test'):write_csv(data/f'processed/taukadial/{split}.csv',[row] if split=='train' else [])
    cross={'tauk_recording':'synthetic-1','tauk_participant_id':'synthetic','tauk_split':'train','source_dataset':'Delaware',
           'source_path':str(source),'source_offset_seconds':'2','full_waveform_corr':'1','identity_evidence':'full_resampled_waveform_correlation'}
    write_csv(legacy/'recording_crosswalk_complete.csv',[cross])
    (legacy/'inventory.json').write_text(json.dumps([
      {'location':str(source),'group':'Delaware_raw','pcm_sha256':pcm_hash(16000,source_samples)},
      {'location':'archive::train/synthetic-1.wav','group':'TAUK_train','pcm_sha256':pcm_hash(16000,samples)}]))
    resolutions=tmp_path/'resolution.jsonl';resolutions.write_text(json.dumps({'recording_id':'synthetic-1',
      'source_offset_seconds':1.5,'offset_method':'different_media_clock','clock_validated':True,'scope_review_complete':True,
      'extraction_timing_complete':True,'CHAT_media_clock_differs_from_waveform_clock':True})+'\n')
    original_open=Path.open
    def guarded(path,*args,**kwargs):
        if 'locked' in path.parts:pytest.fail('locked diagnoses accessed')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    out=data/'reports/identity';report=run_audit(data,legacy,resolutions,out)
    assert report['fresh_constant_offset_supported']==1
    assert report['chat_clock_differs_from_waveform']==1
    with (out/'per_recording.csv').open() as f:result=next(csv.DictReader(f))
    assert result['proposed_offset_seconds']=='2.0' and result['chat_offset_seconds']=='1.5'
    assert not report['locked_diagnoses_read']
    # Hash changes in a materialized original are errors, never accepted on old evidence.
    changed=source_samples.copy();changed[16000*2]+=10;wavfile.write(source,16000,changed)
    with pytest.raises(ValueError,match='PCM'):
        run_audit(data,legacy,resolutions,data/'reports/corrupt')
