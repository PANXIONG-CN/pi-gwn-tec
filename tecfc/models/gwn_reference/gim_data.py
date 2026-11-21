#!/usr/bin/env python3
"""Native two-hour GIM windows, exact year roles and train-only climatology.

The public GIM array has inclusive -180/+180 columns; use 72 unique columns
internally and restore the duplicate for the existing 71x73 scoring interface.
No year outside 2009..2025 is opened. Zeros are valid product values.
"""
from __future__ import annotations
import argparse,hashlib,json
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent.parent
OLD=PROJECT/'manuscript/revisions-20260929/revised/release/assets/data/gim'
EXT=PROJECT/'output/gimtec-v2-20261013'
TRAIN=(2009,2010,2011,2012,2014,2016,2018)
VALIDATION=(2013,2017)
ORIGINAL=(2015,2019,2020,2021,2022)
EXTENSION=(2023,2024,2025)
GROUPS=((2009,2010,2011,2012),(2014,),(2016,),(2018,))
ALL=TRAIN+VALIDATION+ORIGINAL+EXTENSION


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def to_unique(a, check=True):
    a=np.asarray(a)
    if a.shape[-2:]!=(71,73):raise ValueError('expected 71x73 GIM')
    if check and not np.array_equal(a[...,0],a[...,-1],equal_nan=True):raise ValueError('longitude endpoints disagree')
    return a[...,:72]


def restore_grid(a):
    a=np.asarray(a)
    if a.shape[-2:]!=(71,72):raise ValueError('expected 71x72 unique grid')
    return np.concatenate((a,a[...,:1]),axis=-1)


def stamp(day):return int(np.datetime64(day,'s').astype(np.int64))


class GIM:
    def __init__(self,old=OLD,extension=EXT):self.old=Path(old);self.extension=Path(extension)
    def path(self,year):
        year=int(year)
        if year not in ALL:raise ValueError(f'forbidden or unavailable year {year}')
        return (self.old if year<=2022 else self.extension)/f'TEC_{year}.npy'
    @lru_cache(maxsize=17)
    def raw(self,year):
        year=int(year);p=self.path(year);a=np.load(p,mmap_mode='r',allow_pickle=False)
        n=int((np.datetime64(f'{year+1}-01-01')-np.datetime64(f'{year}-01-01')).astype(int))
        if a.shape!=(n,12,71,73) or a.dtype!=np.float32:raise ValueError(f'wrong shape/dtype {p}: {a.shape}, {a.dtype}')
        return a
    def day(self,day,unique=True):
        d=np.datetime64(day,'D');year=int(str(d)[:4]);ix=int((d-np.datetime64(f'{year}-01-01')).astype(int))
        raw=np.asarray(self.raw(year)[ix]);raw=to_unique(raw) if unique else raw
        return raw.astype(np.float32)/np.float32(10.)
    def windows(self,role):
        if role=='train':
            for group in GROUPS:
                start=np.datetime64(f'{group[0]}-01-02');end=np.datetime64(f'{group[-1]+1}-01-01')
                for d in np.arange(start,end,dtype='datetime64[D]'):yield str(d)
        else:
            years={'validation':VALIDATION,'original_evaluation':ORIGINAL,'extension_evaluation':EXTENSION}[role]
            for y in years:
                for d in np.arange(np.datetime64(f'{y}-01-01'),np.datetime64(f'{y+1}-01-01'),dtype='datetime64[D]'):yield str(d)
    def sample(self,day,climate=None):
        d=np.datetime64(day,'D');p=d-np.timedelta64(1,'D')
        history=self.day(p);target=self.day(d);valid=np.isfinite(target)
        hv=np.isfinite(history)
        if not hv.all():raise ValueError('missing history requires an explicitly frozen causal fill rule')
        result=dict(origin=stamp(d),cutoff=stamp(d)-7200,history=history,target=target,valid=valid)
        if climate is not None:
            # Match Japan's history-only amplitude ratio and clipping. This is
            # provisional until its selected recipe and interface are frozen.
            c0=climate[int(str(p)[5:7])-1];c1=climate[int(str(d)[5:7])-1]
            denom=float(c0[hv].sum(dtype=np.float64));numer=float(history[hv].sum(dtype=np.float64))
            if denom<=0 or not np.isfinite(numer):raise ValueError('invalid climatology amplitude')
            alpha=np.float32(np.clip(numer/denom,.3,3.))
            result.update(alpha=alpha,background_history=c0*alpha,background_future=c1*alpha)
        return result


def build_climate(gim):
    sums=np.zeros((12,12,71,72),np.float64);counts=np.zeros_like(sums,dtype=np.int32);maximum=-np.inf; manifest={}
    for year in TRAIN:
        raw=gim.raw(year);p=gim.path(year);manifest[str(year)]={'path':str(p),'sha256':sha(p)}
        dates=np.arange(np.datetime64(f'{year}-01-01'),np.datetime64(f'{year+1}-01-01'),dtype='datetime64[D]')
        months=dates.astype('datetime64[M]').astype(np.int64)%12
        for m in range(12):
            x=to_unique(raw[months==m]).astype(np.float64)/10.;v=np.isfinite(x)
            sums[m]+=np.where(v,x,0.).sum(axis=0);counts[m]+=v.sum(axis=0)
            if v.any():maximum=max(maximum,float(x[v].max()))
        print('climate processed',year,flush=True)
    if np.any(counts==0):raise ValueError('empty train climate bin: define train-only fallback before use')
    climate=(sums/counts).astype(np.float32)
    if not np.isfinite(climate).all() or (climate<=0).any():raise ValueError('invalid positive train climatology')
    dest=HERE/'prepared';dest.mkdir(exist_ok=True)
    path=dest/'train_climatology_v1.npz';assert not path.exists(),'refuse to overwrite existing prepared artifact'
    np.savez_compressed(path,climatology=climate,counts=counts,scale=np.float64(maximum))
    info=dict(status='prepared_provisional',train_years=list(TRAIN),excluded_from_fit=list(VALIDATION+ORIGINAL+EXTENSION),unit='TECU',scale_train_max_tecu=maximum,shape=list(climate.shape),calendar='month x even UTC slot x 71 lat x 72 unique longitudes',source_manifest=manifest,preparation_code_sha256=sha(__file__),climatology_sha256=sha(path),alpha='history ratio sum(valid TEC)/sum(climatology at same valid history); clip [0.3,3]; provisional pending Japan recipe freeze',zero_labels='valid')
    (dest/'TRAIN_CLIMATOLOGY.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(dict(train_max=maximum,clim_shape=list(climate.shape),file=str(path))))


def audit(gim):
    records=[]
    for year in sorted(ALL):
        p=gim.path(year);a=gim.raw(year);seam=np.asarray(a[...,0])-np.asarray(a[...,-1]);f=np.isfinite(a)
        records.append(dict(year=year,path=str(p),sha256=sha(p),shape=list(a.shape),dtype=str(a.dtype),endpoint_unequal=int(np.count_nonzero(seam)),endpoint_max_abs_raw=float(np.max(np.abs(seam))),nonfinite=int((~f).sum()),negative=int((a<0).sum()),zero=int((a==0).sum())))
        assert records[-1]['endpoint_unequal']==0 and records[-1]['nonfinite']==0
        print('audited',year,flush=True)
    counts={r:sum(1 for _ in gim.windows(r)) for r in ['train','validation','original_evaluation','extension_evaluation']}
    assert counts==dict(train=2553,validation=730,original_evaluation=1826,extension_evaluation=1096)
    (HERE/'DATA_AUDIT.json').write_text(json.dumps(dict(status='passed',years=records,window_counts=counts,unit='raw / 10 = TECU',timestamp='12 samples per day 00,02,...,22 UTC',duplicates='each target is restored to 73 longitude columns for existing scoring weighting',prohibited='never open 2026'),indent=2)+'\n')


def selftest(gim):
    a=np.arange(71*72).reshape(71,72);np.testing.assert_array_equal(to_unique(restore_grid(a)),a)
    bad=restore_grid(a).copy();bad[0,-1]+=1
    try:to_unique(bad);raise AssertionError('seam mismatch went undetected')
    except ValueError:pass
    try:gim.path(2026);raise AssertionError('2026 access allowed')
    except ValueError:pass
    for d in ['2013-01-01','2017-01-01','2015-01-01','2020-01-01','2023-01-01','2025-12-31']:
        s=gim.sample(d);assert s['history'].shape==(12,71,72) and s['target'].shape==(12,71,72)
        assert s['cutoff']==s['origin']-7200 and np.isfinite(s['history']).all()
        np.testing.assert_array_equal(restore_grid(s['target']),gim.day(d,unique=False))
    assert list(gim.windows('train'))[0]=='2009-01-02'
    assert '2014-01-01' not in set(gim.windows('train'))
    assert '2010-01-01' in set(gim.windows('train'))
    (HERE/'INTERFACE_CHECK.json').write_text(json.dumps(dict(status='passed',synthetic_seam_round_trip=True,unequal_seam_rejected=True,access_2026_rejected=True,history_cutoff='D-1 22:00 UTC',target='D 00:00..22:00 UTC => +2..+24 h',native_shape=[71,73],unique_shape=[71,72],cross_year_history_checked=True,source_sha256=sha(__file__)),indent=2)+'\n')
    print('global interface checks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','climatology','selftest']);a=parser.parse_args();g=GIM()
    {'audit':audit,'climatology':build_climate,'selftest':selftest}[a.action](g)
