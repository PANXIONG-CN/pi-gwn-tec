"""Read-only explicitly enumerated 2025/2026 inputs. Never discovers other years."""
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
from tecfc.scoring.confirm_records import ROOT, EXP, sha, read

PROJECT=Path(os.environ.get('TECFC_PROJECT','.')).resolve()
DATA=Path(os.environ.get('TECFC_DATA_2026',PROJECT/'output/reserved-2026-exp0024-20261114')).resolve()


def unix(day):return int(np.datetime64(day,'s').astype(np.int64))


class Data:
    def __init__(self):
        proof=read(EXP/'checks/F_HASHES.json');assert proof['status']=='passed'
        self.catalog={r['path']:r for r in proof['files']}
        self.global2025=read(ROOT/'experiments/exp0018/configs/inference.json')['data_files']['2025']
        assert sha(self.global2025['path'])==self.global2025['sha256']
        self.raw2025=np.load(self.global2025['path'],mmap_mode='r',allow_pickle=False)
        self.raw2026=self.array('global/TEC_2026.npy')
        with self.array('global/QUALITY_2026.npz') as z:self.quality={k:z[k].copy() for k in z.files}
        self.oldrows={}
        jp=PROJECT/'output/japan-jpld-isee-20261105/jpld'
        self.oldrows['japan']={r['date']:dict(path=str(jp/r['daily_path']),sha256=r['daily_sha256']) for r in map(json.loads,(jp/'data/DAYS.jsonl').read_text().splitlines()) if r['date'].startswith('2025-')}
        cn=PROJECT/'output/jpld-15min-ab-20261101/data';self.oldrows['china']={}
        for p in (cn/'records/2025').glob('*.json'):
            r=read(p);assert r['date'].startswith('2025-')
            self.oldrows['china'][r['date']]=dict(path=str(cn/r['output']),sha256=r['output_sha256'])
        self.inputs={}

    def array(self,relative):
        r=self.catalog[relative];p=DATA/relative;assert sha(p)==r['sha256']
        return np.load(p,mmap_mode='r' if p.suffix=='.npy' else None,allow_pickle=False)

    def global_day(self,day,unique=True):
        year=int(day[:4]);assert year in [2025,2026]
        ix=int((np.datetime64(day)-np.datetime64(f'{year}-01-01')).astype(int))
        raw=self.raw2025 if year==2025 else self.raw2026
        if not 0<=ix<len(raw):raise KeyError('unpublished_global_day:'+day)
        a=np.asarray(raw[ix],np.float32);np.testing.assert_array_equal(a[...,0],a[...,-1])
        return a[...,:72]/np.float32(10) if unique else a/np.float32(10)

    @lru_cache(100)
    def regional_day(self,region,day):
        assert region in ['china','japan'] and day[:4] in ['2025','2026']
        if day.startswith('2026-'):
            rel=f'{region}_15min/2026/{day}.npz';r=self.catalog[rel];path=DATA/rel
        else:
            r=self.oldrows[region][day];path=Path(r['path'])
        assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        self.inputs[str(path)]=dict(sha256=r['sha256'],date=day)
        return a

    def drivers(self):
        p=PROJECT/'output/gimtec-v2-20261013/drivers_2009_2025_2h.npz'
        ref=read(ROOT/'experiments/exp0015/configs/global_g_de_42.json')['driver']
        assert sha(p)==ref['sha256']
        with np.load(p) as z:a={k:z[k][-31*12:].copy() for k in ['time','Kp','Dst','AE','F107']}
        with self.array('drivers/drivers_2026_2h.npz') as z:b={k:z[k].copy() for k in a}
        return {k:np.concatenate([a[k],b[k]]) for k in a}
