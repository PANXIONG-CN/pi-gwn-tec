"""Fixed train-year-only regional windows and mapped exp0017 anchors for exp0020.

Reads only 2014--2025 observations needed by declared windows. No network and no
training. Execution requires the completed revised A1/A2 admission record.
"""
import argparse,hashlib,json,os
from functools import lru_cache
from pathlib import Path
import numpy as np

REPO=Path(os.environ.get('TECFC_OUTPUT','.')).resolve();PROJECT=Path(os.environ.get('TECFC_PROJECT','.')).resolve();EXP=REPO/'experiments/exp0020'
TRAIN=[2015,2019,2020,2021];SPLITS={'train':TRAIN,'validation':[2022],'evaluation':[2023,2024,2025]}
OFF=np.arange(12,dtype=np.int64)*7200

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def unix(day):return int(np.datetime64(day,'s').astype(np.int64))
def days(years):return [str(d) for y in years for d in np.arange(np.datetime64(f'{y}-01-01'),np.datetime64(f'{y+1}-01-01'))]

def mapping(lat,lon):
    la,lo=np.meshgrid(lat,lon,indexing='ij');yy=(87.5-la.ravel().astype(float))/2.5;xx=(lo.ravel().astype(float)+180)/5
    iy=yy.astype(int);ix=xx.astype(int);dy=yy-iy;dx=xx-ix
    assert iy.min()>=0 and iy.max()+1<71 and ix.min()>=0 and ix.max()+1<73
    return np.stack([iy*73+ix,iy*73+ix+1,(iy+1)*73+ix,(iy+1)*73+ix+1]),np.stack([(1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx])
def remap(x,mp):return sum(np.asarray(x[...,j],np.float64)*w for j,w in zip(*mp))

class Native:
    def __init__(self,region):
        self.region=region;self.inputs={};self.rows={}
        if region=='japan':
            root=PROJECT/'output/japan-jpld-isee-20261105/jpld'
            self.rows={r['date']:dict(path=root/r['daily_path'],sha256=r['daily_sha256']) for r in map(json.loads,(root/'data/DAYS.jsonl').read_text().splitlines())}
        else:
            root=PROJECT/'output/jpld-15min-ab-20261101/data'
            for p in (root/'records').glob('*/*.json'):
                r=read(p)
                if int(r['date'][:4])<=2025:self.rows[r['date']]=dict(path=root/r['output'],sha256=r['output_sha256'])
        archive=PROJECT/'output/jpld-2022-china-japan-20261113'
        verification=read(archive/'VERIFICATION.json');assert verification['status']=='passed'
        assert sha(archive/'MANIFEST.jsonl')==verification['manifest_sha256']
        archive_rows={r['date']:r for r in map(json.loads,(archive/'MANIFEST.jsonl').read_text().splitlines())}
        assert set(archive_rows)==set(days([2022]))
        for d,row in archive_rows.items():
            spec=row['outputs'][region+'_15min'];assert spec['path']==f'{region}_15min/2022/{d}.npz'
            self.rows[d]=dict(path=archive/spec['path'],sha256=spec['sha256'])
        first=self.day('2015-01-01');self.lat=first['lat'];self.lon=first['lon'];self.n=len(self.lat)*len(self.lon)
    @lru_cache(64)
    def day(self,d):
        assert 2014<=int(d[:4])<=2025
        spec=self.rows[d];p=spec['path'];digest=sha(p)
        if spec['sha256'] is not None:assert digest==spec['sha256']
        self.inputs[str(p)]=dict(bytes=p.stat().st_size,sha256=digest,date=d)
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in ['tec','valid','present','time_utc','lat','lon']}
        np.testing.assert_array_equal(a['time_utc'],unix(d)+np.arange(96)*900)
        assert a['tec'].dtype==np.float32 and a['valid'].dtype==bool
        assert not (a['valid']&~np.isfinite(a['tec'])).any()
        if hasattr(self,'lat'):
            np.testing.assert_array_equal(a['lat'],self.lat);np.testing.assert_array_equal(a['lon'],self.lon)
        return a

def fill(a,v,fallback):
    out=a.copy();last=np.full(a.shape[1],-10000);value=np.zeros(a.shape[1],np.float32)
    for k in range(89):
        last[v[k]]=k;value[v[k]]=a[k,v[k]];recent=~v[k]&((k-last)*900<=21600)
        out[k,recent]=value[recent];other=~v[k]&~recent;out[k,other]=fallback[k//8,other]
    assert np.isfinite(out).all();return out

def evaluation_origins(region):
    if region=='china':
        ts=np.load(PROJECT/'output/jpld-regional-completion-20261102/support/index.npz')['origins']
        assert len(ts)==1077
    else:
        ts=np.load(PROJECT/'output/japan-jpld-isee-20261105/jpld/runs/dense_do_42/evaluation_origins.npy')
        assert len(ts)==1091
    assert len(set(ts))==len(ts);return ts

def prepare(region):
    gate=read(EXP/'A_GATE_R1.json');assert gate['status']=='passed'
    cfg=read(EXP/'config.json')
    for name in ['archive_2022_verification','anchor_manifest']:
        spec=cfg['audit_inputs'][name];assert sha(spec['path'])==spec['sha256']
    ar=Native(region);dest=EXP/'artifacts/prepared_r1'/region;dest.mkdir(parents=True,exist_ok=False)
    sums=np.zeros((12,ar.n),np.float64);counts=np.zeros((12,ar.n),np.int64);scale=-np.inf
    for d in days(TRAIN):
        a=ar.day(d);x=a['tec'][::8].reshape(12,ar.n);v=a['valid'][::8].reshape(12,ar.n)
        sums+=np.where(v,x.astype(np.float64),0);counts+=v
        if v.any():scale=max(scale,float(x[v].max()))
    slot_sums=sums.sum(1);slot_counts=counts.sum(1);global_mean=sums.sum()/counts.sum()
    slot_mean=np.divide(slot_sums,slot_counts,out=np.full(12,global_mean),where=slot_counts>0)
    fallback=np.divide(sums,counts,out=np.broadcast_to(slot_mean[:,None],sums.shape).copy(),where=counts>0).astype(np.float32)
    assert scale>0 and np.isfinite(fallback).all()
    np.savez(dest/'statistics.npz',scale=np.float64(scale),fallback=fallback,sums=sums,counts=counts,fit_years=np.array(TRAIN),lat=ar.lat,lon=ar.lon)
    fusion=read(REPO/'experiments/exp0017/FUSED_PREDICTIONS.json');source={}
    for split in ['original_evaluation','extension_evaluation']:
        spec=fusion['splits'][split]
        for key in ['predictions','origins']:assert sha(spec[key])==fusion['files_sha256'][spec[key]]
        ts=np.load(spec['origins']);array=np.load(spec['predictions'],mmap_mode='r');assert array.shape==(len(ts),12,71,73)
        assert len(set(ts))==len(ts)
        source[split]=(array,{int(t):i for i,t in enumerate(ts)})
    mp=mapping(ar.lat,ar.lon);plan={};exclusions=[]
    for split,years in SPLITS.items():
        requested=evaluation_origins(region) if split=='evaluation' else np.array([unix(d) for d in days(years)],np.int64)
        rows=[]
        for origin in requested:
            d=str(np.datetime64(int(origin),'s').astype('datetime64[D]'));h=str(np.datetime64(d)-np.timedelta64(1,'D'))
            a,b=ar.day(h),ar.day(d);why=[]
            if not a['present'][:89].all():why.append('history_frame_absent')
            if not b['present'][::8].all():why.append('target_frame_absent')
            if not b['valid'][::8].any():why.append('target_empty')
            if why:exclusions.append(dict(split=split,date=d,reasons=why));continue
            rows.append(dict(date=d,history=h,origin=int(origin),cutoff=int(origin)-7200))
        if split=='evaluation':np.testing.assert_array_equal([r['origin'] for r in rows],requested)
        plan[split]=rows;folder=dest/split;folder.mkdir()
        layout={'history':(89,np.float32),'target':(12,np.float32),'valid':(12,bool),'anchor':(12,np.float64)}
        arrays={key:np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype=dt,shape=(len(rows),frames,ar.n)) for key,(frames,dt) in layout.items()}
        for i,r in enumerate(rows):
            a,b=ar.day(r['history']),ar.day(r['date']);t=r['origin']
            assert (a['time_utc'][:89]<=r['cutoff']).all()
            np.testing.assert_array_equal(a['time_utc'][:89],t-86400+np.arange(89)*900)
            np.testing.assert_array_equal(b['time_utc'][::8],t+OFF)
            raw=a['tec'][:89].reshape(89,ar.n);valid=a['valid'][:89].reshape(89,ar.n)
            arrays['history'][i]=fill(raw,valid,fallback);arrays['target'][i]=b['tec'][::8].reshape(12,ar.n);arrays['valid'][i]=b['valid'][::8].reshape(12,ar.n)
            f,lookup=source['extension_evaluation' if split=='evaluation' else 'original_evaluation'];j=lookup[t]
            np.testing.assert_array_equal(np.array(t)+OFF,b['time_utc'][::8])
            arrays['anchor'][i]=remap(f[j].reshape(12,5183),mp);assert np.isfinite(arrays['anchor'][i]).all()
            if i%200==0:print(region,split,i,'/',len(rows),flush=True)
        for v in arrays.values():v.flush()
        np.save(folder/'origins.npy',np.array([r['origin'] for r in rows],np.int64))
        write(folder/'rows.json',rows)
    # B: original exp0011 regional common support; no evaluation-trained model exists.
    folder=dest/'evaluation';a=np.load(folder/'anchor.npy',mmap_mode='r');y=np.load(folder/'target.npy',mmap_mode='r');v=np.load(folder/'valid.npy',mmap_mode='r')
    sse=0.;n=0
    for i in range(len(a)):
        err=a[i][v[i]]-y[i][v[i]].astype(float);sse+=np.square(err).sum();n+=len(err)
    rmse=float(np.sqrt(sse/n));ref={'china':6.545185,'japan':4.458810}[region]
    assert n=={'china':36099959,'japan':3770496}[region]
    bcheck=dict(status='passed' if abs(rmse-ref)<=1e-6 else 'failed',region=region,RMSE=rmse,reference=ref,absolute_difference=abs(rmse-ref),n_valid=n,tolerance=1e-6)
    write(dest/'CHECK_B.json',bcheck)
    # Twenty deterministic windows: verify unmodified raw valid values and each imputed cell's causal source.
    allrows=[(split,i,r) for split,rows in plan.items() for i,r in enumerate(rows)];rng=np.random.default_rng(20260924);spot=[]
    for j in rng.choice(len(allrows),20,replace=False):
        split,i,r=allrows[int(j)];native=ar.day(r['history']);raw=native['tec'][:89].reshape(89,ar.n);valid=native['valid'][:89].reshape(89,ar.n)
        actual=np.load(dest/split/'history.npy',mmap_mode='r')[i]
        np.testing.assert_array_equal(actual[valid],raw[valid])
        for frame,node in zip(*np.where(~valid)):
            previous=np.flatnonzero(valid[max(0,frame-24):frame+1,node])
            expected=raw[max(0,frame-24)+previous[-1],node] if len(previous) else fallback[frame//8,node]
            assert actual[frame,node]==expected
        target=ar.day(r['date']);original=target['tec'][::8].reshape(12,ar.n)
        np.testing.assert_array_equal(np.load(dest/split/'target.npy',mmap_mode='r')[i],original)
        np.testing.assert_array_equal(np.load(dest/split/'valid.npy',mmap_mode='r')[i],target['valid'][::8].reshape(12,ar.n))
        spot.append(dict(split=split,origin=r['origin'],history=r['history'],raw_valid_values_checked=int(valid.sum()),imputed_values_checked=int((~valid).sum()),passed=True))
    write(dest/'CHECK_D.json',dict(status='passed',region=region,all_windows_timestamp_assertions=True,windows=sum(map(len,plan.values())),spot_checks=spot,labels_unfilled=True,training_statistics_years=TRAIN))
    not_common=sorted(set(days([2023,2024,2025]))-{r['date'] for r in plan['evaluation']})
    write(dest/'metadata.json',dict(region=region,scale=scale,nodes=ar.n,counts={s:len(rows) for s,rows in plan.items()},exclusions=exclusions,evaluation_days_without_common_historical_predictions=not_common,training_years=TRAIN,validation_years=[2022],evaluation_years=[2023,2024,2025],evaluation_support='fixed common available historical regional prediction origins; exp0011 support',anchor_dtype='float64 bilinear mapping of saved float32 exp0017 fused forecast',label_unit='TECU',source_sha256=sha(__file__)))
    write(dest/'INPUT_MANIFEST.json',ar.inputs)
    write(dest/'FILES.json',{str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='FILES.json'})
    assert bcheck['status']=='passed','B anchor check failed'
    print(json.dumps(dict(region=region,status='prepared_B_D_passed',counts={s:len(rows) for s,rows in plan.items()},anchor_RMSE=rmse)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',choices=['china','japan'],required=True);a=p.parse_args();prepare(a.region)
