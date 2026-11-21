"""Independent NumPy scorer. Open only frozen predictions after candidate freeze."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode=True
SEEDS=[42,1024,2025,3407,5000]
SPLITS=['validation','original_evaluation','extension_evaluation']
GROUPS=['all','unique_longitude','high_TEC','disturbed_day']
YEARS={'validation':[2013,2017],'original_evaluation':[2015,2019,2020,2021,2022],
       'extension_evaluation':[2023,2024,2025]}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sufficient(prediction,truth,valid):
    if not np.isfinite(prediction[:,valid]).all():raise ValueError('Nonfinite required prediction')
    error=np.where(valid,prediction-truth,0.)
    return np.stack([np.broadcast_to(valid,prediction.shape).sum(-1),np.square(error).sum(-1),
                     np.abs(error).sum(-1),error.sum(-1)],axis=-1)


def summary(stats,seeds=None):
    sums=stats.reshape(len(stats),-1,4).sum(1)
    assert np.all(sums[:,0]==sums[0,0])
    n=int(sums[0,0]);values=[]
    for i,(count,sse,sae,bias) in enumerate(sums):
        row=dict(RMSE=float(np.sqrt(sse/count)) if count else None,
                 MAE=float(sae/count) if count else None,bias=float(bias/count) if count else None)
        if seeds is not None:row['seed']=seeds[i]
        values.append(row)
    return dict(n_valid=n,by_seed=values,
        **{key:float(np.mean([v[key] for v in values])) if n else None for key in ('RMSE','MAE','bias')},
        RMSE_seed_sd=float(np.std([v['RMSE'] for v in values],ddof=1)) if n and len(values)>1 else None)


def tables(groups,origins,seeds=None):
    years=origins.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    out={}
    for key,stats in groups.items():
        row=summary(stats,seeds)
        row['yearly']={str(y):summary(stats[:,years==y],seeds) for y in sorted(set(years))}
        row['lead_hours']={str(2*(h+1)):summary(stats[:,:,h],seeds) for h in range(12)}
        out[key]=row
    return out


def draws(origins,block_days=7):
    dates=origins.astype('datetime64[s]').astype('datetime64[D]')
    years=dates.astype('datetime64[Y]').astype(int)+1970;rng=np.random.default_rng(20260924)
    out=[]
    for year in sorted(set(years)):
        start=np.datetime64(f'{year}-01-01');days=int((np.datetime64(f'{year+1}-01-01')-start).astype(int))
        n=(days+block_days-1)//block_days;selected=years==year
        ids=(dates[selected]-start).astype(int)//block_days
        index=rng.integers(n,size=(5000,n));weights=np.zeros((5000,n),np.int32)
        np.add.at(weights,(np.arange(5000)[:,None],index),1)
        out.append((selected,ids,weights))
    return out


def time_rmse(stats,shared_draws):
    daily=stats.sum(2)[...,:2];sample=np.zeros((5000,len(stats),2))
    for selected,ids,weight in shared_draws:
        block=np.zeros((weight.shape[1],len(stats),2))
        for s in range(len(stats)):np.add.at(block[:,s],ids,daily[s,selected])
        sample+=(weight@block.reshape(len(block),-1)).reshape(sample.shape)
    if np.any(sample[:,:,0]==0):return None
    return np.sqrt(sample[:,:,1]/sample[:,:,0]).mean(1)


def paired(a,b,origins,block_days=7):
    assert a.shape==b.shape and np.array_equal(a[...,0],b[...,0])
    am,bm=summary(a),summary(b)
    if not am['n_valid']:return dict(n_valid=0,mean_delta=None,ci95=None)
    delta=np.array([x['RMSE']-y['RMSE'] for x,y in zip(am['by_seed'],bm['by_seed'])])
    shared=draws(origins,block_days);ra,rb=time_rmse(a,shared),time_rmse(b,shared)
    ci=None if ra is None or rb is None else np.quantile(ra-rb,[.025,.975]).tolist()
    years=origins.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    return dict(by_seed_delta=delta.tolist(),seeds=SEEDS if len(delta)==5 else None,
        mean_delta=float(delta.mean()),percent_delta=float(100*delta.mean()/bm['RMSE']),
        improving_seeds=int((delta<0).sum()),worsening_seeds=int((delta>0).sum()),
        leave_one_out_means=[float(np.delete(delta,i).mean()) for i in range(len(delta))] if len(delta)>1 else None,
        ci95=ci,block_days=block_days,draws=5000,rng=20260924,
        yearly_delta={str(y):summary(a[:,years==y])['RMSE']-summary(b[:,years==y])['RMSE']
                      if summary(a[:,years==y])['n_valid'] else None for y in sorted(set(years))},
        interval_scope='Conditional time variation for fixed trained models; distinct from seed variation.')


def prediction_set(records,split):
    arrays=[];origins=None
    for row in records:
        paths=row['splits'][split]
        ts=np.load(paths['origins'],allow_pickle=False)
        if origins is None:origins=ts
        else:np.testing.assert_array_equal(origins,ts)
        pred=np.load(paths['predictions'],mmap_mode='r',allow_pickle=False)
        assert pred.shape==(len(ts),12,71,73) and pred.dtype==np.float32
        arrays.append(pred)
    assert len(arrays)==5
    return origins,arrays


def collect_global(records,split,config,driver_path):
    ts,arrays=prediction_set(records,split)
    expected=np.concatenate([np.arange(np.datetime64(f'{y}-01-01'),np.datetime64(f'{y+1}-01-01'),dtype='datetime64[D]') for y in YEARS[split]]).astype('datetime64[s]').astype(np.int64)
    np.testing.assert_array_equal(ts,expected)
    with np.load(driver_path,allow_pickle=False) as z:
        time=z['time'].astype(np.int64);query=ts[:,None]+np.arange(12)*7200;ix=np.searchsorted(time,query)
        np.testing.assert_array_equal(time[ix],query)
        kp,dst=z['Kp'][ix],z['Dst'][ix]
    if not np.isfinite(kp).all() or not np.isfinite(dst).all():raise ValueError('Missing disturbance driver')
    disturbed=np.any((kp>=5)|(dst<=-50),1)
    cache={};stats={k:np.zeros((5,len(ts),12,4)) for k in GROUPS};ensemble={k:np.zeros((1,len(ts),12,4)) for k in GROUPS}
    for i,t in enumerate(ts):
        d=np.datetime64(int(t),'s').astype('datetime64[D]');year=int(str(d)[:4]);assert year<=2025
        if year not in cache:cache[year]=np.load(config['data_files'][str(year)]['path'],mmap_mode='r',allow_pickle=False)
        day=int((d-np.datetime64(f'{year}-01-01')).astype(int))
        truth=(np.asarray(cache[year][day],np.float32)/np.float32(10)).reshape(12,5183).astype(np.float64)
        pred=np.stack([p[i].reshape(12,5183).astype(np.float64) for p in arrays])
        assert np.array_equal(pred.reshape(5,12,71,73)[...,0],pred.reshape(5,12,71,73)[...,-1])
        valid=np.isfinite(truth);unique=valid.reshape(12,71,73).copy();unique[:,:,-1]=False
        masks=dict(all=valid,unique_longitude=unique.reshape(12,5183),high_TEC=valid&(truth>=60),disturbed_day=valid&disturbed[i])
        mean=pred.mean(0,keepdims=True)
        for key,mask in masks.items():
            stats[key][:,i]=sufficient(pred,truth,mask);ensemble[key][:,i]=sufficient(mean,truth,mask)
        if i%400==0:print('global',records[0]['arm'],split,i,'/',len(ts),flush=True)
    return ts,stats,ensemble


def decisions(pairs,main_table):
    primary=pairs['extension_evaluation']['all']['PI-DO'];old=pairs['original_evaluation']['all']['PI-DO']
    harms={};inconsistent=[]
    for split in SPLITS:
        for group in ('high_TEC','disturbed_day'):
            for pair in ('PI-DO','EMP-DO','PI-EMP'):
                p=pairs[split][group][pair]
                if p['mean_delta'] is None:continue
                trigger=p['mean_delta']>0 and p['worsening_seeds']>=4 and all(x>0 for x in p['leave_one_out_means']) and p['ci95'] is not None and p['ci95'][0]>0
                harms[f'{split}/{group}/{pair}']=bool(trigger)
                # Once-only review is required when repeated year harms appear without
                # the sufficient condition; retain the original seven-day interval.
                if pair=='PI-DO' and not trigger and sum(x is not None and x>0 for x in p['yearly_delta'].values())>=2:
                    inconsistent.append(f'{split}/{group}/{pair}')
    ci=primary['ci95'];cross=ci is not None and ci[0]<=0<=ci[1]
    repeated=any(v for k,v in harms.items() if k.endswith('/PI-DO'))
    improve=(primary['mean_delta']<0 and primary['improving_seeds']>=4 and all(x<0 for x in primary['leave_one_out_means'])
             and ci is not None and ci[1]<0 and old['mean_delta']<=0 and not repeated and not inconsistent)
    flat=not improve and (abs(primary['mean_delta'])<.005*main_table['extension_evaluation']['DO']['all']['RMSE'] or cross)
    worse=primary['mean_delta']>0 and ci is not None and ci[0]>0
    labels=[name for name,yes in [('改善成立',improve),('持平',flat),('变差',worse)] if yes]
    ep=pairs['extension_evaluation']['all']['PI-EMP'];ec=ep['ci95']
    return dict(improvement_established=bool(improve),flat=bool(flat),worse=bool(worse),
        classification=labels[0] if len(labels)==1 else ('规则重叠：'+'、'.join(labels) if labels else '未覆盖：报告条件与效应量'),
        sufficient_repeated_harm=harms,once_only_targeted_review_required=inconsistent,
        PI_not_worse_than_EMP=bool(ep['mean_delta']<=0 or (ec is not None and ec[0]<=0<=ec[1])),
        information_caveat='PI additionally uses Dst. Frozen four-coefficient EMP is not strictly information-matched.')
