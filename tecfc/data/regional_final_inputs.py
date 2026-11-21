"""Training-only refits, immutable anchors, and mandatory B/D/E checks."""
import argparse
from functools import lru_cache
import numpy as np
from tecfc.regional.replay_common import *

old = module(Path(__file__).resolve().parent/'regional_windows.py', '_exp0023_old_prepare')

class Native(old.Native):
    @lru_cache(64)
    def day(self, day):
        if day != '2013-12-31': return super().day(day)
        assert self.region == 'japan', 'Only the inherited Japan boundary history is admitted'
        record = self.rows[day]; p = record['path']; assert sha(p) == record['sha256']
        self.inputs[str(p)] = dict(bytes=p.stat().st_size,sha256=sha(p),date=day)
        with np.load(p,allow_pickle=False) as z: a = {k:z[k].copy() for k in ['tec','valid','present','time_utc','lat','lon']}
        np.testing.assert_array_equal(a['time_utc'],old.unix(day)+np.arange(96)*900)
        np.testing.assert_array_equal(a['lat'],self.lat); np.testing.assert_array_equal(a['lon'],self.lon)
        assert a['tec'].dtype == np.float32 and a['valid'].dtype == bool
        return a

def fit(native, years):
    sums=np.zeros((12,native.n),np.float64); valid_counts=np.zeros((12,native.n),np.int64)
    bins=np.zeros(1,np.int64); scale=-np.inf
    for day in old.days(years):
        raw=native.day(day); y=raw['tec'][::8].reshape(12,native.n); v=raw['valid'][::8].reshape(12,native.n)
        sums+=np.where(v,y.astype(float),0);valid_counts+=v;values=y[v]
        assert (values>=0).all()
        if len(values): scale=max(scale,float(values.max()))
        c=np.bincount(np.floor(values.astype(float)/10).astype(np.int64))
        if len(c)>len(bins): bins=np.pad(bins,(0,len(c)-len(bins)))
        bins[:len(c)]+=c
    slot_sum=sums.sum(1);slot_count=valid_counts.sum(1);mean=sums.sum()/valid_counts.sum()
    slot_mean=np.divide(slot_sum,slot_count,out=np.full(12,mean),where=slot_count>0)
    fallback=np.divide(sums,valid_counts,out=np.broadcast_to(slot_mean[:,None],sums.shape).copy(),where=valid_counts>0).astype(np.float32)
    counts=bins.copy(); upper=len(counts)*10; n=int(counts.sum())
    while len(counts)>1 and counts[-1]<1000: counts[-2]+=counts[-1];counts=counts[:-1]
    assert (counts>0).all() and np.isfinite(fallback).all() and scale>0
    edges=np.r_[np.arange(len(counts))*10,upper].astype(float)
    raw_weight=n/(len(counts)*counts.astype(float)); clipped=np.clip(raw_weight,.1,10)
    normalizer=float(np.dot(counts,clipped)/n);weights=clipped/normalizer
    assert abs(np.dot(counts,weights)/n-1)<1e-12
    record=dict(target_years=years,n_targets=n,raw_10_TECU_counts=bins.tolist(),edges=edges.tolist(),counts=counts.tolist(),
        unclipped_weights=raw_weight.tolist(),clipped_weights=clipped.tolist(),normalizer=normalizer,weights=weights.tolist(),scale=scale,
        target_support='All original valid two-hour labels in training years, including target days excluded because of missing history',
        normalization='count-weighted training-label mean 1 after clipping; no second clipping')
    return fallback,scale,sums,valid_counts,record

def anchors(include_evaluation):
    fusion=read(REPO/'experiments/exp0019/FUSED_PREDICTIONS.json');out={}
    for split in ['original_evaluation']+(['extension_evaluation'] if include_evaluation else []):
        record=fusion['systems']['S2']['splits'][split]
        for k in ['predictions','origins']:assert sha(record[k]) == fusion['files_sha256'][record[k]]
        ts=np.load(record['origins']);assert len(set(ts))==len(ts)
        out[split]=(np.load(record['predictions'],mmap_mode='r'),{int(t):i for i,t in enumerate(ts)})
    record=read(REPO/'experiments/exp0022/ANCHOR_2014.json')
    assert record['kind']=='leave_2014_out_global_S2_anchor' and record['usable_as_anchor']
    assert 2014 not in record['training_years']+record['validation_years']
    assert sha(record['predictions'])==record['sha256'] and sha(record['origins']['path'])==record['origins']['sha256']
    ts=np.load(record['origins']['path']);out['2014']=(np.load(record['predictions'],mmap_mode='r'),{int(t):i for i,t in enumerate(ts)})
    return out

def plan(native, years, evaluation=False):
    stamps=old.evaluation_origins(native.region) if evaluation else np.array([old.unix(d) for d in old.days(years)])
    rows=[];excluded=[]
    for t in stamps:
        day=str(np.datetime64(int(t),'s').astype('datetime64[D]')); history=str(np.datetime64(day)-np.timedelta64(1,'D'))
        why=[]
        if native.region=='china' and day=='2014-01-01': why=['inherited_2014_support_has_no_previous_day']
        else:
            a,b=native.day(history),native.day(day)
            if not a['present'][:89].all():why.append('history_frame_absent')
            if not b['present'][::8].all():why.append('target_frame_absent')
        if why:excluded.append(dict(date=day,reasons=why));continue
        rows.append(dict(date=day,history=history,origin=int(t),cutoff=int(t)-7200))
    return rows,excluded

def build(region, stage):
    assert read(EXP/'INHERITED_CONFIG_AUDIT.json')['status']=='passed'
    assert stage in ['stage1','stage2']
    if stage=='stage2':
        selection=read(EXP/'SELECTION_2022.json');assert selection['regions'][region]['replace']
    years=TRAIN1 if stage=='stage1' else TRAIN2
    dest=PREP/stage/region;dest.mkdir(parents=True,exist_ok=False)
    native=Native(region);fallback,scale,sums,counts,record=fit(native,years)
    np.savez(dest/'statistics.npz',scale=np.float64(scale),fallback=fallback,sums=sums,counts=counts,fit_years=np.array(years),lat=native.lat,lon=native.lon)
    source=anchors(stage=='stage1');mp=old.mapping(native.lat,native.lon)
    plans={};exclusions={};allrows=[]
    splits=['train','validation','evaluation'] if stage=='stage1' else ['train','evaluation']
    # Evaluation input/label arrays are prepared for specified B/D checks only;
    # no trained-model evaluation is possible until a committed regional freeze.
    if stage=='stage2': source.update(anchors(True))
    for split in splits:
        rows,excluded=plan(native,years if split=='train' else [2022] if split=='validation' else [2023,2024,2025],split=='evaluation')
        plans[split]=rows;exclusions[split]=excluded;folder=dest/split;folder.mkdir()
        layout=dict(history=(89,np.float32),target=(12,np.float32),valid=(12,bool),anchor=(12,np.float64))
        arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=dtype,shape=(len(rows),frames,native.n)) for k,(frames,dtype) in layout.items()}
        for i,row in enumerate(rows):
            a,b=native.day(row['history']),native.day(row['date']);t=row['origin']
            np.testing.assert_array_equal(a['time_utc'][:89],t-86400+np.arange(89)*900)
            assert (a['time_utc'][:89]<=row['cutoff']).all()
            np.testing.assert_array_equal(b['time_utc'][::8],t+old.OFF)
            arrays['history'][i]=old.fill(a['tec'][:89].reshape(89,native.n),a['valid'][:89].reshape(89,native.n),fallback)
            arrays['target'][i]=b['tec'][::8].reshape(12,native.n);arrays['valid'][i]=b['valid'][::8].reshape(12,native.n)
            which='2014' if row['date'].startswith('2014-') else 'extension_evaluation' if split=='evaluation' else 'original_evaluation'
            f,lookup=source[which];arrays['anchor'][i]=old.remap(f[lookup[t]].reshape(12,5183),mp)
            assert np.isfinite(arrays['anchor'][i]).all();allrows.append((split,i,row))
        for a in arrays.values():a.flush()
        np.save(folder/'origins.npy',np.array([r['origin'] for r in rows],np.int64));write(folder/'rows.json',rows)
        print(json.dumps(dict(region=region,stage=stage,split=split,windows=len(rows))),flush=True)
    h=np.load(dest/'train/history.npy',mmap_mode='r');p=np.array(h[:,::8],np.float32).reshape(-1)
    assert np.isfinite(p).all();q95,q99=np.quantile(p,[.95,.99],method='linear').tolist()
    record.update(n_persistence=len(p),tau95=q95,tau99=q99,percentile_method='numpy linear; all finite training-window P grid-times; no target mask',counts_by_split={k:len(v) for k,v in plans.items()},exclusions=exclusions)
    write(dest/'TRAINING_STATISTICS.json',record)
    write(dest/'metadata.json',dict(region=region,nodes=native.n,scale=scale,counts={s:len(v) for s,v in plans.items()},training_years=years,validation_years=[2022] if stage=='stage1' else [],evaluation_years=[2023,2024,2025],exclusions=exclusions))
    if stage=='stage1':
        # Same support and unfilled labels as exp0021 and exp0022.
        assert {s:len(v) for s,v in plans.items()}==dict(train=1823 if region=='china' else 1824,validation=365,evaluation=1077 if region=='china' else 1091)
        check_b={}
        for name,split,indices,ref in [('evaluation','evaluation',list(range(len(plans['evaluation']))),{'china':6.418629,'japan':4.259275}[region]),
            ('2014','train',[i for i,r in enumerate(plans['train']) if r['date'].startswith('2014-')],read(REPO/'experiments/exp0022/REGIONAL_SCORES.json')['CN' if region=='china' else 'JP']['methods']['mapped_out_of_sample_anchor']['RMSE'])]:
            folder=dest/split;a=np.load(folder/'anchor.npy',mmap_mode='r');y=np.load(folder/'target.npy',mmap_mode='r');v=np.load(folder/'valid.npy',mmap_mode='r');sse=0.;n=0
            for i in indices:
                error=a[i][v[i]]-y[i][v[i]].astype(float);sse+=np.square(error).sum();n+=len(error)
            rmse=float(np.sqrt(sse/n));check_b[name]=dict(RMSE=rmse,reference=ref,absolute_difference=abs(rmse-ref),windows=len(indices),n_valid=n)
        passed=all(r['absolute_difference']<=1e-6 for r in check_b.values())
        write(EXP/f'checks/B_{region}.json',dict(status='passed' if passed else 'failed',region=region,comparisons=check_b,tolerance=1e-6,scope='prespecified anchor-only check'))
        assert passed,'Check B failed'
        rng=np.random.default_rng(20260924);early=[i for i,(_,_,r) in enumerate(allrows) if r['date'].startswith('2014-')]
        picked=list(rng.choice(early,5,replace=False));rest=[i for i in range(len(allrows)) if i not in picked]
        picked+=list(rng.choice(rest,15,replace=False));spots=[]
        for j in picked:
            split,i,row=allrows[int(j)];a,b=native.day(row['history']),native.day(row['date']);folder=dest/split
            np.testing.assert_array_equal(np.load(folder/'history.npy',mmap_mode='r')[i],old.fill(a['tec'][:89].reshape(89,native.n),a['valid'][:89].reshape(89,native.n),fallback))
            np.testing.assert_array_equal(np.load(folder/'target.npy',mmap_mode='r')[i],b['tec'][::8].reshape(12,native.n))
            np.testing.assert_array_equal(np.load(folder/'valid.npy',mmap_mode='r')[i],b['valid'][::8].reshape(12,native.n))
            assert np.max(a['time_utc'][:89])<=row['cutoff'];spots.append(dict(split=split,**row,passed=True))
        write(EXP/f'checks/D_{region}.json',dict(status='passed',all_windows_causal=True,windows=len(allrows),spot_checks=spots,at_least_five_2014=True))
        previous=read(OLD/'TRAINING_STATISTICS.json')['regions'][region]
        old_counts=previous['raw_10_TECU_counts'];new_counts=record['raw_10_TECU_counts'];width=max(len(old_counts),len(new_counts))
        comparison=[dict(lower_TECU=10*i,upper_TECU=10*(i+1),exp0021=old_counts[i] if i<len(old_counts) else 0,exp0023=new_counts[i] if i<len(new_counts) else 0) for i in range(width)]
        for row in comparison:row['increase']=row['exp0023']-row['exp0021']
        high={str(t):dict(exp0021=sum(old_counts[t//10:]),exp0023=sum(new_counts[t//10:]),increase=sum(new_counts[t//10:])-sum(old_counts[t//10:])) for t in [60,80,100]}
        write(EXP/f'checks/E_{region}.json',dict(status='passed',raw_bin_comparison=comparison,high_level_counts=high,old_merged_bins={k:previous[k] for k in ['edges','counts','weights']},new_merged_bins={k:record[k] for k in ['edges','counts','weights']}))
    write(dest/'INPUT_MANIFEST.json',native.inputs)
    write(dest/'FILES.json',{str(p.relative_to(dest)):spec(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='FILES.json'})
    print(json.dumps(dict(status='prepared',region=region,stage=stage,scale=scale,tau99=q99,bins=record['counts'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',choices=REGIONS,required=True);p.add_argument('--stage',choices=['stage1','stage2'],required=True);a=p.parse_args();build(a.region,a.stage)
