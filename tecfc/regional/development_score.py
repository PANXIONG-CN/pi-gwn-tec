"""One-pass scores, regional 2022 selection and common-draw time intervals."""
import argparse
import numpy as np
from tecfc.regional.replay_common import *

old=module(HERE/'regional_score.py','_exp0023_metrics')
GROUPS=['high_level_P95']+old.GROUPS
old.GROUPS=GROUPS

def array(rec):
    assert sha(rec['path'])==rec['sha256']
    return np.load(rec['path'],mmap_mode='r',allow_pickle=False)

def evaluate(region,period,manifest):
    ts=array(manifest['origins']);data={k:array(v) for k,v in manifest['scoring_arrays'].items()}
    y,valid,history,anchor=[data[k] for k in ['target','valid','history','anchor']]
    descriptors=manifest['methods'];methods=list(descriptors)+['anchor','persistence'];seed_names=[f'{name}::seed{s}' for name in descriptors for s in SEEDS]
    names=methods+seed_names;stats=np.zeros((len(names),len(ts),12,4));groups=np.zeros((len(names),len(ts),len(GROUPS),4))
    p95=float(np.quantile(np.array(history[:,::8]).reshape(-1),.95,method='linear'))
    cache={}
    def cached(rec):
        if rec['path'] not in cache:cache[rec['path']]=array(rec)
        return cache[rec['path']]
    forecasts={k:cached(v['prediction']) for k,v in descriptors.items()}
    seeds={k:[cached(rec) for rec in v['seeds']] for k,v in descriptors.items()}
    gate_histories={k:cached(v['gate_history']) for k,v in descriptors.items()}
    for i in range(len(ts)):
        target=y[i].astype(float);v=valid[i];p=history[i,::8].astype(float)
        masks=[p>=p95,p>=60,target>=60,target>=80,target>=100,(p>=60)&(target>=60),(p<60)&(target>=60),(p>=60)&(target<60)]
        values={k:a[i] for k,a in forecasts.items()};values.update(anchor=anchor[i],persistence=p)
        for name,desc in descriptors.items():
            mask=None if desc['gate']=='G0' else gate_histories[name][i,::8]>=desc['threshold']
            for seed,a in zip(SEEDS,seeds[name]):values[f'{name}::seed{seed}']=a[i] if mask is None else np.where(mask,anchor[i],a[i])
        for j,name in enumerate(names):
            assert np.isfinite(values[name]).all();error=values[name]-target
            stats[j,i]=old.sufficient(error,v);groups[j,i]=np.stack([old.sufficient(error,v&mask).sum(0) for mask in masks])
        if i%300==0:print(json.dumps(dict(region=region,period=period,scored_days=i,total_days=len(ts))),flush=True)
    pi=names.index('persistence')
    tables={name:old.table(stats[j],groups[j],ts,stats[pi],groups[pi]) for j,name in enumerate(names)}
    # Report yearly high-level strata using the same period-wide group cutoffs.
    years=ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    for j,name in enumerate(names):
        tables[name]['group_yearly']={str(year):{g:old.summary(groups[j,years==year,k],groups[pi,years==year,k]) for k,g in enumerate(GROUPS)} for year in sorted(set(years))}
    seed_metrics={}
    for name in descriptors:
        rows=[tables[f'{name}::seed{s}'] for s in SEEDS]
        seed_metrics[name]=old.seed_summary([{k:v for k,v in row.items() if k!='group_yearly'} for row in rows])
        seed_metrics[name]['group_yearly']={year:{group:old.aggregate([row['group_yearly'][year][group] for row in rows]) for group in GROUPS} for year in rows[0]['group_yearly']}
    path=EXP/'artifacts'/f'{region}_{period}_daily_statistics.npz';np.savez_compressed(path,origins=ts,names=np.array(names),stats=stats,groups=groups)
    return dict(days=len(ts),P95=p95,methods={name:dict(**tables[name],seed_sd=None,role=descriptors.get(name,{}).get('role','baseline')) for name in methods},
        seed_metrics=seed_metrics,individual_seeds={name:tables[name] for name in seed_names},sufficient_statistics=spec(path),
        support='same unfilled original valid JPLD product labels and fixed origins; common stage1 causal P for strata and persistence baseline')

def select(results):
    chosen={}
    for region in REGIONS:
        rows=[];methods=results[region]['methods'];anchor=methods['anchor']
        for candidate in CANDIDATES:
            m=methods[candidate];overall=m['all']['RMSE']/anchor['all']['RMSE'];high=m['groups']['high_level_P95']['RMSE']/anchor['groups']['high_level_P95']['RMSE']
            rows.append(dict(candidate=candidate,overall_RMSE=m['all']['RMSE'],high_level_RMSE=m['groups']['high_level_P95']['RMSE'],
                overall_relative_RMSE=overall,high_level_relative_RMSE=high,eligible=high<=1.))
        eligible=[row for row in rows if row['eligible']]
        selected=min(eligible,key=lambda row:row['overall_relative_RMSE']) if eligible else min(rows,key=lambda row:row['high_level_relative_RMSE'])
        arm,gate=selected['candidate'].split('/');baseline={'china':4.009078,'japan':2.469876}[region]
        replace=selected['overall_RMSE']<=baseline*1.005
        reference=methods['exp0021_R1/G1' if region=='china' else 'exp0021_R1/G0']['all']['RMSE']
        assert abs(reference-baseline)<1e-6,'Frozen exp0021 replacement benchmark mismatch'
        chosen[region]=dict(**selected,arm=arm,gate=gate,constraint_satisfied=bool(eligible),replace=replace,candidates=rows,
            baseline_RMSE=baseline,replacement_limit_RMSE=baseline*1.005,change_percent=100*(selected['overall_RMSE']/baseline-1),
            retained_system=None if replace else 'exp0021 '+('R1/G1' if region=='china' else 'R1/G0'))
    return dict(status='passed',selection_status='selected_on_2022_only',at_utc=now(),regions=chosen,
        tie_break='first preregistered candidate in exact unrounded ties',design_reason='increased high-activity training coverage;2022 checks for no material overall degradation',evaluation_scores_read=False)

def intervals(result):
    z=np.load(result['sufficient_statistics']['path']);ts=z['origins'];names=z['names'].tolist();stats=z['stats'];groups=z['groups']
    keep=['final','exp0021_regional','stage1_selected','anchor','persistence'];idx=[names.index(k) for k in keep]
    # [model,day,stratum, sufficient statistic], shared draws for all strata/pairs.
    selected=np.concatenate([stats[idx].sum(2)[:,:,None,:],groups[idx]],axis=2)
    strata=['all']+GROUPS;dates=ts.astype('datetime64[s]').astype('datetime64[D]');years=dates.astype('datetime64[Y]').astype(int)+1970
    rng=np.random.default_rng(20260924);samples=np.zeros((5000,5,len(strata),4));draw_hash=hashlib.sha256()
    for year in sorted(set(years)):
        start=np.datetime64(f'{year}-01-01');n=(int((np.datetime64(f'{year+1}-01-01')-start).astype(int))+6)//7
        mask=years==year;ids=(dates[mask]-start).astype(int)//7;draw=rng.integers(n,size=(5000,n));draw_hash.update(draw.tobytes())
        blocks=np.zeros((n,5,len(strata),4))
        for m in range(5):np.add.at(blocks[:,m],ids,selected[m,mask])
        # Bound memory while preserving one common fixed sample array.
        for lo in range(0,5000,100):samples[lo:lo+100]+=blocks[draw[lo:lo+100]].sum(1)
    result={}
    for j in range(1,5):
        groups_out={}
        for k,label in enumerate(strata):
            point0=old.summary(selected[0,:,k]);pointj=old.summary(selected[j,:,k]);metrics={}
            good=(samples[:,0,k,0]>0)&(samples[:,j,k,0]>0)
            for metric,column in [('RMSE',1),('MAE',2),('bias',3)]:
                if not point0['n_valid'] or not pointj['n_valid']:metrics[metric]=dict(delta=None,ci95=None,valid_replicates=0);continue
                a=samples[good,0,k,column]/samples[good,0,k,0];b=samples[good,j,k,column]/samples[good,j,k,0]
                if metric=='RMSE':a=np.sqrt(a);b=np.sqrt(b)
                metrics[metric]=dict(delta=point0[metric]-pointj[metric],ci95=np.quantile(a-b,[.025,.975]).tolist() if good.any() else None,valid_replicates=int(good.sum()))
            groups_out[label]=metrics
        result['final-'+keep[j]]=dict(delta_RMSE=groups_out['all']['RMSE']['delta'],ci95=groups_out['all']['RMSE']['ci95'],strata=groups_out,
            replicates=5000,calendar_block_days=7,year_stratified=True,rng=20260924,common_draw_sha256=draw_hash.hexdigest(),scope='time variation conditional on frozen models; separate from seed SD')
    return result
