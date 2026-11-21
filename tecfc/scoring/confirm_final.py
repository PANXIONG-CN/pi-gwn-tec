"""One registered scoring pass after prediction commit; no inference or fitting."""
import hashlib
import json
import numpy as np
from tecfc.scoring.confirm_records import ROOT, EXP, SEEDS, sha, read, write, now
from tecfc.scoring.confirm_score import sufficient, summary, bootstrap
from tecfc.scoring.confirm_inputs import Reader, SINGLE

METRICS=['RMSE','MAE','bias']
REGROUPS=['P_ge60','Y_ge60','Y_ge80','Y_ge100','persistent_high','new_high','high_decline','high_level_P95']


def samples(stats,ts):
    dates=np.asarray(ts).astype('datetime64[s]').astype('datetime64[D]')
    years=dates.astype('datetime64[Y]').astype(int)+1970
    if len(set(years))==1:
        values,draw=bootstrap(stats,ts)
        return values,hashlib.sha256(draw.tobytes()).hexdigest()
    rng=np.random.default_rng(20260924);total=np.zeros((5000,len(stats),4));digest=hashlib.sha256()
    for year in sorted(set(years)):
        mask=years==year;ids=(dates[mask]-np.datetime64(f'{year}-01-01')).astype(int)//7;n=int(ids.max())+1
        blocks=np.zeros((n,len(stats),4))
        for j,a in enumerate(stats):np.add.at(blocks[:,j],ids,a[mask])
        draws=rng.integers(n,size=(5000,n));digest.update(draws.tobytes())
        for lo in range(0,5000,100):total[lo:lo+100]+=blocks[draws[lo:lo+100]].sum(1)
    return np.sqrt(total[:,:,1]/total[:,:,0]),digest.hexdigest()


def ci(x):return np.quantile(x,[.025,.975]).tolist()


def disturbance(spec,dates):
    assert sha(spec['path'])==spec['sha256']
    hourly={}
    with open(spec['path']) as f:
        for line in f:
            w=line.split()
            if not w:continue
            year,doy,hour=map(int,w[:3]);date=np.datetime64(f'{year}-01-01')+np.timedelta64(doy-1,'D')
            if str(date) not in {str(d) for d in dates}:continue
            kp=float(w[38]);dst=float(w[40]);kp=np.nan if kp==99 else kp/10.;dst=np.nan if dst==99999 else dst
            key=str(date);hourly.setdefault(key,{})[hour]=(kp,dst)
    known=[];active=[];missing=[]
    for date in dates:
        rows=hourly.get(str(date),{});good=len(rows)==24 and np.isfinite(list(rows.values())).all()
        known.append(good);active.append(bool(good and any(k>=5 or d<=-50 for k,d in rows.values())))
        if not good:missing.append(str(date))
    return np.array(active),dict(source=spec,cadence='all 24 hourly OMNI2 records per target UTC day',
        version='Kp: OMNI2/GFZ, final status not separately declared; Dst: January–July provisional, August–October 29 quick-look; October 30 version not independently confirmed',
        rule='any hourly Kp>=5 or Dst<=-50; any missing Kp/Dst hour excludes date only from this stratum',
        complete_days=int(sum(known)),disturbed_days=int(sum(active)),excluded_dates=missing)


def with_improvement(stat,persistence,gstats=None):
    row=summary(stat);p=summary(persistence)['RMSE']
    row['RMSE_improvement_percent_vs_persistence']=100*(1-row['RMSE']/p) if row['n_valid'] and p else None
    if gstats is not None:
        refs={str(s):summary(gstats[j])['RMSE'] for j,s in enumerate(SEEDS)}
        row['RMSE_improvement_percent_vs_G_single_seed']={s:100*(1-row['RMSE']/v) if row['n_valid'] and v else None for s,v in refs.items()}
        mean=float(np.mean(list(refs.values()))) if all(v is not None for v in refs.values()) else None
        row['RMSE_improvement_percent_vs_G_single_seed_metric_mean']=100*(1-row['RMSE']/mean) if row['n_valid'] and mean else None
    return row


def table(stats,groupstats,names,ts,main_names,domain,group_names,subset=None):
    use=np.ones(len(ts),bool) if subset is None else np.asarray(subset)
    dates=ts[use].astype('datetime64[s]').astype('datetime64[D]');months=dates.astype('datetime64[M]').astype(int)%12+1
    a=stats[:,use];gg=groupstats[:,use] if groupstats is not None else None
    daily=a.sum(2);draws,drawsha=samples(daily,ts[use]);pidx=names.index('persistence') if 'persistence' in names else None
    gi=[names.index('G::seed'+str(s)) for s in SEEDS] if domain=='global' and 'G::seed42' in names else None
    rows={}
    for j,name in enumerate(names):
        def report(s,p,gs):return with_improvement(s,p,gs) if p is not None else summary(s)
        p=a[pidx] if pidx is not None else None;g=a[gi] if gi is not None else None
        overall=report(a[j],p,g);overall['RMSE_ci95']=ci(draws[:,j])
        row=dict(all=overall,
            monthly={str(m):report(a[j,months==m],p[months==m] if p is not None else None,g[:,months==m] if g is not None else None) for m in sorted(set(months))},
            horizons={str(2*(k+1)):report(a[j,:,k],p[:,k] if p is not None else None,g[:,:,k] if g is not None else None) for k in range(12)})
        if gg is not None:
            row['groups']={label:report(gg[j,:,k],gg[pidx,:,k] if pidx is not None else None,gg[gi,:,k] if gi is not None else None) for k,label in enumerate(group_names)}
        rows[name]=row
    pairs={}
    comparisons=([('S2','G5'),('G-plus-M5','G5'),('G-geo5','G5'),('PI5','G5'),('EMP5','G5')]
                 if domain=='global' and 'G5' in names else [('S1','S2')] if domain=='global' else
                 [('anchor','persistence'),('final','anchor'),('final','persistence'),('final','exp0021_regional'),('final','R1-B_G0'),('mapped_G5','persistence')])
    for first,second in comparisons:
        i,j=names.index(first),names.index(second)
        pairs[first+'-'+second]=dict(first=first,second=second,delta_RMSE=rows[first]['all']['RMSE']-rows[second]['all']['RMSE'],ci95=ci(draws[:,i]-draws[:,j]))
    seed_metrics={};paired={}
    bases=SINGLE if domain=='global' and gi is not None else ['final','exp0021_regional','R1-B_G0'] if domain!='global' else []
    for base in bases:
        keys=[base+'::seed'+str(s) for s in SEEDS]
        seed_metrics[base]={k:dict(by_seed=[rows[n]['all'][k] for n in keys],mean=float(np.mean([rows[n]['all'][k] for n in keys])),sample_sd=float(np.std([rows[n]['all'][k] for n in keys],ddof=1))) for k in METRICS}
        if domain=='global' and base!='G':
            ids=[names.index(n) for n in keys];d=[rows[n]['all']['RMSE']-rows['G::seed'+str(s)]['all']['RMSE'] for n,s in zip(keys,SEEDS)]
            paired[base+'-G']=dict(seeds=SEEDS,delta_RMSE_by_seed=d,improved_seeds=int(np.sum(np.asarray(d)<0)),
                mean_individual_RMSE_delta=float(np.mean(d)),sample_sd=float(np.std(d,ddof=1)),
                mean_individual_delta_ci95=ci(np.mean(draws[:,ids]-draws[:,gi],axis=1)),criterion_uses='ensemble prediction difference; individual metric mean shown separately')
    return dict(days=len(dates),date_first=str(dates[0]),date_last=str(dates[-1]),months=sorted(int(m) for m in set(months)),
        methods={n:rows[n] for n in main_names},individual_seeds={n:v for n,v in rows.items() if '::seed' in n},
        seed_metrics=seed_metrics,paired_seed_differences=paired,comparisons=pairs,
        time_intervals=dict(block_days=7,origin='January 1 of each calendar year',replicates=5000,rng=20260924,
            ci='95% percentile',common_draw_sha256=drawsha,year_stratified=len(set(dates.astype('datetime64[Y]')))>1,
            scope='time variation conditional on frozen models; seed sample SD is separate'))


def evaluate(key,desc):
    reader=Reader(desc);idx=reader.selected;ts=reader.ts[idx];names=list(desc['methods']);domain=desc['domain'];is2026=desc['period']=='2026';sensitivity=key.startswith('S1')
    stats=np.zeros((len(names),len(idx),12,4));group_names=[];p95=None;dist=None;driver_info=None
    if is2026 and not sensitivity:
        if domain=='global':
            group_names=['unique_longitude_72','Y_ge60','disturbed_day'];dist,driver_info=disturbance(desc['driver_hourly'],reader.dates[idx])
        else:
            group_names=REGROUPS
            p95=float(np.quantile(np.stack([reader.prediction('persistence',i) for i in idx]).reshape(-1),.95,method='linear'))
    groups=np.zeros((len(names),len(idx),len(group_names),4)) if group_names else None
    gate_count=gate_total=gate_valid_count=gate_valid_total=0
    for k,i in enumerate(idx):
        y,v=reader.label(i);masks=[]
        if group_names:
            if domain=='global':
                unique=v.copy();unique[...,-1]=False;masks=[unique,v&(y>=60),v&dist[k]]
            else:
                p=reader.prediction('persistence',i)
                masks=[v&m for m in [p>=60,y>=60,y>=80,y>=100,(p>=60)&(y>=60),(p<60)&(y>=60),(p>=60)&(y<60),p>=p95]]
                if desc['gate_threshold'] is not None:
                    gate=reader.array(desc['gate_history'])[i,::8]>=desc['gate_threshold']
                    gate_count+=int(gate.sum());gate_total+=gate.size;gate_valid_count+=int((gate&v).sum());gate_valid_total+=int(v.sum())
        for j,name in enumerate(names):
            pred=reader.prediction(name,i);stats[j,k]=sufficient(pred,y,v)
            if groups is not None:
                groups[j,k]=np.stack([sufficient(pred,y,m).sum(0) for m in masks])
        if k%100==0:print(json.dumps(dict(dataset=key,scored_windows=k,total=len(idx))),flush=True)
    dest=EXP/'artifacts/scores';dest.mkdir(parents=True,exist_ok=True);path=dest/(key+'.npz');assert not path.exists()
    np.savez_compressed(path,origins=ts,names=np.array(names),stats=stats,groups=groups if groups is not None else np.empty(0),group_names=np.array(group_names))
    result=dict(full=table(stats,groups,names,ts,desc['main_methods'],domain,group_names),
        statistics=dict(path=str(path),sha256=sha(path)),P95=p95,P95_support='all finite common causal P grid-times, without future target mask' if p95 is not None else None,
        disturbance=driver_info)
    if domain=='global' and not sensitivity:
        half=reader.months[idx]<=6
        result['January_June']=table(stats,groups,names,ts,desc['main_methods'],domain,group_names,half)
    if domain!='global' and is2026:
        rows=result['full']['methods'];f=rows['final']['groups']['persistent_high']['bias'];a=rows['anchor']['groups']['persistent_high']['bias']
        result['persistent_high_bias']=dict(final=f,anchor=a,signed_change=f-a if f is not None and a is not None else None,
            absolute_bias_change=abs(f)-abs(a) if f is not None and a is not None else None)
        result['gate_replacement']=dict(threshold=desc['gate_threshold'],replaced_grid_times=gate_count,total_grid_times=gate_total,
            fraction=gate_count/gate_total if gate_total else 0.,valid_label_replaced_grid_times=gate_valid_count,
            valid_label_grid_times=gate_valid_total,valid_label_fraction=gate_valid_count/gate_valid_total if gate_valid_total else 0.)
    return result


def classify(pair,physics=False):
    delta=pair['delta_RMSE'];low,high=pair['ci95']
    if physics:label='与开发期不一致' if delta<0 and high<0 else '与开发期一致'
    else:label='未确认' if delta>=0 else '确认' if high<0 else '方向一致'
    return dict(**pair,classification=label)


def main():
    predictions=read(EXP/'FROZEN_PREDICTIONS_2026.json')
    spec=read(EXP/'SCORING_INPUTS.json')
    assert spec['predictions_sha256']==sha(EXP/'FROZEN_PREDICTIONS_2026.json')
    for p,r in predictions['files'].items():assert sha(p)==r['sha256']
    datasets={}
    for key,desc in spec['datasets'].items():
        datasets[key]=evaluate(key,desc);write(EXP/(key+'_SCORES.json'),datasets[key])
    g=datasets['global_2026']['full']['comparisons']
    criteria={k:classify(g[p],k=='K4') for k,p in [('K1','S2-G5'),('K2','G-plus-M5-G5'),('K3','G-geo5-G5'),('K4','PI5-G5')]}
    for region in ['china','japan']:
        pairs=datasets[region+'_2026']['full']['comparisons']
        criteria[region]={k:classify(pairs[p]) for k,p in [('K5','anchor-persistence'),('K6','final-anchor'),('K7','final-persistence')]}
    result=dict(experiment='exp0025',status='scored_pending_independent_verification',at_utc=now(),datasets=datasets,criteria=criteria,
        K4_basis='PI five-seed mean prediction minus DO/G five-seed mean prediction; paired individual RMSE differences reported separately',
        S1_status='evaluable' if 'S1_2026' in datasets else 'not_evaluable',S1_label='驱动版本与训练不一致的敏感性结果',
        caveats=['2026 one-time confirmation; used data no longer independent of this experiment',
                 'Global: 2026-01-01 to 2026-10-31; regional: 2026-01-01 to 2026-06-30',
                 'IGS GIM and JPLD are mapping products, not independent ground truth',
                 'JPLD monthly archive final status is not officially declared',
                 '2023-2025 participated in development; matched-month comparison is descriptive'],
        predictions_sha256=spec['predictions_sha256'],systems_modified_after_scoring=False,next_experiment_started=False)
    write(EXP/'SCORES_2026_FIRST.json',result)
    print(json.dumps(dict(status=result['status'],criteria=criteria),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
