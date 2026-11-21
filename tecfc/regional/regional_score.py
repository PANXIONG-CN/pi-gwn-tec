"""One CPU scoring pass over completely frozen exp0020 predictions."""
import hashlib,json,os
from pathlib import Path
import numpy as np

REPO=Path(os.environ.get('TECFC_OUTPUT','.')).resolve();EXP=REPO;PROJECT=REPO
SEEDS=[42,1024,2025,3407,5000]
GROUPS=['P_ge60','Y_ge60','Y_ge80','Y_ge100','persistent_high','new_high','high_decline']
METRICS=['RMSE','MAE','bias']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def array(spec):
    p=Path(spec['path']);assert sha(p)==spec['sha256'];return np.load(p,mmap_mode='r',allow_pickle=False)
def sufficient(error,valid):
    e=np.where(valid,error,0.)
    return np.stack([valid.sum(-1),np.square(e).sum(-1),np.abs(e).sum(-1),e.sum(-1)],axis=-1)
def summary(a,persistence=None):
    n,sse,sae,bias=np.asarray(a).reshape(-1,4).sum(0);n=int(n)
    out=dict(n_valid=n,RMSE=float(np.sqrt(sse/n)) if n else None,MAE=float(sae/n) if n else None,bias=float(bias/n) if n else None)
    base=summary(persistence)['RMSE'] if persistence is not None else None
    out['RMSE_improvement_percent_vs_persistence']=100*(1-out['RMSE']/base) if n and base else None
    return out
def table(a,g,ts,p,pg):
    years=ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    return dict(all=summary(a,p),yearly={str(y):summary(a[years==y],p[years==y]) for y in sorted(set(years))},
        horizons={str(2*(i+1)):summary(a[:,i],p[:,i]) for i in range(12)},
        groups={name:summary(g[:,i],pg[:,i]) for i,name in enumerate(GROUPS)})
def seed_summary(tables):
    sample=tables[0];out={}
    for key,value in sample.items():
        if isinstance(value,dict) and key not in ['all']:
            out[key]={sub:aggregate([t[key][sub] for t in tables]) for sub in value}
        else:out[key]=aggregate([t[key] for t in tables])
    return out
def aggregate(values):
    assert len(values)==5 and len({v['n_valid'] for v in values})==1
    n=values[0]['n_valid'];out=dict(n_valid=n,seeds=SEEDS)
    for key in METRICS+['RMSE_improvement_percent_vs_persistence']:
        v=[row[key] for row in values]
        measurable=n and all(x is not None for x in v)
        out[key]=dict(mean=float(np.mean(v)) if measurable else None,sample_sd=float(np.std(v,ddof=1)) if measurable else None,by_seed=v)
    return out

def intervals(allstats,names,ts):
    keep=['R1_ensemble','anchor','R0_ensemble','equal_mix'];selected=np.stack([allstats[names.index(k)].sum(1) for k in keep])
    dates=ts.astype('datetime64[s]').astype('datetime64[D]');years=dates.astype('datetime64[Y]').astype(int)+1970
    rng=np.random.default_rng(20260924);samples=np.zeros((5000,4,2))
    draw_hash=hashlib.sha256()
    for year in sorted(set(years)):
        start=np.datetime64(f'{year}-01-01');n=(int((np.datetime64(f'{year+1}-01-01')-start).astype(int))+6)//7
        mask=years==year;ids=(dates[mask]-start).astype(int)//7;draw=rng.integers(n,size=(5000,n));draw_hash.update(draw.tobytes())
        blocks=np.zeros((n,4,2))
        for m in range(4):np.add.at(blocks[:,m],ids,selected[m,mask][:,[1,0]])
        samples+=blocks[draw].sum(1)
    scores=np.sqrt(samples[:,:,0]/samples[:,:,1]);out={}
    for j in range(1,4):
        delta=summary(selected[0])['RMSE']-summary(selected[j])['RMSE']
        out['R1_ensemble-'+keep[j]]=dict(delta_RMSE=delta,ci95=np.quantile(scores[:,0]-scores[:,j],[.025,.975]).tolist(),
            replicates=5000,calendar_block_days=7,year_stratified=True,rng=20260924,common_draw_sha256=draw_hash.hexdigest(),scope='time variation conditional on these frozen models; distinct from seed SD')
    return out

def evaluate(region,period,manifest):
    split='validation' if period=='2022' else 'evaluation';folder=EXP/'artifacts/prepared_r1'/region/split
    ts=array(manifest['origins']);np.testing.assert_array_equal(ts,np.load(folder/'origins.npy'))
    for name,specification in manifest['scoring_arrays'].items():
        assert Path(specification['path'])==folder/(name+'.npy') and sha(specification['path'])==specification['sha256']
    y=np.load(folder/'target.npy',mmap_mode='r');v=np.load(folder/'valid.npy',mmap_mode='r')
    h=np.load(folder/'history.npy',mmap_mode='r');anchor=np.load(folder/'anchor.npy',mmap_mode='r')
    forecasts={name:array(spec) for name,spec in manifest['predictions'].items()}
    names=[f'{arm}_{s}' for arm in ['R0','R1'] for s in SEEDS]+['R0_ensemble','R1_ensemble','anchor','persistence','equal_mix']
    historical={name:array(spec) for name,spec in manifest.get('historical_predictions',{}).items()}
    if period=='2023-2025':names+=['historical_A_native_mean'] if region=='china' else ['historical_dense_do_mean','historical_dense_de_mean']
    assert set(forecasts)==set(names[:10])
    stats=np.zeros((len(names),len(ts),12,4));groups=np.zeros((len(names),len(ts),7,4))
    for i,t in enumerate(ts):
        valid=v[i];target=y[i].astype(np.float64);p=h[i,::8].astype(np.float64)
        f={name:np.asarray(a[i],np.float64).reshape(12,-1) for name,a in forecasts.items()}
        f.update({arm+'_ensemble':np.mean([f[f'{arm}_{s}'] for s in SEEDS],axis=0) for arm in ['R0','R1']})
        f.update(anchor=anchor[i],persistence=p,equal_mix=(f['R0_ensemble']+anchor[i])/2)
        if historical:
            for group in (['A_native'] if region=='china' else ['dense_do','dense_de']):
                values=[np.asarray(historical[f'{group}_{s}'][i],np.float64).reshape(12,-1) for s in (SEEDS[:3] if region=='china' else SEEDS)]
                f['historical_'+group+'_mean']=np.mean(values,axis=0)
        masks=[p>=60,target>=60,target>=80,target>=100,(p>=60)&(target>=60),(p<60)&(target>=60),(p>=60)&(target<60)]
        assert set(f)==set(names)
        for j,name in enumerate(names):
            assert np.isfinite(f[name]).all();e=f[name]-target
            stats[j,i]=sufficient(e,valid)
            groups[j,i]=np.stack([sufficient(e,valid&mask).sum(0) for mask in masks])
        if i%200==0:print(region,period,i,'/',len(ts),flush=True)
    persistence=names.index('persistence');tables={name:table(stats[j],groups[j],ts,stats[persistence],groups[persistence]) for j,name in enumerate(names)}
    ensembles={name:tables[name] for name in names[10:]}
    for value in ensembles.values():value['seed_sd']=None
    out=dict(days=len(ts),methods=ensembles,seed_metrics={arm:seed_summary([tables[f'{arm}_{s}'] for s in SEEDS]) for arm in ['R0','R1']},
        individual_seeds={name:tables[name] for name in names[:10]},time_block_intervals=intervals(stats,names,ts),
        support='same original effective JPLD label mask and fixed regional origin set for all methods',historical_reference_label=None if period=='2022' else ('A_native 三种子均值预测（含驱动输入、训练年 2014–2020）；与 R0 同时存在数据年份和输入通道差异，只作描述，不作归因' if region=='china' else 'Dense DO/DE 五种子均值预测；训练年不同，仅作描述'))
    np.savez_compressed(EXP/'artifacts'/f'{region}_{period}_daily_statistics.npz',origins=ts,names=np.array(names),stats=stats,groups=groups)
    return out

def decision(periods):
    q1=all(x['methods']['R1_ensemble']['all']['RMSE']<x['methods']['anchor']['all']['RMSE'] for x in periods.values())
    q2=all(x['methods']['R1_ensemble']['all']['RMSE']<x['methods']['R0_ensemble']['all']['RMSE'] for x in periods.values())
    q3={}
    for period,x in periods.items():
        r=x['methods']['R1_ensemble']['groups']['persistent_high']['bias'];a=x['methods']['anchor']['groups']['persistent_high']['bias']
        delta=None if r is None or a is None else abs(r)-abs(a)
        q3[period]=dict(R1_bias=r,anchor_bias=a,absolute_bias_change=delta,classification=None if delta is None else ('持平' if abs(delta)<.5 else '变小' if delta<0 else '变大'))
    larger=any(x['classification']=='变大' for x in q3.values());observed=all(x['classification'] is not None for x in q3.values())
    if not q1:classification='区域残差没有超过零样本的锚，区域上直接使用全球融合预测。'
    elif larger:classification='总体改善以高值为代价，需要让残差随水平受限后再用。'
    elif q2 and observed:classification='锚定残差设计成立，可用最终的全球融合预测作为锚重训。'
    else:classification='未落入预先写定的三类；原样报告 Q1–Q3，不新增判读规则。'
    return dict(Q1=q1,Q2=q2,Q3=q3,classification=classification)
