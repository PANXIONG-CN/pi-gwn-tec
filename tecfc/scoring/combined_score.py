"""Fixed descriptive statistics, using the unchanged exp0013 score kernels."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(os.environ.get('TECFC_OUTPUT','.')).resolve();EXP=ROOT/'experiments/exp0018'
SEEDS=[42,1024,2025,3407,5000]
SPLITS=['validation','original_evaluation','extension_evaluation']


def reference():
    source=Path(__file__).resolve().parent/'pooled.py'
    spec=importlib.util.spec_from_file_location('_exp0018_original_scorer',source)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    return ref


def dump_daily(ref,arm,split,ts,stats,ensemble):
    path=EXP/f'artifacts/statistics/{arm}_{split}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,origins=ts,**stats,**{'ensemble_'+k:v for k,v in ensemble.items()})
    return str(path)


def load_daily(ref,arm,split):
    with np.load(EXP/f'artifacts/statistics/{arm}_{split}.npz',allow_pickle=False) as z:
        return z['origins'].copy(),{k:z[k].copy() for k in ref.GROUPS},{k:z['ensemble_'+k].copy() for k in ref.GROUPS}


def enrich(value,deterministic=False):
    if isinstance(value,dict):
        if 'by_seed' in value:
            if deterministic:
                value.pop('RMSE_seed_sd',None)
                value['prediction_metrics']=value.pop('by_seed')[0]
            else:
                for name in ('RMSE','MAE','bias'):
                    a=[v[name] for v in value['by_seed']]
                    value[name+'_seed_sd']=float(np.std(a,ddof=1)) if all(x is not None for x in a) else None
        for x in list(value.values()):enrich(x,deterministic)
    return value


ARMS=('G-plus-M','G-plus-S','C-plus')
BASES=('G','C','G-long','G-geo','G-MAE','C-in')
COMPARISONS={'G-plus-M':['G','G-long','G-geo','G-MAE','G-plus-S'],
             'G-plus-S':['G','G-long','G-geo','G-MAE'],'C-plus':['C-in','C']}


def inputs_and_anchors(ref):
    frozen=ref.read(EXP/'FROZEN_CANDIDATES.json')
    if not frozen['all_training_complete']:raise ValueError('Wait for all training')
    paths=ref.read(EXP/'configs/inference.json');upstream=ref.read(ROOT/'experiments/exp0017/FROZEN_PREDICTIONS.json')
    fused=ref.read(ROOT/'experiments/exp0017/FUSED_PREDICTIONS.json');records={a:upstream['arms'][a] for a in BASES};files={}
    def checked(p,h):
        if ref.sha(p)!=h:raise ValueError('Changed scoring input '+str(p))
        files[str(p)]=h
    for row in paths['data_files'].values():checked(row['path'],row['sha256'])
    driver=ROOT.parent/'output/gimtec-v2-20261013/drivers_2009_2025_2h.npz';checked(driver,ref.sha(driver))
    for name in ('score.py','recheck.py','freeze_predictions.py','operations.py','delivery.py'):
        p=ROOT/'src/fusion_members/exp0018'/name
        checked(p,ref.sha(p))
    for arm,rows in records.items():
        assert [r['seed'] for r in rows]==SEEDS
        for row in rows:
            for split in SPLITS:
                for field in ('predictions','origins'):
                    p=row['splits'][split][field];checked(p,upstream['files_sha256'][p])
    for split in ('validation','extension_evaluation'):
        for field in ('predictions','origins'):
            p=fused['splits'][split][field];checked(p,fused['files_sha256'][p])
    sources={a:('exp0013' if a=='G' else 'exp0015' if a in ('G-long','G-geo') else 'exp0016') for a in BASES if a!='C'}
    expected={}
    for arm,exp in sources.items():
        p=ROOT/f'experiments/{exp}/result.json';checked(p,ref.sha(p));expected[arm]=ref.read(p)
    for name in ('FROZEN_PREDICTIONS.json','FUSED_PREDICTIONS.json'):
        p=ROOT/'experiments/exp0017'/name;checked(p,ref.sha(p))
    ref.write(EXP/'SCORING_INPUTS.json',dict(status='frozen_scoring_inputs',source_commit=frozen['source_commit'],
        files_sha256=files,
        driver=str(driver),reference_arms=records,exp0017_adopted=fused,candidate_freeze_sha256=ref.sha(EXP/'FROZEN_CANDIDATES.json')))
    rows={}
    for split in SPLITS:
        rows[split]={}
        for arm in BASES:
            ts,stats,ens=ref.collect_global(records[arm],split,paths,str(driver))
            single=ref.summary(stats['all'],SEEDS);ensemble=ref.summary(ens['all'])
            row=dict(single_model_five_seed_mean_RMSE=single['RMSE'],ensemble_RMSE=ensemble['RMSE'])
            if arm in expected:
                source=expected[arm];a=source['main_table'][split][arm]['all']['RMSE'];b=source['ensemble_table'][split][arm]['all']['RMSE']
                assert abs(single['RMSE']-a)<1e-6 and abs(ensemble['RMSE']-b)<1e-6
                row.update(single_expected=a,ensemble_expected=b,single_difference=single['RMSE']-a,ensemble_difference=ensemble['RMSE']-b,passed=True)
            if arm=='G' and split!='validation':
                explicit={'original_evaluation':2.233885478,'extension_evaluation':4.729342137}[split]
                assert abs(single['RMSE']-explicit)<1e-6;row['explicit_G_anchor']=explicit
            rows[split][arm]=row;dump_daily(ref,arm,split,ts,stats,ens)
    ref.write(EXP/'ANCHORS.json',dict(status='passed',splits=rows,tolerance=1e-6,
        note='All requested historical seed-mean and ensemble anchors recomputed from frozen predictions before any new scores.'))
    print(json.dumps(dict(status='anchors_passed')),flush=True)


def correlations(ref,records,reference_records,split,config):
    ts,arrays=ref.prediction_set(records,split);others=[]
    for record in reference_records:
        paths=record['splits'][split];np.testing.assert_array_equal(ts,np.load(paths['origins'],allow_pickle=False))
        others.append(np.load(paths['predictions'],mmap_mode='r',allow_pickle=False))
    sums=np.zeros(6,np.float64);cache={}
    for i,t in enumerate(ts):
        day=np.datetime64(int(t),'s').astype('datetime64[D]');year=int(str(day)[:4]);assert year<=2025
        if year not in cache:cache[year]=np.load(config['data_files'][str(year)]['path'],mmap_mode='r',allow_pickle=False)
        ix=int((day-np.datetime64(f'{year}-01-01')).astype(int));y=(np.asarray(cache[year][ix],np.float32)/np.float32(10)).astype(np.float64);valid=np.isfinite(y)
        a=np.mean([p[i].astype(np.float64) for p in arrays],axis=0)[valid]-y[valid]
        b=np.mean([p[i].astype(np.float64) for p in others],axis=0)[valid]-y[valid]
        sums += [a.size,a.sum(),b.sum(),np.square(a).sum(),np.square(b).sum(),(a*b).sum()]
    n,sa,sb,saa,sbb,sab=sums;va=saa/n-(sa/n)**2;vb=sbb/n-(sb/n)**2;cov=sab/n-sa*sb/n**2
    return dict(n=int(n),Pearson=float(cov/np.sqrt(va*vb)),member_bias=float(sa/n),reference_bias=float(sb/n),
                member_error_variance=float(va),reference_error_variance=float(vb),covariance=float(cov))


def score(ref):
    manifest=ref.read(EXP/'FROZEN_PREDICTIONS.json');inputs=ref.read(EXP/'SCORING_INPUTS.json')
    if ref.read(EXP/'ANCHORS.json')['status']!='passed':raise ValueError('Historical anchors required')
    for p,h in {**manifest['files_sha256'],**inputs['files_sha256']}.items():
        if ref.sha(p)!=h:raise ValueError('Changed frozen scoring input '+p)
    config=ref.read(EXP/'configs/inference.json');main={};ensemble={};paired={};errors={};statistics={};additivity={}
    for split in SPLITS:
        main[split]={};ensemble[split]={};paired[split]={};allstats={};allens={}
        for arm in BASES+ARMS:
            if arm in BASES:ts,stats,ens=load_daily(ref,arm,split)
            else:
                ts,stats,ens=ref.collect_global(manifest['arms'][arm],split,config,inputs['driver'])
                dump_daily(ref,arm,split,ts,stats,ens)
            allstats[arm]=stats;allens[arm]=ens
            main[split][arm]=enrich(ref.tables(stats,ts,SEEDS));ensemble[split][arm]=enrich(ref.tables(ens,ts),True)
            path=EXP/f'artifacts/statistics/{arm}_{split}.npz';statistics[arm+'/'+split]=dict(path=str(path),sha256=ref.sha(path))
        for arm,controls in COMPARISONS.items():
            for control in controls:
                key=arm+'_minus_'+control;paired[split][key]={}
                for group in ref.GROUPS:
                    a,b=allstats[arm][group],allstats[control][group]
                    value=ref.paired(a,b,ts)
                    value['lead_hours']={}
                    for h in range(12):
                        aa,bb=ref.summary(a[:,:,h]),ref.summary(b[:,:,h])
                        delta=[x['RMSE']-y['RMSE'] if x['RMSE'] is not None else None for x,y in zip(aa['by_seed'],bb['by_seed'])]
                        value['lead_hours'][str(2*(h+1))]=dict(by_seed_delta=delta,mean_delta=float(np.mean(delta)) if delta[0] is not None else None)
                    paired[split][key][group]=value
        rmse=lambda a:main[split][a]['all']['RMSE']
        improvement={a:rmse('G')-rmse(a) for a in ('G-long','G-geo','G-MAE')}
        additivity[split]={}
        for arm,components in [('G-plus-M',('G-long','G-geo','G-MAE')),('G-plus-S',('G-long','G-geo'))]:
            actual=rmse('G')-rmse(arm);expected=sum(improvement[a] for a in components)
            additivity[split][arm]=dict(reference='G',actual_improvement=actual,single_component_improvements={a:improvement[a] for a in components},
                sum_of_component_improvements=expected,actual_minus_sum=actual-expected,interpretation='descriptive same-backbone comparison; differing training windows are inherited')
        actual=rmse('C')-rmse('C-plus');proxy=rmse('C')-rmse('C-in')+improvement['G-long']
        additivity[split]['C-plus']=dict(reference='C',actual_improvement=actual,sum_of_component_improvements=None,
            strict_additivity_identifiable=False,missing_component='C with only extra historical TEC channels was not run',
            cross_backbone_proxy_sum=proxy,actual_minus_proxy=actual-proxy,
            proxy_components={'C_minus_Cin':rmse('C')-rmse('C-in'),'G_minus_Glong':improvement['G-long']})
        if split in ('validation','extension_evaluation'):
            errors[split]={}
            for arm in ARMS:
                errors[split][arm]={
                    'G5':correlations(ref,manifest['arms'][arm],inputs['reference_arms']['G'],split,config),
                    'exp0017_adopted':correlations(ref,manifest['arms'][arm],[inputs['exp0017_adopted']],split,config)}
        ref.write(EXP/'SCORING_PROGRESS.json',dict(completed_split=split,checked_at_utc=datetime.now(timezone.utc).isoformat()))
    result=dict(experiment='exp0018',status='complete_scoring',source_commit=manifest['source_commit'],
        frozen_predictions_sha256=ref.sha(EXP/'FROZEN_PREDICTIONS.json'),anchors=ref.read(EXP/'ANCHORS.json'),
        main_table=main,ensemble_table=ensemble,paired=paired,additivity=additivity,error_correlations=errors,daily_statistics=statistics,
        training=ref.read(EXP/'FROZEN_CANDIDATES.json')['formal_runs'],dropped_training_windows=26,members_discarded=0,
        no_fusion_performed=True,no_weight_selection=True,
        operational_incidents=ref.read(EXP/'OPERATIONAL_INCIDENTS.json') if (EXP/'OPERATIONAL_INCIDENTS.json').exists() else {'incidents':[]},
        caveats=['2023–2025 has participated in development; not independent confirmation.',
                 'All members delivered regardless of standalone accuracy.',
                 'C-plus has no C-long component control: strict additivity is not identifiable; cross-backbone proxy is labelled.',
                 'Seven-day intervals describe time variation conditional on trained models, separate from seed sample SD.',
                 'Retrospectively finalized Kp/Dst is used only for post-freeze disturbance strata.'])
    ref.write(EXP/'result.json',result);print(json.dumps(dict(status='complete_scoring',arms=ARMS)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['anchors','score']);a=p.parse_args();ref=reference()
    inputs_and_anchors(ref) if a.action=='anchors' else score(ref)
