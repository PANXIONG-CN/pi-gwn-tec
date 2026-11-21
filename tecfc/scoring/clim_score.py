"""Fixed descriptive statistics, using the unchanged exp0013 score kernels."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(os.environ.get('TECFC_OUTPUT','.')).resolve();EXP=ROOT/'experiments/exp0016'
SEEDS=[42,1024,2025,3407,5000]
SPLITS=['validation','original_evaluation','extension_evaluation']


def reference():
    source=Path(__file__).resolve().parent/'pooled.py'
    spec=importlib.util.spec_from_file_location('_exp0016_original_scorer',source)
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


def inputs_and_anchors(ref):
    frozen=ref.read(EXP/'FROZEN_CANDIDATES.json')
    if not frozen['all_training_complete']:raise ValueError('Wait for all training')
    paths=ref.read(EXP/'configs/inference.json');anchor=ref.read(EXP/'ANCHOR_PREDICTIONS.json')
    if anchor['candidate_freeze_sha256']!=ref.sha(EXP/'FROZEN_CANDIDATES.json'):raise ValueError('Wrong anchor identity')
    files={}
    for row in paths['data_files'].values():
        if int(Path(row['path']).stem.split('_')[-1]) not in range(2009,2026):raise ValueError('Forbidden observation')
        if ref.sha(row['path'])!=row['sha256']:raise ValueError('Input hash mismatch')
        files[row['path']]=row['sha256']
    driver=ROOT.parent/'output/gimtec-v2-20261013/drivers_2009_2025_2h.npz';files[str(driver)]=ref.sha(driver)
    for name in ('score.py','recheck.py','freeze_predictions.py','operations.py','delivery.py'):
        p=ROOT/'src/fusion_members/exp0016'/name
        files[str(p)]=ref.sha(p)
    for p,h in anchor['files_sha256'].items():
        if ref.sha(p)!=h:raise ValueError('Historical prediction changed')
        files[p]=h
    ref.write(EXP/'SCORING_INPUTS.json',dict(status='frozen_scoring_inputs',source_commit=frozen['source_commit'],
        
        files_sha256=files,driver=str(driver),candidate_freeze_sha256=ref.sha(EXP/'FROZEN_CANDIDATES.json')))
    expected_mean={'original_evaluation':2.233885478,'extension_evaluation':4.729342137}
    expected_ensemble={'validation':2.31396563758096,'original_evaluation':2.136731879698973,'extension_evaluation':4.49271502063866}
    rows={}
    for split in SPLITS:
        ts,stats,ens=ref.collect_global(anchor['arms']['G'],split,paths,str(driver))
        single=ref.summary(stats['all'],SEEDS);ensemble=ref.summary(ens['all'])
        delta=ensemble['RMSE']-expected_ensemble[split]
        assert abs(delta)<1e-6
        row=dict(ensemble_RMSE=ensemble['RMSE'],ensemble_expected=expected_ensemble[split],ensemble_difference=delta)
        if split in expected_mean:
            delta=single['RMSE']-expected_mean[split];assert abs(delta)<1e-6
            row.update(single_model_five_seed_mean_RMSE=single['RMSE'],single_expected=expected_mean[split],single_difference=delta)
        rows[split]=dict(passed=True,**row);dump_daily(ref,'G',split,ts,stats,ens)
    ref.write(EXP/'ANCHORS.json',dict(status='passed',splits=rows,tolerance=1e-6,
        note='Historical numbers independently reproduced before scoring any new predictions.'))
    print(json.dumps(dict(status='anchors_passed',splits=rows)),flush=True)


def correlations(ref,records,reference_records,split,config):
    ts,arrays=ref.prediction_set(records,split);gts,garrays=ref.prediction_set(reference_records,split)
    np.testing.assert_array_equal(ts,gts);s=np.zeros(6,np.float64);cache={}
    for i,t in enumerate(ts):
        day=np.datetime64(int(t),'s').astype('datetime64[D]');year=int(str(day)[:4]);assert year<=2025
        if year not in cache:cache[year]=np.load(config['data_files'][str(year)]['path'],mmap_mode='r',allow_pickle=False)
        ix=int((day-np.datetime64(f'{year}-01-01')).astype(int))
        y=(np.asarray(cache[year][ix],np.float32)/np.float32(10)).astype(np.float64);valid=np.isfinite(y)
        a=np.mean([p[i].astype(np.float64) for p in arrays],axis=0)[valid]-y[valid]
        b=np.mean([p[i].astype(np.float64) for p in garrays],axis=0)[valid]-y[valid]
        s += [a.size,a.sum(),b.sum(),np.square(a).sum(),np.square(b).sum(),(a*b).sum()]
    n,sa,sb,saa,sbb,sab=s;va=saa/n-(sa/n)**2;vb=sbb/n-(sb/n)**2;cov=sab/n-sa*sb/n**2
    return dict(n=int(n),Pearson=float(cov/np.sqrt(va*vb)),member_error_variance=float(va),G5_error_variance=float(vb),
                member_bias=float(sa/n),G5_bias=float(sb/n),cross_moment_including_bias=float(sab/n),covariance=float(cov))


def score(ref):
    manifest=ref.read(EXP/'FROZEN_PREDICTIONS.json');anchor=ref.read(EXP/'ANCHOR_PREDICTIONS.json')
    if ref.read(EXP/'ANCHORS.json')['status']!='passed':raise ValueError('Historical anchors required')
    paths=ref.read(EXP/'configs/inference.json');inputs=ref.read(EXP/'SCORING_INPUTS.json')
    for p,h in manifest['files_sha256'].items():
        if ref.sha(p)!=h:raise ValueError('Frozen scoring input changed '+p)
    main_table={};ensemble_table={};paired={};ensemble_pairs={};errors={};statistics={}
    for split in SPLITS:
        ts,g,ge=load_daily(ref,'G',split)
        main_table[split]={'G':enrich(ref.tables(g,ts,SEEDS))}
        ensemble_table[split]={'G':enrich(ref.tables(ge,ts),True)}
        paired[split]={};ensemble_pairs[split]={}
        statistics['G/'+split]=dict(path=str(EXP/f'artifacts/statistics/G_{split}.npz'),sha256=ref.sha(EXP/f'artifacts/statistics/G_{split}.npz'))
        for arm in ('G-MAE','C-in'):
            newts,stats,ens=ref.collect_global(manifest['arms'][arm],split,paths,inputs['driver']);np.testing.assert_array_equal(newts,ts)
            saved=dump_daily(ref,arm,split,ts,stats,ens);statistics[arm+'/'+split]=dict(path=saved,sha256=ref.sha(saved))
            main_table[split][arm]=enrich(ref.tables(stats,ts,SEEDS));ensemble_table[split][arm]=enrich(ref.tables(ens,ts),True)
            paired[split][arm]={group:ref.paired(stats[group],g[group],ts) for group in ref.GROUPS}
            for group in ref.GROUPS:
                paired[split][arm][group]['lead_hours']={}
                for h in range(12):
                    a=ref.summary(stats[group][:,:,h]);b=ref.summary(g[group][:,:,h])
                    delta=[x['RMSE']-y['RMSE'] if x['RMSE'] is not None else None for x,y in zip(a['by_seed'],b['by_seed'])]
                    paired[split][arm][group]['lead_hours'][str(2*(h+1))]=dict(by_seed_delta=delta,mean_delta=float(np.mean(delta)) if delta[0] is not None else None)
            ensemble_pairs[split][arm]={}
            for group in ref.GROUPS:
                row=ref.paired(ens[group],ge[group],ts)
                ensemble_pairs[split][arm][group]={k:row[k] for k in ('mean_delta','ci95') if k in row}
                ensemble_pairs[split][arm][group].update(deterministic_predictions=True,block_days=7,draws=5000,rng=20260924)
            if split in ('validation','extension_evaluation'):
                errors.setdefault(split,{})[arm]=correlations(ref,manifest['arms'][arm],anchor['arms']['G'],split,paths)
        ref.write(EXP/'SCORING_PROGRESS.json',dict(completed_split=split,checked_at_utc=datetime.now(timezone.utc).isoformat()))
    result=dict(experiment='exp0016',status='complete_scoring',source_commit=manifest['source_commit'],
        frozen_predictions_sha256=ref.sha(EXP/'FROZEN_PREDICTIONS.json'),anchors=ref.read(EXP/'ANCHORS.json'),
        main_table=main_table,ensemble_table=ensemble_table,paired_vs_G=paired,ensemble_paired_vs_G5=ensemble_pairs,
        error_structure_vs_G5=errors,daily_statistics=statistics,dropped_training_windows=0,members_discarded=0,
        training=ref.read(EXP/'FROZEN_CANDIDATES.json')['formal_runs'],
        caveats=['2023–2025 has participated in development; it is not independent confirmation.',
                 'Single-model metrics are five-seed mean and sample SD; five-seed ensembles are deterministic predictions.',
                 'All members are delivered regardless of standalone accuracy. No weights or fusion choices were fitted.',
                 'Disturbance labels use retrospectively finalized Kp/Dst only for post-freeze diagnostics; neither new member consumes these drivers.'])
    ref.write(EXP/'result.json',result);print(json.dumps(dict(status='complete_scoring',arms=['G-MAE','C-in'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['anchors','score']);a=p.parse_args();ref=reference()
    inputs_and_anchors(ref) if a.action=='anchors' else score(ref)
