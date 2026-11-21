"""Descriptive fixed-member scores; no fusion, ranking, tuning, or elimination."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)


def load_reference(repo):
    path=Path(__file__).resolve().parent/'pooled.py'
    spec=importlib.util.spec_from_file_location('exp0015_reference_score',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def with_sds(value):
    if isinstance(value,dict):
        if 'by_seed' in value and len(value['by_seed'])>1:
            for key in ('RMSE','MAE','bias'):
                values=[x[key] for x in value['by_seed']]
                value[key+'_seed_sd']=float(np.std(values,ddof=1)) if all(x is not None for x in values) else None
        for v in value.values():with_sds(v)
    return value


def verify_split(records,split,manifest):
    for record in records:
        for key in ('predictions','origins'):
            p=record['splits'][split][key]
            if sha(p)!=manifest['files_sha256'][p]:raise ValueError('Frozen file changed: '+p)


def g_anchor(repo,split,ref,config):
    exp=repo/'experiments/exp0015';source=read(repo/'experiments/exp0013/FROZEN_PREDICTIONS.json');records=source['arms']['G']
    verify_split(records,split,source)
    for y in ref.YEARS[split]:
        row=config['data_files'][str(y)];assert sha(row['path'])==row['sha256']
    ts,stats,ensemble=ref.collect_global(records,split,config,config['driver']['path'])
    expected_g5={'validation':2.31396563758096,'original_evaluation':2.136731879698973,'extension_evaluation':4.49271502063866}[split]
    observed=ref.summary(ensemble['all'])['RMSE'];single=ref.summary(stats['all'])['RMSE']
    expected_single={'original_evaluation':2.233885478,'extension_evaluation':4.729342137}.get(split)
    row=dict(status='passed',G5=dict(expected=expected_g5,observed=observed,difference=observed-expected_g5),
        G_single_five_seed_mean=dict(expected=expected_single,observed=single,difference=None if expected_single is None else single-expected_single),n_windows=len(ts),tolerance=1e-6)
    if abs(observed-expected_g5)>=1e-6 or (expected_single is not None and abs(single-expected_single)>=1e-6):raise ValueError('G anchor failed')
    np.savez_compressed(exp/f'artifacts/G_{split}_daily.npz',origins=ts,**stats,**{'ensemble_'+k:v for k,v in ensemble.items()})
    return row,ts,stats,ensemble


def error_structure(records,g_records,split,config,ref):
    ts,arrays=ref.prediction_set(records,split);gt,gs=ref.prediction_set(g_records,split);np.testing.assert_array_equal(ts,gt)
    n=0;s=np.zeros(2);ss=np.zeros(2);cross=0.;raw={}
    for i,t in enumerate(ts):
        d=np.datetime64(int(t),'s').astype('datetime64[D]');year=int(str(d)[:4]);idx=int((d-np.datetime64(f'{year}-01-01')).astype(int))
        if year not in raw:raw[year]=np.load(config['data_files'][str(year)]['path'],mmap_mode='r',allow_pickle=False)
        y=(np.asarray(raw[year][idx],np.float32)/np.float32(10)).astype(np.float64);mask=np.isfinite(y)
        a=np.mean(np.stack([x[i].astype(np.float64) for x in arrays]),axis=0)[mask]-y[mask]
        b=np.mean(np.stack([x[i].astype(np.float64) for x in gs]),axis=0)[mask]-y[mask]
        n+=a.size;s+=[a.sum(),b.sum()];ss+=[np.dot(a,a),np.dot(b,b)];cross+=np.dot(a,b)
    mean=s/n;variance=ss/n-mean*mean;moment=cross/n;cov=moment-mean[0]*mean[1]
    return dict(Pearson=float(cov/np.sqrt(variance.prod())),error_means=mean.tolist(),error_variances=variance.tolist(),
        mean_squared_errors=(ss/n).tolist(),cross_moment_including_bias=float(moment),covariance=float(cov),n_valid=n,
        order=[records[0]['arm']+'5','G5'],grid='71x73, pooled all leads and windows')


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--validation-anchor',action='store_true');p.add_argument('--anchors-only',action='store_true');a=p.parse_args()
    repo=a.repo.resolve();exp=repo/'experiments/exp0015';ref=load_reference(repo);cfg=read(exp/'configs/infer_g_de_42.json')
    anchors=read(exp/'ANCHORS.json') if (exp/'ANCHORS.json').exists() else {}
    if a.validation_anchor:
        row,*_=g_anchor(repo,'validation',ref,cfg);anchors['validation']=row;write(exp/'ANCHORS.json',anchors);print(json.dumps(row));return
    candidate=read(exp/'FROZEN_CANDIDATES.json');assert candidate['all_training_complete'] and len(candidate['new_candidates'])==15
    stats={};ensembles={};origins={}
    for split in ref.SPLITS:
        row,ts,s,e=g_anchor(repo,split,ref,cfg);anchors[split]=row;write(exp/'ANCHORS.json',anchors)
        stats[(split,'G')]=s;ensembles[(split,'G')]=e;origins[split]=ts
    if a.anchors_only:print(json.dumps(dict(status='all_historical_anchors_passed')));return
    manifest=read(exp/'FROZEN_PREDICTIONS.json');assert manifest['candidate_freeze_sha256']==sha(exp/'FROZEN_CANDIDATES.json')
    arms=['G-DE','G-long','G-geo'];main_table={};ensemble_table={};pairs={};ensemble_pairs={};structure={}
    g_records=read(repo/'experiments/exp0013/FROZEN_PREDICTIONS.json')['arms']['G']
    for split in ref.SPLITS:
        for arm in arms:
            verify_split(manifest['arms'][arm],split,manifest)
            ts,s,e=ref.collect_global(manifest['arms'][arm],split,cfg,cfg['driver']['path']);np.testing.assert_array_equal(ts,origins[split]);stats[(split,arm)]=s;ensembles[(split,arm)]=e
        ts=origins[split]
        main_table[split]={arm:with_sds(ref.tables(stats[(split,arm)],ts,ref.SEEDS)) for arm in ['G']+arms}
        ensemble_table[split]={arm:ref.tables(ensembles[(split,arm)],ts) for arm in ['G']+arms}
        pairs[split]={group:{arm+'-G':ref.paired(stats[(split,arm)][group],stats[(split,'G')][group],ts) for arm in arms} for group in ref.GROUPS}
        ensemble_pairs[split]={arm+'5-G5':ref.paired(ensembles[(split,arm)]['all'],ensembles[(split,'G')]['all'],ts) for arm in arms}
        np.savez_compressed(exp/f'artifacts/{split}_daily.npz',origins=ts,**{arm+'_'+group:stats[(split,arm)][group] for arm in ['G']+arms for group in ref.GROUPS},
            **{arm+'5_'+group:ensembles[(split,arm)][group] for arm in ['G']+arms for group in ref.GROUPS})
        if split in ('validation','extension_evaluation'):
            structure[split]={arm:error_structure(manifest['arms'][arm],g_records,split,cfg,ref) for arm in arms}
        write(exp/'SCORING_PROGRESS.json',dict(completed_split=split,status='scoring'))
    result=dict(experiment='exp0015',status='scored',source_commit=candidate['source_commit'],scorer_sha256=sha(__file__),
        checks=read(exp/'CHECKS.json'),anchors=anchors,main_table=main_table,ensemble_table=ensemble_table,paired=pairs,ensemble_paired=ensemble_pairs,error_structure=structure,
        training_windows=dict(G_DE=2553,G_long=2527,G_geo=2553),dropped_train_windows=read(exp/'CHECK_B.json')['arms']['G-long']['dropped_train_windows'],
        all_members_retained=True,no_fusion_performed=True,candidate_freeze_sha256=sha(exp/'FROZEN_CANDIDATES.json'),prediction_freeze_sha256=sha(exp/'FROZEN_PREDICTIONS.json'),
        limitations=['Driving indices are retrospectively finalized products.','2023–2025 participated in development; not independent confirmation.',
            'Time bootstrap conditions on fixed trained models; seed variability is reported separately.','Single-member scores are descriptive and do not select or eliminate members.'])
    write(exp/'result.json',result);print(json.dumps(dict(status='scored',arms=arms)))


if __name__=='__main__':main()
