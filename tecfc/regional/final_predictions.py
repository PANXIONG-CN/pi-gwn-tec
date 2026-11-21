"""Hash-bound means and gates; no metric computation."""
import argparse
import numpy as np
from tecfc.regional.replay_common import *

def array(record):
    assert sha(record['path'])==record['sha256']
    return np.load(record['path'],mmap_mode='r',allow_pickle=False)

def freeze(period):
    first=read(EXP/'FROZEN_CHECKPOINTS.json');split='validation' if period=='2022' else 'evaluation'
    final=None if period=='2022' else read(EXP/'FROZEN_REGIONAL_V2.json')
    selected=None if period=='2022' else read(EXP/'SELECTION_2022.json')
    historical=read(OLD/'FROZEN_PREDICTIONS.json');stats1=read(EXP/'TRAINING_STATISTICS.json')['regions']
    oldstats=read(OLD/'TRAINING_STATISTICS.json')['regions'];regions={}
    for region in REGIONS:
        folder=PREP/'stage1'/region/split;origins=np.load(folder/'origins.npy')
        previous=historical['regions'][region][period]
        np.testing.assert_array_equal(array(previous['origins']),origins)
        for key in ['target','valid']:
            a=array(previous['scoring_arrays'][key]);b=np.load(folder/(key+'.npy'),mmap_mode='r')
            assert a.shape==b.shape
            for i in range(len(a)):np.testing.assert_array_equal(a[i],b[i])
        dest=EXP/'artifacts/predictions'/region/period;dest.mkdir(parents=True,exist_ok=False)
        anchor=np.load(folder/'anchor.npy',mmap_mode='r');history=np.load(folder/'history.npy',mmap_mode='r')
        members={};methods={}
        for arm in ARMS:
            seed_specs=[]
            for seed in SEEDS:
                model=next(m for m in first['models'] if m['region']==region and m['arm']==arm and m['seed']==seed)
                if period=='2022':record=model['validation_predictions'];ts=array(model['validation_origins'])
                else:
                    record=spec(output(region+'_predict')/f'stage1_{arm}_{seed}.npy');ts=np.load(output(region+'_predict')/'stage1_origins.npy')
                np.testing.assert_array_equal(ts,origins);assert array(record).shape==anchor.shape
                seed_specs.append(record);members[f'stage1_{arm}_{seed}']=record
            arrays=[array(rec) for rec in seed_specs]
            for gate in ['G0','G1']:
                path=dest/f'{arm}_{gate}.npy';mean=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=anchor.shape)
                threshold=None if gate=='G0' else stats1[region]['tau99']
                for i in range(len(origins)):
                    p=np.mean([a[i] for a in arrays],axis=0)
                    mean[i]=p if threshold is None else np.where(history[i,::8]>=threshold,anchor[i],p)
                mean.flush();methods[f'{arm}/{gate}']=dict(prediction=spec(path),seeds=seed_specs,gate=gate,threshold=threshold,
                    gate_history=spec(folder/'history.npy'),role='candidate',source='exp0023_stage1')
        def old_method(candidate):
            arm,gate=candidate.split('/')
            return dict(prediction=previous['candidates'][candidate],seeds=[previous['predictions'][f'{arm}_{seed}'] for seed in SEEDS],gate=gate,
                threshold=oldstats[region]['tau99'] if gate=='G1' else None,gate_history=previous['scoring_arrays']['history'],source='exp0021',role='baseline')
        if period=='2022':
            for candidate in ['R1/G0','R1/G1']:methods['exp0021_'+candidate]=old_method(candidate)
        else:
            chosen=selected['regions'][region]['candidate']
            methods['stage1_selected']=dict(methods[chosen],role='第一阶段所选，未并入2022，仅作参照')
            for candidate in CANDIDATES:methods[candidate]['role']='第一阶段所选' if candidate==chosen else '未选中，仅供诊断'
            methods['exp0021_unified']=old_method('R1/G1')
            methods['exp0021_regional']=old_method('R1/G1' if region=='china' else 'R1/G0')
            methods['exp0021_R1-B/G0']=old_method('R1-B/G0')
            choice=final['regions'][region]
            if choice['replace']:
                seed_specs=[spec(output(region+'_predict')/f'stage2_{choice["arm"]}_{seed}.npy') for seed in SEEDS]
                np.testing.assert_array_equal(np.load(output(region+'_predict')/'stage2_origins.npy'),origins)
                arrays=[array(rec) for rec in seed_specs];h=np.load(PREP/'stage2'/region/'evaluation/history.npy',mmap_mode='r')
                for a in arrays:assert a.shape==anchor.shape
                a2=np.load(PREP/'stage2'/region/'evaluation/anchor.npy',mmap_mode='r')
                path=dest/'final.npy';mean=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=anchor.shape)
                for i in range(len(origins)):
                    np.testing.assert_array_equal(anchor[i],a2[i]);p=np.mean([a[i] for a in arrays],axis=0)
                    mean[i]=p if choice['gate']=='G0' else np.where(h[i,::8]>=choice['threshold'],anchor[i],p)
                mean.flush();methods['final']=dict(prediction=spec(path),seeds=seed_specs,gate=choice['gate'],threshold=choice['threshold'],
                    gate_history=spec(PREP/'stage2'/region/'evaluation/history.npy'),source='exp0023_stage2',role='定稿系统，正式结果')
            else:methods['final']=dict(methods['exp0021_regional'],role='定稿系统，保留exp0021')
        for name,rec in methods.items():
            assert array(rec['prediction']).shape==anchor.shape
            for seed_spec in rec['seeds']:assert array(seed_spec).shape==anchor.shape
        regions[region]={period:dict(origins=spec(folder/'origins.npy'),scoring_arrays={key:spec(folder/(key+'.npy')) for key in ['target','valid','history','anchor']},
            methods=methods,members=members)}
    record=dict(status='frozen',at_utc=now(),period=period,regions=regions,source_sha256=sha(__file__))
    if period=='2022':write(EXP/'FROZEN_VALIDATION_PREDICTIONS.json',record)
    else:
        previous=read(EXP/'FROZEN_VALIDATION_PREDICTIONS.json')
        for region in REGIONS:record['regions'][region].update(previous['regions'][region])
        record['regional_system_sha256']=sha(EXP/'FROZEN_REGIONAL_V2.json');write(EXP/'FROZEN_PREDICTIONS.json',record)
    print(json.dumps(dict(status='predictions_frozen',period=period)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('period',choices=['2022','2023-2025']);freeze(p.parse_args().period)
