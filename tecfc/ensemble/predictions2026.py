"""Prepare, transport and compose the registered predictions without scoring."""
import argparse
import json
from pathlib import Path
import tarfile
import numpy as np
from tecfc.scoring.confirm_records import ROOT, EXP, SEEDS, sha, read, write, now
from tecfc.data.data2026 import Data
from tecfc.features.confirm_inputs import global_input, regional_history, core
from tecfc.ensemble.apply import mapping, remap, fuse


def spec(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)


def array(p):
    if isinstance(p,dict):assert sha(p['path'])==p['sha256'];p=p['path']
    return np.load(p,mmap_mode='r',allow_pickle=False)


def save(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.save(p,a);return spec(p)


def returned(job):
    meta=EXP/'artifacts/ops';r=read(meta/(job+'.recovered.json'))
    assert r['status']=='complete' and r['hash_verified']
    assert (meta/(job+'.released.json')).exists() and (meta/(job+'.watchdog_unregistered.json')).exists()
    dest=Path(r['local_root']);manifest=read(meta/(job+'.recovery_manifest.json'))
    for name,s in manifest['files'].items():assert sha(dest/name)==s['sha256']
    assert read(dest/'outputs/COMPLETE.json')['status']=='passed'
    return dest/'outputs'


class Drivers:
    def __init__(self,data):
        self.data=data.drivers();c=read(ROOT/'experiments/exp0015/DRIVER_SCALER.json')
        self.mean=np.array([c['channels'][k]['mean'] for k in core.f15.DRIVER_ORDER],np.float32)
        self.std=np.array([c['channels'][k]['standard_deviation'] for k in core.f15.DRIVER_ORDER],np.float32)
    def standardized(self,times):
        ix=np.searchsorted(self.data['time'],times);np.testing.assert_array_equal(self.data['time'][ix],times)
        out=np.empty((len(times),5112,5),np.float32)
        for j,k in enumerate(core.f15.DRIVER_ORDER[:-1]):out[:,:,j]=self.data[k][ix,None]
        out[:,:,4]=np.clip(core.f15.signed_cos_sza(times,core.f15.LON,core.f15.LAT),0.,1.)
        assert np.isfinite(out).all()
        return (out.reshape(len(times),71,72,5)-self.mean)/self.std


def prepare(domain):
    job='predict_'+domain
    frozen=read(ROOT/'experiments/exp0025/FROZEN_SYSTEMS.json');scope=read(ROOT/'experiments/exp0025/SCOPE_2026.json')
    assert not (EXP/'FROZEN_PREDICTIONS_2026.json').exists(),'Never regenerate committed predictions'
    for letter in 'ABCDEFG':assert read(ROOT/f'experiments/exp0025/checks/{letter}.json')['status']=='passed'
    for row in frozen['files']:assert sha(row['path'])==row['sha256'],'Frozen system changed: '+row['path']
    reference=read(ROOT/'experiments/exp0025/CHECK_C_INPUTS.json')
    dest=EXP/f'artifacts/{job}_bundle';dest.mkdir(parents=True,exist_ok=False);files={};models=[];origins={};scoring={}
    def add(path,name,expected=None):
        r=spec(path)
        if expected is not None:assert r['sha256']==expected
        files[name]=r
    def keep(name,value):
        r=save(dest/name,value);files[name]=r;return name
    for name,r in reference['files'].items():
        if name.startswith(('repo/','assets/')) or (domain=='regional' and name.startswith('legacy/')):add(r['path'],name,r['sha256'])
    data=Data()
    if domain=='global':
        cfg=read(ROOT/'experiments/exp0018/configs/inference.json')
        with np.load(Path(cfg['asset_root'])/'prepared/train_climatology_v1.npz') as z:climate=z['climatology'].copy()
        drivers=Drivers(data);input_names={}
        for family in sorted({m['family'] for m in frozen['models'] if m['domain']=='global'}):
            rows=scope['plans']['S1' if family=='G-DE' else 'global']['included'];dates=[r['date'] for r in rows]
            if not dates:continue
            key='G' if family in ['G','PI','EMP'] else 'G-plus' if family in ['G-plus-M','G-plus-S'] else family
            if key not in input_names:
                xx=np.stack([global_input(day,family,data.global_day,climate,drivers) for day in dates])
                input_names[key]=keep('inputs/'+key+'.npy',xx)
            origins[family]=save(dest/f'origins_{family}.npy',np.array([r['origin'] for r in rows],np.int64))
            for member in [m for m in frozen['models'] if m['domain']=='global' and m['family']==family]:
                ident=f'global_{family}_{member["seed"]}';cp='checkpoints/'+ident+'.pt';add(member['checkpoint'],cp,member['checkpoint_sha256'])
                models.append(dict(id=ident,domain='global',family=family,seed=member['seed'],checkpoint=cp,input=input_names[key],batch_size=32))
            print(json.dumps(dict(prepared_family=family,windows=len(dates))),flush=True)
        rows=scope['plans']['global']['included']
        p=np.stack([data.global_day(str(np.datetime64(r['date'])-np.timedelta64(1,'D')),unique=False) for r in rows])
        scoring['persistence']=save(dest/'persistence.npy',p)
    else:
        global_manifest=read(ROOT/'experiments/exp0025/GLOBAL_PREDICTIONS_READY.json')
        s2=array(global_manifest['methods']['S2']);g5=array(global_manifest['methods']['G5']);gt=array(global_manifest['origins']);lookup={int(t):i for i,t in enumerate(gt)}
        for region in ['china','japan']:
            rows=scope['plans'][region]['included'];ts=np.array([r['origin'] for r in rows],np.int64)
            origins[region]=save(dest/f'origins_{region}.npy',ts)
            raw=[data.regional_day(region,r['history']) for r in rows];target=[data.regional_day(region,r['date']) for r in rows]
            stats={
                'R1':ROOT/f'experiments/exp0021/artifacts/prepared/{region}/statistics.npz',
                'R2':ROOT/f'experiments/exp0023/artifacts/prepared/stage2/{region}/statistics.npz',
                'common':ROOT/f'experiments/exp0023/artifacts/prepared/stage1/{region}/statistics.npz'}
            histories={};scoring[region]={}
            for recipe,path in stats.items():
                with np.load(path) as z:
                    fallback=z['fallback'].copy();scale=float(z['scale']);lat=z['lat'];lon=z['lon']
                n=len(lat)*len(lon)
                history=np.stack([regional_history(r['tec'][:89].reshape(89,n),r['valid'][:89].reshape(89,n),fallback) for r in raw])
                histories[recipe]=(keep(f'inputs/{region}_{recipe}.npy',history),scale)
                scoring[region]['history_'+recipe]=files[histories[recipe][0]]
            mp=mapping(lat,lon)
            anchors={name:np.stack([remap(a[lookup[int(t)]].reshape(12,5183),mp) for t in ts]) for name,a in [('S2',s2),('G5',g5)]}
            anchor_name=keep(f'inputs/{region}_anchor.npy',anchors['S2'])
            scoring[region]['anchor']=files[anchor_name]
            scoring[region]['mapped_G5']=save(dest/f'{region}_mapped_G5.npy',anchors['G5'])
            scoring[region]['target']=save(dest/f'{region}_target.npy',np.stack([r['tec'][::8].reshape(12,n) for r in target]))
            scoring[region]['valid']=save(dest/f'{region}_valid.npy',np.stack([r['valid'][::8].reshape(12,n) for r in target]))
            scoring[region]['persistence']=save(dest/f'{region}_persistence.npy',array(scoring[region]['history_common'])[:,::8])
            for member in [m for m in frozen['models'] if m['domain']==region]:
                ident=f'{region}_{member["family"]}_{member["seed"]}';cp='checkpoints/'+ident+'.pt';add(member['checkpoint'],cp,member['checkpoint_sha256'])
                inp,scale=histories['R2' if member['family']=='R2' else 'R1']
                models.append(dict(id=ident,domain=region,family=member['family'],seed=member['seed'],checkpoint=cp,input=inp,scale=scale,anchor=anchor_name,batch_size=4))
            print(json.dumps(dict(prepared_region=region,windows=len(rows))),flush=True)
    write(dest/'JOB.json',dict(stage='prediction',models=models,files=files))
    write(EXP/f'INPUTS_{job}.json',dict(status='frozen',stage=job,models=models,files=files,origins=origins,scoring=scoring,
          scope_sha256=sha(EXP/'SCOPE_2026.json'),inference_source_sha256=sha(ROOT/'src/confirm2026/infer.py'),no_error_metrics=True))
    archive=EXP/f'artifacts/{job}.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,r in sorted(files.items()):tar.add(r['path'],arcname=name,recursive=False)
        tar.add(dest/'JOB.json',arcname='JOB.json',recursive=False)
    write(EXP/f'PAYLOAD_{job}.json',spec(archive));print(json.dumps(dict(job=job,models=len(models),payload_bytes=archive.stat().st_size)),flush=True)


def compose_global():
    output=returned('predict_global');inputs=read(ROOT/'experiments/exp0025/INPUTS_predict_global.json')
    dest=EXP/'artifacts/predictions/global';dest.mkdir(parents=True,exist_ok=False)
    scope=read(ROOT/'experiments/exp0025/SCOPE_2026.json');ts=array(inputs['origins']['G']);s1ts=array(inputs['origins']['G-DE']) if 'G-DE' in inputs['origins'] else np.array([],np.int64)
    members={m['id']:spec(output/(m['id']+'.npy')) for m in inputs['models']}
    arrays={k:array(v) for k,v in members.items()};means={};methods={}
    families=sorted({m['family'] for m in inputs['models']})
    for family in families:
        means[family]=np.mean(np.stack([np.asarray(arrays[f'global_{family}_{s}'],np.float64) for s in SEEDS]),axis=0)
        if family in ['G','G-plus-M','G-geo','PI','EMP','C','C-in','C-plus']:
            methods[family+'5']=save(dest/(family+'5.npy'),means[family])
    definitions={s:read(ROOT/f'experiments/exp0019/FROZEN_FUSION_{s}.json') for s in ['S2','S1']}
    s2=[]
    for i in range(len(ts)):s2.append(fuse({f:a[i] for f,a in means.items() if f!='G-DE'},definitions['S2']))
    methods['S2']=save(dest/'S2.npy',np.stack(s2));methods['persistence']=inputs['scoring']['persistence']
    if len(s1ts):
        lookup={int(t):i for i,t in enumerate(ts)};s1=[]
        for i,t in enumerate(s1ts):
            values={f:a[i if f=='G-DE' else lookup[int(t)]] for f,a in means.items()};s1.append(fuse(values,definitions['S1']))
        methods['S1']=save(dest/'S1.npy',np.stack(s1))
    record=dict(status='predictions_ready_not_scored',origins=inputs['origins']['G'],S1_origins=inputs['origins'].get('G-DE'),methods=methods,members=members,
                scope_sha256=sha(EXP/'SCOPE_2026.json'),no_error_metrics=True)
    write(EXP/'GLOBAL_PREDICTIONS_READY.json',record)


def compose_regional():
    output=returned('predict_regional');inputs=read(ROOT/'experiments/exp0025/INPUTS_predict_regional.json');frozen=read(ROOT/'experiments/exp0025/FROZEN_SYSTEMS.json')
    dest=EXP/'artifacts/predictions';regions={}
    for region in ['china','japan']:
        data=inputs['scoring'][region];anchor=array(data['anchor']);methods=dict(anchor=data['anchor'],mapped_G5=data['mapped_G5'],persistence=data['persistence']);members={};raw_members={}
        for family,name,rule in [('R2','final','main'),('R1','exp0021_regional','reference'),('R1-B','R1-B_G0','diagnostic')]:
            specs=[spec(output/f'{region}_{family}_{s}.npy') for s in SEEDS];arrays=[array(r) for r in specs]
            gate=frozen['regional_systems'][region][rule];h=array(data['history_R2' if family=='R2' else 'history_R1']);mask=None if gate['threshold'] is None else h[:,::8]>=gate['threshold']
            mean=np.mean(arrays,axis=0)
            if mask is not None:mean=np.where(mask,anchor,mean)
            methods[name]=save(dest/region/(name+'.npy'),mean)
            for s,a,r in zip(SEEDS,arrays,specs):
                raw_members[family+'_seed'+str(s)]=r
                key=name+'_seed'+str(s)
                members[key]=r if mask is None else save(dest/region/(key+'.npy'),np.where(mask,anchor,a))
        regions[region]=dict(origins=inputs['origins'][region],methods=methods,members=members,raw_members=raw_members,scoring_arrays=data,
                             gates=frozen['regional_systems'][region])
    write(EXP/'REGIONAL_PREDICTIONS_READY.json',dict(status='predictions_ready_not_scored',regions=regions,no_error_metrics=True))


def freeze_predictions():
    assert not (EXP/'FROZEN_PREDICTIONS_2026.json').exists()
    g=read(ROOT/'experiments/exp0025/GLOBAL_PREDICTIONS_READY.json');r=read(ROOT/'experiments/exp0025/REGIONAL_PREDICTIONS_READY.json')
    scope=read(ROOT/'experiments/exp0025/SCOPE_2026.json')
    np.testing.assert_array_equal(array(g['origins']),[x['origin'] for x in scope['plans']['global']['included']])
    for name,s in g['methods'].items():
        count=len(scope['plans']['S1' if name=='S1' else 'global']['included'])
        assert array(s).shape==(count,12,71,73)
    for name,s in g['members'].items():
        count=len(scope['plans']['S1' if name.startswith('global_G-DE_') else 'global']['included'])
        assert array(s).shape==(count,12,71,73)
    if g['S1_origins'] is not None:np.testing.assert_array_equal(array(g['S1_origins']),[x['origin'] for x in scope['plans']['S1']['included']])
    for region,row in r['regions'].items():
        np.testing.assert_array_equal(array(row['origins']),[x['origin'] for x in scope['plans'][region]['included']])
        target=array(row['scoring_arrays']['target']);valid=array(row['scoring_arrays']['valid'])
        assert target.shape==valid.shape and target.shape[:2]==(len(array(row['origins'])),12)
        assert valid.dtype==bool and np.isfinite(target[valid]).all()
        for s in list(row['methods'].values())+list(row['members'].values())+list(row['raw_members'].values()):assert array(s).shape==target.shape
    files={}
    def visit(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:
                a=array(x);files[x['path']]=dict(**x,shape=list(a.shape),dtype=str(a.dtype));assert np.isfinite(a).all() or 'target' in Path(x['path']).name
            else:
                for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(g);visit(r)
    write(EXP/'FROZEN_PREDICTIONS_2026.json',dict(status='frozen',at_utc=now(),global_=g,regions=r['regions'],files=files,
          systems_sha256=sha(EXP/'FROZEN_SYSTEMS.json'),scope_sha256=sha(EXP/'SCOPE_2026.json'),all_member_deterministic_repeats_passed=True,
          error_metrics_computed=False,regeneration_prohibited=True,
          prediction_packaging_source_sha256=sha(__file__),
          deterministic_recovery_manifests={job:spec(EXP/'artifacts/ops'/(job+'.recovery_manifest.json')) for job in ['predict_global','predict_regional']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare_global','prepare_regional','compose_global','compose_regional','freeze']);a=p.parse_args()
    if a.stage.startswith('prepare_'):prepare(a.stage.removeprefix('prepare_'))
    else:{'compose_global':compose_global,'compose_regional':compose_regional,'freeze':freeze_predictions}[a.stage]()
