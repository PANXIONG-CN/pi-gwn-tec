"""One fixed exp0021 job. The R1 path calls exp0020's unchanged numerical code."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import inherited_formal as old
from train import sha, write

SEEDS = [42,1024,2025,3407,5000]

def setup(root, region, arm, seed, out, stage):
    objects = old.setup(root, region, 'R1', seed, out)
    device, tr, va, model, opt, scheduler, loader = objects
    rec = json.loads((root/'TRAINING_STATISTICS.json').read_text())['regions'][region]
    model.bin_boundaries = torch.tensor(rec['edges'][1:-1], device=device, dtype=torch.float32)
    model.bin_weights = torch.tensor(rec['weights'], device=device, dtype=torch.float32)
    cfg = json.loads((out/'config.json').read_text())
    cfg.update(arm=arm, stage=stage, epochs=3 if stage=='a' else 1 if stage=='ce' else 200,
        experiment='exp0021', source_sha256=sha(__file__), inherited_source_sha256=sha(old.__file__),
        loss='masked_TECU_MAE' if arm=='R1' else 'level_balanced_weighted_masked_TECU_MAE',
        weights_record_sha256=sha(root/'TRAINING_STATISTICS.json'), anchor='exp0017' if stage=='a' else 'exp0019/S2')
    write(out/'config.json',cfg)
    return objects

def step(model, batch, optimizer, device, arm):
    if arm=='R1': return old.step(model,batch,optimizer,device)
    x,y,v,a,_=batch; x,y,v,a=x.to(device),y.to(device),v.to(device),a.to(device)
    optimizer.zero_grad(set_to_none=True); out=model(x,a)
    assert torch.isfinite(out).all() and bool(v.any())
    target=y.float()[v]
    weight=model.bin_weights[torch.bucketize(target,model.bin_boundaries,right=True)]
    loss=((out.float()[v]-target).abs()*weight).mean()
    if model.region=='japan': model.legacy.assign_gradients(loss,None,list(model.parameters()),0.)
    else: loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); optimizer.step()
    return float(loss.detach()),int(v.sum())

def epoch(model,loader,opt,device,arm):
    model.train(); total=count=0
    for batch in loader:
        loss,n=step(model,batch,opt,device,arm)
        assert np.isfinite(loss); total+=loss*n; count+=n
    return total/count

def complete(out, **record):
    write(out/'complete.json',dict(status='complete',**record,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def equivalence(root, job, region):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False)
    device,tr,va,model,opt,scheduler,loader=setup(root,region,'R1',42,out,'a')
    refs=json.loads((root/'reference_epochs.json').read_text())[region]; results=[]
    with (out/'epochs.jsonl').open('x') as log:
        for e in range(1,4):
            loss=epoch(model,loader,opt,device,'R1'); val=old.evaluate(model,va,device); scheduler.step(val['RMSE'])
            row=dict(epoch=e,training_MAE=loss,validation=val,reference_RMSE=refs[e-1],absolute_difference=abs(val['RMSE']-refs[e-1]))
            results.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    passed=all(r['absolute_difference']<1e-6 for r in results)
    write(out/'CHECK_A.json',dict(status='passed' if passed else 'failed',region=region,epochs=results,
        absolute_tolerance_exclusive=1e-6,anchor='exp0020/exp0017',diagnostic_only=True))
    complete(out,stage='a',checks_passed=passed)
    return 0 if passed else 2

def checks(root,job,region):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False); c={}; paths={}; e={}
    for arm in ['R1','R1-B']:
        sub=out/arm;sub.mkdir()
        device,tr,va,model,opt,scheduler,loader=setup(root,region,arm,42,sub,'ce')
        initial=old.evaluate(model,va,device,compare_anchor=True)
        passed=initial['max_abs_prediction_minus_anchor']==0 and initial['prediction_minus_anchor_RMS']==0
        c[arm]=dict(status='passed' if passed else 'failed',validation=initial,pointwise_exact=passed)
        write(out/'CHECK_C.json',dict(status='passed' if all(x['status']=='passed' for x in c.values()) else 'failed',region=region,arms=c))
        assert passed,'Check C failed'
        model.legacy.seed_all(42)
        loss=epoch(model,loader,opt,device,arm)
        path=out/f'{arm}_one_epoch_validation.npy'; val=old.evaluate(model,va,device,path)
        paths[arm]=path;e[arm]=dict(loss=loss,validation=val)
        del model,opt,scheduler,loader;torch.cuda.empty_cache()
    a=np.load(paths['R1'],mmap_mode='r'); b=np.load(paths['R1-B'],mmap_mode='r')
    sse=0.;n=0
    for i in range(len(a)):
        d=b[i]-a[i];sse+=np.square(d).sum();n+=d.size
    rms=float(np.sqrt(sse/n));passed=rms>0
    write(out/'CHECK_E.json',dict(status='passed' if passed else 'failed',region=region,seed=42,epochs=1,
        prediction_difference_RMS=rms,arms=e,diagnostic_weights_discarded=True))
    complete(out,stage='ce',checks_passed=passed)
    return 0 if passed else 2

def train(root,job,region,arm,seed):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False)
    device,tr,va,model,opt,scheduler,loader=setup(root,region,arm,seed,out,'train')
    best=np.inf;selected=None;started=time.monotonic()
    with (out/'epochs.jsonl').open('x') as log:
        for e in range(1,201):
            lr=opt.param_groups[0]['lr']; loss=epoch(model,loader,opt,device,arm)
            val=old.evaluate(model,va,device);score=val['RMSE'];scheduler.step(score)
            if score<best:
                best=score;selected=e
                torch.save(dict(state_dict=model.state_dict(),epoch=e,validation_RMSE=best),out/'best.pt')
            row=dict(epoch=e,training_loss=loss,validation=val,selected_epoch=selected,lr=lr,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    saved=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict'],strict=True)
    val=old.evaluate(model,va,device,out/'validation_predictions.npy');assert abs(val['RMSE']-best)<1e-9
    np.save(out/'validation_origins.npy',va.origins)
    complete(out,region=region,arm=arm,seed=seed,epochs=200,selected_epoch=selected,validation=val,evaluation_accessed=False)
    return 0

def predict(root,job,region):
    frozen=json.loads((root/'FROZEN_CHECKPOINTS.json').read_text());assert frozen['all_twenty_complete']
    regional=json.loads((root/'FROZEN_REGIONAL.json').read_text());assert regional['status']=='frozen'
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False);ds=old.Windows(root,region,'evaluation');device=torch.device('cuda');rows=[]
    for candidate in frozen['models']:
        if candidate['region']!=region:continue
        arm,seed=candidate['arm'],candidate['seed'];checkpoint=root/'checkpoints'/f'{arm}_{seed}.pt'
        assert sha(checkpoint)==candidate['checkpoint_sha256']
        model=old.Model(root,region,'R1',seed,device,ds.scale)
        state=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(state['state_dict'],strict=True)
        dest=np.lib.format.open_memmap(out/f'{arm}_{seed}_predictions.npy',mode='w+',dtype=np.float64,shape=(len(ds),12,ds.n));model.eval()
        with torch.no_grad():
            for x,y,v,a,ii in DataLoader(ds,batch_size=4,shuffle=False):
                p=model(x.to(device),a.to(device)).cpu().numpy()[...,0];assert np.isfinite(p).all();dest[ii.numpy()]=p
        dest.flush();rows.append(dict(arm=arm,seed=seed,selected_epoch=state['epoch']));del model,dest;torch.cuda.empty_cache()
    assert len(rows)==10;np.save(out/'origins.npy',ds.origins)
    complete(out,region=region,stage='predict',models=rows,scoring_performed=False)
    return 0

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--region',choices=['china','japan'],required=True);p.add_argument('--stage',choices=['a','ce','train','predict'],required=True)
    p.add_argument('--arm',choices=['R1','R1-B'],default='R1');p.add_argument('--seed',type=int,choices=SEEDS,default=42);a=p.parse_args()
    if a.stage=='a': code=equivalence(a.root,a.job,a.region)
    elif a.stage=='ce': code=checks(a.root,a.job,a.region)
    elif a.stage=='train':code=train(a.root,a.job,a.region,a.arm,a.seed)
    else:code=predict(a.root,a.job,a.region)
    sys.exit(code)
