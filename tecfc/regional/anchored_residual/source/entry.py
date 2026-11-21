"""Fixed stage-1 learning and validation-free stage-2 LR replay."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import inherited_formal as old
import inherited_entry as previous
from train import sha, write

SEEDS=[42,1024,2025,3407,5000]

class Windows(old.Windows):
    def __init__(self,root,region,split):
        self.root=root/'prepared'/region;self.meta=json.loads((self.root/'metadata.json').read_text())
        self.scale=self.meta['scale'];self.n=self.meta['nodes'];self.split=split
        assert split!='validation' or self.meta['validation_years']==[2022]
        self.arrays={k:np.load(self.root/split/(k+'.npy'),mmap_mode='r') for k in ['history','target','valid','anchor','origins']}
        self.origins=self.arrays['origins'];assert len(self.origins)==self.meta['counts'][split]
        years=self.origins.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
        key={'train':'training_years','validation':'validation_years','evaluation':'evaluation_years'}[split]
        assert set(years)==set(self.meta[key]) and np.max(years)<=2025

old.Windows=Windows

def setup(root,region,arm,seed,out,stage):
    replay=stage in ['f','refit']
    if not replay:
        objects=previous.setup(root,region,'R1' if arm=='R2' else 'R1-B',seed,out,'train')
    else:
        device=torch.device('cuda')
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        tr=Windows(root,region,'train');va=Windows(root,region,'validation') if stage=='f' else None
        if stage=='refit':
            assert tr.meta['validation_years']==[] and not (root/'prepared'/region/'validation').exists()
            assert tr.meta['training_years']==[2014,2015,2019,2020,2021,2022]
        model=old.Model(root,region,'R1',seed,device,tr.scale)
        if seed==42:
            assert model.r0_initial_tensor_sha256==json.loads((root/'legacy_input_identity.json').read_text())['initial_tensor_sha256']
        opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=0.)
        model.legacy.seed_all(seed)
        loader=DataLoader(tr,batch_size=32,shuffle=True,drop_last=False,generator=torch.Generator().manual_seed(seed))
        rec=json.loads((root/'TRAINING_STATISTICS.json').read_text())['regions'][region]
        model.bin_boundaries=torch.tensor(rec['edges'][1:-1],device=device,dtype=torch.float32)
        model.bin_weights=torch.tensor(rec['weights'],device=device,dtype=torch.float32)
        objects=(device,tr,va,model,opt,None,loader)
    device,tr,va,model,opt,scheduler,loader=objects
    cfg=dict(experiment='exp0023',region=region,arm=arm,seed=seed,stage=stage,
        training_years=tr.meta['training_years'],validation_years=[] if va is None else [2022],
        train_windows=len(tr),validation_windows=0 if va is None else len(va),scale=tr.scale,
        history_channels=['TEC'],anchor_input=True,driver_input=False,batch_size=32,evaluation_batch_size=4,
        initial_tensor_sha256=model.initial_tensor_sha256,r0_initial_tensor_sha256=model.r0_initial_tensor_sha256,
        initial_source=model.initial_source,source_sha256=sha(__file__),inherited_entry_sha256=sha(previous.__file__),
        inherited_formal_sha256=sha(old.__file__),statistics_sha256=sha(root/'TRAINING_STATISTICS.json'),
        loss='masked_TECU_MAE' if arm=='R2' else 'level_balanced_weighted_masked_TECU_MAE',
        lr_mode='same_region_arm_seed_stage1_epoch_start_replay' if replay else 'inherited_ReduceLROnPlateau_2022_RMSE',
        validation_metrics_during_training=va is not None,optimizer_steps_per_epoch=len(loader),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    write(out/'config.json',cfg)
    return objects

def finish(out,**record):
    write(out/'complete.json',dict(status='complete',**record,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def epoch(model,loader,opt,device,arm):
    return previous.epoch(model,loader,opt,device,'R1' if arm=='R2' else 'R1-B')

def train(root,job,region,arm,seed,stage):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False)
    device,tr,va,model,opt,scheduler,loader=setup(root,region,arm,seed,out,stage)
    refs=None;sequence=None
    if stage=='a':
        refs=json.loads((root/'reference_epochs.json').read_text())[region];epochs=3
        assert tr.meta['training_years']==[2015,2019,2020,2021]
    elif stage in ['f','refit']:
        record=json.loads((root/'LR_REPLAY.json').read_text());selected=record['regions'][region]
        assert selected['arm']==arm;sequence=selected['seeds'][str(seed)]['learning_rates'];epochs=selected['fixed_epochs']
        assert len(sequence)>=epochs and epochs>=1
        if stage=='f':refs=selected['seeds']['42']['epoch_records'];assert seed==42
    else: epochs=200
    write(out/'budget.json',dict(epochs=epochs,optimizer_steps=epochs*len(loader),validation_metrics_computed=stage!='refit',diagnostic_only=stage in ['a','f']))
    best=np.inf;selected_epoch=None;started=time.monotonic();comparisons=[]
    with (out/'epochs.jsonl').open('x') as log:
        for e in range(1,epochs+1):
            if sequence is not None:
                values=sequence[e-1];assert len(values)==len(opt.param_groups)
                for group,value in zip(opt.param_groups,values):group['lr']=value
            lrs=[float(group['lr']) for group in opt.param_groups]
            loss=epoch(model,loader,opt,device,arm)
            # Formal refit never instantiates a validation dataset, evaluates a
            # validation batch, creates a scheduler, or selects a checkpoint.
            val=None if stage=='refit' else old.evaluate(model,va,device)
            if scheduler is not None:scheduler.step(val['RMSE'])
            if stage=='train' and val['RMSE']<best:
                best=val['RMSE'];selected_epoch=e
                torch.save(dict(state_dict=model.state_dict(),epoch=e,validation_RMSE=best),out/'best.pt')
            row=dict(epoch=e,training_loss=loss,validation=val,lr=lrs[0],optimizer_group_lrs=lrs,
                selected_epoch=selected_epoch,optimizer_steps=e*len(loader),elapsed_seconds=time.monotonic()-started)
            if stage=='a':row['reference_RMSE']=refs[e-1];row['RMSE_absolute_difference']=abs(val['RMSE']-refs[e-1])
            if stage=='f':
                row['loss_absolute_difference']=abs(loss-refs[e-1]['training_loss'])
                row['RMSE_absolute_difference']=abs(val['RMSE']-refs[e-1]['validation']['RMSE'])
            if stage in ['a','f']:comparisons.append(row)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    if stage in ['a','f']:
        passed=all(r['RMSE_absolute_difference']<1e-6 and (stage=='a' or r['loss_absolute_difference']<1e-6) for r in comparisons)
        write(out/f'CHECK_{stage.upper()}.json',dict(status='passed' if passed else 'failed',region=region,arm=arm,seed=seed,
            epochs=epochs,rows=comparisons,absolute_tolerance_exclusive=1e-6,diagnostic_weights_excluded=True))
        finish(out,stage=stage,checks_passed=passed,diagnostic_only=True)
        return 0 if passed else 2
    if stage=='train':
        saved=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict'],strict=True)
        val=old.evaluate(model,va,device,out/'validation_predictions.npy');assert abs(val['RMSE']-best)<1e-9
        np.save(out/'validation_origins.npy',va.origins)
    else:
        assert va is None and scheduler is None and selected_epoch is None
        torch.save(dict(state_dict=model.state_dict(),epoch=epochs,validation_RMSE=None,selection='fixed_final_epoch'),out/'final.pt')
    finish(out,stage=stage,region=region,arm=arm,seed=seed,epochs=epochs,selected_epoch=selected_epoch,
        fixed_epochs=epochs if stage=='refit' else None,validation=val,validation_metrics_computed=stage!='refit',
        optimizer_steps=epochs*len(loader),evaluation_accessed=False)
    return 0

def zero(root,job,region):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False);results={}
    for arm in ['R2','R2-B']:
        sub=out/arm;sub.mkdir()
        device,tr,va,model,opt,scheduler,loader=setup(root,region,arm,42,sub,'c')
        val=old.evaluate(model,va,device,compare_anchor=True)
        passed=val['max_abs_prediction_minus_anchor']==0 and val['prediction_minus_anchor_RMS']==0
        results[arm]=dict(status='passed' if passed else 'failed',pointwise_equal=passed,validation=val)
        del model,opt,scheduler,loader;torch.cuda.empty_cache()
    passed=all(r['pointwise_equal'] for r in results.values())
    write(out/'CHECK_C.json',dict(status='passed' if passed else 'failed',region=region,arms=results))
    finish(out,stage='c',checks_passed=passed,optimizer_steps=0)
    return 0 if passed else 2

def predict(root,job,region):
    frozen=json.loads((root/'FROZEN_REGIONAL_V2.json').read_text());assert frozen['status']=='frozen'
    manifest=json.loads((root/'PREDICT_MODELS.json').read_text());device=torch.device('cuda')
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False);rows=[]
    for row in manifest['models']:
        if row['region']!=region:continue
        ds=Windows(root/row['prepared_root'],region,'evaluation')
        checkpoint=root/row['checkpoint'];assert sha(checkpoint)==row['checkpoint_sha256']
        model=old.Model(root,region,'R1',row['seed'],device,ds.scale)
        state=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(state['state_dict'],strict=True)
        path=out/(row['id']+'.npy');dest=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(ds),12,ds.n));model.eval()
        with torch.no_grad():
            for x,y,v,a,ii in DataLoader(ds,batch_size=4,shuffle=False):
                p=model(x.to(device),a.to(device)).cpu().numpy()[...,0];assert np.isfinite(p).all();dest[ii.numpy()]=p
        dest.flush();rows.append(dict(id=row['id'],seed=row['seed'],epoch=state['epoch'],stage=row['stage']))
        np.save(out/(row['stage']+'_origins.npy'),ds.origins)
        del model,dest;torch.cuda.empty_cache()
    finish(out,region=region,stage='predict',models=rows,scoring_performed=False)
    return 0

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--region',choices=['china','japan'],required=True);p.add_argument('--stage',choices=['a','c','train','f','refit','predict'],required=True)
    p.add_argument('--arm',choices=['R2','R2-B'],default='R2');p.add_argument('--seed',type=int,choices=SEEDS,default=42);a=p.parse_args()
    if a.stage=='c':code=zero(a.root,a.job,a.region)
    elif a.stage=='predict':code=predict(a.root,a.job,a.region)
    else:code=train(a.root,a.job,a.region,a.arm,a.seed,a.stage)
    sys.exit(code)
