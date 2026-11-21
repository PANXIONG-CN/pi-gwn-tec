"""Fixed exp0020 R0/R1 training, C/E checks, and post-freeze prediction.

Training sees only prepared training/2022 windows. Evaluation arrays are opened
only by the separate predict stage after a committed all-model checkpoint freeze.
"""
import argparse,copy,json,platform,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from train import load_legacy,sha,write,digest

SEEDS=[42,1024,2025,3407,5000]


class Windows(Dataset):
    def __init__(self,root,region,split):
        self.root=root/'prepared'/region;self.meta=json.loads((self.root/'metadata.json').read_text())
        self.scale=self.meta['scale'];self.n=self.meta['nodes'];self.split=split
        self.arrays={k:np.load(self.root/split/(k+'.npy'),mmap_mode='r') for k in ['history','target','valid','anchor','origins']}
        self.origins=self.arrays['origins'];assert len(self.origins)==self.meta['counts'][split]
        years=self.origins.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
        assert set(years)==set({'train':[2015,2019,2020,2021],'validation':[2022],'evaluation':[2023,2024,2025]}[split])
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        return (torch.from_numpy(np.array(self.arrays['history'][i]/np.float32(self.scale),dtype=np.float32)[...,None]),
                torch.from_numpy(np.array(self.arrays['target'][i],dtype=np.float32)[...,None]),
                torch.from_numpy(np.array(self.arrays['valid'][i],dtype=bool)[...,None]),
                torch.from_numpy(np.array(self.arrays['anchor'][i],dtype=np.float64)[...,None]),i)


class Model(torch.nn.Module):
    def __init__(self,root,region,arm,seed,device,scale):
        super().__init__();legacy,path=load_legacy(root,region);legacy.seed_all(seed)
        self.region,self.arm,self.scale,self.legacy=region,arm,scale,legacy
        if region=='japan':
            ar=SimpleNamespace(prepared=path/'jpld/prepared',n=288)
            base=legacy.model_for(ar,'dense_do',device,seed)
            self.initial_source=sha(ar.prepared/f'initial_tec_{seed}.pt')
        else:
            initial=path/'shared'/f'initial_{seed}.pt'
            assert initial.exists(),'Every scientific arm must load the same frozen seed initialization file'
            state=torch.load(initial,map_location='cpu',weights_only=True);self.initial_source=sha(initial)
            state=copy.deepcopy(state)
            state['end_conv_2.weight']=state['end_conv_2.weight'][legacy.COMMON].clone()
            state['end_conv_2.bias']=state['end_conv_2.bias'][legacy.COMMON].clone()
            base=legacy.model_for(path,'A_native',device)
            base.load_state_dict(state,strict=True)
            first=base.start_conv;replacement=torch.nn.Conv2d(1,32,(1,1)).to(device)
            with torch.no_grad():replacement.weight.copy_(first.weight[:,:1]);replacement.bias.copy_(first.bias)
            base.start_conv=replacement;base.feature_dim=1
        self.base=base;self.anchor=None;self.r0_initial_tensor_sha256=digest(base)
        if arm=='R1':
            old=base.end_conv_1;extra=torch.nn.Conv2d(268,512,(1,1)).to(device)
            with torch.no_grad():extra.weight.zero_();extra.weight[:,:256].copy_(old.weight);extra.bias.copy_(old.bias)
            base.end_conv_1=extra
            with torch.no_grad():base.end_conv_2.weight.zero_();base.end_conv_2.bias.zero_()
            base.end_conv_1.register_forward_pre_hook(self.inject)
        self.initial_tensor_sha256=digest(base)
    def inject(self,module,args):
        x=args[0];assert self.anchor is not None and self.anchor.shape[1]==12 and x.shape[1]==256 and x.shape[-1]==1
        anchor=(self.anchor.float()/self.scale).to(dtype=x.dtype)
        return (torch.cat([x,anchor],dim=1),)
    def forward(self,x,anchor):
        self.anchor=anchor if self.arm=='R1' else None
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.device.type=='cuda'):
            residual=self.base(x)
        out=residual.float()*self.scale
        return anchor+out.double() if self.arm=='R1' else out


def step(model,batch,optimizer,device):
    if model.arm=='R0':
        x,y,v,a,ii=batch
        if model.region=='japan':
            loss,n,_,_=model.legacy.step(model.base,(x,y,v,ii,ii),optimizer,SimpleNamespace(scale=model.scale),device)
            return loss,n
        return model.legacy.step(model.base,(x,y,v,ii),optimizer,model.scale,device)
    x,y,v,a,_=batch;x,y,v,a=x.to(device),y.to(device),v.to(device),a.to(device)
    optimizer.zero_grad(set_to_none=True);out=model(x,a);assert torch.isfinite(out).all() and bool(v.any())
    loss=(out.float()[v]-y.float()[v]).abs().mean()
    if model.region=='japan':model.legacy.assign_gradients(loss,None,list(model.parameters()),0.)
    else:loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    return float(loss.detach()),int(v.sum())


class LegacyView(Dataset):
    def __init__(self,ds,region):
        self.ds,self.region,self.scale=ds,region,ds.scale
        self.ar=SimpleNamespace(n=ds.n);self.dense_out=False
    def __len__(self):return len(self.ds)
    def __getitem__(self,i):
        x,y,v,a,ii=self.ds[i]
        return (x,y,v,int(self.ds.origins[i]),ii) if self.region=='japan' else (x,y,v,ii)


@torch.no_grad()
def evaluate(model,ds,device,path=None,compare_anchor=False):
    if model.arm=='R0':
        assert not compare_anchor
        temp=None if path is None else Path(path).with_name('legacy_validation_float32.npy')
        old=model.legacy.evaluate(model.base,LegacyView(ds,model.region),device,temp)
        result=old if model.region=='japan' else old['common12']
        if temp is not None:
            original=np.load(temp,mmap_mode='r');saved=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(ds),12,ds.n))
            for i in range(len(ds)):saved[i]=original[i,...,0]
            saved.flush();del saved,original;temp.unlink()
        return result
    model.eval();total=np.zeros(4,np.float64);difference_sse=0.;difference_n=0;maximum=0.
    saved=None if path is None else np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(ds),12,ds.n))
    for x,y,v,a,ii in DataLoader(ds,batch_size=4,shuffle=False,num_workers=0):
        out=model(x.to(device),a.to(device)).detach().cpu().numpy()[...,0]
        valid=v.numpy()[...,0];target=y.numpy()[...,0];assert np.isfinite(out).all()
        error=out[valid].astype(np.float64)-target[valid].astype(np.float64)
        total+=[np.square(error).sum(),np.abs(error).sum(),error.sum(),error.size]
        if compare_anchor:
            d=out-a.numpy()[...,0];difference_sse+=float(np.square(d).sum());difference_n+=d.size;maximum=max(maximum,float(np.abs(d).max()))
        if saved is not None:saved[ii.numpy()]=out
    if saved is not None:saved.flush()
    value=dict(RMSE=float(np.sqrt(total[0]/total[3])),MAE=float(total[1]/total[3]),bias=float(total[2]/total[3]),sse=float(total[0]),n_valid=int(total[3]))
    if compare_anchor:value.update(prediction_minus_anchor_RMS=float(np.sqrt(difference_sse/difference_n)),max_abs_prediction_minus_anchor=maximum)
    return value


def setup(root,region,arm,seed,out):
    device=torch.device('cuda');assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tr=Windows(root,region,'train');va=Windows(root,region,'validation')
    model=Model(root,region,arm,seed,device,tr.scale)
    if seed==42:
        legacy_identity=json.loads((root/'legacy_input_identity.json').read_text())
        assert model.r0_initial_tensor_sha256==legacy_identity['initial_tensor_sha256'],'R0 backbone must equal the A1/A2-admitted seed42 initialization'
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10,min_lr=0.)
    model.legacy.seed_all(seed);loader=DataLoader(tr,batch_size=32,shuffle=True,drop_last=False,generator=torch.Generator().manual_seed(seed))
    write(out/'config.json',dict(region=region,arm=arm,seed=seed,epochs=200,train_windows=len(tr),validation_windows=len(va),
          training_years=[2015,2019,2020,2021],validation_years=[2022],history_channels=['TEC'],driver_input=False,
          anchor_input=arm=='R1',scale=tr.scale,initial_source=model.initial_source,r0_initial_tensor_sha256=model.r0_initial_tensor_sha256,
          initial_tensor_sha256=model.initial_tensor_sha256,parameters=sum(p.numel() for p in model.parameters()),torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),source_sha256=sha(__file__)))
    return device,tr,va,model,optimizer,scheduler,loader


def check(root,job,region):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False)
    device,tr,va,model,opt,scheduler,loader=setup(root,region,'R1',42,out)
    initial=evaluate(model,va,device,compare_anchor=True)
    # Exact zero residual implies identical mapped anchors pointwise, not merely rounded metrics.
    assert initial['max_abs_prediction_minus_anchor']==0. and initial['prediction_minus_anchor_RMS']==0.
    totals=np.zeros(2)
    for i in range(len(va)):
        v=va.arrays['valid'][i];err=va.arrays['anchor'][i][v]-va.arrays['target'][i][v].astype(float)
        totals+=[np.square(err).sum(),err.size]
    baseline=float(np.sqrt(totals[0]/totals[1]));delta=abs(initial['RMSE']-baseline)
    c=dict(status='passed' if delta<1e-6 else 'failed',region=region,untrained_R1=initial,anchor_RMSE=baseline,absolute_difference=delta)
    write(out/'CHECK_C.json',c);assert c['status']=='passed'
    model.legacy.seed_all(42);model.train();total=count=0
    for batch in loader:
        loss,n=step(model,batch,opt,device);total+=loss*n;count+=n
    val=evaluate(model,va,device,compare_anchor=True);rms=val['prediction_minus_anchor_RMS']
    e=dict(status='passed' if rms>0 else 'failed',region=region,seed=42,epochs=1,training_MAE=total/count,validation=val,diagnostic_weights_discarded=True)
    write(out/'CHECK_E.json',e);assert e['status']=='passed'
    if region=='china':
        # Original study stored only three seeds. Generate each added seed once,
        # then distribute these exact untrained bytes to both formal arms.
        records={}
        for seed in [3407,5000]:
            model.legacy.seed_all(seed)
            initial=model.legacy.model_for(root/'legacy/china','B_native',torch.device('cpu'),initial=True)
            path=out/f'initial_{seed}.pt';torch.save(initial.state_dict(),path)
            records[str(seed)]=dict(file=path.name,sha256=sha(path),initial_tensor_sha256=digest(initial),optimizer_steps=0)
            del initial
        write(out/'ADDED_SEED_INITIALIZATIONS.json',dict(status='frozen_untrained',source='original six-channel B_native initializer',seeds=records))
    write(out/'complete.json',dict(status='complete',stage='ce',region=region,checks_passed=True,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


def train(root,job,region,arm,seed):
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False)
    device,tr,va,model,opt,scheduler,loader=setup(root,region,arm,seed,out)
    best=np.inf;selected=None;started=time.monotonic()
    with (out/'epochs.jsonl').open('x') as log:
        for epoch in range(1,201):
            model.train();total=count=0;lr=opt.param_groups[0]['lr']
            for batch in loader:
                loss,n=step(model,batch,opt,device);total+=loss*n;count+=n
            val=evaluate(model,va,device);score=val['RMSE'];scheduler.step(score)
            if score<best:
                best=score;selected=epoch;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_RMSE=best),out/'best.pt')
            row=dict(epoch=epoch,training_MAE=total/count,validation=val,selected_epoch=selected,lr=lr,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    saved=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict'],strict=True)
    val=evaluate(model,va,device,out/'validation_predictions.npy');assert abs(val['RMSE']-best)<1e-9
    np.save(out/'validation_origins.npy',va.origins)
    write(out/'complete.json',dict(status='complete',region=region,arm=arm,seed=seed,epochs=200,selected_epoch=selected,
          validation=val,evaluation_accessed=False,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


def predict(root,job,region):
    frozen=json.loads((root/'FROZEN_CHECKPOINTS.json').read_text());assert frozen['all_twenty_complete']
    out=root/'outputs'/job;out.mkdir(parents=True,exist_ok=False);ds=Windows(root,region,'evaluation');device=torch.device('cuda')
    rows=[]
    for candidate in frozen['models']:
        if candidate['region']!=region:continue
        arm,seed=candidate['arm'],candidate['seed'];checkpoint=root/'checkpoints'/f'{region}_{arm.lower()}_{seed}.pt'
        assert sha(checkpoint)==candidate['checkpoint_sha256']
        model=Model(root,region,arm,seed,device,ds.scale)
        state=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(state['state_dict'],strict=True)
        # Predictions are generated here; all cross-model evaluation scoring is deferred to the single CPU score pass.
        dest=np.lib.format.open_memmap(out/f'{arm}_{seed}_predictions.npy',mode='w+',dtype=np.float64,shape=(len(ds),12,ds.n))
        model.eval()
        with torch.no_grad():
            for x,y,v,a,ii in DataLoader(ds,batch_size=4,shuffle=False):
                p=model(x.to(device),a.to(device)).cpu().numpy()[...,0];assert np.isfinite(p).all();dest[ii.numpy()]=p
        dest.flush();rows.append(dict(arm=arm,seed=seed,selected_epoch=state['epoch']));del model,dest;torch.cuda.empty_cache()
    assert len(rows)==10;np.save(out/'origins.npy',ds.origins)
    write(out/'complete.json',dict(status='complete',region=region,models=rows,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--region',choices=['china','japan'],required=True);p.add_argument('--stage',choices=['ce','train','predict'],required=True)
    p.add_argument('--arm',choices=['R0','R1'],default='R1');p.add_argument('--seed',type=int,choices=SEEDS,default=42);a=p.parse_args()
    if a.stage=='ce':check(a.root,a.job,a.region)
    elif a.stage=='train':train(a.root,a.job,a.region,a.arm,a.seed)
    else:predict(a.root,a.job,a.region)
