"""One shared, node-parameterized Japan training/prediction entrypoint."""
import argparse, copy, gc, hashlib, json, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data import HERE, SEEDS, ARMS, Archive, NativeDays, prepare, sha, write, fill, unix, role, now
from lib.normalization import MinMax01Scaler
from lib.physics.ppinn_projected import PPinnLoss
from lib.physics.projection import assign_gradients
from lib.physics.drivers_projected import DriversStore
from lib.physics.driver_scaler import FrozenDriverScaler
from model.GWN.GWN import GWNET

DILATIONS=[1,2,4,8,16,32,16,16]
TERMS={'nonneg','rot','roti','diffusion','chemistry','night'}

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def has_drivers(arm):return arm in ['dense_de','dense_pi_drivers','coarse_de']
def has_pi(arm):return arm in ['dense_pi','dense_pi_drivers']
def init_group(arm):return 'drivers' if has_drivers(arm) else 'tec'
def tensor_digest(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def amp(device):return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')

class RawDrivers:
    def __init__(self,ar,device):
        p=ar.prepared/'drivers.npz';self.store=DriversStore(str(p),device=device,strict=True,expected_sha256=sha(p))
        self.time=self.store.time.astype('datetime64[s]').astype(np.int64);self.n=ar.n
        self.scaler=FrozenDriverScaler.load(ar.prepared/'driver_scaler.json',expected_file_sha256=sha(ar.prepared/'driver_scaler.json'),expected_source_sha256=sha(p),expected_dataset=ar.product+'_japan')
    def raw(self,origins):
        times=np.asarray(origins,np.int64);start=np.searchsorted(self.time,times);idx=start[:,None]+np.arange(12)
        assert idx.max()<len(self.time) and np.array_equal(self.time[idx],times[:,None]+np.arange(12)*7200)
        return self.store.slice_batched(start,12,self.n)

class Windows(Dataset):
    def __init__(self,ar,split,arm,rows=None):
        self.ar,self.arm,self.scale=ar,arm,ar.scale
        if rows is None:
            support=json.loads((HERE/'SUPPORT.json').read_text());assert support['status']=='frozen'
            assert support['products'][ar.product]['daily_manifest_sha256']==sha(ar.root/'data/DAYS.jsonl')
            rows=support['rows']
        self.rows=[r for r in rows if r['role']==split and r['included']];self.days=[r['date'] for r in self.rows]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        d=self.days[i];coarse=self.arm=='coarse_de';h=self.ar.history(d,coarse)
        x=np.empty((*h.shape,6 if has_drivers(self.arm) else 1),np.float32);x[:,:,0]=h/np.float32(self.scale)
        if has_drivers(self.arm):x[:,:,1:]=self.ar.history_drivers(d,coarse)
        y,v=self.ar.target(d)
        return torch.from_numpy(x),torch.from_numpy(y[...,None]),torch.from_numpy(v[...,None]),unix(d),i

def model_for(ar,arm,device,seed=None,initial=False):
    args=SimpleNamespace(adj_mx=np.load(ar.prepared/'adjacency.npy'),num_nodes=ar.n,dropout=.3,blocks=4,layers=2,gcn_bool=True,addaptadj=True,adjtype='doubletransition',randomadj=not initial,aptonly=False,kernel_size=2,nhid=32,residual_channels=32,dilation_channels=32,input_window=12 if arm=='coarse_de' else 89,output_window=12,driver_fusion='none')
    # Explicit causal driver construction in Archive replaces the legacy 12-step wrapper.
    m=GWNET(args,device,6 if has_drivers(arm) else 1,1).to(device)
    if arm!='coarse_de':
        for f,g,d in zip(m.filter_convs,m.gate_convs,DILATIONS):f.dilation=g.dilation=(1,d)
        m.receptive_field=1+sum(DILATIONS)
    if seed is not None:
        m.load_state_dict(torch.load(ar.prepared/f'initial_{init_group(arm)}_{seed}.pt',map_location=device,weights_only=True),strict=True)
    return m

def prepare_initial(ar):
    records={}
    for s in SEEDS:
        for group,arm in [('tec','dense_do'),('drivers','dense_de')]:
            p=ar.prepared/f'initial_{group}_{s}.pt'
            if not p.exists():
                seed_all(s);m=model_for(ar,arm,torch.device('cpu'),initial=True);torch.save(m.state_dict(),p);del m
            records[p.name]=sha(p)
    write(ar.prepared/'INITIALIZATIONS.json',records)

def make_physics(ar,device):
    p=ar.prepared/'adjacency.npy';h=sha(p)
    physics=PPinnLoss(MinMax01Scaler(0.,ar.scale),torch.as_tensor(np.load(p),dtype=torch.float32,device=device),dataset=ar.product+'_japan',normalizer_id=ar.product+'_japan_trainmax_v1',model_key='pi_driver',dt_minutes=120,tec_ref=50.,t_ref_sec=7200.,rot_cap_tecu_per_min=.5,roti_w=3,roti_cap_scale=.7,kappa_nd=.05,alpha_nd=.01,beta_nd=.01,ae_factor=.001,rot_cap_k=.01,use_nonneg=True,use_rot=True,use_roti=True,use_diffusion=True,use_chemistry=True,use_night=True,use_dynamic_rot_cap=True,use_advection=False,first_transition_policy='forecast_internal_only',strict_paper_run=True,adjacency_sha256=h,expected_adjacency_sha256=h).to(device)
    for p in physics.uncert.log_vars.values():p.requires_grad_(False);p.data.zero_()
    return physics

def step(m,b,opt,ar,device,physics=None,drivers=None,rho=.01):
    x,y,v,origins,_=b;x,y,v=x.to(device),y.to(device),v.to(device);opt.zero_grad(set_to_none=True)
    with amp(device):p=m(x)
    assert torch.isfinite(p).all() and v.any()
    loss=(p.float()[v]*ar.scale-y.float()[v]).abs().mean();physical=None;values={}
    if physics is not None and rho:
        _,terms=physics(p.float(),None,drivers=drivers.raw(origins));assert set(terms)==TERMS
        physical=.5*sum(terms.values());values={k:float(t.detach()) for k,t in terms.items()}
    projection=assign_gradients(loss,physical,list(m.parameters()),rho if physical is not None else 0.)
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step()
    return float(loss.detach()),int(v.sum()),projection,values

@torch.no_grad()
def evaluate(m,ds,device,path=None):
    m.eval();tot=np.zeros(4,np.float64);n=0
    saved=None if path is None else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ds),12,ds.ar.n,1))
    for x,y,v,origins,ii in DataLoader(ds,batch_size=4,shuffle=False):
        with amp(device):p=m(x.to(device))
        pred=p.float().cpu().numpy()*np.float32(ds.scale);assert np.isfinite(pred).all()
        e=pred[v.numpy()].astype(np.float64)-y.numpy()[v.numpy()].astype(np.float64)
        tot+=np.array([np.square(e).sum(),np.abs(e).sum(),e.sum(),e.size]);n+=len(x)
        if saved is not None:saved[ii.numpy()]=pred
    assert n==len(ds) and tot[3]>0
    if saved is not None:saved.flush()
    return dict(RMSE=float(np.sqrt(tot[0]/tot[3])),MAE=float(tot[1]/tot[3]),bias=float(tot[2]/tot[3]),sse=float(tot[0]),sae=float(tot[1]),error_sum=float(tot[2]),n_valid=int(tot[3]))

def fixture_rows(ar):
    rows=json.loads((ar.prepared/'CANDIDATES.json').read_text());kept=[r for r in rows if r['included'] and r['role']=='train'];assert kept
    return kept[:1]

def check(ar,device):
    prepare_initial(ar);rows=fixture_rows(ar);drivers=RawDrivers(ar,device);checks={};initial={}
    x=np.full((89,2),np.nan,np.float32);v=np.zeros(x.shape,bool);x[0,0]=7;v[0,0]=True;x[70,1]=99;v[70,1]=True
    f=fill(x,v,np.full(x.shape,2,np.float32),900)
    assert np.all(f[:25,0]==7) and f[25,0]==2 and np.all(f[:70,1]==2)
    assert all(int(r['date'][:4])<2026 for r in ar.native.rows.values())
    for arm in ARMS:
        seed_all(42);ds=Windows(ar,'train',arm,rows);b=list(next(iter(DataLoader(ds,batch_size=1))))
        b[1]=b[1].clone();b[2]=b[2].clone();b[1][0,0,0,0]=float('nan');b[2][0,0,0,0]=False
        m=model_for(ar,arm,device,42);initial[arm]=tensor_digest(m)
        physics=make_physics(ar,device) if has_pi(arm) else None;opt=torch.optim.Adam(m.parameters(),lr=.003,weight_decay=0.)
        m.train();loss,n,proj,terms=step(m,b,opt,ar,device,physics,drivers)
        assert np.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        if has_pi(arm):assert proj['auxiliary_gradient_norm']>0 and set(terms)==TERMS and physics.transition_count(12)==11
        m.eval();xx=b[0].to(device).detach().requires_grad_(True);pred=m(xx)
        grad=torch.autograd.grad(pred.square().mean(),xx)[0];assert torch.all(grad[:,:,:,0].abs().sum((0,2))>0)
        state=copy.deepcopy(m.state_dict());reloaded=model_for(ar,arm,device);reloaded.load_state_dict(state);reloaded.eval()
        with torch.no_grad():torch.testing.assert_close(pred,reloaded(xx),rtol=0,atol=0)
        checks[arm]=dict(loss=loss,n_valid=n,shape=list(pred.shape),initial_tensor_sha256=initial[arm],all_input_frames_have_gradients=True,projection=proj,terms=terms)
        del m,reloaded,opt,pred,xx,physics,b;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    assert initial['dense_do']==initial['dense_pi'] and initial['dense_de']==initial['dense_pi_drivers']
    # Rho=0 must reproduce the supervised update exactly, including dropout RNG.
    ds=Windows(ar,'train','dense_do',rows);b=next(iter(DataLoader(ds,batch_size=1)))
    m=model_for(ar,'dense_do',device,42);other=copy.deepcopy(m);physics=make_physics(ar,device)
    opt1=torch.optim.Adam(m.parameters(),lr=.003);opt2=torch.optim.Adam(other.parameters(),lr=.003)
    seed_all(42);step(m,b,opt1,ar,device);seed_all(42);step(other,b,opt2,ar,device,physics,drivers,rho=0)
    assert tensor_digest(m)==tensor_digest(other)
    # Exercise every residual away from its inactive region, with a graph to prediction.
    p=(torch.randn(1,12,ar.n,1,device=device)*5).requires_grad_(True);raw=drivers.raw(np.array([unix(rows[0]['date'])]));raw['cosSZA']=torch.full_like(raw['cosSZA'],-.5)
    _,terms=physics(p,None,drivers=raw);term_grads={}
    for k,t in terms.items():
        g=torch.autograd.grad(t,p,retain_graph=True)[0];assert torch.isfinite(g).all() and g.abs().sum()>0;term_grads[k]=float(g.norm())
    info=dict(status='passed',product=ar.product,num_nodes=ar.n,device=str(device),fixture_only=True,checks=checks,six_hour_fill_observation_age=True,paired_initialization=True,rho_zero_supervised_equivalent=True,term_gradient_norms=term_grads,source_sha256=sha(__file__),data_code_sha256=sha(HERE/'data.py'))
    write(ar.root/('GPU_CHECK.json' if device.type=='cuda' else 'CPU_CHECK.json'),info);print(json.dumps(dict(status='passed',product=ar.product,device=str(device))),flush=True)

def benchmark(ar):
    device=torch.device('cuda');assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    prepare_initial(ar);rows=fixture_rows(ar);ds=Windows(ar,'train','dense_pi_drivers',rows*32);b=next(iter(DataLoader(ds,batch_size=32)))
    m=model_for(ar,'dense_pi_drivers',device,42);physics=make_physics(ar,device);drivers=RawDrivers(ar,device);opt=torch.optim.Adam(m.parameters(),lr=.003)
    torch.cuda.reset_peak_memory_stats();dur=[]
    for i in range(4):
        m.train();start=time.monotonic();step(m,b,opt,ar,device,physics,drivers);torch.cuda.synchronize();dur.append(time.monotonic()-start)
    info=dict(status='passed',product=ar.product,num_nodes=ar.n,batch_size=32,precision='bf16',seconds_per_train_step=float(np.mean(dur[1:])),step_seconds=dur,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,fixture_only=True,scientific_runs=0,code_sha256=sha(__file__))
    write(ar.root/'BENCHMARK.json',info);print(json.dumps(info),flush=True)

def train(ar,arm,seed):
    assert ar.identity['scientific_ready']
    for product in ['jpld','isee']:
        for name in ['CPU_CHECK.json','BENCHMARK.json']:
            check=json.loads((HERE.parent/product/name).read_text());assert check['status']=='passed'
    device=torch.device('cuda');assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    out=ar.root/'runs'/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=False)
    seed_all(seed);tr=Windows(ar,'train',arm);va=Windows(ar,'validation',arm);assert len(tr)>0 and len(va)>0
    m=model_for(ar,arm,device,seed);drivers=RawDrivers(ar,device) if has_pi(arm) else None;physics=make_physics(ar,device) if has_pi(arm) else None
    opt=torch.optim.Adam(m.parameters(),lr=.003,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10,min_lr=0.)
    seed_all(seed);dl=DataLoader(tr,batch_size=32,shuffle=True,drop_last=False,generator=torch.Generator().manual_seed(seed))
    cfg=dict(product=ar.product,dataset=ar.product+'_japan',normalizer_id=ar.product+'_japan_trainmax_v1',arm=arm,seed=seed,protocol_sha256=sha(HERE/'PROTOCOL.json'),support_sha256=sha(HERE/'SUPPORT.json'),source_sha256=sha(__file__),data_source_sha256=sha(HERE/'data.py'),initial_sha256=sha(ar.prepared/f'initial_{init_group(arm)}_{seed}.pt'),initial_tensor_sha256=tensor_digest(m),preprocessing_identity_sha256=sha(ar.prepared/'IDENTITY.json'),train_windows=len(tr),validation_windows=len(va),num_nodes=ar.n,parameters=sum(p.numel() for p in m.parameters()),torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),rho=.01 if has_pi(arm) else 0.,physics_constants=physics.constants if physics else None,epochs=200,batch_size=32,precision='bf16',evaluation_batch_size=4)
    write(out/'config.json',cfg);best=np.inf;selected=None;start=time.monotonic()
    with (out/'epochs.jsonl').open('x') as log:
        for epoch in range(1,201):
            m.train();total=count=0;lr=opt.param_groups[0]['lr'];term_sums={k:0. for k in TERMS} if physics else {};projections=0
            for b in dl:
                loss,n,proj,terms=step(m,b,opt,ar,device,physics,drivers);total+=loss*n;count+=n
                for k,t in terms.items():term_sums[k]+=t*n
                projections+=int(proj.get('projection_triggered',False))
            val=evaluate(m,va,device);score=val['RMSE'];scheduler.step(score)
            if score<best:
                best=score;selected=epoch;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,validation_rmse=best),out/'best.pt')
            row=dict(epoch=epoch,training_MAE=total/count,validation=val,lr=lr,next_lr=opt.param_groups[0]['lr'],selected_epoch=selected,physics_terms={k:t/count for k,t in term_sums.items()},projection_steps=projections,elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(dict(product=ar.product,arm=arm,seed=seed,**row)),flush=True)
    state=torch.load(out/'best.pt',map_location=device,weights_only=True);m.load_state_dict(state['state_dict'],strict=True)
    val=evaluate(m,va,device,out/'validation_predictions.npy');assert abs(val['RMSE']-best)<1e-9
    ev=Windows(ar,'evaluation',arm);metrics=evaluate(m,ev,device,out/'evaluation_predictions.npy')
    np.save(out/'validation_origins.npy',np.array([unix(d) for d in va.days],np.int64));np.save(out/'evaluation_origins.npy',np.array([unix(d) for d in ev.days],np.int64))
    write(out/'complete.json',dict(status='complete',product=ar.product,arm=arm,seed=seed,epochs=200,selected_epoch=selected,validation=val,evaluation=metrics,files={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))

def main():
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','cpu-check','gpu-check','benchmark','train','verify']);ap.add_argument('--product',choices=['jpld','isee'],required=True);ap.add_argument('--root',type=Path,default=HERE.parent);ap.add_argument('--fixture',action='store_true');ap.add_argument('--arm',choices=ARMS);ap.add_argument('--seed',type=int,choices=SEEDS);ap.add_argument('--drivers',type=Path)
    a=ap.parse_args();root=a.root/a.product
    if a.action=='prepare':
        source=a.drivers or HERE.parents[2]/'manuscript/revisions-20260929/revised/release/assets/data/drivers/gim_2009_2025.npz'
        print(json.dumps(prepare(root,a.product,source,partial=a.fixture)),flush=True)
        ar=Archive(root,a.product,a.fixture);prepare_initial(ar);return
    ar=Archive(root,a.product,a.fixture)
    if a.action in ['cpu-check','gpu-check']:check(ar,torch.device('cpu' if a.action=='cpu-check' else 'cuda'))
    elif a.action=='benchmark':benchmark(ar)
    elif a.action=='verify':
        from score import verify
        verify(ar,a.arm,a.seed)
    else:
        assert not a.fixture and a.arm and a.seed;train(ar,a.arm,a.seed)

if __name__=='__main__':main()
