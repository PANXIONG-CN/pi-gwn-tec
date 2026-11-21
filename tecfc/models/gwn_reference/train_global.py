"""Global matched GWN, full TEC and no PI; C training protocol preserved."""
from __future__ import annotations
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from gim_data import GIM,HERE,sha,restore_grid
from global_model import GlobalGWN


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


class Windows(Dataset):
 def __init__(self,role,base=None):
  if base is None and (HERE/'data/gim').exists():base=HERE/'data'
  self.gim=GIM() if base is None else GIM(base/'gim',base/'gim_extension')
  self.days=list(self.gim.windows(role));self.role=role
  with np.load(HERE/'prepared/train_climatology_v1.npz') as z:self.scale=float(z['scale'])
  assert self.scale==147.1
 def __len__(self):return len(self.days)
 def __getitem__(self,i):
  d=self.days[i];s=self.gim.sample(d);scale=np.float32(self.scale)
  return dict(history=torch.from_numpy(np.array(s['history']/scale,dtype=np.float32)),
   target=torch.from_numpy(np.array(restore_grid(s['target']),dtype=np.float32)),
   valid=torch.from_numpy(np.array(restore_grid(s['valid']),dtype=bool)),origin=int(s['origin']),index=i)


def forward(model,b,arm,device):
 assert arm=='G'
 return model(b['history'].to(device),restore=True)


def supervised(pred,b,scale,device):
 valid=b['valid'].to(device);y=b['target'].to(device)
 if not bool(valid.any()) or not torch.isfinite(pred).all():raise ValueError('empty target or nonfinite prediction')
 return nn.functional.smooth_l1_loss(pred.float()[valid],y.float()[valid]/scale,beta=1.)


def amp(device):return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')


@torch.no_grad()
def evaluate(model,ds,arm,device,batch_size,path=None):
 model.eval();sums=np.zeros(4,np.float64)
 dest=None if path is None else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ds),12,71,73))
 for b in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
  with amp(device): pred=forward(model,b,arm,device)
  pred=pred.float().cpu().numpy()*np.float32(ds.scale);truth=b['target'].numpy();valid=b['valid'].numpy()
  if not np.isfinite(pred).all():raise ValueError('nonfinite predictions')
  e=pred[valid].astype(np.float64)-truth[valid].astype(np.float64)
  sums += [np.square(e).sum(),np.abs(e).sum(),e.sum(),e.size]
  if dest is not None:dest[b['index'].numpy()]=pred
 if dest is not None:dest.flush()
 if sums[3]==0:raise ValueError('empty evaluation')
 return dict(RMSE=float(np.sqrt(sums[0]/sums[3])),MAE=float(sums[1]/sums[3]),bias=float(sums[2]/sums[3]),sse=float(sums[0]),sae=float(sums[1]),error_sum=float(sums[2]),n_valid=int(sums[3]))


def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def atomic_save(obj,path):
 tmp=path.with_name(path.name+'.partial');torch.save(obj,tmp);os.replace(tmp,path)


def load_freeze(identity):
 p=HERE/'SCIENTIFIC_FREEZE.json'
 if not p.exists():raise RuntimeError('matched GWN recipe is not frozen')
 f=json.loads(p.read_text());assert f['status']=='frozen' and identity in f['identities']
 cfg=f['identities'][identity];assert cfg['model_arm']=='G' and cfg['physics']=='none'
 return f,cfg


def gpu_preflight(identity,cfg,batch_size):
 device=torch.device('cuda');assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 ds=Windows('train');b=next(iter(DataLoader(ds,batch_size=batch_size)));seed_all(int(cfg['seed']))
 model=GlobalGWN(device).to(device);initial=digest(model);model.train();b['history'].requires_grad_(True)
 torch.cuda.reset_peak_memory_stats()
 with amp(device):pred=forward(model,b,'G',device)
 loss=supervised(pred,b,ds.scale,device);loss.backward();torch.cuda.synchronize()
 grad=b['history'].grad.abs().sum((0,2,3));assert torch.isfinite(grad).all() and (grad>0).all()
 assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
 assert tuple(pred.shape)==(batch_size,12,71,73) and torch.equal(pred[...,0],pred[...,-1])
 info=dict(status='passed',identity=identity,device=str(device),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
  bf16=torch.cuda.is_bf16_supported(),batch_size=batch_size,shape=list(pred.shape),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
  loss=float(loss.detach()),all_12_history_frames_have_gradient=True,initial_tensor_sha256=initial,
  scope='G full training batch forward and backward in train mode; no optimizer step')
 write(HERE/'runs'/identity/'GPU_CHECK.json',info)
 del model,pred,loss,b;torch.cuda.empty_cache()
 return info


def train(identity):
 f,cfg=load_freeze(identity);seed=int(cfg['seed']);arm=cfg['model_arm'];device=torch.device('cuda')
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 ds=Windows('train');va=Windows('validation')
 # A copied deployment must match every audited local source year before use.
 audit=json.loads((HERE/'DATA_AUDIT.json').read_text())
 for rec in audit['years']:
  assert sha(ds.gim.path(rec['year']))==rec['sha256'], f"changed GIM source {rec['year']}"
 check=gpu_preflight(identity,cfg,int(f['batch_size']))
 seed_all(seed);model=GlobalGWN(device).to(device)
 initial=digest(model);assert initial==check['initial_tensor_sha256']
 opt=torch.optim.Adam(model.parameters(),lr=float(f['lr']),weight_decay=float(f['weight_decay']))
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10,min_lr=0.)
 seed_all(seed);generator=torch.Generator().manual_seed(seed)
 dl=DataLoader(ds,batch_size=int(f['batch_size']),shuffle=True,drop_last=False,generator=generator,num_workers=0)
 out=HERE/'runs'/identity;out.mkdir(parents=True,exist_ok=True);assert not (out/'complete.json').exists()
 config=dict(identity=identity,seed=seed,model_arm=arm,physics=cfg['physics'],physics_weight=float(cfg.get('physics_weight',0)),
  epochs=int(f['epochs']),batch_size=int(f['batch_size']),precision='bf16',optimizer=f"Adam lr={f['lr']} weight_decay={f['weight_decay']}",
  gradient_clip=float(f['gradient_clip']),scheduler='ReduceLROnPlateau factor=.9 patience=5 threshold=1e-5 abs cooldown=10',
  selection='first strictly lowest pooled validation RMSE',loss='SmoothL1 beta=1 on normalized full TEC at valid targets',
  train_windows=len(ds),validation_windows=len(va),scale=ds.scale,initial_tensor_sha256=initial,
  parameters=sum(p.numel() for p in model.parameters()),freeze_sha256=sha(HERE/'SCIENTIFIC_FREEZE.json'),
  gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
  graph_sha256=sha(HERE/'prepared/graph_svd_v1.npz'),backbone_sha256=sha(HERE/'model/GWN/GWN.py'),matching_protocol_sha256=f['matching_protocol_sha256'])
 if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
 else:write(out/'config.json',config)
 last=out/'last.pt';epoch0=0;best=float('inf');selected=0;best_state=None
 if last.exists():
  ck=torch.load(last,map_location=device,weights_only=False);epoch0=ck['epoch'];best=ck['best'];selected=ck['selected'];best_state=ck['best_state']
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
  random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu())
  torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']]);generator.set_state(ck['loader_rng'].cpu())
  log=out/'epochs.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
  rows=[r for r in rows if r['epoch']<epoch0]+[ck['row']];assert [r['epoch'] for r in rows]==list(range(1,epoch0+1))
  log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 start=time.monotonic()
 for epoch in range(epoch0+1,int(f['epochs'])+1):
  model.train();total=count=neg_count=neg_total=0;neg_sq=0.;minimum=float('inf');lr=opt.param_groups[0]['lr']
  for b in dl:
   opt.zero_grad(set_to_none=True)
   with amp(device):pred=forward(model,b,arm,device)
   with torch.no_grad():
    tec=pred.float()*ds.scale;negative=tec<0;neg_count+=int(negative.sum());neg_total+=tec.numel();neg_sq+=float(tec[negative].square().sum());minimum=min(minimum,float(tec.min()))
   sup=supervised(pred,b,ds.scale,device);loss=sup
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(f['gradient_clip']),error_if_nonfinite=True);opt.step()
   n=int(b['valid'].sum());total+=float(sup.detach())*n;count+=n
  val=evaluate(model,va,arm,device,int(f['eval_batch_size']));score=val['RMSE'];scheduler.step(score)
  if score<best:best=score;selected=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  row=dict(epoch=epoch,training_loss=total/count,validation=val,lr=lr,next_lr=opt.param_groups[0]['lr'],selected_epoch=selected,
   elapsed_this_process_seconds=time.monotonic()-start,training_nonnegative_diagnostic=dict(negative_count=neg_count,total_count=neg_total,negative_squared_sum_TECU2=neg_sq,minimum_TECU=minimum))
  state=dict(epoch=epoch,best=best,selected=selected,best_state=best_state,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state(),row=row)
  atomic_save(state,last)
  with (out/'epochs.jsonl').open('a') as log:log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
  print(json.dumps(dict(identity=identity,**row)),flush=True)
 model.load_state_dict(best_state);atomic_save(dict(state_dict=best_state,epoch=selected,validation_rmse=best),out/'best.pt')
 results={}
 for split in ('validation','original_evaluation','extension_evaluation'):
  d=va if split=='validation' else Windows(split)
  results[split]=evaluate(model,d,arm,device,int(f['eval_batch_size']),out/f'{split}_predictions.npy')
  np.save(out/f'{split}_origins.npy',np.array([int(np.datetime64(day,'s').astype(np.int64)) for day in d.days],np.int64))
 assert abs(results['validation']['RMSE']-best)<1e-8
 write(out/'complete.json',dict(status='complete',identity=identity,seed=seed,epochs=int(f['epochs']),selected_epoch=selected,**results,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
 print(json.dumps(dict(status='complete',identity=identity)),flush=True)


if __name__=='__main__':
 torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('action',choices=['train']);p.add_argument('--identity');a=p.parse_args()
 if not a.identity:p.error('--identity required')
 train(a.identity)
