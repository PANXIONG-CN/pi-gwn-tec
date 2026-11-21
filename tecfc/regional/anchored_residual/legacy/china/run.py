"""Four fixed dense-history GWN arms; isolated from the manuscript's runtime."""
import argparse
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.physics.solar import canonical_cos_sza_float64
from model.GWN.GWN import GWNET

ROOT = Path(__file__).resolve().parent
COMMON = np.arange(7, 96, 8)
DILATIONS = [1, 2, 4, 8, 16, 32, 16, 16]
ARMS = ['A_held', 'A_native', 'B_held', 'B_native']
SEEDS = [42, 1024, 2025]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = path.with_suffix(path.suffix+'.tmp')
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    p.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fill(a, valid, fallback, step):
    out = a.copy()
    last = np.full(a.shape[1], -10000)
    values = np.zeros(a.shape[1], np.float32)
    for t in range(len(a)):
        last[valid[t]] = t; values[valid[t]] = a[t, valid[t]]
        recent = ~valid[t] & ((t-last)*step <= 21600)
        out[t, recent] = values[recent]
        other = ~valid[t] & ~recent
        out[t, other] = fallback[t, other]
    assert np.isfinite(out).all()
    return out


class Archive:
    def __init__(self, root):
        self.root = root
        with np.load(root/'shared/tec_statistics.npz') as z:
            self.scale = float(z['scale']); self.fallback = z['fallback'].astype(np.float32)
        with np.load(root/'shared/drivers.npz') as z:
            self.drivers = {k: z[k] for k in ['time','AE','Dst','Kp','F107','lat_deg','lon_deg']}
        meta = json.loads((root/'shared/driver_scaler.json').read_text())
        self.means = np.array([meta['channels'][k]['mean'] for k in meta['channel_order']], np.float32)
        self.stds = np.array([meta['channels'][k]['standard_deviation'] for k in meta['channel_order']], np.float32)

    @lru_cache(maxsize=None)
    def read(self, name):
        with np.load(self.root/'data'/name) as z:
            return z['tec'].reshape(96,2800), z['valid'].reshape(96,2800)

    @lru_cache(maxsize=None)
    def history(self, name, native):
        a, v = self.read(name)
        if native:
            return fill(a[:89], v[:89], self.fallback[np.arange(89)//8], 900)
        return fill(a[::8], v[::8], self.fallback, 7200)[np.arange(89)//8]

    @lru_cache(maxsize=None)
    def driver_values(self, origin):
        times = origin-86400+np.arange(89)*900
        ix = np.searchsorted(self.drivers['time'], times, side='right')-1
        assert np.all(ix >= 0)
        source_times = self.drivers['time'][ix]
        assert np.all(source_times <= times) and np.all(times-source_times < 7200)
        assert times[-1] == origin-7200
        result = np.empty((89,2800,5), np.float32)
        for i, k in enumerate(['AE','Dst','Kp','F107']):
            result[:,:,i] = self.drivers[k][ix,None]
        result[:,:,4] = canonical_cos_sza_float64(times, self.drivers['lon_deg'].astype(np.float64), self.drivers['lat_deg'].astype(np.float64))
        result = (result-self.means)/self.stds
        assert np.isfinite(result).all()
        return result

    def target(self, row):
        h, hv = self.read(row['history']); y, yv = self.read(row['target'])
        return np.concatenate([h[89:],y[:89]]), np.concatenate([hv[89:],yv[:89]])

    def persistence(self, row):
        # D-2 tail supplies the first seven target slots; all values are visible at C.
        a2, v2 = self.read(row['previous2']); h, hv = self.read(row['history'])
        f = self.fallback[np.arange(96)//8]
        p2 = fill(a2, v2, f, 900)
        ph = self.history(row['history'], True)
        return np.concatenate([p2[89:], ph])


class Windows(Dataset):
    def __init__(self, archive, role, arm, rows=None):
        self.archive, self.arm, self.scale = archive, arm, archive.scale
        if rows is None:
            rows = [json.loads(s) for s in (archive.root/'data/WINDOWS.jsonl').read_text().splitlines()]
        self.rows = [r for r in rows if r['role']==role and r['included']]
        self.native = arm.endswith('native'); self.dense_out = arm.startswith('B')

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x = np.empty((89,2800,6), np.float32)
        x[:,:,0] = self.archive.history(row['history'], self.native)/np.float32(self.scale)
        x[:,:,1:] = self.archive.driver_values(row['origin'])
        y, v = self.archive.target(row)
        if not self.dense_out: y, v = y[COMMON], v[COMMON]
        return torch.from_numpy(x), torch.from_numpy(y[...,None]), torch.from_numpy(v[...,None]), i


def model_for(root, arm, device, seed=None, initial=False):
    args = SimpleNamespace(adj_mx=np.load(root/'shared/adjacency.npy'), num_nodes=2800,
        dropout=.3, blocks=4, layers=2, gcn_bool=True, addaptadj=True,
        adjtype='doubletransition', randomadj=not initial, aptonly=False, kernel_size=2,
        nhid=32, residual_channels=32, dilation_channels=32, input_window=89,
        output_window=96 if arm.startswith('B') else 12, driver_fusion='none')
    # Six channels are assembled with explicit timestamps above. The legacy
    # 12-frame DriverHistoryBatch is intentionally not used for this new task.
    model = GWNET(args, device, 6, 1).to(device)
    for f,g,d in zip(model.filter_convs, model.gate_convs, DILATIONS):
        f.dilation = g.dilation = (1,d)
    model.receptive_field = 1+sum(DILATIONS)
    if seed is not None:
        state = torch.load(root/f'shared/initial_{seed}.pt', map_location=device, weights_only=True)
        if arm.startswith('A'):
            state['end_conv_2.weight'] = state['end_conv_2.weight'][COMMON].clone()
            state['end_conv_2.bias'] = state['end_conv_2.bias'][COMMON].clone()
        model.load_state_dict(state, strict=True)
    return model


def prepare_shared(root):
    shared = root/'shared'; shared.mkdir(exist_ok=True)
    old = root.parents[1]/'output/jpld-china-point-20261031/shared'
    for name in ['adjacency.npy','tec_statistics.npz','driver_scaler.json','drivers.npz']:
        dest=shared/name
        if not dest.exists(): shutil.copyfile(old/name,dest)
        assert sha(dest)==sha(old/name)
    for seed in SEEDS:
        dest=shared/f'initial_{seed}.pt'
        if not dest.exists():
            seed_all(seed); m=model_for(root,'B_native',torch.device('cpu'),initial=True)
            torch.save(m.state_dict(),dest); del m
    write(shared/'identity.json',dict(protocol_sha256=sha(root/'PROTOCOL.json'),
          files={p.name:sha(p) for p in shared.iterdir() if p.name!='identity.json'}))


def amp(device):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')


def step(model, batch, optimizer, scale, device):
    x,y,v,_=batch; x=x.to(device); y=y.to(device); v=v.to(device)
    optimizer.zero_grad(set_to_none=True)
    with amp(device): p=model(x)
    loss=(p.float()[v]*scale-y.float()[v]).abs().mean()
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite masked loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), int(v.sum())


@torch.no_grad()
def evaluate(model, ds, device, path=None):
    model.eval(); sums=np.zeros((2,3),np.float64)
    horizon=96 if ds.dense_out else 12
    saved=None if path is None else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ds),horizon,2800,1))
    for x,y,v,ii in DataLoader(ds,batch_size=4,shuffle=False):
        with amp(device): pred=model(x.to(device))
        p=pred.float().cpu().numpy()*np.float32(ds.scale)
        assert np.isfinite(p).all()
        yy=y.numpy(); vv=v.numpy()
        for j,sel in enumerate([slice(None),COMMON if ds.dense_out else slice(None)]):
            valid=vv[:,sel]; e=p[:,sel][valid].astype(np.float64)-yy[:,sel][valid].astype(np.float64)
            sums[j] += [np.square(e).sum(),np.abs(e).sum(),e.size]
        if saved is not None: saved[ii.numpy()]=p
    if saved is not None: saved.flush()
    def metrics(s): return dict(RMSE=float(np.sqrt(s[0]/s[2])),MAE=float(s[1]/s[2]),sse=float(s[0]),sae=float(s[1]),n_valid=int(s[2]))
    return dict(native=metrics(sums[0]),common12=metrics(sums[1]))


def smoke_rows(root):
    records={json.loads(p.read_text())['date']:json.loads(p.read_text()) for p in (root/'data/records').glob('*/*.json')}
    days=['2014-01-01','2014-01-02','2014-01-03']
    return [dict(day=days[2],role='train',included=True,origin=int(np.datetime64(days[2],'s').astype(np.int64)),
                 previous2=records[days[0]]['output'],history=records[days[1]]['output'],target=records[days[2]]['output'])]


def smoke(root, cuda=False):
    device=torch.device('cuda' if cuda else 'cpu'); archive=Archive(root); rows=smoke_rows(root)
    # Compare the two head shapes in full FP32. cuDNN TF32 otherwise uses
    # shape-dependent rounding despite identical shared weights.
    if cuda:
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    checks={}; initial=[]
    for arm in ARMS:
        seed_all(42); ds=Windows(archive,'train',arm,rows)
        m=model_for(root,arm,device,42); batch=list(next(iter(DataLoader(ds,batch_size=1))))
        batch[1]=batch[1].clone(); batch[2]=batch[2].clone()
        batch[1][0,0,0,0]=float('nan'); batch[2][0,0,0,0]=False
        opt=torch.optim.Adam(m.parameters(),lr=.003)
        m.eval()
        with torch.no_grad(): before=m(batch[0].to(device)).float().cpu()
        initial.append(before[:,COMMON] if arm.startswith('B') else before)
        m.train(); loss,n=step(m,batch,opt,ds.scale,device)
        checks[arm]=dict(loss=loss,n_valid=n,shape=list(before.shape),parameters=sum(p.numel() for p in m.parameters()))
        if arm.endswith('native'):
            m.eval(); x=batch[0].to(device).detach().requires_grad_(True)
            grad=torch.autograd.grad(m(x).square().mean(),x)[0]
            assert torch.all(grad[:,:,:,0].abs().sum((0,2))>0)
            checks[arm]['all_89_TEC_frames_influence_output']=True
        del m,opt,batch;gc.collect()
        if cuda: torch.cuda.empty_cache()
    # Across routes, identical input yields identical initial common12 output.
    torch.testing.assert_close(initial[0],initial[2],rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(initial[1],initial[3],rtol=1e-5,atol=1e-7)
    row=rows[0]; y,v=archive.target(row)
    with np.load(root/'data'/row['target']) as z:
        np.testing.assert_array_equal(y[COMMON],z['tec'][::8].reshape(12,2800))
        np.testing.assert_array_equal(v[COMMON],z['valid'][::8].reshape(12,2800))
    assert np.isfinite(archive.persistence(row)).all()
    write(root/('GPU_SMOKE.json' if cuda else 'CPU_SMOKE.json'),dict(status='passed',checks=checks,
        common_time_initial_predictions_match=True,comparison_precision='FP32 without TF32',source_sha256=sha(Path(__file__))))


def benchmark(root):
    device=torch.device('cuda'); archive=Archive(root); rows=smoke_rows(root)
    ds=Windows(archive,'train','B_native',rows*32)
    b=next(iter(DataLoader(ds,batch_size=32)))
    m=model_for(root,'B_native',device,42); opt=torch.optim.Adam(m.parameters(),lr=.003)
    torch.cuda.reset_peak_memory_stats(); durations=[]
    for i in range(4):
        start=time.time();m.train(); step(m,b,opt,ds.scale,device)
        torch.cuda.synchronize();durations.append(time.time()-start)
    info=dict(status='passed',batch_size=32,seconds_per_train_step=float(np.mean(durations[1:])),
        step_seconds=durations,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda)
    write(root/'BENCHMARK.json',info); print(json.dumps(info),flush=True)


def train(root,arm,seed):
    device=torch.device('cuda'); seed_all(seed)
    out=root/'runs'/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    archive=Archive(root); tr=Windows(archive,'train',arm); va=Windows(archive,'validation',arm)
    model=model_for(root,arm,device,seed)
    opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10,min_lr=0.)
    seed_all(seed)
    dl=DataLoader(tr,batch_size=32,shuffle=True,drop_last=False,generator=torch.Generator().manual_seed(seed))
    cfg=dict(arm=arm,seed=seed,protocol=protocol,protocol_sha256=sha(root/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),
        initial_sha256=sha(root/f'shared/initial_{seed}.pt'),preprocessing_identity_sha256=sha(root/'shared/identity.json'),
        windows_sha256=sha(root/'data/WINDOWS.jsonl'),train_windows=len(tr),validation_windows=len(va),
        parameters=sum(p.numel() for p in model.parameters()),torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    write(out/'config.json',cfg); best=float('inf'); selected=None; start=time.time()
    with (out/'epochs.jsonl').open('x') as log:
        for epoch in range(1,201):
            model.train();total=0.;count=0; lr=opt.param_groups[0]['lr']
            for b in dl:
                loss,n=step(model,b,opt,tr.scale,device); total+=loss*n;count+=n
            val=evaluate(model,va,device); score=val['common12']['RMSE']; scheduler.step(score)
            if score<best:
                best=score;selected=epoch
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_common12_rmse=best),out/'best.pt')
            row=dict(epoch=epoch,training_MAE=total/count,validation=val,lr=lr,selected_epoch=selected,elapsed_seconds=time.time()-start)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
    state=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(state['state_dict'],strict=True)
    val=evaluate(model,va,device,out/'validation_predictions.npy')
    assert abs(val['common12']['RMSE']-best)<1e-9
    # Replay is accessed by the model only after checkpoint and policy are frozen.
    del tr,va,dl,archive;gc.collect()
    replay=Windows(Archive(root),'replay',arm)
    metrics=evaluate(model,replay,device,out/'replay_predictions.npy')
    write(out/'complete.json',dict(status='complete',arm=arm,seed=seed,epochs=200,selected_epoch=selected,
        validation=val,replay=metrics,files={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','smoke','gpu-smoke','benchmark','train'])
    ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--seed',type=int,choices=SEEDS)
    a=ap.parse_args()
    if a.action=='prepare':prepare_shared(a.root)
    elif a.action in ['smoke','gpu-smoke']:smoke(a.root,a.action=='gpu-smoke')
    elif a.action=='benchmark':benchmark(a.root)
    else:train(a.root,a.arm,a.seed)
