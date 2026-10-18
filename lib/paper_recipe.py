"""Portable execution of the retained GIM recipes through model/Run.py.

Legacy adaptive runs keep the original trainer. The matched projection runs
use their saved direct-output architecture, native auxiliary and selection rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import copy
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import yaml
from lib.normalization import MinMax01Scaler
from lib.physics.driver_scaler import DRIVER_CHANNEL_ORDER, DriverHistoryBatch, FrozenDriverScaler, validate_history_visibility
from lib.paper_statistics import horizon_statistics


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Windows(Dataset):
    """Daily origins and complete targets within one continuous segment."""
    def __init__(self, values_tecu, first_time, horizon, scale=147.1):
        self.values = values_tecu
        self.first_time = int(np.datetime64(first_time, 's').astype(np.int64))
        self.horizon = horizon
        self.scale = np.float32(scale)
    def __len__(self):
        return max(0, (len(self.values)-12-self.horizon)//12+1)
    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        j = i*12
        x = np.asarray(self.values[j:j+12], dtype=np.float32)/self.scale
        y = np.asarray(self.values[j+12:j+12+self.horizon], dtype=np.float32)/self.scale
        return torch.from_numpy(x[..., None]), torch.from_numpy(y[..., None]), self.first_time+(j+12)*7200


def read_year(root, year, dataset="gim"):
    """Published yearly GIM arrays use 0.1 TECU; network inputs use TECU/147.1."""
    path = Path(root)/f'TEC_{year}.npy'
    if not path.exists():
        path = Path(root)/f'TEC_context_{year}_12_31.npy'
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    return np.asarray(a, dtype=np.float32).reshape(-1, 5183 if dataset=="gim" else 2911)/np.float32(10 if dataset=="gim" else 1)


def datasets(root, years, horizon, training=False, dataset="gim"):
    scale=147.1 if dataset=="gim" else 300.
    segments = []
    if training:
        groups = [[2009, 2010, 2011, 2012], [2014], [2016], [2018]] if dataset=="gim" else [[y] for y in [2014,2016,2018,2020,2022,2023]]
        for group in groups:
            values = np.concatenate([read_year(root, y, dataset) for y in group])
            segments.append(Windows(values, f'{group[0]}-01-01', horizon, scale))
    else:
        for year in years:
            prefix = read_year(root, year-1, dataset)[-12:]
            values = np.concatenate([prefix, read_year(root, year, dataset)])
            segments.append(Windows(values, f'{year-1}-12-31', horizon, scale))
    return ConcatDataset(segments)


class Drivers:
    """Read exact timestamp slices; standardization never refits at evaluation."""
    def __init__(self, path, scaler_path, device, dataset='gim'):
        from lib.physics.drivers_projected import DriversStore
        self.nodes = 5183 if dataset=="gim" else 2911
        self.source_sha = sha256(path)
        self.store = DriversStore(str(path), device=device, strict=True, expected_sha256=self.source_sha)
        if getattr(self.store, 'synthetic', False):
            raise ValueError('real drivers are required')
        self.time = np.asarray(self.store.time).astype('datetime64[s]').astype(np.int64)
        self.device = device
        self.scaler = None
        if scaler_path:
            # A separately supplied later-year archive uses the original training statistics.
            meta = json.loads(Path(scaler_path).read_text())
            self.scaler = FrozenDriverScaler.load(scaler_path, expected_file_sha256=sha256(scaler_path),
                expected_source_sha256=meta['source_driver_sha256'], expected_dataset=dataset)
    def raw(self, times, steps):
        times = np.asarray(times, dtype=np.int64)
        start = np.searchsorted(self.time, times)
        index = start[:, None] + np.arange(steps)[None, :]
        if index.max() >= len(self.time) or not np.array_equal(self.time[index], times[:, None]+np.arange(steps)*7200):
            raise ValueError('driver timestamp window is unavailable')
        return self.store.slice_batched(start, steps, self.nodes)
    def history(self, origins):
        origins = np.asarray(origins, dtype=np.int64)
        times = origins[:, None]-np.arange(12,0,-1)*7200
        validate_history_visibility(times, origins, expected_history_steps=12)
        raw = self.raw(origins-12*7200, 12)
        values = torch.stack([raw[k][:,:,None].expand(-1,-1,self.nodes) if raw[k].ndim==2 else raw[k] for k in DRIVER_CHANNEL_ORDER], -1)
        if self.scaler is None:
            raise ValueError('a saved training-year driver scaler is required')
        # Fit-source identity remains in the scaler; input-source identity describes
        # the archive actually read, including separately assembled later years.
        means = torch.as_tensor(self.scaler.means, dtype=values.dtype, device=values.device)
        stds = torch.as_tensor(self.scaler.standard_deviations, dtype=values.dtype, device=values.device)
        values = (values-means)/stds
        return DriverHistoryBatch(values, torch.as_tensor(times), torch.as_tensor(origins), DRIVER_CHANNEL_ORDER,
            self.source_sha, self.scaler.file_sha256, 'train_zscore')


def build_predictor(config, adjacency, device, drivers=None, load_only=False):
    model = config.get('model', 'GWN')
    if model == 'GNNTrans':
        from model.GNNTrans.gnntrans import GNNTransPyramid
        return GNNTransPyramid().to(device)
    if model == 'Mixer':
        from lib.external_mixer_model import Mix4SimVP
        return Mix4SimVP().to(device)
    from model.GWN.GWN import GWNET
    from model.GWN.driver_fusion import required_film_config
    fusion = config['driver_fusion']
    args = SimpleNamespace(adj_mx=adjacency, num_nodes=len(adjacency), dropout=.3,
        blocks=4, layers=2, gcn_bool=True, addaptadj=True, adjtype='doubletransition',
        randomadj=load_only, aptonly=False, kernel_size=2, nhid=32, residual_channels=32,
        dilation_channels=32, input_window=12, output_window=config['horizon'], driver_fusion=fusion,
        input_mask=config.get('input_mask'), driver_scaling='train_zscore',
        predictor_driver_channels=list(DRIVER_CHANNEL_ORDER),
        driver_source_sha256=drivers.source_sha if drivers and drivers.scaler else config.get('driver_source_sha256'),
        driver_scaler_sha256=drivers.scaler.file_sha256 if drivers and drivers.scaler else config.get('driver_scaler_sha256'),
        **required_film_config())
    return GWNET(args, device, 6 if fusion=='zscore_concat' else 1, 1).to(device)


def build_physics(adjacency, device):
    from lib.physics.ppinn_projected import PPinnLoss
    module = PPinnLoss(MinMax01Scaler(0,147.1), torch.as_tensor(adjacency, device=device),
        dataset='gim', normalizer_id='gim_tec01_147p1_v1', model_key='pi_driver',
        use_chemistry=True, use_night=True, use_dynamic_rot_cap=True, use_advection=False).to(device)
    for p in module.parameters():
        p.requires_grad_(False)
        p.data.zero_()
    return module


def predict(model, config, x, origins, drivers):
    if config.get('model') == 'Mixer':
        from lib.external_mixer import prepare_mix4_history, restore_gim_tecu
        times = (np.asarray(origins)[:,None]-np.arange(12,0,-1)*7200).astype('datetime64[s]')
        h = (x*147.1).reshape(-1,12,1,71,73)
        return restore_gim_tecu(model(prepare_mix4_history(h,times))).reshape(-1,12,5183,1)/147.1
    history = drivers.history(origins) if config['use_drivers_input'] else None
    if config.get('model') == 'GNNTrans':
        return model(x, history.values)
    return model(x, history) if history is not None else model(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-config', required=True)
    parser.add_argument('--action', choices=['evaluate','train'], default='evaluate')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--adjacency', required=True)
    parser.add_argument('--drivers')
    parser.add_argument('--driver-scaler')
    parser.add_argument('--checkpoint')
    parser.add_argument('--years', type=int, nargs='+', default=[2015,2019,2020,2021,2022])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    a = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(a.paper_config).read_text())
    dataset=cfg.get('dataset','gim'); scale=147.1 if dataset=='gim' else 300.
    nodes=5183 if dataset=='gim' else 2911
    def data(years, training=False):
        return datasets(a.data_root,years,cfg['horizon'],training=training,dataset=dataset)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device = torch.device(a.device)
    adjacency = np.load(a.adjacency, allow_pickle=False)
    drivers = Drivers(a.drivers, a.driver_scaler, device, dataset=dataset) if cfg['use_drivers_input'] or (a.action=='train' and cfg.get('rho',0)) else None
    model = build_predictor(cfg, adjacency, device, drivers, load_only=a.action=='evaluate')
    batch = int(cfg['batch_size']) if a.action=='train' else int(cfg.get('eval_batch_size',1))
    def loader(ds, shuffle=False, offset=0):
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=torch.Generator().manual_seed(a.seed+offset))
    def amp():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda' and cfg.get('bf16',True))
    @torch.no_grad()
    def evaluate(ds, save=False, predictor=None):
        predictor=model if predictor is None else predictor
        predictor.eval(); sse=0.; n=0
        if save:
            shape=(len(ds),cfg['horizon'],nodes,1)
            arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype=np.float32,shape=shape) for k in ['predictions','truth']}
            origins_all=[]; position=0
        for x,y,origins in loader(ds):
            x,y=x.to(device),y.to(device)
            with amp(): pred=predict(predictor,cfg,x,origins,drivers)
            if not torch.isfinite(pred).all(): raise ValueError('nonfinite prediction')
            mask=torch.isfinite(y); error=(pred.float()*scale-y.float()*scale).double()
            sse+=float(error[mask].square().sum()); n+=int(mask.sum())
            if save:
                count=len(x); arrays['predictions'][position:position+count]=(pred.float()*scale).cpu().numpy()
                arrays['truth'][position:position+count]=(y.float()*scale).cpu().numpy()
                origins_all.extend(origins.tolist()); position+=count
        if save:
            for arr in arrays.values(): arr.flush()
            np.save(out/'origins.npy',np.asarray(origins_all,dtype=np.int64))
            mask=np.isfinite(arrays['truth']); np.save(out/'valid.npy',mask)
            (out/'metrics.json').write_text(json.dumps(horizon_statistics(arrays['predictions'],arrays['truth'],mask),indent=2))
        return float(np.sqrt(sse/n))
    started=time.perf_counter()
    if a.action=='evaluate':
        if not a.checkpoint: parser.error('--checkpoint is required for evaluation')
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint_weights(state),strict=True)
        evaluate(data(a.years),save=True)
    else:
        if cfg.get('model','GWN') == 'Mixer':
            parser.error('external comparators use the saved published/adapted weights; see their configuration notes')
        from lib.physics.projection import assign_gradients
        physics=build_physics(adjacency,device) if cfg.get('rho',0) else None
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg.get('weight_decay',0.))
        schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10,min_lr=0.)
        tr=loader(data([],training=True),True)
        va=data([2013,2017] if dataset=='gim' else [2017,2021]); best=float('inf'); stale=0
        is_gnn=cfg.get('model')=='GNNTrans'
        ema=copy.deepcopy(model).eval() if is_gnn else None
        if is_gnn:
            for parameter in ema.parameters(): parameter.requires_grad_(False)
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.1,betas=(.9,.999),eps=1e-8)
        for epoch in range(1,int(cfg['epochs'])+1):
            model.train()
            if is_gnn:
                e=epoch-1
                lr=2e-4*(e+1)/20 if e<20 else (1e-6 if e>=970 else 1e-6+(2e-4-1e-6)*.5*(1+math.cos(math.pi*(e-20)/950)))
                for group in optimizer.param_groups: group["lr"]=lr
            for x,y,origins in tr:
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                with amp(): pred=predict(model,cfg,x,origins,drivers)
                valid=torch.isfinite(y)
                loss=supervised_mae(pred.float()*scale,y.float()*scale,valid,dataset)
                physical=None
                if physics is not None:
                    _,terms=physics(pred.float(),None,drivers=drivers.raw(origins,cfg['horizon']))
                    physical=.5*sum(terms.values())
                assign_gradients(loss,physical,list(model.parameters()),float(cfg.get('rho',0)))
                if not is_gnn:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                if is_gnn:
                    with torch.no_grad():
                        for e,p in zip(ema.parameters(),model.parameters()): e.mul_(.99996).add_(p,alpha=.00004)
            score=evaluate(va,predictor=ema); stale+=1
            if not is_gnn: schedule.step(score)
            state=dict(state_dict=(ema if is_gnn else model).state_dict(),candidate_config=cfg,seed=a.seed,epoch=epoch,validation_rmse=score)
            if score < best:
                best=score; stale=0; torch.save(state,out/'best.pt')
            with (out/'epochs.jsonl').open('a') as handle: handle.write(json.dumps(dict(epoch=epoch,validation_rmse=score))+'\n')
            if cfg.get('early_stop_patience') and stale>=cfg['early_stop_patience'] and (not is_gnn or epoch>=20): break
        torch.save(state,out/'last.pt')
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg,seed=a.seed,years=a.years if a.action=='evaluate' else None,
        action=a.action,torch=torch.__version__,numpy=np.__version__,elapsed_seconds=time.perf_counter()-started,
        checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None),indent=2))


def checkpoint_weights(state):
    """Read the tensor container used by saved native and legacy checkpoints."""
    for key in ('state_dict', 'model_state_dict', 'model'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if state and all(k.startswith("predictor.") for k in state):
        state = {k[len("predictor."):]: v for k, v in state.items()}
    return state


def supervised_mae(pred, truth, valid, dataset="gim"):
    error=(pred-truth).abs()
    if dataset=="china":
        lower=torch.tensor(300.-.001,dtype=torch.float32,device=truth.device)
        upper=torch.tensor(300.+.001,dtype=torch.float32,device=truth.device)
        saturated=(truth>=lower)&(truth<=upper)
        error=torch.where(saturated,error*.3,error)
    return error[valid].mean()
