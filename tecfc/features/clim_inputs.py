"""Frozen exp0016 inputs and CG adapter; importing this module opens no data."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.dont_write_bytecode = True
from tecfc.training import global_train as base

ARMS = ('G-MAE', 'C-in')
GEOMETRY = dict(
    formula='NOAA fractional-year Fourier solar declination and equation of time',
    precision='float64 calculation, one cast to float32; signed, no daylight clipping',
    gamma='2*pi/N_year*(zero_based_DOY+(UTC_hour-12)/24)',
    declination='0.006918-0.399912*cos(g)+0.070257*sin(g)-0.006758*cos(2g)+0.000907*sin(2g)-0.002697*cos(3g)+0.00148*sin(3g)',
    equation_of_time_minutes='229.18*(0.000075+0.001868*cos(g)-0.032077*sin(g)-0.014615*cos(2g)-0.040849*sin(2g))',
    hour_angle='radians((UTC_minutes+equation_of_time_minutes+4*longitude)/4-180)',
    cosSZA='sin(latitude)*sin(declination)+cos(latitude)*cos(declination)*cos(hour_angle)',
    latitude='87.5,85,...,-87.5 degrees north (step -2.5)',
    longitude='-180,-175,...,175 degrees east (72 unique columns)',
    history='D-1 00,02,...,22 UTC',
    alpha='float32(clip(sum(history[finite],float64)/sum(climate[finite],float64),0.3,3))')


def solar(day):
    day = np.datetime64(day, 'D')
    start = day.astype('datetime64[Y]').astype('datetime64[D]')
    stop = (day.astype('datetime64[Y]') + np.timedelta64(1, 'Y')).astype('datetime64[D]')
    n = int((stop-start).astype(int)); doy = int((day-start).astype(int))
    hour = np.arange(0, 24, 2, dtype=np.float64)
    g = 2*np.pi/n*(doy+(hour-12)/24)
    dec = (0.006918-0.399912*np.cos(g)+0.070257*np.sin(g)
           -0.006758*np.cos(2*g)+0.000907*np.sin(2*g)
           -0.002697*np.cos(3*g)+0.00148*np.sin(3*g))
    eq = 229.18*(0.000075+0.001868*np.cos(g)-0.032077*np.sin(g)
                -0.014615*np.cos(2*g)-0.040849*np.sin(2*g))
    lat = np.deg2rad(87.5-2.5*np.arange(71, dtype=np.float64))[None,:,None]
    lon = (-180+5*np.arange(72, dtype=np.float64))[None,None,:]
    ha = np.deg2rad((hour[:,None,None]*60+eq[:,None,None]+4*lon)/4-180)
    dec = dec[:,None,None]
    return (np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha)).astype(np.float32)


@contextmanager
def cg_modules(root):
    root = Path(root)
    names = ('periodic', 'attentions', 'simvp', 'cg_model', '_exp0016_cg')
    previous = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    try:
        for name, file in zip(names, ('periodic.py', 'shared/attentions.py', 'shared/simvp.py',
                                     'shared/cg_model.py', 'global_model.py')):
            module = base.load_module(name, root/file)
        class CGInput(module.GlobalAmplitudeConditionedSimVP):
            # Same modules, parameters, initialization and operation order as C.
            # Only the supplied three-channel tensor changes; no future add-back.
            def forward(self, x, restore=False):
                if x.ndim != 5 or x.shape[1:] != (12,3,71,72):
                    raise ValueError('C input must be [batch,12,3,71,72]')
                b,t,c,h,w = x.shape
                enc = self.attn(self.spatial_enc(x.reshape(b*t,c,h,w)))
                _,hidden,hh,ww = enc.shape
                compressed = self.time_adapter(enc.reshape(b,t,hidden,hh,ww).permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()
                latent = self.temporal(compressed.reshape(b,self.compressed_frames*hidden,hh,ww))
                pred = self.time_proj(latent).reshape(b*self.output_frames,hidden,hh,ww)
                pred = module.restore_height(self.spatial_dec(pred)).reshape(b,self.output_frames,71,72)
                return module.restore_longitude(pred) if restore else pred
        yield CGInput
    finally:
        sys.path[:] = old_path
        for name, previous_module in previous.items():
            if previous_module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = previous_module


@contextmanager
def modules(config, device):
    with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, gwn):
        if config['arm'] in ('G-control', 'G-MAE'):
            yield gim, lambda: gwn.GlobalGWN(device)
        else:
            with cg_modules(config['cg_reference_root']) as cls:
                yield gim, cls


class Windows(Dataset):
    def __init__(self, role, config, gim, *, inference=False):
        roles = ('validation','original_evaluation','extension_evaluation') if inference else ('train','validation')
        if role not in roles: raise ValueError('Forbidden role: '+role)
        expected = set(range(2009,2026)) if inference else set(base.ALLOWED_YEARS)
        if set(map(int, config['data_files'])) != expected: raise ValueError('Year whitelist mismatch')
        files = config['data_files']
        class Restricted(gim.GIM):
            def path(self, year):
                year = int(year)
                if year not in expected or year > 2025: raise ValueError('Forbidden year')
                return Path(files[str(year)]['path'])
        self.gim = Restricted(); self.days = list(self.gim.windows(role)); self.role = role
        self.restore_grid = gim.restore_grid; self.arm = config['arm']; self.scale = 147.1
        with np.load(Path(config['asset_root'])/'prepared/train_climatology_v1.npz', allow_pickle=False) as z:
            if float(z['scale']) != self.scale: raise ValueError('Changed scale')
            self.climate = z['climatology'].copy() if self.arm == 'C-in' else None
        if len(self.days) != dict(train=2553,validation=730,original_evaluation=1826,extension_evaluation=1096)[role]:
            raise ValueError('Changed windows')
        for day in self.days:
            origin = int(np.datetime64(day,'s').astype(np.int64))
            times = origin-86400+np.arange(12,dtype=np.int64)*7200
            assert np.all(times <= origin-7200) and times[-1] == origin-7200
            for date in (np.datetime64(day,'D'), np.datetime64(day,'D')-np.timedelta64(1,'D')):
                self.gim.path(int(str(date)[:4]))

    def __len__(self): return len(self.days)

    def __getitem__(self, index):
        day = np.datetime64(self.days[index], 'D'); previous = day-np.timedelta64(1,'D')
        sample = self.gim.sample(day)  # Never request background_future from GIM.
        history = np.array(sample['history']/np.float32(self.scale), dtype=np.float32)
        if self.arm in ('C-in','C-control'):
            if self.arm == 'C-control':
                background, geom = np.zeros_like(history), np.zeros_like(history)
            else:
                climate = self.climate[int(str(previous)[5:7])-1]
                valid = np.isfinite(sample['history'])
                numerator = sample['history'][valid].sum(dtype=np.float64)
                denominator = climate[valid].sum(dtype=np.float64)
                if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError('Invalid historical alpha support')
                alpha = np.float32(np.clip(numerator/denominator,.3,3.))
                background = np.array(climate*alpha/np.float32(self.scale),np.float32)
                geom = solar(previous)
            history = np.stack((history, background, geom),axis=1)
        return dict(history=torch.from_numpy(history),
                    target=torch.from_numpy(np.array(self.restore_grid(sample['target']),np.float32)),
                    valid=torch.from_numpy(np.array(self.restore_grid(sample['valid']),bool)),
                    origin=int(sample['origin']),index=index)


def scientific_config(config):
    result = base.scientific_config(config)
    result.pop('cg_reference_root')
    for row in result['reference_epochs'].values(): row.pop('path')
    return result


def validate(config, *, inference=False):
    if config['schema'] != 'exp0016_fixed200_v1' or config['arm'] not in ARMS+('G-control','C-control'):
        raise ValueError('Unregistered configuration')
    if config['seed'] not in base.SEEDS or config['geometry'] != GEOMETRY:
        raise ValueError('Changed seed/geometry')
    if config['loss'] != ('mae' if config['arm']=='G-MAE' else 'smooth_l1'):
        raise ValueError('Changed supervised loss')
    expected = set(range(2009,2026)) if inference else set(base.ALLOWED_YEARS)
    if set(map(int,config['data_files'])) != expected: raise ValueError('Changed role access')
    for key, root in (('reference_sha256','reference_root'),('cg_reference_sha256','cg_reference_root'),('asset_sha256','asset_root')):
        for name, h in config[key].items():
            if base.sha(Path(config[root])/name) != h: raise ValueError('Changed asset '+name)
    for row in list(config['data_files'].values())+list(config['reference_epochs'].values()):
        if base.sha(row['path']) != row['sha256']: raise ValueError('Changed explicit input')
