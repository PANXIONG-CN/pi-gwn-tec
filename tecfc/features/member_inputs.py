"""Daily causal information channels; historical TEC labels remain role restricted."""
import json
from pathlib import Path
import types
import numpy as np
import torch
from torch.utils.data import Dataset
from tecfc.features.geometry import signed_cos_sza

ARMS=('G-DE','G-long','G-geo')
CHANNELS={'G':1,'G-DE':6,'G-long':4,'G-geo':6}
DRIVER_ORDER=('AE','Dst','Kp','F107','cosSZA')
TRAIN=(2009,2010,2011,2012,2014,2016,2018)
DEV=set(TRAIN+(2013,2017))
LAT,LON=np.meshgrid(np.linspace(87.5,-87.5,71,dtype=np.float64),np.arange(-180,180,5,dtype=np.float64),indexing='ij')
LAT,LON=LAT.reshape(-1),LON.reshape(-1)


def history_times(day,lag=1):
    origin=int(np.datetime64(day,'s').astype(np.int64))
    ts=origin-lag*86400+np.arange(12,dtype=np.int64)*7200
    if not np.all(ts<=origin-7200):raise ValueError('Future history')
    return ts


def geometry(times):
    times=np.asarray(times,np.int64);dt=times.astype('datetime64[s]')
    first=dt.astype('datetime64[Y]');next_year=first+np.timedelta64(1,'Y')
    seconds=(dt-first.astype('datetime64[s]')).astype(np.int64).astype(np.float64)
    seconds_year=(next_year.astype('datetime64[s]')-first.astype('datetime64[s]')).astype(np.int64).astype(np.float64)
    season=2*np.pi*seconds/seconds_year
    local=np.mod(np.mod(times,86400).astype(np.float64)[:,None]/3600+LON[None,:]/15,24)
    angle=2*np.pi*local/24
    out=np.empty((len(times),5112,5),np.float32)
    out[:,:,0]=np.sin(season)[:,None];out[:,:,1]=np.cos(season)[:,None]
    out[:,:,2]=np.sin(angle);out[:,:,3]=np.cos(angle)
    out[:,:,4]=signed_cos_sza(times,LON,LAT)
    return out.reshape(len(times),71,72,5)


class Drivers:
    def __init__(self,config):
        self.config=config
        with np.load(config['driver']['path'],allow_pickle=False) as z:
            self.time=np.array(z['time'],np.int64)
            self.values={k:np.array(z[k],np.float32) for k in DRIVER_ORDER[:-1]}
        if not np.all(np.diff(self.time)==7200) or self.time[-1]>=int(np.datetime64('2026-01-01','s').astype(np.int64)):
            raise ValueError('Unexpected driver grid')
        scaler=json.loads(Path(config['scaler']['path']).read_text())
        if scaler['grid_shape']!=[71,72] or scaler['channel_order']!=list(DRIVER_ORDER):raise ValueError('Wrong driver scaler')
        self.mean=np.array([scaler['channels'][k]['mean'] for k in DRIVER_ORDER],np.float32)
        self.std=np.array([scaler['channels'][k]['standard_deviation'] for k in DRIVER_ORDER],np.float32)
    def raw(self,times):
        ix=np.searchsorted(self.time,times)
        if np.any(ix>=len(self.time)) or not np.array_equal(self.time[ix],times):raise ValueError('Missing driver history')
        out=np.empty((len(times),5112,5),np.float32)
        for j,k in enumerate(DRIVER_ORDER[:-1]):out[:,:,j]=self.values[k][ix,None]
        out[:,:,4]=np.clip(signed_cos_sza(times,LON,LAT),0.,1.)
        if not np.isfinite(out).all():raise ValueError('Missing driver value')
        return out.reshape(len(times),71,72,5)
    def standardized(self,times):return (self.raw(times)-self.mean)/self.std


def member_forward(self,history,restore=True):
    if history.ndim!=5 or tuple(history.shape[1:])!=(12,71,72,self.input_channels):raise ValueError('Wrong member input')
    p=self.gwn(history.reshape(history.shape[0],12,5112,self.input_channels)).reshape(history.shape[0],12,71,72)
    return torch.cat((p,p[...,:1]),dim=-1) if restore else p


def build_model(architecture,device,arm):
    if arm=='G':return architecture.GlobalGWN(device)
    dim=CHANNELS[arm];original=architecture.GWNET
    # Only constructor input width changes; all original RNG/order/graph code runs.
    def constructor(args,device,dim_in,dim_out):return original(args,device,dim_in=dim,dim_out=dim_out)
    try:
        architecture.GWNET=constructor
        model=architecture.GlobalGWN(device)
    finally:architecture.GWNET=original
    model.input_channels=dim;model.forward=types.MethodType(member_forward,model)
    return model


class Windows(Dataset):
    def __init__(self,role,config,gim,evaluation=False):
        self.config=config;self.role=role;self.arm=config['arm'];self.scale=147.1;self.restore=gim.restore_grid
        self.evaluation=evaluation
        if role not in (('validation','original_evaluation','extension_evaluation') if evaluation else ('train','validation')):
            raise ValueError('Forbidden role')
        files=config['data_files'];allowed=set(range(2009,2026)) if evaluation else DEV
        class RestrictedGIM(gim.GIM):
            def path(self,year):
                if type(year) is not int or year not in allowed:raise ValueError('Forbidden label year')
                return Path(files[str(year)]['path'])
        self.gim=RestrictedGIM();original=list(self.gim.windows(role));self.dropped=[]
        self.days=[];self.history_access=set();self.history_only_cache={}
        for day in original:
            lags=(1,2,3,27) if self.arm=='G-long' else (1,)
            if any(str(np.datetime64(day)-np.timedelta64(lag,'D'))<'2009-01-01' for lag in lags):
                if role!='train':raise ValueError('Incomplete evaluation history')
                self.dropped.append(day);continue
            for lag in lags:
                ts=history_times(day,lag);hday=str(ts[0].astype('datetime64[s]').astype('datetime64[D]'))
                self.history_access.add(hday)
                if int(hday[:4]) not in allowed:
                    row=config['history_only_files'].get(hday[:4])
                    if row is None or hday not in row['allowed_dates']:raise ValueError('History date not authorized: '+hday)
            self.days.append(day)
        expected={'train':2527 if self.arm=='G-long' else 2553,'validation':730,'original_evaluation':1826,'extension_evaluation':1096}[role]
        if len(self.days)!=expected:raise ValueError('Changed windows')
        self.drivers=Drivers(config) if self.arm=='G-DE' else None
    def __len__(self):return len(self.days)
    def history_day(self,day,cutoff):
        times=history_times(str(np.datetime64(day)+np.timedelta64(1,'D')))
        if np.any(times>cutoff):raise ValueError('Noncausal TEC history')
        year=int(day[:4])
        if year in (set(range(2009,2026)) if self.evaluation else DEV):return self.gim.day(day)
        row=self.config['history_only_files'].get(str(year))
        if row is None or day not in row['allowed_dates']:raise ValueError('History-only date forbidden')
        if year not in self.history_only_cache:
            self.history_only_cache[year]=np.load(row['path'],mmap_mode='r',allow_pickle=False)
        a=self.history_only_cache[year]
        index=int((np.datetime64(day)-np.datetime64(f'{year}-01-01')).astype(int))
        x=np.asarray(a[index],np.float32)
        if not np.array_equal(x[...,0],x[...,-1]):raise ValueError('History seam differs')
        return x[...,:72]/np.float32(10)
    def __getitem__(self,index):
        day=self.days[index];sample=self.gim.sample(day);cutoff=sample['cutoff'];times=history_times(day)
        x=np.array(sample['history']/np.float32(self.scale),np.float32)
        if self.arm=='G-long':
            parts=[x]
            for lag in (2,3,27):
                hist=str(np.datetime64(day)-np.timedelta64(lag,'D'))
                parts.append(self.history_day(hist,cutoff)/np.float32(self.scale))
            x=np.stack(parts,axis=-1)
        elif self.arm=='G-DE':x=np.concatenate((x[...,None],self.drivers.standardized(times)),axis=-1)
        elif self.arm=='G-geo':x=np.concatenate((x[...,None],geometry(times)),axis=-1)
        if not np.isfinite(x).all():raise ValueError('Nonfinite history')
        return dict(history=torch.from_numpy(x),target=torch.from_numpy(np.array(self.restore(sample['target']),np.float32)),
            valid=torch.from_numpy(np.array(self.restore(sample['valid']),bool)),origin=int(sample['origin']),index=index)


def validate_config(config):
    from tecfc.training import global_train as base
    base.validate_config(config)
    if config['arm'] not in CHANNELS:raise ValueError('Foreign member')
    if set(config['history_only_files'])!={'2015'}:raise ValueError('History whitelist changed')
    expected=[str(x) for x in np.arange(np.datetime64('2015-12-06'),np.datetime64('2016-01-01'))]
    if config['history_only_files']['2015']['allowed_dates']!=expected:raise ValueError('2015 history dates changed')
    if config['recipe']['epochs']!=200 or config['recipe']['scale']!=147.1:raise ValueError('Recipe changed')


def verify_assets(config,base):
    base.verify_assets(config)
    for key in ('driver','scaler'):
        if base.sha(config[key]['path'])!=config[key]['sha256']:raise ValueError('Changed '+key)
    for row in config['history_only_files'].values():
        if base.sha(row['path'])!=row['sha256']:raise ValueError('Changed history source')
