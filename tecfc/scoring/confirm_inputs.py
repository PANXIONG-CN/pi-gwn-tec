"""Bind final scoring to immutable predictions and original product arrays."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from tecfc.scoring.confirm_records import ROOT, EXP, SEEDS, sha, write, read
from tecfc.ensemble.apply import mapping, remap

FAMILIES=['G','G-plus-M','G-geo','PI','EMP','C','C-in','C-plus']
SINGLE=FAMILIES[:5]


class Reader:
    def __init__(self,desc):
        self.desc=desc;self.cache={};self.lookups={};self.maps={}
        self.ts=self.array(desc['origins']);self.dates=self.ts.astype('datetime64[s]').astype('datetime64[D]')
        self.months=self.dates.astype('datetime64[M]').astype(int)%12+1
        self.selected=np.flatnonzero(self.months<=desc['max_month'])
    def array(self,spec):
        path=spec['path']
        if path not in self.cache:
            assert sha(path)==spec['sha256'],'Scoring input hash mismatch: '+path
            self.cache[path]=np.load(path,mmap_mode='r',allow_pickle=False)
        return self.cache[path]
    def raw(self,years,day):
        year=str(day)[:4];ix=int((day-np.datetime64(year+'-01-01')).astype(int))
        return np.asarray(self.array(years[year])[ix],np.float32)/np.float32(10)
    def label(self,i):
        d=self.desc['target']
        if d['kind']=='global_raw':
            y=self.raw(d['years'],self.dates[i]);return y,np.isfinite(y)
        return self.array(d['array'])[i],self.array(self.desc['valid'])[i]
    def prediction(self,name,i):
        d=self.desc['methods'][name]
        if d.get('kind')=='global_prior':return self.raw(d['years'],self.dates[i]-np.timedelta64(1,'D'))
        j=i
        if 'source_origins' in d:
            key=d['source_origins']['path']
            if key not in self.lookups:self.lookups[key]={int(t):j for j,t in enumerate(self.array(d['source_origins']))}
            j=self.lookups[key][int(self.ts[i])]
        arrays=[self.array(a) for a in d['arrays']]
        value=np.asarray(arrays[0][j],np.float64) if len(arrays)==1 else np.mean([np.asarray(a[j],np.float64) for a in arrays],axis=0)
        if d.get('stride'):value=value[::d['stride']]
        if 'mapping_statistics' in d:
            key=d['mapping_statistics']['path']
            if key not in self.maps:
                z=self.array(d['mapping_statistics']);self.maps[key]=mapping(z['lat'],z['lon'])
            value=remap(value.reshape(12,5183),self.maps[key])
        if d.get('threshold') is not None:
            mask=self.array(d['gate_history'])[i,::8]>=d['threshold'];value=np.where(mask,self.array(d['gate_anchor'])[i],value)
        return value
