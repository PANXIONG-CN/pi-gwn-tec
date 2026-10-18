"""Paper-based GNNTrans pyramid under explicit, disclosed reconstruction rules.

NOT an official implementation or an exact recovery of the authors' code.
Read ARCHITECTURE.md for the ambiguous-source choices. Only one architecture.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model.GNNTrans.components import GIMGridAdapter


def sphere_xyz(lat, lon):
    return torch.stack((lat.cos()*lon.cos(), lat.cos()*lon.sin(), lat.sin()),-1)


def coordinates():
    lat,lon=GIMGridAdapter.internal_coordinates()
    xyz=sphere_xyz(lat*math.pi/180,lon*math.pi/180)
    result=[]
    for level in range(4):
        result.append(xyz)
        if level<3:
            xyz=F.avg_pool2d(xyz.permute(2,0,1)[None],2,2)[0].permute(1,2,0)
            xyz=F.normalize(xyz,dim=-1)
    return result


def fixed_neighbors(xyz, reduction, k=9):
    """Spherical chord distance; deterministic stable index ordering of ties.

    Reduction pools only candidate keys/values, not query nodes. An exactly
    colocated candidate is eligible; k counts the selected candidates.
    """
    reduced=F.avg_pool2d(xyz.permute(2,0,1)[None],reduction,reduction)[0].permute(1,2,0)
    reduced=F.normalize(reduced,dim=-1)
    q=xyz.reshape(-1,3).numpy(); p=reduced.reshape(-1,3).numpy()
    # Round numerical pole/chord ties at 1e-12 then stable sort by node index.
    distance=np.round(((q[:,None,:]-p[None,:,:])**2).sum(-1),12)
    return torch.from_numpy(np.argsort(distance,axis=1,kind='stable')[:,:k].copy()).long()


class DropPath(nn.Module):
    def __init__(self,p):super().__init__();self.p=p
    def forward(self,x):
        if not self.training or not self.p:return x
        keep=(torch.rand((x.shape[0],1,1,1),device=x.device)>=self.p).to(x.dtype)
        return x*keep/(1-self.p)


class GraphBlock(nn.Module):
    def __init__(self,c,xyz,reduction):
        super().__init__();self.reduction=reduction
        self.register_buffer('neighbors',fixed_neighbors(xyz,reduction))
        self.pre=nn.Conv2d(c,c,1)
        self.update=nn.Conv2d(c*2,c,1)
        self.fc1=nn.Conv2d(c,c*4,1);self.fc2=nn.Conv2d(c*4,c,1)
        self.drop=DropPath(.1)
    def forward(self,x):
        z=self.pre(x)
        candidates=F.avg_pool2d(z,self.reduction,self.reduction).flatten(2)
        center=z.flatten(2)
        difference=candidates[:,:,self.neighbors]-center[:,:,:,None]
        aggregate=torch.cat((center,difference.max(-1).values),1).reshape(x.shape[0],-1,*x.shape[-2:])
        g=F.gelu(self.update(aggregate))
        # Explicit Eq8 reconstruction, not claimed exact transcription:
        # skip1 x->u; skip2 u->(v+u); skip3 x->output.
        u=g+x
        v=F.gelu(self.fc2(F.gelu(self.fc1(u))))
        return x+self.drop(F.gelu(v+u))


class ConditionTokens(nn.Module):
    """Four activity feature tokens per historical frame, no single-key softmax."""
    def __init__(self,width,temporal):
        super().__init__();self.temporal=temporal
        self.encode=nn.Sequential(nn.Linear(1,width),nn.GELU(),nn.Linear(width,width))
        self.feature=nn.Parameter(torch.zeros(4,width))
        self.time=nn.Parameter(torch.zeros(12,width)) if temporal else None
        nn.init.normal_(self.feature,std=.02)
        if temporal:nn.init.normal_(self.time,std=.02)
    def forward(self,global_history):
        # [B,12,4] => [B,12,4,D]
        z=self.encode(global_history[...,None])+self.feature
        if self.time is not None:z=z+self.time[None,:,None,:]
        return z


class EncoderBlock(nn.Module):
    def __init__(self,first):
        super().__init__();d=12*64
        self.cross=nn.MultiheadAttention(d,8,dropout=.1,batch_first=True) if first else None
        self.cross_norm=nn.LayerNorm(d) if first else None
        self.self_norm=nn.LayerNorm(d)
        self.attention=nn.MultiheadAttention(d,8,dropout=.1,batch_first=True)
        self.ffn_norm=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,d),nn.Dropout(.1))
        self.dropout=nn.Dropout(.1)
    def forward(self,z,conditions):
        if self.cross is not None:
            q=self.cross_norm(z);z=z+self.dropout(self.cross(q,conditions,conditions,need_weights=False)[0])
        q=self.self_norm(z);z=z+self.dropout(self.attention(q,q,q,need_weights=False)[0])
        return z+self.ffn(self.ffn_norm(z))


class GNNTransPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        xyz=coordinates()
        self.input=nn.Conv2d(2,8,1) # TEC plus spatial cosSZA; other 4 drivers condition attention
        lat,lon=GIMGridAdapter.internal_coordinates();lat=lat*math.pi/180;lon=lon*math.pi/180
        key=(lon*72+lat.sin())*72
        div=10000**(torch.arange(0,8,2,dtype=torch.float64)/8)
        pe=torch.empty(8,72,72,dtype=torch.float64)
        pe[0::2]=torch.sin(key[None]/div[:,None,None]);pe[1::2]=torch.cos(key[None]/div[:,None,None])
        self.register_buffer('spatial_encoding',pe.float()[None])
        self.stem_conditions=ConditionTokens(8,False)
        self.stem_cross=nn.MultiheadAttention(8,2,dropout=.1,batch_first=True)
        self.stem=nn.ModuleList(GraphBlock(c,xyz[i],r) for i,(c,r) in enumerate(zip([8,16,32],[4,2,1])))
        self.down=nn.ModuleList(nn.Conv2d(a,b,2,stride=2) for a,b in [(8,16),(16,32),(32,64)])
        self.temporal_conditions=ConditionTokens(768,True)
        self.position=nn.Parameter(torch.zeros(1,81,768));nn.init.normal_(self.position,std=.02)
        self.encoder=nn.ModuleList(EncoderBlock(i==0) for i in range(3))
        self.up=nn.ModuleList(nn.Conv2d(a,b,1) for a,b in [(64,32),(32,16),(16,8)])
        self.head=nn.ModuleList(GraphBlock(c,xyz[i],r) for c,i,r in [(32,2,1),(16,1,2),(8,0,4)])
        self.output=nn.Conv2d(8,1,1)

    def forward(self,tec,drivers):
        # common protocol [B,12,5183,1] and [B,12,5183,5], history already validated
        if tec.ndim!=4 or tec.shape[1:]!=(12,5183,1):raise ValueError('TEC shape must be [B,12,5183,1]')
        if drivers.shape!=(tec.shape[0],12,5183,5):raise ValueError('driver shape must be [B,12,5183,5]')
        if not bool(torch.isfinite(drivers).all()):raise ValueError('nonfinite driver')
        b=tec.shape[0]
        globals_=drivers[:,:,0,:4]
        if not bool(torch.equal(drivers[:,:,:,:4],globals_[:,:,None,:].expand(-1,-1,5183,-1))):
            raise ValueError('AE/Dst/Kp/F107 must be spatially invariant')
        tec72=GIMGridAdapter.encode(tec[...,0].reshape(b,12,71,73))
        # cosSZA is analytic and +180/-180 roundoff can differ. Keep both TEC
        # endpoint checks exact, but periodic cosSZA accepts a documented 1e-5
        # scaled tolerance before choosing the -180 representation.
        cos=drivers[...,4].reshape(b,12,71,73)
        if not torch.allclose(cos[...,0],cos[...,-1],atol=1e-5,rtol=0):raise ValueError('cosSZA seam disagreement')
        cunique=cos[...,:-1]
        cpole=cunique[...,:1,:].mean(-1,keepdim=True).expand(b,12,1,72)
        cos72=torch.cat((cpole,cunique),-2)
        x=self.input(torch.stack((tec72,cos72),2).reshape(b*12,2,72,72))+self.spatial_encoding
        q=x.flatten(2).transpose(1,2)
        cond=self.stem_conditions(globals_).reshape(b*12,4,8)
        x=x+self.stem_cross(q,cond,cond,need_weights=False)[0].transpose(1,2).reshape(b*12,8,72,72)
        skips=[]
        for graph,down in zip(self.stem,self.down):x=graph(x);skips.append(x);x=down(x)
        # TableI: B x HW/64 x (TC): 81 spatial tokens, 768 time/channel features.
        z=x.reshape(b,12,64,81).permute(0,3,1,2).reshape(b,81,768)+self.position
        cond=self.temporal_conditions(globals_).reshape(b,48,768)
        for block in self.encoder:z=block(z,cond)
        x=z.reshape(b,81,12,64).permute(0,2,3,1).reshape(b*12,64,9,9)
        for up,graph,skip in zip(self.up,self.head,reversed(skips)):
            x=up(F.interpolate(x,scale_factor=2,mode='nearest'))+skip
            x=graph(x)
        y=self.output(x).reshape(b,12,72,72)
        return GIMGridAdapter.decode(y).reshape(b,12,5183,1)
