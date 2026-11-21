"""Longitude-periodic building blocks for the unique 71x72 GIM grid.

These are interface primitives, not a frozen global model or a training recipe.
Latitude retains the same zero/negative-infinity padding as the relevant PyTorch
convolution/pooling operator. Transposed-convolution spill is wrapped before one
bias is applied. Latitude's 72nd decoded row is cropped by restore_height().
"""
import torch
from torch import nn
from torch.nn import functional as F

class LongitudeConv2d(nn.Conv2d):
    def forward(self,x):
        ph,pw=self.padding
        if pw:x=F.pad(x,(pw,pw,0,0),mode='circular')
        if ph:x=F.pad(x,(0,0,ph,ph),mode='constant',value=0.)
        return F.conv2d(x,self.weight,self.bias,self.stride,(0,0),self.dilation,self.groups)

class LongitudeMaxPool2d(nn.MaxPool2d):
    def forward(self,x):
        ph,pw=(self.padding,self.padding) if isinstance(self.padding,int) else self.padding
        if pw:x=F.pad(x,(pw,pw,0,0),mode='circular')
        if ph:x=F.pad(x,(0,0,ph,ph),mode='constant',value=float('-inf'))
        return F.max_pool2d(x,self.kernel_size,self.stride,0,self.dilation,self.ceil_mode,self.return_indices)

class LongitudeConvTranspose2d(nn.ConvTranspose2d):
    def forward(self,x,output_size=None):
        if output_size is not None:raise ValueError('output_size is unsupported')
        if self.kernel_size!=(4,4) or self.stride!=(2,2) or self.padding!=(1,1) or self.output_padding!=(0,0) or self.dilation!=(1,1):
            raise ValueError('only the frozen SimVP 4x4 stride-2 decoder geometry is supported')
        z=F.conv_transpose2d(x,self.weight,None,self.stride,(1,0),self.output_padding,self.groups,self.dilation)
        # Raw output has one longitude column of spill on both sides.
        core=z[...,1:-1]
        core=core+F.pad(z[...,-1:],(0,core.shape[-1]-1))+F.pad(z[...,:1],(core.shape[-1]-1,0))
        if self.bias is not None:core=core+self.bias[None,:,None,None]
        return core

def restore_height(x):
    if x.shape[-2:]!=(72,72):raise ValueError('expected 72x72 decoded grid')
    return x[...,:71,:]

def restore_longitude(x):
    if x.shape[-2:]!=(71,72):raise ValueError('expected 71x72 unique grid')
    return torch.cat((x,x[...,:1]),dim=-1)
