"""Provisional native-cadence global port of the Japan CG SimVP backbone.

Not a frozen scientific recipe. Imported Japan parameters and nonperiodic
latitude rules are retained; time adapter is 12->compressed_frames because GIM
is natively two-hourly, longitude layers are periodic, and 71 rows are restored.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from periodic import LongitudeConv2d,LongitudeConvTranspose2d,LongitudeMaxPool2d,restore_height,restore_longitude

sys.path.insert(0,str(Path(__file__).resolve().parent/'shared'))
from cg_model import AmplitudeConditionedSimVP


def periodicize(module):
    # Swap classes of newly instantiated modules, preserving parameters, names,
    # initialization and random-generator state; the shared files are untouched.
    for sub in module.modules():
        if type(sub) is nn.Conv2d:sub.__class__=LongitudeConv2d
        elif type(sub) is nn.ConvTranspose2d:sub.__class__=LongitudeConvTranspose2d
        elif type(sub) is nn.MaxPool2d:sub.__class__=LongitudeMaxPool2d
    return module


class GlobalAmplitudeConditionedSimVP(AmplitudeConditionedSimVP):
    def __init__(self,hidden=32,compressed_frames=12,output_frames=12):
        super().__init__(hidden=hidden,compressed_frames=compressed_frames,output_frames=output_frames)
        self.time_adapter=nn.Linear(12,compressed_frames,bias=False)
        periodicize(self)
    def forward(self,history,background_history,background_future,geometry,arm,restore=False):
        if arm not in ('C','D','E'):raise ValueError('arm must be C/D/E')
        if history.ndim!=4 or history.shape[1:]!=(12,71,72):raise ValueError('history must be [batch,12,71,72]')
        if background_history.shape!=history.shape or geometry.shape!=history.shape or background_future.shape!=history.shape:raise ValueError('condition shape mismatch')
        zero=torch.zeros_like(history)
        x=torch.stack((history if arm=='C' else history-background_history,
            zero if arm=='C' else background_history,geometry if arm=='E' else zero),dim=2)
        b,t,c,h,w=x.shape
        enc=self.attn(self.spatial_enc(x.reshape(b*t,c,h,w)))
        _,hidden,hh,ww=enc.shape
        compressed=self.time_adapter(enc.reshape(b,t,hidden,hh,ww).permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()
        latent=self.temporal(compressed.reshape(b,self.compressed_frames*hidden,hh,ww))
        pred=self.time_proj(latent).reshape(b*self.output_frames,hidden,hh,ww)
        pred=restore_height(self.spatial_dec(pred)).reshape(b,self.output_frames,71,72)
        full=pred if arm=='C' else pred+background_future
        return restore_longitude(full) if restore else full
