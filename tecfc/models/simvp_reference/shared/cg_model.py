"""Japan adaptation of submitted SimVP: shared 89->12 encoded-frame mixer."""
import torch
from torch import nn
from attentions import CBAM
from simvp import InceptionModule

class AmplitudeConditionedSimVP(nn.Module):
    """Input/output are normalized TEC, explicitly retaining background amplitude.

    All C/D/E models have the same parameters, including three input channels.
    C: [full TEC,0,0]; D: [TEC-background,background,0];
    E: [TEC-background,background,cos(SZA)].  D/E add the future background
    to the network residual.  The spatial grid is geographic row/column order;
    all convolutions use the original nonperiodic zero-padding scheme.
    """
    def __init__(self, hidden=32, compressed_frames=12, output_frames=12):
        super().__init__()
        self.hidden = hidden
        self.compressed_frames = compressed_frames
        self.output_frames = output_frames
        self.spatial_enc = nn.Sequential(nn.Conv2d(3,hidden,3,stride=2,padding=1),nn.GELU(),
                                         nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU())
        self.attn = CBAM(hidden)
        self.time_adapter = nn.Linear(89,compressed_frames,bias=False)
        self.temporal = nn.Sequential(InceptionModule(hidden*compressed_frames),
                                      InceptionModule(hidden*compressed_frames))
        self.time_proj = nn.Conv2d(hidden*compressed_frames,hidden*output_frames,1)
        self.spatial_dec = nn.Sequential(nn.ConvTranspose2d(hidden,hidden,4,stride=2,padding=1),nn.GELU(),
                                         nn.Conv2d(hidden,1,3,padding=1))

    def forward(self, history, background_history, background_future, geometry, arm):
        assert arm in ('C','D','E')
        assert history.ndim == 4 and history.shape[1:] == (89,16,18)
        assert background_history.shape == geometry.shape == history.shape
        assert background_future.shape == (history.shape[0],12,16,18)
        zero = torch.zeros_like(history)
        x = torch.stack((history if arm == 'C' else history-background_history,
                         zero if arm == 'C' else background_history,
                         geometry if arm == 'E' else zero),dim=2)
        batch, frames, channels, height, width = x.shape
        enc = self.attn(self.spatial_enc(x.reshape(batch*frames,channels,height,width)))
        _, hidden, hh, ww = enc.shape
        enc = enc.reshape(batch,frames,hidden,hh,ww).permute(0,2,3,4,1)
        compressed = self.time_adapter(enc).permute(0,4,1,2,3).contiguous()
        latent = self.temporal(compressed.reshape(batch,self.compressed_frames*hidden,hh,ww))
        pred = self.time_proj(latent).reshape(batch*self.output_frames,hidden,hh,ww)
        pred = self.spatial_dec(pred).reshape(batch,self.output_frames,height,width)
        return pred if arm == 'C' else pred + background_future
