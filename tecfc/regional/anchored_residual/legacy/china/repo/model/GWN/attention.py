"""One nodewise causal temporal cross-attention layer; no scientific search."""
import math
import torch
from torch import nn


def real_time_mapping(receptive_field, history_steps=12):
    """Apply GWNET's left pad, then extra left pad or tail crop to time labels.

    -1 denotes architectural padding, never a driver token. Reject any crop
    that loses one of the twelve real observations.
    """
    if history_steps != 12 or receptive_field <= 0:
        raise ValueError('expected twelve historical observations and positive RF')
    labels = [-1] + list(range(history_steps))
    if len(labels) < receptive_field:
        labels = [-1] * (receptive_field - len(labels)) + labels
    else:
        labels = labels[-receptive_field:]
    if [t for t in labels if t >= 0] != list(range(12)):
        raise ValueError('GWN crop loses required real historical observations')
    return tuple(labels), tuple(labels.index(t) for t in range(12))


class NodewiseCausalTemporalCrossAttention(nn.Module):
    architecture_id = 'gim_nodewise_causal_temporal_crossattn_v1'

    def __init__(self, chunk_size=32768):
        super().__init__()
        if not 0 < chunk_size <= 32768:
            raise ValueError('invalid independent node-sequence chunk size')
        self.chunk_size = chunk_size
        self.query = nn.Linear(32, 32, bias=True)
        self.key = nn.Linear(5, 32, bias=True)
        self.value = nn.Linear(5, 32, bias=True)
        self.output = nn.Linear(32, 32, bias=True)
        position = torch.zeros(12, 32, dtype=torch.float32)
        angles = torch.arange(12, dtype=torch.float32)[:, None] / (
            10000. ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        position[:, 0::2] = angles.sin()
        position[:, 1::2] = angles.cos()
        self.register_buffer('position', position)
        self.register_buffer('future_mask', torch.ones(12, 12, dtype=torch.bool).triu(1))
        self.reset_parameters()

    def reset_parameters(self):
        for projection in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, hidden, drivers):
        if hidden.ndim != 4 or drivers.ndim != 4:
            raise ValueError('expected H[B,32,N,T_rf], D[B,12,N,5]')
        b, c, n, rf = hidden.shape
        if c != 32 or drivers.shape != (b, 12, n, 5):
            raise ValueError('attention shape mismatch')
        if drivers.device != hidden.device:
            raise ValueError('attention device mismatch')
        if not torch.isfinite(hidden).all() or not torch.isfinite(drivers).all():
            raise ValueError('attention input must be finite')
        _, positions = real_time_mapping(rf)
        indices = torch.tensor(positions, device=hidden.device)
        h = hidden.index_select(-1, indices).permute(0, 2, 3, 1).reshape(b*n, 12, 32)
        d = drivers.permute(0, 2, 1, 3).reshape(b*n, 12, 5)
        increments = []
        # AMP conversion is explicit: projections through output projection are
        # float32, including positional addition, logits and softmax. Only the
        # final increment is cast to the backbone dtype before residual addition.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, b*n, self.chunk_size):
                hs = h[start:start+self.chunk_size].float()
                ds = d[start:start+self.chunk_size].float()
                q = (self.query(hs) + self.position).reshape(-1, 12, 4, 8).transpose(1, 2)
                k = (self.key(ds) + self.position).reshape(-1, 12, 4, 8).transpose(1, 2)
                v = self.value(ds).reshape(-1, 12, 4, 8).transpose(1, 2)
                logits = (q @ k.transpose(-1, -2)) / math.sqrt(8)
                weights = logits.masked_fill(self.future_mask, float('-inf')).softmax(-1)
                attended = (weights @ v).transpose(1, 2).reshape(-1, 12, 32)
                increments.append(self.output(attended))
        increment = torch.cat(increments, 0).reshape(b, n, 12, 32).permute(0, 3, 1, 2).to(hidden.dtype)
        # No padding query is computed; preserve even nonzero start_conv bias.
        fused_real = hidden.index_select(-1, indices) + increment
        result = hidden.index_copy(-1, indices, fused_real)
        if not torch.isfinite(result).all():
            raise ValueError('attention output must be finite')
        return result
