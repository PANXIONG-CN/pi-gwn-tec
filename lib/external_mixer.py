"""History-only GIM bridge for the released SimVP mix_4channel checkpoint.

Source: Liu et al. Channel Mixer v1.0, SimVP._predict first autoregressive
block and SimVP_Model.forward. This is an external pretrained comparator,
not a CAC-R1 selected model or a declaration of its unknown training exposure.
The caller supplies an audited, weight-bound SimVP_Model; this module does
not import external code, discover files, select a checkpoint, or read targets.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn


def prepare_mix4_history(history_tecu: torch.Tensor, history_time_utc: np.ndarray) -> torch.Tensor:
    """Map [B,12,1,71,73] physical TECU and [B,12] UTC datetime64 to mix4.

    Longitude runs -180..180, latitude 87.5..-87.5. Inputs must already
    represent the caller's valid historical grid. Zero TEC is a valid value.
    UTC timestamps must be explicit numpy datetime64 values, two hours apart;
    they are never inferred from filenames or shifted to the external data.
    """
    if history_tecu.ndim != 5 or tuple(history_tecu.shape[1:]) != (12, 1, 71, 73):
        raise ValueError('expected history TECU shape [B,12,1,71,73]')
    if history_tecu.shape[0] < 1 or not history_tecu.is_floating_point():
        raise ValueError('history TECU must be a nonempty floating-point tensor')
    if not torch.isfinite(history_tecu).all() or (history_tecu < 0).any():
        raise ValueError('history TECU must be finite and nonnegative')
    times = np.asarray(history_time_utc)
    if times.shape != tuple(history_tecu.shape[:2]) or times.dtype.kind != 'M':
        raise ValueError('history timestamps must be UTC datetime64 with shape [B,12]')
    if np.isnat(times).any():
        raise ValueError('history timestamps cannot contain NaT')
    seconds = times.astype('datetime64[s]')
    if not np.array_equal(times, seconds) or not np.all(np.diff(seconds, axis=1) == np.timedelta64(7200, 's')):
        raise ValueError('history timestamps must follow exact two-hour UTC cadence')

    batch = history_tecu.shape[0]
    tec = torch.nn.functional.pad(history_tecu[..., :72].to(torch.float32), (0, 0, 0, 1))
    tec = tec.clamp(max=200.0) / 200.0
    cols = torch.arange(72, device=tec.device, dtype=tec.dtype).view(1, 1, 1, 1, 72) / 72.0
    rows = torch.arange(72, device=tec.device, dtype=tec.dtype).view(1, 1, 1, 72, 1) / 72.0
    years = times.astype('datetime64[Y]').astype(np.int64) + 1970
    year = torch.as_tensor(years.copy(), device=tec.device, dtype=tec.dtype).view(batch, 12, 1, 1, 1) / 2500.0
    grid = (batch, 12, 1, 72, 72)
    channels = torch.cat((tec, cols.expand(grid), rows.expand(grid), year.expand(grid)), dim=2)

    # Author reshape_patch(x.permute(B,T,H,W,C), patch_size=72) followed
    # by mix4's view/transpose. With one full-grid patch, reshape_patch
    # preserves the BHWC flattened order. Keep actual history T=12 here;
    # the released wrapper's total_length=60 includes future auxiliaries.
    mixed = channels.permute(0, 1, 3, 4, 2).contiguous()
    mixed = mixed.reshape(batch, 12, 144, 36, 2, 2).transpose(3, 4).contiguous()
    mixed = mixed.reshape(batch, 12, 4, 72, 72)
    return torch.cat((tec, mixed), dim=2)


def restore_gim_tecu(model_output: torch.Tensor) -> torch.Tensor:
    """Restore physical cells and the cyclic seam, without clipping forecasts."""
    if model_output.ndim != 5 or tuple(model_output.shape[1:]) != (12, 1, 72, 72):
        raise ValueError('expected first SimVP block [B,12,1,72,72]')
    physical = model_output[..., :71, :] * 200.0
    return torch.cat((physical, physical[..., :1]), dim=-1)


def load_mix4_state_dict(model: nn.Module, checkpoint_path: str | Path) -> None:
    """CPU-load only the released tensor state dict and require an exact match.

    The caller must bind this path to the independently checked complete
    checkpoint SHA and exact audited model configuration. No pickle fallback,
    custom globals, optimizer restoration, or source execution is supported.
    """
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError('expected a tensor-only external model state_dict')
    model.load_state_dict(state, strict=True)


class ChannelMixerGIMAdapter(nn.Module):
    """Evaluate one audited SimVP mix4 first block using historical inputs only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, history_tecu: torch.Tensor, history_time_utc: np.ndarray) -> torch.Tensor:
        self.model.eval()
        inputs = prepare_mix4_history(history_tecu, history_time_utc)
        output = self.model(inputs)
        if output.shape[0] != history_tecu.shape[0]:
            raise ValueError('external model changed the batch size')
        return restore_gim_tecu(output)
