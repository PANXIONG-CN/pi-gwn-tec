"""Verified common-grid and driver primitives used by the full model.

The single disclosed paper-based pyramid is implemented in model.py; see
ARCHITECTURE.md for explicit reconstruction choices and source limitations.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lib.paper_statistics import masked_statistics as masked_regression_statistics
from lib.physics.driver_scaler import (
    DRIVER_CHANNEL_ORDER, FrozenDriverScaler, extract_strict_driver_history,
)

HISTORY = HORIZON = 12
GRID = (71, 73)
N = 5183


class GIMGridAdapter:
    """North-to-south 71x73 -> north-pole-extended 72x72 and back.

    Internal longitudes are -180 through +175. The paper's last-column label
    disagrees with its stated orientation; we use the repository coordinate
    order, explicitly remove +180, and restore it from -180. The synthetic
    north-pole row is the longitudinal mean of the 87.5-degree row, never a
    target. This keeps its graph-feature contribution without evaluating it.
    """

    @staticmethod
    def encode(history: torch.Tensor) -> torch.Tensor:
        # Last axes [H,W]; leading axes may be B,T or B,T,C.
        if history.shape[-2:] != GRID or not history.is_floating_point():
            raise ValueError('expected floating [...,71,73] physical grid')
        if not bool(torch.isfinite(history).all()):
            raise ValueError('missing TEC requires an explicit upstream input-mask policy')
        if not bool(torch.equal(history[..., 0], history[..., -1])):
            raise ValueError('longitude endpoints differ; do not silently discard information')
        unique = history[..., :-1]
        pole = unique[..., :1, :].mean(-1, keepdim=True).expand(*unique.shape[:-2], 1, 72)
        return torch.cat((pole, unique), dim=-2)

    @staticmethod
    def decode(prediction: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-2:] != (72, 72):
            raise ValueError('expected [...,72,72] internal grid')
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError('nonfinite prediction')
        physical = prediction[..., 1:, :]
        return torch.cat((physical, physical[..., :1]), dim=-1)

    @staticmethod
    def internal_coordinates() -> tuple[torch.Tensor, torch.Tensor]:
        lat = torch.cat((torch.tensor([90.], dtype=torch.float64),
                         torch.linspace(87.5, -87.5, 71, dtype=torch.float64)))
        lon = torch.arange(-180., 180., 5., dtype=torch.float64)
        return torch.meshgrid(lat, lon, indexing='ij')

    @staticmethod
    def restore_and_score(prediction_normalized, target_tecu, target_valid):
        """Score only on the unmodified physical truth/mask; no clipping."""
        pred = GIMGridAdapter.decode(prediction_normalized) * 147.1
        if tuple(target_tecu.shape) != tuple(pred.shape):
            raise ValueError('target must be original 71x73 grid with matching batch/lead')
        return masked_regression_statistics(pred.detach().cpu().numpy(), target_tecu, target_valid)


class MaxRelativeConvolution(nn.Module):
    """Equation (7), with caller-supplied neighbor indices.

    For x [B,N,C], neighbors [N,K], compute linear([x_i,max_j(x_j-x_i)]).
    Graph construction is deliberately not decided by this primitive.
    Bias is explicit because the article omits bias terms in its equations.
    """
    def __init__(self, in_channels: int, out_channels: int, *, bias: bool):
        super().__init__()
        self.update = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x, neighbors):
        if x.ndim != 3 or neighbors.ndim != 2 or neighbors.shape[0] != x.shape[1]:
            raise ValueError('expected x[B,N,C] and neighbors[N,K]')
        if neighbors.dtype != torch.long or neighbors.numel() == 0:
            raise ValueError('nonempty int64 graph required')
        if bool((neighbors < 0).any()) or bool((neighbors >= x.shape[1]).any()):
            raise ValueError('neighbor out of range')
        relative = x[:, neighbors] - x[:, :, None, :]
        return self.update(torch.cat((x, relative.max(dim=2).values), dim=-1))


def common_driver_history(timeline_values, timeline_timestamps, forecast_origins,
                          *, scaler: FrozenDriverScaler):
    """Reuse validated history extraction/scaling without reading future values.

    This CPU interface accepts an existing timeline. A production streaming
    loader must construct history directly; no full node-expanded timeline is
    proposed for deployment.
    """
    return extract_strict_driver_history(
        timeline_values, timeline_timestamps, forecast_origins,
        history_steps=12, cadence_seconds=7200,
        channel_order=DRIVER_CHANNEL_ORDER,
        source_driver_sha256=scaler.source_driver_sha256,
        scaling='train_zscore', scaler=scaler,
        bound_driver_scaler_sha256=scaler.file_sha256,
    )
