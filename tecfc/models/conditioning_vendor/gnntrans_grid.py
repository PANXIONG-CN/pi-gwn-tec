"""GIMGridAdapter copied verbatim from pi-gwn-tec/model/GNNTrans/components.py (encode, decode, coordinates)."""
from __future__ import annotations
import torch

GRID = (71, 73)


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
