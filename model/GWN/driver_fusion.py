"""Production CAC-R1 history-only FiLM primitives for GWNET."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


FILM_ARCHITECTURE_ID = "cac_r1_gwn_history_film_v1"
FILM_ARCHITECTURE_DIGEST_SHA256 = (
    "1eda27a88a166b261b1a93a9d57f26c61cadb38e80052979e94794c47f1a3e13"
)
FILM_INSERTION_POINT = "after_start_conv_before_block0"
FILM_PROJECTION_ID = "Conv2d_1x1_D5_to_2Cres"
FILM_DRIVER_CHANNELS = 5


def expected_film_parameter_count(residual_channels: int) -> int:
    channels = int(residual_channels)
    if channels <= 0:
        raise ValueError("residual_channels must be positive")
    return (FILM_DRIVER_CHANNELS * 2 * channels) + (2 * channels)


def align_film_driver_history(
    driver_history: torch.Tensor,
    *,
    receptive_field: int,
) -> torch.Tensor:
    """Mirror GWNET's temporal path using architectural zero-padding only."""

    if not isinstance(driver_history, torch.Tensor) or driver_history.ndim != 4:
        raise ValueError("driver history must have shape [B,L,N,5]")
    if any(int(size) <= 0 for size in driver_history.shape):
        raise ValueError("driver history dimensions must all be positive")
    if int(driver_history.size(-1)) != FILM_DRIVER_CHANNELS:
        raise ValueError(
            f"driver channel mismatch: expected {FILM_DRIVER_CHANNELS}, "
            f"got {int(driver_history.size(-1))}"
        )
    if not torch.is_floating_point(driver_history):
        raise ValueError("driver history must have a floating dtype")
    if not bool(torch.isfinite(driver_history).all()):
        raise ValueError("driver history contains non-finite or missing values")
    target_length = int(receptive_field)
    if target_length <= 0:
        raise ValueError("receptive_field must be positive")

    aligned = driver_history.permute(0, 3, 2, 1).contiguous()
    aligned = F.pad(aligned, (1, 0, 0, 0))
    input_length = int(aligned.size(-1))
    if input_length < target_length:
        aligned = F.pad(aligned, (target_length - input_length, 0, 0, 0))
    else:
        aligned = aligned[..., -target_length:].contiguous()
    return aligned


class HistoryResidualGate(nn.Module):
    """Independent fixed-recipe residual gate; not a frozen CAC-R1 candidate."""

    architecture_id = "gim_history_residual_gate_v1"

    def __init__(self, residual_channels: int):
        super().__init__()
        if int(residual_channels) <= 0:
            raise ValueError("residual channels must be positive")
        self.value = nn.Conv2d(5, residual_channels, 1, bias=True)
        self.gate = nn.Conv2d(residual_channels + 5, residual_channels, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden, aligned_drivers):
        if hidden.ndim != 4 or aligned_drivers.ndim != 4:
            raise ValueError("gate inputs must be four-dimensional")
        if aligned_drivers.shape[1] != 5 or hidden.shape[1] != self.value.out_channels:
            raise ValueError("gate channel mismatch")
        if hidden.shape[0:1] + hidden.shape[2:] != aligned_drivers.shape[0:1] + aligned_drivers.shape[2:]:
            raise ValueError("gate batch/node/time mismatch")
        if hidden.device != aligned_drivers.device:
            raise ValueError("gate device mismatch")
        if not torch.isfinite(hidden).all() or not torch.isfinite(aligned_drivers).all():
            raise ValueError("gate inputs must be finite")
        value = self.value(aligned_drivers)
        gate = torch.sigmoid(self.gate(torch.cat((hidden, aligned_drivers), dim=1)))
        output = hidden + gate * value
        if not torch.isfinite(output).all():
            raise ValueError("gate output must be finite")
        return output


class HistoryFiLM(nn.Module):
    """Frozen biased 1x1 projection and pointwise FiLM modulation."""

    architecture_id = FILM_ARCHITECTURE_ID
    architecture_digest_sha256 = FILM_ARCHITECTURE_DIGEST_SHA256
    driver_channels = FILM_DRIVER_CHANNELS

    def __init__(self, residual_channels: int):
        super().__init__()
        residual_channels = int(residual_channels)
        if residual_channels <= 0:
            raise ValueError("residual_channels must be positive")
        self.residual_channels = residual_channels
        self.projection = nn.Conv2d(
            in_channels=self.driver_channels,
            out_channels=2 * residual_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def assert_exact_zero_initialization(self) -> None:
        if int(torch.count_nonzero(self.projection.weight).item()) != 0:
            raise ValueError("FiLM projection weight is not exactly zero-initialized")
        if int(torch.count_nonzero(self.projection.bias).item()) != 0:
            raise ValueError("FiLM projection bias is not exactly zero-initialized")

    def project(self, aligned_drivers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(aligned_drivers, torch.Tensor) or aligned_drivers.ndim != 4:
            raise ValueError("aligned drivers must have shape [B,5,N,T_rf]")
        if int(aligned_drivers.size(1)) != self.driver_channels:
            raise ValueError("aligned driver channel mismatch")
        if not torch.is_floating_point(aligned_drivers):
            raise ValueError("aligned drivers must have a floating dtype")
        if not bool(torch.isfinite(aligned_drivers).all()):
            raise ValueError("aligned drivers contain non-finite values")
        film = self.projection(aligned_drivers)
        if not bool(torch.isfinite(film).all()):
            raise ValueError("FiLM projection produced non-finite values")
        gamma_raw, beta = torch.split(film, self.residual_channels, dim=1)
        return torch.tanh(gamma_raw), beta

    def forward(self, hidden: torch.Tensor, aligned_drivers: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 4:
            raise ValueError("hidden state must have shape [B,Cres,N,T_rf]")
        if int(hidden.size(1)) != self.residual_channels:
            raise ValueError("hidden residual-channel mismatch")
        if not torch.is_floating_point(hidden) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("hidden state contains non-finite or non-floating values")
        if hidden.device != aligned_drivers.device:
            raise ValueError("FiLM hidden/driver device mismatch")
        gamma, beta = self.project(aligned_drivers)
        if tuple(gamma.shape) != tuple(hidden.shape):
            raise ValueError(
                f"FiLM/hidden shape mismatch: hidden={tuple(hidden.shape)}, "
                f"gamma={tuple(gamma.shape)}"
            )
        output = hidden * (1.0 + gamma) + beta
        if not bool(torch.isfinite(output).all()):
            raise ValueError("FiLM modulation produced non-finite values")
        return output


def required_film_config() -> dict[str, Any]:
    """Return the exact flat runtime fields required by ``GWNET``."""

    return {
        "film_architecture_id": FILM_ARCHITECTURE_ID,
        "film_architecture_digest_sha256": FILM_ARCHITECTURE_DIGEST_SHA256,
        "film_insertion_point": FILM_INSERTION_POINT,
        "film_projection": FILM_PROJECTION_ID,
        "film_zero_init": True,
        "film_gamma_activation": "tanh",
        "film_beta_activation": "identity",
        "film_apply_count": 1,
    }


__all__ = [
    "FILM_ARCHITECTURE_DIGEST_SHA256",
    "FILM_ARCHITECTURE_ID",
    "FILM_DRIVER_CHANNELS",
    "FILM_INSERTION_POINT",
    "FILM_PROJECTION_ID",
    "HistoryFiLM",
    "align_film_driver_history",
    "expected_film_parameter_count",
    "required_film_config",
]
