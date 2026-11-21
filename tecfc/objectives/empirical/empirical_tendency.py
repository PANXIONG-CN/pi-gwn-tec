"""Frozen same-information TEC tendency; NumPy and PyTorch arrays are supported.

Keep COEFFICIENTS.json alongside this module.  Inputs use raw AE and F107,
float32 NOAA cosSZA at the previous state's UTC, and u=TEC/50. No fit here.
"""
from pathlib import Path
import json
import numpy as np

with Path(__file__).with_name('COEFFICIENTS.json').open() as _file:
    COEFFICIENTS = tuple(json.load(_file)['coefficients'])


def empirical_tendency(u, ae, f107, cos_sza):
    """Return nondimensional two-hour q; arrays must be broadcast-compatible."""
    b0, b1, b2, b3 = COEFFICIENTS
    daylight = cos_sza.clamp_min(0.) if hasattr(cos_sza, 'clamp_min') else np.maximum(cos_sza, 0.)
    return b0 + b1*u + b2*(ae*u) + b3*(f107*daylight)


def internal_residual(pred_tecu, ae, f107, cos_sza):
    """Residual for 11 internal transitions from 12 targets.

    pred_tecu and cos_sza: [batch, 12, *grid], ae/f107: [batch, 12].
    Return [batch, 11, *grid]. For valid-label comparisons, reduce only where
    valid[:, :-1] & valid[:, 1:] is true; valid zero TEC is retained. In a
    training loss, convert model output to TECU before calling and apply the
    same valid mask and reduction to both empirical and chemistry arms.
    """
    if pred_tecu.ndim < 3 or pred_tecu.shape[1] != 12 or cos_sza.shape != pred_tecu.shape:
        raise ValueError('prediction and cosSZA must have matching [B,12,*grid] shapes')
    if tuple(ae.shape) != tuple(pred_tecu.shape[:2]) or tuple(f107.shape) != tuple(pred_tecu.shape[:2]):
        raise ValueError('AE and F107 must have shape [B,12]')
    previous = pred_tecu[:, :-1] / 50.
    following = pred_tecu[:, 1:] / 50.
    temporal_shape = (pred_tecu.shape[0], 11) + (1,) * (pred_tecu.ndim - 2)
    tendency = empirical_tendency(previous, ae[:, :-1].reshape(temporal_shape),
                                  f107[:, :-1].reshape(temporal_shape), cos_sza[:, :-1])
    return following - previous - tendency
