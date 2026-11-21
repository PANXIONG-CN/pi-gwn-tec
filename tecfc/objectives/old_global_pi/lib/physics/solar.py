"""Frozen CAC-R1 solar-geometry calculation used by production drivers.

The Task 151 candidate-instantiation generator is already hash-bound by the
frozen protocol lock.  Its float64 NOAA implementation therefore remains the
authoritative implementation; this module gives production code a stable,
named entry point without changing that generator's bytes.
"""

from __future__ import annotations

import numpy as np


COS_SZA_ALGORITHM_ID = "noaa_fractional_year_365_float64_daylight_clip_v1"


def canonical_cos_sza_float64(
    times_sec: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> np.ndarray:
    """Return daylight-clipped cos(SZA) using the frozen float64 operation order.

    ``times_sec`` is flattened by the frozen implementation and the result is
    always ``[K, N]``.  Callers retain any leading timestamp shape metadata and
    reshape only after this function returns.
    """

    # Deliberately import the hash-bound Task 151 implementation instead of
    # copying its formula.  This prevents production/scaler numerical drift
    # while preserving all Task 151 generator and protocol-lock hashes.
    from tools.cac_r1.instantiate_candidates import _cos_sza_chunk

    result = _cos_sza_chunk(times_sec, lon_deg, lat_deg)
    if not isinstance(result, np.ndarray) or result.dtype != np.float64:
        raise RuntimeError(
            "frozen cosSZA implementation must return a float64 numpy array"
        )
    return result


__all__ = ["COS_SZA_ALGORITHM_ID", "canonical_cos_sza_float64"]
