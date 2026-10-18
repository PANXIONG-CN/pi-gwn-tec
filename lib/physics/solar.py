"""NOAA fractional-year solar geometry, preserving the saved scaler operation order."""
import numpy as np

COS_SZA_ALGORITHM_ID = "noaa_fractional_year_365_float64_daylight_clip_v1"

def canonical_cos_sza_float64(times: np.ndarray, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Canonical float64 NOAA fractional-year cos(SZA), clipped to daylight."""
    timestamps = np.asarray(times, dtype=np.int64).reshape(-1)
    dt = timestamps.astype("datetime64[s]")
    days = dt.astype("datetime64[D]")
    year_start = dt.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (days - year_start).astype(np.int64).astype(np.float64) + 1.0
    seconds_in_day = np.mod(timestamps, 86400).astype(np.float64)
    hour = np.floor(seconds_in_day / 3600.0)
    minute = np.floor(np.mod(seconds_in_day, 3600.0) / 60.0)

    gamma = 2.0 * np.pi * (day_of_year - 1.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    time_minutes = hour * 60.0 + minute + eqtime
    hour_angle_deg = (
        (time_minutes / 4.0 - 180.0).reshape(-1, 1)
        + lon_deg.reshape(1, -1)
    )
    hour_angle = np.deg2rad(hour_angle_deg)
    latitude = np.deg2rad(lat_deg).reshape(1, -1)
    declination = decl.reshape(-1, 1)
    cos_sza = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    return np.clip(cos_sza, 0.0, 1.0)
