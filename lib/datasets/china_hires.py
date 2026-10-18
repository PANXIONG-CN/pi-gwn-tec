"""China high-resolution data loading with calendar-year DataLoaders.

Source: data/china_hires/*.hdf5 (two-hour cadence, 41 x 71 regional grid).
Year segments follow the GIM loader's convention. Default evaluation roles
are test=2015 and ood={2019, 2024}. Sliding windows return (x, y, start_idx)
so the trainer can align historical driver slices."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import h5py

from lib.TrainInits import seed_worker
from lib.load_dataset import time_add
from lib.normalization import MinMax01Scaler, NScaler, StandardScaler, MinMax11Scaler, ColumnMinMaxScaler


H_CN = 41
W_CN = 71
N_CN = H_CN * W_CN
INTERVAL_MIN = 120


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _data_dir() -> str:
    return os.path.join(_repo_root(), 'data', 'china_hires')


def _year_path(year: int) -> str:
    return os.path.join(_data_dir(), f"CRIM_SW2hr_AI_v1.2_{int(year)}_DataDrivenRange_CN.hdf5")


def _resolve_china_hires_data_root(root: str) -> str:
    root = str(root or '').strip()
    if not root:
        return ''
    if os.path.isabs(root):
        return root
    return os.path.join(_repo_root(), root)


def _load_year_tec_flat_from_npy(year: int, data_root: str) -> np.ndarray:
    """Load china_hires yearly TEC from npy-only QC root.

    Expected file: <data_root>/TEC_<YEAR>.npy with shape [days,12,41,71] float32 TECU.
    Returns: [T,N] float32 where T=days*12 and N=41*71.
    """
    data_root = _resolve_china_hires_data_root(data_root)
    if not data_root:
        raise ValueError("data_root is empty for npy loader")
    path = os.path.join(data_root, f"TEC_{int(year)}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode='r')
    if arr.dtype != np.float32:
        raise ValueError(f"Unexpected TEC dtype for {year}: {arr.dtype} (expected float32)")
    if arr.ndim != 4 or arr.shape[1:] != (12, H_CN, W_CN):
        raise ValueError(f"Unexpected TEC shape for {year}: {arr.shape} (expected [days,12,41,71])")
    tec = np.asarray(arr, dtype=np.float32)  # materialize; downstream expects ndarray
    days = int(tec.shape[0])
    tec = tec.reshape(days * 12, H_CN, W_CN).astype(np.float32, copy=False)
    if not np.isfinite(tec).all():
        raise ValueError(f"Non-finite TEC detected in npy for year={year}: {path}")
    return tec.reshape(days * 12, -1).astype(np.float32, copy=False)


def _load_year_tec_flat(year: int) -> np.ndarray:
    path = _year_path(year)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, 'r') as f:
        tec = f['ionosphere/TEC'][:]  # [T, H, W]
        if tec.ndim != 3 or tec.shape[1:] != (H_CN, W_CN):
            raise ValueError(f"Unexpected TEC shape for {year}: {tec.shape}")
        tec = tec.astype(np.float32)
        fv = f['ionosphere/TEC'].attrs.get('_FillValue', None)
        if fv is not None:
            # Mask provider fill values defensively; the source QA reported no filled cells.
            tec = np.where(np.isclose(tec, float(fv), rtol=0.0, atol=1e-3), np.nan, tec)
    T = tec.shape[0]
    return tec.reshape(T, -1)  # [T, N]


def load_year_tec_flat_tecu(year: int, *, china_hires_data_root: str = "") -> np.ndarray:
    """Load china_hires yearly TEC in TECU domain as [T,N] float32.

    When `china_hires_data_root` is provided, load from QC npy-only root:
      <data_root>/TEC_<YEAR>.npy (shape [days,12,41,71], float32 TECU)
    Otherwise, load from raw HDF5 under data/china_hires/.
    """
    qc_root = _resolve_china_hires_data_root(china_hires_data_root)
    if qc_root:
        return _load_year_tec_flat_from_npy(int(year), qc_root)
    return _load_year_tec_flat(int(year))


def _with_time_features(data_flat: np.ndarray, year: int, interval_min: int) -> np.ndarray:
    # data_flat: [T, N]
    week_start = datetime(int(year), 1, 1, tzinfo=timezone.utc).isoweekday()  # 1..7 (Mon..Sun)
    day_data, week_data, _ = time_add(
        data_flat,
        week_start=week_start,
        interval=interval_min,
        weekday_only=False,
        holiday_list=[],
    )
    day_data = np.expand_dims(day_data, axis=-1).astype(np.int16)
    week_data = np.expand_dims(week_data, axis=-1).astype(np.int16)
    base = np.expand_dims(data_flat, axis=-1).astype(np.float32)
    return np.concatenate([base, day_data, week_data], axis=-1).astype(np.float32)  # [T,N,3]


def _year_start_idx(year: int, start_year: int, interval_min: int) -> int:
    """Return global index of `year-01-01 00:00` in a continuous cadence timeline."""
    t0 = datetime(int(start_year), 1, 1, tzinfo=timezone.utc)
    t = datetime(int(year), 1, 1, tzinfo=timezone.utc)
    dt = t - t0
    return int(dt.total_seconds() // float(interval_min * 60))


def _year_len_steps(year: int, start_year: int, interval_min: int) -> int:
    return _year_start_idx(year + 1, start_year, interval_min) - _year_start_idx(year, start_year, interval_min)


def _fit_scalers(train_data: np.ndarray, input_base_dim: int, normalizer: str, column_wise: bool):
    # Follow the GIM loader scaler conventions, with a fixed 300-TECU regional scale.
    if normalizer == 'tec01_300':
        scaler_data = MinMax01Scaler(0.0, 300.0)
        scaler_day = NScaler()
        scaler_week = NScaler()
    elif normalizer == 'tec01':
        # Keep the 147.1-TECU scale used by the global pipeline.
        scaler_data = MinMax01Scaler(0.0, 147.1)
        scaler_day = NScaler()
        scaler_week = NScaler()
    elif normalizer == 'std':
        if column_wise:
            mean = train_data.mean(axis=0, keepdims=True)
            std = train_data.std(axis=0, keepdims=True)
            scaler_data = StandardScaler(mean[..., :input_base_dim], std[..., :input_base_dim])
            scaler_day = StandardScaler(mean[..., input_base_dim:input_base_dim + 1], std[..., input_base_dim:input_base_dim + 1])
            scaler_week = StandardScaler(mean[..., input_base_dim + 1:input_base_dim + 2], std[..., input_base_dim + 1:input_base_dim + 2])
        else:
            td = train_data[..., :input_base_dim]
            dd = train_data[..., input_base_dim:input_base_dim + 1]
            wd = train_data[..., input_base_dim + 1:input_base_dim + 2]
            scaler_data = StandardScaler(td.mean(), td.std())
            scaler_day = StandardScaler(dd.mean(), dd.std())
            scaler_week = StandardScaler(wd.mean(), wd.std())
    elif normalizer == 'max01':
        if column_wise:
            minimum = train_data.min(axis=0, keepdims=True)
            maximum = train_data.max(axis=0, keepdims=True)
        else:
            minimum = train_data.min()
            maximum = train_data.max()
        scaler_data = MinMax01Scaler(minimum, maximum)
        scaler_day = NScaler()
        scaler_week = NScaler()
    elif normalizer == 'max11':
        if column_wise:
            minimum = train_data.min(axis=0, keepdims=True)
            maximum = train_data.max(axis=0, keepdims=True)
        else:
            minimum = train_data.min()
            maximum = train_data.max()
        scaler_data = MinMax11Scaler(minimum, maximum)
        scaler_day = NScaler()
        scaler_week = NScaler()
    elif normalizer == 'cmax':
        scaler_data = ColumnMinMaxScaler(train_data.min(axis=0), train_data.max(axis=0))
        scaler_day = NScaler()
        scaler_week = NScaler()
    else:
        scaler_data = NScaler()
        scaler_day = NScaler()
        scaler_week = NScaler()
    return scaler_data, scaler_day, scaler_week


class _SegmentWindowDataset(Dataset):
    """Year-segment windows returning (x, y, global_start_idx) for driver alignment."""

    def __init__(
        self,
        seg: np.ndarray,  # [T, N, C]
        lag: int,
        horizon: int,
        single: bool,
        scaler_data,
        scaler_day,
        scaler_week,
        input_base_dim: int,
        stride: int = 1,
        global_start_idx: int = 0,
    ):
        super().__init__()
        self.seg = seg
        self.lag = int(lag)
        self.horizon = int(horizon)
        self.single = bool(single)
        self.scaler_data = scaler_data
        self.scaler_day = scaler_day
        self.scaler_week = scaler_week
        self.input_base_dim = int(input_base_dim)
        self.stride = max(1, int(stride))
        self.global_start_idx = int(global_start_idx)
        T = int(seg.shape[0])
        end = max(0, T - (self.lag + self.horizon) + 1)
        self.length = 0 if end <= 0 else (end + self.stride - 1) // self.stride

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s = int(idx) * self.stride
        e_x = s + self.lag
        if self.single:
            s_y = s + self.lag + self.horizon - 1
            e_y = s + self.lag + self.horizon
        else:
            s_y = s + self.lag
            e_y = s + self.lag + self.horizon
        x = self.seg[s:e_x]
        y = self.seg[s_y:e_y]
        xb = self.scaler_data.transform(x[..., :self.input_base_dim])
        yb = self.scaler_data.transform(y[..., :self.input_base_dim])
        xd = self.scaler_day.transform(x[..., self.input_base_dim:self.input_base_dim + 1])
        yd = self.scaler_day.transform(y[..., self.input_base_dim:self.input_base_dim + 1])
        xw = self.scaler_week.transform(x[..., self.input_base_dim + 1:self.input_base_dim + 2])
        yw = self.scaler_week.transform(y[..., self.input_base_dim + 1:self.input_base_dim + 2])
        x = np.concatenate([xb, xd, xw], axis=-1)
        y = np.concatenate([yb, yd, yw], axis=-1)
        start_idx = self.global_start_idx + s
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), int(start_idx)


def _drivers_out_path() -> str:
    return os.path.join(_data_dir(), 'drivers', 'drivers_2014_2024_2h_cn.npz')


def _fill_f107_daily_linear(f107_2h: np.ndarray, fill_value: float, steps_per_day: int = 12) -> np.ndarray:
    """Interpolate missing daily F10.7 values and repeat them at two-hour cadence.

    F10.7 is constant within each day (twelve two-hour samples).
    A missing day has the fill value in every sample."""
    x = np.asarray(f107_2h, dtype=np.float32).reshape(-1)
    if steps_per_day <= 0:
        return x
    days = int(x.shape[0] // steps_per_day)
    if days * steps_per_day != x.shape[0]:
        # Return the original sequence when samples do not align to complete days.
        return x
    m = x.reshape(days, steps_per_day)
    # A day is missing only when all of its samples equal the fill value.
    day_missing = np.all(np.isclose(m, float(fill_value), rtol=0.0, atol=1e-3), axis=1)
    daily = m[:, 0].astype(np.float32)
    day_valid = (~day_missing) & np.isfinite(daily)
    if int(day_valid.sum()) == 0:
        return x
    idx = np.arange(days, dtype=np.float32)
    filled_daily = np.interp(idx, idx[day_valid], daily[day_valid]).astype(np.float32)
    out = np.repeat(filled_daily[:, None], steps_per_day, axis=1).reshape(-1)
    return out.astype(np.float32)


def ensure_china_hires_drivers_npz(overwrite: bool = False) -> str:
    """Build or reuse the china_hires driver NPZ for DriversStore slicing.

    Contents: time [T] int64 UTC seconds; AE/Dst/Kp/F107 [T] float32;
    grid_shape [2] (H, W); lon_deg/lat_deg [N] float32 flattened coordinates
    for solar geometry. Missing F10.7 days use linear daily interpolation."""
    out_path = _drivers_out_path()
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path) and (not overwrite):
        return out_path

    years = list(range(2014, 2025))
    # grid meta from any year file
    lat = None
    lon = None
    for y in years:
        p = _year_path(y)
        if os.path.exists(p):
            with h5py.File(p, 'r') as f:
                lat = np.asarray(f['coordinates/latitude'][:], dtype=np.float32).reshape(-1)
                lon = np.asarray(f['coordinates/longitude'][:], dtype=np.float32).reshape(-1)
            break
    if lat is None or lon is None:
        raise FileNotFoundError('No china_hires HDF5 files found to infer grid.')
    if lat.size != H_CN or lon.size != W_CN:
        raise ValueError(f"Unexpected grid meta: lat={lat.size}, lon={lon.size} (expected {H_CN},{W_CN})")
    Lon, Lat = np.meshgrid(lon.astype(np.float32), lat.astype(np.float32))
    lon_deg = Lon.reshape(-1).astype(np.float32)
    lat_deg = Lat.reshape(-1).astype(np.float32)

    time_all = []
    ae_all = []
    dst_all = []
    kp_all = []
    f107_all = []

    for y in years:
        path = _year_path(y)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with h5py.File(path, 'r') as f:
            t = np.asarray(f['coordinates/time'][:], dtype=np.float64).reshape(-1)
            # Round float64 UTC seconds to int64, then verify the sampling interval.
            t = np.round(t).astype(np.int64)
            if t.size > 1:
                dt = np.diff(t)
                if not np.all(dt == int(INTERVAL_MIN * 60)):
                    raise ValueError(f"Unexpected time step in {os.path.basename(path)}: {np.unique(dt)[:5]}")

            ae_ds = f['space_weather_indices/AE_Index']
            dst_ds = f['space_weather_indices/Dst_Index']
            kp_ds = f['space_weather_indices/Kp_Index']
            f107_ds = f['space_weather_indices/F107_Index']

            ae = np.asarray(ae_ds[:], dtype=np.float32).reshape(-1)
            dst = np.asarray(dst_ds[:], dtype=np.float32).reshape(-1)
            kp_raw = np.asarray(kp_ds[:], dtype=np.float32).reshape(-1)
            kp_scale = float(kp_ds.attrs.get('scale_factor', 0.1))
            # Mask Kp fill values in the stored domain before applying its scale factor.
            kp_fill = kp_ds.attrs.get('_FillValue', None)
            if kp_fill is not None:
                kp_raw = np.where(np.isclose(kp_raw, float(kp_fill), rtol=0.0, atol=1e-3), np.nan, kp_raw).astype(np.float32)
            kp = (kp_raw * kp_scale).astype(np.float32)

            f107 = np.asarray(f107_ds[:], dtype=np.float32).reshape(-1)
            f107_fill = float(f107_ds.attrs.get('_FillValue', 999.9))
            f107 = _fill_f107_daily_linear(f107, f107_fill, steps_per_day=int(24 * 60 // INTERVAL_MIN))

            # Mask fill values in the remaining indices.
            def _mask_fill(arr: np.ndarray, ds) -> np.ndarray:
                fv = ds.attrs.get('_FillValue', None)
                if fv is None:
                    return arr
                return np.where(np.isclose(arr, float(fv), rtol=0.0, atol=1e-3), np.nan, arr).astype(np.float32)

            ae = _mask_fill(ae, ae_ds)
            dst = _mask_fill(dst, dst_ds)
            # F10.7 fill values have already been interpolated.

        time_all.append(t)
        ae_all.append(ae)
        dst_all.append(dst)
        kp_all.append(kp)
        f107_all.append(f107)

    time = np.concatenate(time_all, axis=0).astype(np.int64)
    AE = np.concatenate(ae_all, axis=0).astype(np.float32)
    Dst = np.concatenate(dst_all, axis=0).astype(np.float32)
    Kp = np.concatenate(kp_all, axis=0).astype(np.float32)
    F107 = np.concatenate(f107_all, axis=0).astype(np.float32)

    # Verify continuity across calendar years.
    if time.size > 1:
        dt = np.diff(time)
        if not np.all(dt == int(INTERVAL_MIN * 60)):
            raise ValueError(f"Concatenated drivers timeline has non-2h gaps: {np.unique(dt)[:5]}")

    np.savez_compressed(
        out_path,
        time=time,
        AE=AE,
        Dst=Dst,
        Kp=Kp,
        F107=F107,
        grid_shape=np.asarray([H_CN, W_CN], dtype=np.int16),
        lon_deg=lon_deg,
        lat_deg=lat_deg,
    )
    return out_path


def build_china_hires_dataloaders(
    args,
    normalizer: Optional[str] = None,
    single: bool = False,
    stride: int = 1,
    prefix_boundary: bool = True,
):
    """Build the china_hires train/validation/evaluation dataloaders.

    Default year roles (Table 1 of the article):
      - train:      2014, 2016, 2018, 2020, 2022, 2023
      - validation: 2017, 2021
      - evaluation: 2015 (Jan-Jun calibration and Jul-Dec evaluation, see
                    lib.uq.conformal.china_calendar_roles), 2019, 2024

    Returns:
      train_loader, val_loader, {'test': ..., 'ood': ...},
      scaler_data, scaler_day, scaler_week, None"""
    def _split_profile_key() -> str:
        v = getattr(args, 'split_profile', None)
        if v is None:
            return 'default'
        s = str(v).strip().lower()
        return s or 'default'

    def _split_cfg(profile: str) -> Tuple[List[int], List[int], List[int], List[int], Dict[int, int]]:
        """Return (train_years, val_years, test_years, ood_years, prefix_src)."""
        if profile in ('d', 'batchd', 'd-split'):
            # Alternative D profile with its own year roles and preceding-year histories.
            train = [2014, 2016, 2018, 2020]
            val = [2017, 2021]
            test = [2015]
            ood = [2019, 2022]
            prefix = {
                2015: 2014,
                2017: 2016,
                2019: 2018,
                2021: 2020,
                2022: 2021,
            }
            return train, val, test, ood, prefix
        # Default calendar-year roles.
        train = [2014, 2016, 2018, 2020, 2022, 2023]
        val = [2017, 2021]
        test = [2015]
        ood = [2019, 2024]
        prefix = {
            2015: 2014,
            2017: 2016,
            2019: 2018,
            2021: 2020,
            2024: 2023,
        }
        return train, val, test, ood, prefix

    split_profile = _split_profile_key()
    train_years, val_years, test_years, ood_years, prefix_src = _split_cfg(split_profile)
    setattr(args, 'split_profile', split_profile)

    # Ensure essential attrs exist on args for downstream modules
    setattr(args, 'interval', int(getattr(args, 'interval', INTERVAL_MIN)))
    setattr(args, 'week_day', int(getattr(args, 'week_day', 7)))
    setattr(args, 'height', int(getattr(args, 'height', H_CN)))
    setattr(args, 'width', int(getattr(args, 'width', W_CN)))

    interval_min = int(getattr(args, 'interval', INTERVAL_MIN))
    if interval_min != INTERVAL_MIN:
        # The source cadence is two hours; an explicit interval override is retained.
        pass

    # Preload year segments so preceding-year histories can be selected.
    year_cache: Dict[int, np.ndarray] = {}
    qc_root = _resolve_china_hires_data_root(getattr(args, 'china_hires_data_root', '') or '')
    if qc_root:
        print(f"[china_hires] Using npy-only data_root: {qc_root}")

    # Prepare driver slices for predictor inputs, physical losses or diagnostics.
    try:
        if bool(getattr(args, 'use_drivers', False)) or bool(getattr(args, 'use_pinn', False)):
            ensure_china_hires_drivers_npz(overwrite=False)
    except Exception:
        # Continue if drivers are unavailable; the trainer uses its existing zero-driver fallback.
        pass

    def pack_year(y: int) -> np.ndarray:
        if qc_root:
            flat = _load_year_tec_flat_from_npy(int(y), qc_root)  # [T,N]
        else:
            flat = _load_year_tec_flat(int(y))  # [T,N]
        seg = _with_time_features(flat, int(y), interval_min=interval_min)  # [T,N,3]
        if getattr(args, 'debug', False):
            max_steps = int(getattr(args, 'debug_max_steps', 1000))
            if seg.shape[0] > max_steps:
                seg = seg[:max_steps]
        return seg

    for y in sorted(set(train_years + val_years + test_years + ood_years)):
        year_cache[int(y)] = pack_year(int(y))

    def maybe_prefix(y: int, seg: np.ndarray) -> np.ndarray:
        if not bool(prefix_boundary):
            return seg
        src = prefix_src.get(int(y), None)
        if src is None:
            return seg
        tail = year_cache[int(src)][-int(args.lag):]
        return np.concatenate([tail, seg], axis=0)

    # Fit scalers on training targets only, excluding preceding-year history.
    train_for_scaler = np.concatenate([year_cache[y] for y in train_years], axis=0)
    norm = normalizer if normalizer is not None else str(getattr(args, 'normalizer', 'std'))
    scaler_data, scaler_day, scaler_week = _fit_scalers(
        train_for_scaler,
        int(getattr(args, 'input_base_dim', 1)),
        norm,
        bool(getattr(args, 'column_wise', False)),
    )

    base_year = 2014

    def y0(y: int) -> int:
        return _year_start_idx(int(y), base_year, interval_min)

    def ylen(y: int) -> int:
        return _year_len_steps(int(y), base_year, interval_min)

    def seg_ds(seg: np.ndarray, global_start_idx: int) -> Dataset:
        return _SegmentWindowDataset(
            seg,
            int(args.lag),
            int(args.horizon),
            bool(single),
            scaler_data,
            scaler_day,
            scaler_week,
            int(getattr(args, 'input_base_dim', 1)),
            stride=int(stride),
            global_start_idx=int(global_start_idx),
        )

    # Keep each training year as a separate segment.
    train_ds_list = [seg_ds(year_cache[y], y0(y)) for y in train_years]

    # Optionally prepend preceding-year history and shift the global index by lag.
    val_segs = [maybe_prefix(y, year_cache[y]) for y in val_years]
    test_segs = [maybe_prefix(y, year_cache[y]) for y in test_years]
    ood_segs = [maybe_prefix(y, year_cache[y]) for y in ood_years]

    def start_with_prefix(y: int) -> int:
        if not bool(prefix_boundary):
            return y0(y)
        src = prefix_src.get(int(y), None)
        if src is None:
            return y0(y)
        return y0(src) + ylen(src) - int(args.lag)

    val_starts = [start_with_prefix(y) for y in val_years]
    test_starts = [start_with_prefix(y) for y in test_years]
    ood_starts = [start_with_prefix(y) for y in ood_years]

    val_ds_list = [seg_ds(seg, st) for seg, st in zip(val_segs, val_starts)]
    test_ds_list = [seg_ds(seg, st) for seg, st in zip(test_segs, test_starts)]
    ood_ds_list = [seg_ds(seg, st) for seg, st in zip(ood_segs, ood_starts)]

    train_dataset = ConcatDataset(train_ds_list) if train_ds_list else None
    val_dataset = ConcatDataset(val_ds_list) if val_ds_list else None
    test_dataset = ConcatDataset(test_ds_list) if test_ds_list else None
    ood_dataset = ConcatDataset(ood_ds_list) if ood_ds_list else None

    num_workers = min(8, max(0, (os.cpu_count() or 8) - 2))

    def _seeded_dl_kwargs(offset: int = 0):
        try:
            g = torch.Generator()
            g.manual_seed(int(getattr(args, 'seed', 0)) + int(offset))
            return {'generator': g, 'worker_init_fn': seed_worker}
        except Exception:
            return {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        **_seeded_dl_kwargs(0),
    ) if train_dataset is not None else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        **_seeded_dl_kwargs(1),
    ) if val_dataset is not None else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        **_seeded_dl_kwargs(2),
    ) if test_dataset is not None else None

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        **_seeded_dl_kwargs(3),
    ) if ood_dataset is not None else None

    # Return separate test and ood loaders for the trainer.
    return train_loader, val_loader, {'test': test_loader, 'ood': ood_loader}, scaler_data, scaler_day, scaler_week, None


def load(args) -> np.ndarray:
    """Plugin-style loader for the non-year-split fallback.

    By default, concatenate 2014-2024 and return float32 [T, N].
    Use -year_split True with build_china_hires_dataloaders for year roles."""
    years = list(range(2014, 2025))
    parts = []
    for y in years:
        parts.append(_load_year_tec_flat(y))
    data = np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, N_CN), dtype=np.float32)
    # Set the common dataset metadata.
    setattr(args, 'interval', int(getattr(args, 'interval', INTERVAL_MIN)))
    setattr(args, 'week_day', int(getattr(args, 'week_day', 7)))
    setattr(args, 'height', int(getattr(args, 'height', H_CN)))
    setattr(args, 'width', int(getattr(args, 'width', W_CN)))
    return data
