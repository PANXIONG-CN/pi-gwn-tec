"""
GIMtec 预训练（GPT‑ST）专用 DataLoader：按年份划分训练/验证/测试。

- 训练: 2009, 2010, 2011, 2012, 2014, 2016, 2018
- 验证: 2013, 2017
- 测试: 2015, 2019, 2020, 2021, 2022

数据处理流程：
- 逐年读取 `../data/GIMtec/TEC_YYYY.npy`，转换为 [T, N]，N = 71*73 = 5183
- 为每年生成 day/week 通道，拼接得到 [T, N, 3]
- 按年份拼接成 train/val/test 三段
- 用训练段拟合归一化器（支持 tec01/std 等），并对三段生成窗口后再分别缩放
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from datetime import datetime, timezone

from lib.add_window import Add_Window_Horizon
from lib.load_dataset import time_add
from lib.normalization import MinMax01Scaler, NScaler, StandardScaler, MinMax11Scaler, ColumnMinMaxScaler
from lib.TrainInits import seed_worker


H, W = 71, 73
N = H * W


def _load_year_raw(base_dir: str, year: int) -> np.ndarray:
    path = os.path.join(base_dir, f"TEC_{year}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    arr = arr.astype(np.float32) / 10.0  # 与 vendor 保持一致的基础缩放
    if arr.ndim == 4:
        d0, d1, h, w = arr.shape
        arr = arr.reshape(d0 * d1, h, w)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected dim for {path}: {arr.shape}")
    if arr.shape[-2:] != (H, W):
        raise ValueError(f"Unexpected grid size for {year}: {arr.shape}")
    T = arr.shape[0]
    return arr.reshape(T, N)  # [T, N]


def _with_time_features(data_flat: np.ndarray, interval_min: int, week_start: int = 4):
    # data_flat: [T, N]
    day_data, week_data, _ = time_add(data_flat, week_start, interval=interval_min, weekday_only=False, holiday_list=[])
    if day_data.ndim == 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([np.expand_dims(data_flat, axis=-1), day_data, week_data], axis=-1)  # [T, N, 3]
    else:
        raise ValueError('time_add returned unexpected dims')
    return data


def _fit_scalers(train_data: np.ndarray, input_base_dim: int, normalizer: str, column_wise: bool):
    # 参考 lib/dataloader.normalize_dataset 的逻辑，返回三类 scaler
    if normalizer == 'tec01':
        # base: [0, 147.1] → [0,1]；day/week 不变
        scaler_data = MinMax01Scaler(0.0, 147.1)
        scaler_day = NScaler()
        scaler_week = NScaler()
    elif normalizer == 'std':
        if column_wise:
            mean = train_data.mean(axis=0, keepdims=True)
            std = train_data.std(axis=0, keepdims=True)
            scaler_data = StandardScaler(mean[..., :input_base_dim], std[..., :input_base_dim])
            scaler_day = StandardScaler(mean[..., input_base_dim:input_base_dim+1], std[..., input_base_dim:input_base_dim+1])
            scaler_week = StandardScaler(mean[..., input_base_dim+1:input_base_dim+2], std[..., input_base_dim+1:input_base_dim+2])
        else:
            td = train_data[..., :input_base_dim]
            dd = train_data[..., input_base_dim:input_base_dim+1]
            wd = train_data[..., input_base_dim+1:input_base_dim+2]
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
    else:
        # 其余保持与通用一致：不缩放或列缩放
        scaler_data = NScaler()
        scaler_day = NScaler()
        scaler_week = NScaler()
    return scaler_data, scaler_day, scaler_week


def _make_windows_stride(data: np.ndarray, lag: int, horizon: int, single: bool, stride: int = 1):
    """
    以给定 stride 生成滑窗。
    data: [T, N, C]
    返回: x:[B, lag, N, C], y:[B, horizon, N, C]
    """
    T = data.shape[0]
    X, Y = [], []
    end = T - (lag + horizon) + 1
    if stride <= 0:
        stride = 1
    if single:
        # 预测单步（保持和通用接口一致）
        for s in range(0, end, stride):
            X.append(data[s:s+lag])
            Y.append(data[s+lag+ horizon - 1:s+lag+ horizon])
    else:
        for s in range(0, end, stride):
            X.append(data[s:s+lag])
            Y.append(data[s+lag:s+lag+horizon])
    return np.array(X), np.array(Y)


def _transform_xy(x: np.ndarray, y: np.ndarray, input_base_dim: int, scaler_data, scaler_day, scaler_week):
    xb = scaler_data.transform(x[:, :, :, :input_base_dim])
    yb = scaler_data.transform(y[:, :, :, :input_base_dim])
    xd = scaler_day.transform(x[:, :, :, input_base_dim:input_base_dim+1])
    yd = scaler_day.transform(y[:, :, :, input_base_dim:input_base_dim+1])
    xw = scaler_week.transform(x[:, :, :, input_base_dim+1:input_base_dim+2])
    yw = scaler_week.transform(y[:, :, :, input_base_dim+1:input_base_dim+2])
    x = np.concatenate([xb, xd, xw], axis=-1)
    y = np.concatenate([yb, yd, yw], axis=-1)
    return x, y


def _to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    num_workers = min(8, max(0, (os.cpu_count() or 8) - 2))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                        pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                        persistent_workers=True if num_workers > 0 else False)
    return loader


def _year_start_idx(year: int, start_year: int, interval_min: int) -> int:
    """Return global index of `year-01-01 00:00` in a continuous cadence timeline."""
    t0 = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    t = datetime(year, 1, 1, tzinfo=timezone.utc)
    dt = (t - t0).total_seconds()
    return int(dt // (interval_min * 60))


def _year_len_steps(year: int, start_year: int, interval_min: int) -> int:
    return _year_start_idx(year + 1, start_year, interval_min) - _year_start_idx(year, start_year, interval_min)


class _SegmentWindowDataset(Dataset):
    """Streaming窗口数据集：针对单个连续年段数据 [T, N, 3]，按给定 stride 产出 (X,Y)。"""
    def __init__(self, seg: np.ndarray, lag: int, horizon: int, single: bool,
                 scaler_data, scaler_day, scaler_week, input_base_dim: int, stride: int = 1, global_start_idx: int = 0):
        super().__init__()
        self.seg = seg  # [T, N, 3]
        self.lag = lag
        self.horizon = horizon
        self.single = single
        self.scaler_data = scaler_data
        self.scaler_day = scaler_day
        self.scaler_week = scaler_week
        self.input_base_dim = input_base_dim
        self.stride = max(1, int(stride))
        self.global_start_idx = int(global_start_idx)
        T = seg.shape[0]
        end = max(0, T - (lag + horizon) + 1)
        self.length = 0 if end <= 0 else (end + self.stride - 1) // self.stride

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s = idx * self.stride
        e_x = s + self.lag
        if self.single:
            s_y = s + self.lag + self.horizon - 1
            e_y = s + self.lag + self.horizon
        else:
            s_y = s + self.lag
            e_y = s + self.lag + self.horizon
        x = self.seg[s:e_x]
        y = self.seg[s_y:e_y]
        # 组分缩放
        xb = self.scaler_data.transform(x[..., :self.input_base_dim])
        yb = self.scaler_data.transform(y[..., :self.input_base_dim])
        xd = self.scaler_day.transform(x[..., self.input_base_dim:self.input_base_dim+1])
        yd = self.scaler_day.transform(y[..., self.input_base_dim:self.input_base_dim+1])
        xw = self.scaler_week.transform(x[..., self.input_base_dim+1:self.input_base_dim+2])
        yw = self.scaler_week.transform(y[..., self.input_base_dim+1:self.input_base_dim+2])
        x = np.concatenate([xb, xd, xw], axis=-1)
        y = np.concatenate([yb, yd, yw], axis=-1)
        start_idx = self.global_start_idx + s
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), int(start_idx)


def build_gimtec_pretrain_dataloaders(args, normalizer=None, single=False, stride: int = 1, prefix_boundary: bool = True):
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    base_dir = os.path.join(_repo, 'data', 'GIMtec')
    train_years = [2009, 2010, 2011, 2012, 2014, 2016, 2018]
    val_years = [2013, 2017]
    test_years = [2015, 2019, 2020, 2021, 2022]

    interval_min = getattr(args, 'interval', 120)
    week_start = getattr(args, 'week_start', 4)
    # ensure attributes exist on args for model usage
    setattr(args, 'interval', interval_min)
    setattr(args, 'week_day', getattr(args, 'week_day', 7))

    def pack_years(years):
        parts = []
        for y in years:
            flat = _load_year_raw(base_dir, y)  # [T, N]
            parts.append(_with_time_features(flat, interval_min=interval_min, week_start=week_start))
        data = np.concatenate(parts, axis=0) if parts else None
        # debug mode: limit timesteps to reduce memory for smoke runs
        if data is not None and getattr(args, 'debug', False):
            max_steps = int(getattr(args, 'debug_max_steps', 1000))
            if data.shape[0] > max_steps:
                data = data[:max_steps]
        return data

    # 构建各段数据（含可选边界前缀）
    # 训练段：与 vendor 一致，分为四个 segment：2009-2012, 2014, 2016, 2018
    tr_segments = [pack_years([2009, 2010, 2011, 2012]), pack_years([2014]), pack_years([2016]), pack_years([2018])]
    # 验证段：2013 前缀来自 2012 段；2017 前缀来自 2016 段
    va_2013 = pack_years([2013])
    va_2017 = pack_years([2017])
    if prefix_boundary:
        tr_2009_2012 = tr_segments[0]
        va_2013 = np.concatenate([tr_2009_2012[-args.lag:], va_2013], axis=0)
        tr_2016 = tr_segments[2]
        va_2017 = np.concatenate([tr_2016[-args.lag:], va_2017], axis=0)
    # 测试段：按 vendor 级联前缀
    te_2015 = pack_years([2015])
    te_2019 = pack_years([2019])
    te_2020 = pack_years([2020])
    te_2021 = pack_years([2021])
    te_2022 = pack_years([2022])
    if prefix_boundary:
        tr_2014 = tr_segments[1]
        tr_2018 = tr_segments[3]
        te_2015 = np.concatenate([tr_2014[-args.lag:], te_2015], axis=0)
        te_2019 = np.concatenate([tr_2018[-args.lag:], te_2019], axis=0)
        te_2020 = np.concatenate([te_2019[-args.lag:], te_2020], axis=0)
        te_2021 = np.concatenate([te_2020[-args.lag:], te_2021], axis=0)
        te_2022 = np.concatenate([te_2021[-args.lag:], te_2022], axis=0)

    # 全局时间索引（drivers_2009_2022_2h）对齐：以 2009-01-01 00:00 为起点
    base_year = 2009
    def y0(y: int) -> int:
        return _year_start_idx(y, base_year, interval_min)

    def ylen(y: int) -> int:
        return _year_len_steps(y, base_year, interval_min)

    # 供 scaler 拟合的训练并集
    data_train_for_scaler = np.concatenate([seg for seg in tr_segments if seg is not None], axis=0)

    # 拟合 scaler（仅用训练集）
    norm = normalizer if normalizer is not None else getattr(args, 'normalizer', 'std')
    scaler_data, scaler_day, scaler_week = _fit_scalers(data_train_for_scaler, args.input_base_dim, norm, getattr(args, 'column_wise', False))

    # 流式 DataLoader：每个 segment 构建一个窗口数据集，训练/验证/测试分别按段 Concat
    def seg_ds(seg, global_start_idx: int):
        return _SegmentWindowDataset(
            seg,
            args.lag,
            args.horizon,
            single,
            scaler_data,
            scaler_day,
            scaler_week,
            args.input_base_dim,
            stride=stride,
            global_start_idx=global_start_idx,
        )

    # Segment start indices (global 2h cadence index)
    tr_starts = [y0(2009), y0(2014), y0(2016), y0(2018)]
    if prefix_boundary:
        va_starts = [y0(2012) + ylen(2012) - args.lag, y0(2016) + ylen(2016) - args.lag]
        te_starts = [
            y0(2014) + ylen(2014) - args.lag,  # 2015 prefixed by 2014 tail
            y0(2018) + ylen(2018) - args.lag,  # 2019 prefixed by 2018 tail
            y0(2019) + ylen(2019) - args.lag,  # 2020 prefixed by 2019 tail
            y0(2020) + ylen(2020) - args.lag,  # 2021 prefixed by 2020 tail
            y0(2021) + ylen(2021) - args.lag,  # 2022 prefixed by 2021 tail
        ]
    else:
        va_starts = [y0(2013), y0(2017)]
        te_starts = [y0(2015), y0(2019), y0(2020), y0(2021), y0(2022)]

    train_ds_list = [seg_ds(seg, st) for (seg, st) in zip(tr_segments, tr_starts) if seg is not None]
    val_ds_list = [seg_ds(seg, st) for (seg, st) in zip([va_2013, va_2017], va_starts) if seg is not None]
    test_ds_list = [seg_ds(seg, st) for (seg, st) in zip([te_2015, te_2019, te_2020, te_2021, te_2022], te_starts) if seg is not None]

    train_dataset = ConcatDataset(train_ds_list) if train_ds_list else None
    val_dataset = ConcatDataset(val_ds_list) if val_ds_list else None
    test_dataset = ConcatDataset(test_ds_list) if test_ds_list else None

    num_workers = min(8, max(0, (os.cpu_count() or 8) - 2))
    def _seeded_dl_kwargs(offset: int = 0):
        try:
            g = torch.Generator()
            g.manual_seed(int(getattr(args, 'seed', 0)) + int(offset))
            return {'generator': g, 'worker_init_fn': seed_worker}
        except Exception:
            return {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                              persistent_workers=True if num_workers > 0 else False,
                              **_seeded_dl_kwargs(0)) if train_dataset is not None else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                            persistent_workers=True if num_workers > 0 else False,
                            **_seeded_dl_kwargs(1)) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                             persistent_workers=True if num_workers > 0 else False,
                             **_seeded_dl_kwargs(2)) if test_dataset is not None else None

    return train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, None
