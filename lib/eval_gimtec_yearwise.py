import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from lib.datasets.gimtec_vendor import _load_year, FIXED_MIN, FIXED_MAX, H, W
from lib.load_dataset import time_add
from lib.normalization import MinMax01Scaler, NScaler


def _prefix_concat(base, prefix_src, time_step):
    return np.concatenate([prefix_src[-time_step:], base], axis=0)


def _split_with_stride(flat: np.ndarray, lag: int, horizon: int, step: int):
    """flat: [T, N] -> X:[B, lag, N], Y:[B, horizon, N]"""
    X, Y = [], []
    T = flat.shape[0]
    for i in range(0, T - lag - horizon + 1, step):
        X.append(flat[i:i+lag])
        Y.append(flat[i+lag:i+lag+horizon])
    return np.array(X), np.array(Y)


def build_year_loader(args, year, time_step, predict_step, prefix_from=None, batch_size=None):
    # Resolve GIMtec data directory relative to repo root to avoid cwd issues
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    base_dir = os.path.join(repo_root, 'data', 'GIMtec')
    y = _load_year(base_dir, year)
    if prefix_from is not None:
        y_prefix = _load_year(base_dir, prefix_from)
        y = _prefix_concat(y, y_prefix, time_step)
    # Flatten to [T, N]
    T_total = y.shape[0]
    base_flat = y.reshape(T_total, H * W)
    # Add day/week features to align with GPT-ST eval pipeline
    interval_min = getattr(args, 'interval', 5)
    week_start = getattr(args, 'week_start', 4)
    day_data, week_data, _ = time_add(base_flat, week_start, interval=interval_min, weekday_only=False, holiday_list=[])
    # Windows with step=predict_step to match vendor
    xb, yb = _split_with_stride(base_flat, time_step, predict_step, step=predict_step)
    xd, yd = _split_with_stride(day_data, time_step, predict_step, step=predict_step)
    xw, yw = _split_with_stride(week_data, time_step, predict_step, step=predict_step)
    # Compose [B, T, N, 3]
    x = np.concatenate([xb[..., None], xd[..., None], xw[..., None]], axis=-1)
    y_t = np.concatenate([yb[..., None], yd[..., None], yw[..., None]], axis=-1)
    ds = _TensorDataset(x, y_t)
    loader = DataLoader(ds, batch_size=(batch_size or args.batch_size), shuffle=False,
                        drop_last=False, pin_memory=torch.cuda.is_available())
    scaler_data = MinMax01Scaler(FIXED_MIN, FIXED_MAX)
    return loader, scaler_data


class _TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluate_per_year(model, trainer):
    """Evaluate CSA-WTConvLSTM on GIMtec year by year and log metrics.
    Matches original dataset splits and prefix stitching.
    """
    args = trainer.args
    logger = trainer.logger
    time_step = args.lag
    predict_step = args.horizon

    # Val years
    val_years = [(2013, [2009, 2010, 2011, 2012]), (2017, [2016])]
    # Test years
    test_years = [(2015, [2014]), (2019, [2018]), (2020, [2019]), (2021, [2020]), (2022, [2021])]

    def _run_for_year(year, prefixes):
        prefix_from = prefixes[-1] if prefixes else None
        loader, scaler = build_year_loader(args, year, time_step, predict_step, prefix_from=prefix_from,
                                           batch_size=args.batch_size)
        trainer.test(model, args, loader, scaler, logger)

    logger.info('==== Year-wise Validation Metrics ====')
    for year, prefixes in val_years:
        logger.info(f'-- Year {year} --')
        _run_for_year(year, prefixes)

    logger.info('==== Year-wise Test Metrics ====')
    for year, prefixes in test_years:
        logger.info(f'-- Year {year} --')
        _run_for_year(year, prefixes)


def compute_yearwise_metrics(model, trainer):
    """Return JSON-style metrics overall and per-year for GIMtec.
    Metrics include normalized MSE/RMSE (on [0,1]) and real TECU MAE/RMSE.
    """
    import json
    args = trainer.args
    time_step = args.lag
    predict_step = args.horizon

    years = {
        '2015': ['2014'],
        '2019': ['2018'],
        '2020': ['2019'],
        '2021': ['2020'],
        '2022': ['2021'],
    }

    FIX_MAX = 147.1

    def _eval_loader(loader):
        import torch
        from lib.metrics import MSE_torch, MAE_torch, RMSE_torch
        # 确保模型在正确设备并处于 eval 状态
        net = model if model is not None else trainer.model
        try:
            net = net.to(args.device)
        except Exception:
            pass
        was_train = net.training
        net.eval()
        y_true_norm_all = []
        y_pred_norm_all = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(args.device, non_blocking=True).float()
                y = y.to(args.device, non_blocking=True).float()
                # 仅评估基础通道（TECU）；[B,T,N,1]
                y = y[..., :1]
                out = net(x, label=None)[0]
                if out.dim() == 3:
                    out = out.unsqueeze(-1)
                y_true_norm_all.append(y)
                y_pred_norm_all.append(out)
        # 还原训练态
        if was_train:
            net.train()
        y_true_norm = torch.cat(y_true_norm_all, dim=0)
        y_pred_norm = torch.cat(y_pred_norm_all, dim=0)
        mse_norm = float(MSE_torch(y_pred_norm, y_true_norm, None).item())
        rmse_norm = float(mse_norm ** 0.5)
        # to TECU
        y_true = y_true_norm * FIX_MAX
        y_pred = y_pred_norm * FIX_MAX
        rmse = float(RMSE_torch(y_pred, y_true, None).item())
        mae, _ = MAE_torch(y_pred, y_true, None)
        mae = float(mae.item())
        count = int(y_true.shape[0] * y_true.shape[1])
        return mse_norm, rmse_norm, rmse, mae, count

    overall = {
        'mse_norm': 0.0,
        'rmse_norm': 0.0,
        'rmse_real_TECU': 0.0,
        'mae_real_TECU': 0.0,
        'relative_error_percent': 0.0,
        'count_frames': 0,
        'shape_per_frame': [1, getattr(args, 'height', 71), getattr(args, 'width', 73)],
        'per_year': {}
    }

    total_mse_sum = 0.0
    total_rmse_sum = 0.0
    total_mae_sum = 0.0
    total_frames = 0

    # per year
    for year, prefixes in years.items():
        loader, _ = build_year_loader(args, int(year), time_step, predict_step, prefix_from=int(prefixes[-1]))
        mse_n, rmse_n, rmse_tecu, mae_tecu, frames = _eval_loader(loader)
        overall['per_year'][year] = {
            'mse_norm': float(mse_n),
            'rmse_norm': float(rmse_n),
            'rmse_real_TECU': float(rmse_tecu),
            'mae_real_TECU': float(mae_tecu),
            'relative_error_percent': float(rmse_n * 100.0),
            'windows': int(frames // predict_step),
        }
        total_mse_sum += mse_n * frames
        total_rmse_sum += rmse_n * frames
        total_mae_sum += mae_tecu * frames
        total_frames += frames

    # overall (weighted by frames)
    if total_frames > 0:
        mse_norm = total_mse_sum / total_frames
        rmse_norm = total_rmse_sum / total_frames
        mae_real = total_mae_sum / total_frames
        overall['mse_norm'] = float(mse_norm)
        overall['rmse_norm'] = float(rmse_norm)
        overall['rmse_real_TECU'] = float(rmse_norm * FIX_MAX)
        overall['mae_real_TECU'] = float(mae_real)
        overall['relative_error_percent'] = float(rmse_norm * 100.0)
        overall['count_frames'] = int(total_frames)
    return overall
