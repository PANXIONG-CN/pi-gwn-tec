import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lib.normalization import MinMax01Scaler, NScaler
from lib.TrainInits import seed_worker


FIXED_MIN = 0.0
FIXED_MAX = 147.1
H = 71
W = 73


def _load_year(base_dir, year):
    path = os.path.join(base_dir, f"TEC_{year}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    arr = arr.astype(np.float32) / 10.0
    if arr.ndim == 4:
        # e.g., [D,M,H,W] -> flatten to T,H,W
        d0, d1, h, w = arr.shape
        arr = arr.reshape(d0 * d1, h, w)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected dim for {path}: {arr.shape}")
    # MinMax to [0,1] with fixed max
    arr = (arr - FIXED_MIN) / (FIXED_MAX - FIXED_MIN)
    return arr


def _split_data(data, time_step, predict_step):
    x, y = [], []
    T = data.shape[0]
    for i in range(0, T - time_step - predict_step + 1, predict_step):
        x.append(data[i:i + time_step])
        y.append(data[i + time_step:i + time_step + predict_step])
    return np.array(x, dtype='float32'), np.array(y, dtype='float32')


class _TensorDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_csa_vendor_dataloaders(args):
    # Resolve GIMtec data directory relative to repo root to avoid cwd issues
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    base_dir = os.path.join(_repo, 'data', 'GIMtec')
    time_step = args.lag
    predict_step = args.horizon

    # Train years
    y2009 = _load_year(base_dir, 2009)
    y2010 = _load_year(base_dir, 2010)
    y2011 = _load_year(base_dir, 2011)
    y2012 = _load_year(base_dir, 2012)
    y2014 = _load_year(base_dir, 2014)
    y2016 = _load_year(base_dir, 2016)
    y2018 = _load_year(base_dir, 2018)
    train_2009_2012 = np.concatenate([y2009, y2010, y2011, y2012], axis=0)

    # Val years with prefix
    y2013 = _load_year(base_dir, 2013)
    y2017 = _load_year(base_dir, 2017)
    val_2013 = np.concatenate([train_2009_2012[-time_step:], y2013], axis=0)
    val_2017 = np.concatenate([y2016[-time_step:], y2017], axis=0)

    # Test years with cascading prefix
    y2015 = _load_year(base_dir, 2015)
    y2019 = _load_year(base_dir, 2019)
    y2020 = _load_year(base_dir, 2020)
    y2021 = _load_year(base_dir, 2021)
    y2022 = _load_year(base_dir, 2022)
    test_2015 = np.concatenate([y2014[-time_step:], y2015], axis=0)
    test_2019 = np.concatenate([y2018[-time_step:], y2019], axis=0)
    test_2020 = np.concatenate([test_2019[-time_step:], y2020], axis=0)
    test_2021 = np.concatenate([test_2020[-time_step:], y2021], axis=0)
    test_2022 = np.concatenate([test_2021[-time_step:], y2022], axis=0)

    # Split windows with step=predict_step
    tr_x1, tr_y1 = _split_data(train_2009_2012, time_step, predict_step)
    tr_x2, tr_y2 = _split_data(y2014, time_step, predict_step)
    tr_x3, tr_y3 = _split_data(y2016, time_step, predict_step)
    tr_x4, tr_y4 = _split_data(y2018, time_step, predict_step)
    va_x1, va_y1 = _split_data(val_2013, time_step, predict_step)
    va_x2, va_y2 = _split_data(val_2017, time_step, predict_step)
    te_x1, te_y1 = _split_data(test_2015, time_step, predict_step)
    te_x2, te_y2 = _split_data(test_2019, time_step, predict_step)
    te_x3, te_y3 = _split_data(test_2020, time_step, predict_step)
    te_x4, te_y4 = _split_data(test_2021, time_step, predict_step)
    te_x5, te_y5 = _split_data(test_2022, time_step, predict_step)

    train_x = np.concatenate([tr_x1, tr_x2, tr_x3, tr_x4], axis=0)
    train_y = np.concatenate([tr_y1, tr_y2, tr_y3, tr_y4], axis=0)
    val_x = np.concatenate([va_x1, va_x2], axis=0)
    val_y = np.concatenate([va_y1, va_y2], axis=0)
    test_x = np.concatenate([te_x1, te_x2, te_x3, te_x4, te_x5], axis=0)
    test_y = np.concatenate([te_y1, te_y2, te_y3, te_y4, te_y5], axis=0)

    # Flatten to [B, T, N, 1]
    Btr, Ttr = train_x.shape[:2]
    train_x = train_x.reshape(Btr, Ttr, H * W, 1)
    train_y = train_y.reshape(Btr, predict_step, H * W, 1)
    Bv, Tv = val_x.shape[:2]
    val_x = val_x.reshape(Bv, Tv, H * W, 1)
    val_y = val_y.reshape(Bv, predict_step, H * W, 1)
    Bte, Tte = test_x.shape[:2]
    test_x = test_x.reshape(Bte, Tte, H * W, 1)
    test_y = test_y.reshape(Bte, predict_step, H * W, 1)

    # Scalers: data base channel minmax, extras are identity
    scaler_data = MinMax01Scaler(FIXED_MIN, FIXED_MAX)
    scaler_day = NScaler()
    scaler_week = NScaler()

    # DataLoaders (CPU tensors + pin_memory + workers)
    num_workers = min(12, max(0, (os.cpu_count() or 12) - 2))
    def _seeded_dl_kwargs(offset: int = 0):
        try:
            g = torch.Generator()
            g.manual_seed(int(getattr(args, 'seed', 0)) + int(offset))
            return {'generator': g, 'worker_init_fn': seed_worker}
        except Exception:
            return {}
    train_ds = _TensorDataset(train_x, train_y)
    val_ds = _TensorDataset(val_x, val_y)
    test_ds = _TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                              persistent_workers=True if num_workers > 0 else False,
                              **_seeded_dl_kwargs(0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                            persistent_workers=True if num_workers > 0 else False,
                            **_seeded_dl_kwargs(1))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                             persistent_workers=True if num_workers > 0 else False,
                             **_seeded_dl_kwargs(2))

    return train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, None
