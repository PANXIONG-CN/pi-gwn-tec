import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
from lib.TrainInits import seed_worker

def normalize_dataset(data, normalizer, input_base_dim, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            scaler = StandardScaler(mean, std)
            data[:, :, 0:input_base_dim] = scaler.transform(data[:, :, 0:input_base_dim])
        else:
            data_ori = data[:, :, 0:input_base_dim]
            data_day = data[:, :, input_base_dim:input_base_dim+1]
            data_week = data[:, :, input_base_dim+1:input_base_dim+2]

            mean_data = data_ori.mean()
            std_data = data_ori.std()
            mean_day = data_day.mean()
            std_day = data_day.std()
            mean_week = data_week.mean()
            std_week = data_week.std()

            scaler_data = StandardScaler(mean_data, std_data)
            data_ori = scaler_data.transform(data_ori)
            scaler_day = StandardScaler(mean_day, std_day)
            data_day = scaler_day.transform(data_day)
            scaler_week = StandardScaler(mean_week, std_week)
            data_week = scaler_week.transform(data_week)
            data = np.concatenate([data_ori, data_day, data_week], axis=-1)
            print(mean_data, std_data, mean_day, std_day, mean_week, std_week)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    elif normalizer == 'tec01':
        # Follow CSA-WTConvLSTM: base channel scaled to [0,1] by fixed (min=0, max=147.1)
        # keep day/week unchanged (identity scaler), since original CSA input has no extra features
        minimum = 0.0
        maximum = 147.1
        scaler_data = MinMax01Scaler(minimum, maximum)
        # compose data back only for logging purpose
        data_ori = scaler_data.transform(data[:, :, 0:input_base_dim])
        data_day = data[:, :, input_base_dim:input_base_dim+1]
        data_week = data[:, :, input_base_dim+1:input_base_dim+2]
        scaler_day = NScaler()
        scaler_week = NScaler()
        data = np.concatenate([data_ori, data_day, data_week], axis=-1)
        print('Normalize the dataset by TEC MinMax01 with fixed [0, 147.1]')
    elif normalizer == 'tec01_300':
        # China hires: base TEC capped at 300, scale to [0,1] by fixed (min=0, max=300)
        minimum = 0.0
        maximum = 300.0
        scaler_data = MinMax01Scaler(minimum, maximum)
        data_ori = scaler_data.transform(data[:, :, 0:input_base_dim])
        data_day = data[:, :, input_base_dim:input_base_dim+1]
        data_week = data[:, :, input_base_dim+1:input_base_dim+2]
        scaler_day = NScaler()
        scaler_week = NScaler()
        data = np.concatenate([data_ori, data_day, data_week], axis=-1)
        print('Normalize the dataset by TEC MinMax01 with fixed [0, 300]')
    else:
        raise ValueError
    return data, scaler_data, scaler_day, scaler_week, None
    # return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(args, X, Y, batch_size, shuffle=True, drop_last=True):
    # Keep tensors on CPU; move to device per-batch inside Trainer to reduce GPU memory footprint
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(X, Y)
    import os
    num_workers = min(8, max(0, (os.cpu_count() or 8) - 2))
    # 适度提高预取，减少输入管道抖动
    pf = 4 if num_workers > 0 else None
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    # Deterministic shuffling/worker seeding (best-effort)
    try:
        g = torch.Generator()
        g.manual_seed(int(getattr(args, 'seed', 0)))
        dl_kwargs['generator'] = g
        dl_kwargs['worker_init_fn'] = seed_worker
    except Exception:
        pass
    if pf is not None:
        dl_kwargs['prefetch_factor'] = pf
    dataloader = torch.utils.data.DataLoader(data, **dl_kwargs)
    return dataloader


class _WindowDataset(torch.utils.data.Dataset):
    """On-the-fly sliding window dataset to avoid materializing all windows in memory."""
    def __init__(self, data, start_idx, end_idx, lag, horizon, single,
                 scaler_data, scaler_day, scaler_week, input_base_dim):
        super().__init__()
        self.data = data  # numpy array [T, N, D]
        self.start = start_idx
        self.end = end_idx
        self.lag = lag
        self.horizon = horizon
        self.single = single
        self.scaler_data = scaler_data
        self.scaler_day = scaler_day
        self.scaler_week = scaler_week
        self.input_base_dim = input_base_dim
        self.length = max(0, (self.end - self.start) - (self.lag + self.horizon) + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s = self.start + idx
        e_x = s + self.lag
        if self.single:
            s_y = s + self.lag + self.horizon - 1
            e_y = s + self.lag + self.horizon
        else:
            s_y = s + self.lag
            e_y = s + self.lag + self.horizon
        x = self.data[s:e_x]
        y = self.data[s_y:e_y]
        # scaling per channel group to match original pipeline
        xb = self.scaler_data.transform(x[..., :self.input_base_dim])
        yb = self.scaler_data.transform(y[..., :self.input_base_dim])
        xd = self.scaler_day.transform(x[..., self.input_base_dim:self.input_base_dim+1])
        yd = self.scaler_day.transform(y[..., self.input_base_dim:self.input_base_dim+1])
        xw = self.scaler_week.transform(x[..., self.input_base_dim+1:self.input_base_dim+2])
        yw = self.scaler_week.transform(y[..., self.input_base_dim+1:self.input_base_dim+2])
        x = np.concatenate([xb, xd, xw], axis=-1)
        y = np.concatenate([yb, yd, yw], axis=-1)
        # return global start index for driver slicing (optional third element)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), s

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    # CSA系列（含优化版）+ GIMtec: 使用 vendor 流水线，完全对齐 VendorCode/原仓
    ds_key = str(getattr(args, 'dataset', '')).lower()
    # NOTE: 部分数据集名称本身含下划线（如 china_hires / nyc_taxi），不能简单 split('_')[0]。
    if ds_key.startswith('gimtec'):
        ds_base = 'gimtec'
    elif ds_key.startswith('tec'):
        ds_base = 'tec'
    elif ds_key.startswith('china_hires'):
        ds_base = 'china_hires'
    else:
        ds_base = ds_key
    model_key = str(getattr(args, 'model', '')).lower()
    # CSA 原版仍使用 vendor 流水线；Optimized/TEC_MoLLM 使用年段切分流水线（可选 stride=horizon）
    if ds_base in ['gimtec', 'tec']:
        if model_key == 'csa_wtconvlstm':
            from lib.datasets.gimtec_vendor import build_csa_vendor_dataloaders
            return build_csa_vendor_dataloaders(args)
        if model_key in ['optimizedcsa_wtconvlstm', 'tec_mollm']:
            from lib.datasets.gimtec_pretrain import build_gimtec_pretrain_dataloaders
            stride = args.horizon if bool(getattr(args, 'stride_horizon', False)) else 1
            prefix_boundary = bool(getattr(args, 'prefix_boundary', True))
            return build_gimtec_pretrain_dataloaders(
                args,
                normalizer=normalizer,
                single=single,
                stride=stride,
                prefix_boundary=prefix_boundary,
            )
    # GPT-ST 预训练：GIMtec/TEC 按年份划分的专用管线
    if getattr(args, 'mode', '') == 'pretrain' and ds_base in ['gimtec', 'tec']:
        from lib.datasets.gimtec_pretrain import build_gimtec_pretrain_dataloaders
        stride = args.horizon if getattr(args, 'stride_horizon', False) else 1
        prefix_boundary = getattr(args, 'prefix_boundary', True)
        return build_gimtec_pretrain_dataloaders(args, normalizer=normalizer, single=single,
                                                 stride=stride, prefix_boundary=prefix_boundary)
    # 监督预测可选：与预训练一致的“按年段”切分（GIMtec/TEC）
    if getattr(args, 'mode', '') != 'pretrain' and ds_base in ['gimtec', 'tec'] and getattr(args, 'year_split', False):
        from lib.datasets.gimtec_pretrain import build_gimtec_pretrain_dataloaders
        stride = args.horizon if getattr(args, 'stride_horizon', False) else 1
        prefix_boundary = getattr(args, 'prefix_boundary', True)
        return build_gimtec_pretrain_dataloaders(args, normalizer=normalizer, single=single,
                                                 stride=stride, prefix_boundary=prefix_boundary)
    # China High-Res: 年份切分（train/val/test/ood）
    if ds_base in ['china_hires'] and getattr(args, 'year_split', False):
        from lib.datasets.china_hires import build_china_hires_dataloaders
        stride = args.horizon if getattr(args, 'stride_horizon', False) else 1
        prefix_boundary = getattr(args, 'prefix_boundary', True)
        return build_china_hires_dataloaders(args, normalizer=normalizer, single=single,
                                             stride=stride, prefix_boundary=prefix_boundary)
    # load raw st dataset: [T, N, D]
    data = load_st_dataset(args.dataset, args)

    # split dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    print('============', data_train.shape, data_val.shape, data_test.shape)
    _, scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(
        data_train, normalizer, args.input_base_dim, args.column_wise)

    # Heuristic: for very large graphs/time spans (e.g., GIMtec), avoid materializing all windows
    T_total, N_total = data.shape[0], data.shape[1]
    use_streaming = (args.dataset.lower() in ['gimtec', 'tec']) or (T_total * N_total > 5_000_000)

    if use_streaming:
        # Build streaming datasets using original data reference to avoid duplication
        def _range_info(split_array):
            # returns start,end index of the split in the original data
            start = 0
            if split_array is data_val:
                start = data_train.shape[0]
            elif split_array is data_test:
                start = data_train.shape[0] + data_val.shape[0]
            end = start + split_array.shape[0]
            return start, end

        tr_s, tr_e = 0, data_train.shape[0]
        va_s, va_e = tr_e, tr_e + data_val.shape[0]
        te_s, te_e = va_e, va_e + data_test.shape[0]

        train_ds = _WindowDataset(data, tr_s, tr_e, args.lag, args.horizon, single,
                                   scaler_data, scaler_day, scaler_week, args.input_base_dim)
        val_ds = _WindowDataset(data, va_s, va_e, args.lag, args.horizon, single,
                                 scaler_data, scaler_day, scaler_week, args.input_base_dim)
        test_ds = _WindowDataset(data, te_s, te_e, args.lag, args.horizon, single,
                                  scaler_data, scaler_day, scaler_week, args.input_base_dim)
        import os
        num_workers = min(8, max(0, (os.cpu_count() or 8) - 2))
        # Deterministic shuffling/worker seeding (best-effort)
        def _seeded_dl_kwargs():
            try:
                g = torch.Generator()
                g.manual_seed(int(getattr(args, 'seed', 0)))
                return {'generator': g, 'worker_init_fn': seed_worker}
            except Exception:
                return {}
        train_dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
            pin_memory=torch.cuda.is_available(), num_workers=num_workers, persistent_workers=True if num_workers > 0 else False,
            **_seeded_dl_kwargs())
        val_dataloader = None if len(val_ds) == 0 else torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=torch.cuda.is_available(), num_workers=num_workers, persistent_workers=True if num_workers > 0 else False,
            **_seeded_dl_kwargs())
        test_dataloader = None if len(test_ds) == 0 else torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=torch.cuda.is_available(), num_workers=num_workers, persistent_workers=True if num_workers > 0 else False,
            **_seeded_dl_kwargs())
        print('Train:', len(train_ds), 'samples; Val:', len(val_ds), 'Test:', len(test_ds))
        return train_dataloader, val_dataloader, test_dataloader, scaler_data, scaler_day, scaler_week, scaler_holiday
    else:
        # Old path: materialize windows in memory (fine for small datasets)
        x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
        x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
        x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)

        print('Train: ', x_tra.shape, y_tra.shape)
        print('Val: ', x_val.shape, y_val.shape)
        print('Test: ', x_test.shape, y_test.shape)

        x_tra_data = scaler_data.transform(x_tra[:, :, :, :args.input_base_dim])
        y_tra_data = scaler_data.transform(y_tra[:, :, :, :args.input_base_dim])
        x_tra_day = scaler_day.transform(x_tra[:, :, :, args.input_base_dim:args.input_base_dim+1])
        y_tra_day = scaler_day.transform(y_tra[:, :, :, args.input_base_dim:args.input_base_dim+1])
        x_tra_week = scaler_week.transform(x_tra[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
        y_tra_week = scaler_week.transform(y_tra[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
        x_tra = np.concatenate([x_tra_data, x_tra_day, x_tra_week], axis=-1)
        y_tra = np.concatenate([y_tra_data, y_tra_day, y_tra_week], axis=-1)

        x_val_data = scaler_data.transform(x_val[:, :, :, :args.input_base_dim])
        y_val_data = scaler_data.transform(y_val[:, :, :, :args.input_base_dim])
        x_val_day = scaler_day.transform(x_val[:, :, :, args.input_base_dim:args.input_base_dim+1])
        y_val_day = scaler_day.transform(y_val[:, :, :, args.input_base_dim:args.input_base_dim+1])
        x_val_week = scaler_week.transform(x_val[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
        y_val_week = scaler_week.transform(y_val[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
        x_val = np.concatenate([x_val_data, x_val_day, x_val_week], axis=-1)
        y_val = np.concatenate([y_val_data, y_val_day, y_val_week], axis=-1)

        x_test_data = scaler_data.transform(x_test[:, :, :, :args.input_base_dim])
        y_test_data = scaler_data.transform(y_test[:, :, :, :args.input_base_dim])
        x_test_day = scaler_day.transform(x_test[:, :, :, args.input_base_dim:args.input_base_dim+1])
        y_test_day = scaler_day.transform(y_test[:, :, :, args.input_base_dim:args.input_base_dim+1])
        x_test_week = scaler_week.transform(x_test[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
        y_test_week = scaler_week.transform(y_test[:, :, :, args.input_base_dim+1:args.input_base_dim+2])
        x_test = np.concatenate([x_test_data, x_test_day, x_test_week], axis=-1)
        y_test = np.concatenate([y_test_data, y_test_day, y_test_week], axis=-1)

        train_dataloader = data_loader(args, x_tra, y_tra, args.batch_size, shuffle=True, drop_last=False)
        val_dataloader = None if len(x_val) == 0 else data_loader(args, x_val, y_val, args.batch_size, shuffle=False, drop_last=False)
        test_dataloader = data_loader(args, x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
        return train_dataloader, val_dataloader, test_dataloader, scaler_data, scaler_day, scaler_week, scaler_holiday
