"""exp0027: 96-hour (48-frame) global windows, models and the PI auxiliary.

Only the output length changes. GWNET already exposes output_window; it sets
only the out_channels of end_conv_2, the last module its constructor creates.
GlobalGWN hard-codes 12, so we pass 48 through the original constructor and
reshape 48 frames. Inputs, losses and the recipe are those of base G, exp0018
G-plus-M and the exp0013 PI arm. With horizon=12 everything reduces to base G.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch.utils.data import Dataset

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SRC = HERE.parent
from tecfc.training import global_train as base  # noqa: E402

_previous = sys.modules.get('geometry')
base.load_module('geometry', SRC / 'features/geometry.py')
f15 = base.load_module('_exp0027_f15', SRC / 'features/member_inputs.py')
if _previous is None: sys.modules.pop('geometry', None)
else: sys.modules['geometry'] = _previous

ARMS = ('H-G', 'H-PI', 'H-GM')
SEEDS = base.SEEDS
HORIZON = 48
SCALE = 147.1
TRAIN_YEARS = (2009, 2010, 2011, 2012, 2014, 2016, 2018)
ROLE_YEARS = {'train': TRAIN_YEARS, 'validation': (2013, 2017),
              'original_evaluation': (2015, 2019, 2020, 2021, 2022),
              'extension_evaluation': (2023, 2024, 2025), 'evaluation_2026': (2026,)}
EVALUATION_ROLES = ('validation', 'original_evaluation', 'extension_evaluation', 'evaluation_2026')
LABEL_END = {'evaluation_2026': '2026-10-31'}  # authorized 2026 label range
DEV = set(TRAIN_YEARS + ROLE_YEARS['validation'])
TRAIN_GROUPS = ((2009, 2010, 2011, 2012), (2014,), (2016,), (2018,))
LAGS = {'H-G': (1,), 'H-PI': (1,), 'H-GM': (1, 2, 3, 27)}
CHANNELS = {'H-G': 1, 'H-PI': 1, 'H-GM': 9}
LOSS = {'H-G': 'smooth_l1', 'H-PI': 'smooth_l1', 'H-GM': 'mae'}
RHO = {'H-G': 0., 'H-PI': .01, 'H-GM': 0.}
COUNTS = {48: {'H-G': dict(train=2541, validation=724, original_evaluation=1820, extension_evaluation=1093, evaluation_2026=301),
               'H-GM': dict(train=2515, validation=724, original_evaluation=1820, extension_evaluation=1093, evaluation_2026=301)},
          12: {'H-G': dict(train=2553, validation=730)}}
COUNTS[48]['H-PI'] = COUNTS[48]['H-G']
CANDIDATES = dict(train=2553, validation=730, original_evaluation=1826, extension_evaluation=1096, evaluation_2026=304)


def stamp(day):
    return int(np.datetime64(day, 's').astype(np.int64))


def history_times(day, lag):
    return stamp(day) - lag * 86400 + 7200 * np.arange(12, dtype=np.int64)


def target_times(day, horizon):
    return stamp(day) + 7200 * np.arange(horizon, dtype=np.int64)


def restore(a):
    a = np.asarray(a)
    if a.shape[-2:] != (71, 72): raise ValueError('expected 71x72 unique grid')
    return np.concatenate((a, a[..., :1]), axis=-1)


def candidates(role):
    """Base G origin calendars (gim_data.GIM.windows) plus Jan-Oct 2026."""
    if role == 'train':
        out = []
        for group in TRAIN_GROUPS:
            out += [str(d) for d in np.arange(np.datetime64(f'{group[0]}-01-02'), np.datetime64(f'{group[-1] + 1}-01-01'))]
        return out
    if role == 'evaluation_2026':
        return [str(d) for d in np.arange(np.datetime64('2026-01-01'), np.datetime64('2026-11-01'))]
    return [str(d) for y in ROLE_YEARS[role] for d in np.arange(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01'))]


def windows(role, arm, horizon, whitelist=()):
    """Return kept origins and dropped (origin, reasons); calendar only, no arrays."""
    years, end = set(ROLE_YEARS[role]), LABEL_END.get(role)
    kept, dropped = [], []
    for day in candidates(role):
        reasons = []
        tdays = np.unique(target_times(day, horizon).astype('datetime64[s]').astype('datetime64[D]'))
        if any(int(str(d)[:4]) not in years for d in tdays) or (end is not None and str(tdays[-1]) > end):
            reasons.append('target_outside_role')
        for lag in LAGS[arm]:
            h = str(np.datetime64(day) - np.timedelta64(lag, 'D'))
            if h < '2009-01-01':
                if role != 'train': raise ValueError('Incomplete evaluation history')
                reasons.append('history_before_2009'); break
            if role in ('train', 'validation') and int(h[:4]) not in DEV and h not in whitelist:
                raise ValueError('History date not authorized: ' + h)
        (dropped if reasons else kept).append((day, reasons) if reasons else day)
    expected = COUNTS[horizon][arm][role]
    if len(kept) != expected or len(kept) + len(dropped) != CANDIDATES[role]:
        raise ValueError(f'Changed windows {arm}/{role}: {len(kept)}')
    return kept, dropped


class Raw:
    """Explicit year files only; reproduces gim_data.GIM.day exactly."""
    def __init__(self, config, allowed, history_only=None):
        self.files, self.allowed, self.history_only, self.cache = config['data_files'], set(allowed), history_only or {}, {}

    def array(self, year):
        if year not in self.cache:
            row = self.files.get(str(year)) if year in self.allowed else self.history_only.get(str(year))
            a = np.load(row['path'], mmap_mode='r', allow_pickle=False)
            days = 304 if year == 2026 else int((np.datetime64(f'{year + 1}-01-01') - np.datetime64(f'{year}-01-01')).astype(int))
            if a.shape != (days, 12, 71, 73) or a.dtype != np.float32: raise ValueError(f'wrong array {year}: {a.shape}')
            self.cache[year] = a
        return self.cache[year]

    def day(self, day, unique=True):
        d = np.datetime64(day, 'D'); year = int(str(d)[:4])
        if year not in self.allowed:
            row = self.history_only.get(str(year))
            if row is None or str(d) not in row['allowed_dates']: raise ValueError('Forbidden date ' + str(d))
        if year == 2026 and str(d) > LABEL_END['evaluation_2026']: raise ValueError('Outside authorized 2026 range')
        raw = np.asarray(self.array(year)[int((d - np.datetime64(f'{year}-01-01')).astype(int))])
        if not np.array_equal(raw[..., 0], raw[..., -1], equal_nan=True): raise ValueError('longitude endpoints disagree')
        raw = raw[..., :72] if unique else raw
        return raw.astype(np.float32) / np.float32(10.)


class Windows(Dataset):
    """Training/validation windows with labels, or label-free inference windows."""
    def __init__(self, role, config, arm, horizon, *, inference=False):
        if arm not in ARMS or horizon not in (12, 48) or (horizon == 12 and arm != 'H-G'): raise ValueError('Unregistered arm/horizon')
        if role not in (('original_evaluation', 'extension_evaluation', 'evaluation_2026') if inference else ('train', 'validation')):
            raise ValueError('Forbidden role/mode')
        self.role, self.arm, self.horizon, self.inference = role, arm, horizon, inference
        with np.load(Path(config['asset_root']) / 'prepared/train_climatology_v1.npz', allow_pickle=False) as z:
            self.scale = float(z['scale'])
        if self.scale != SCALE: raise ValueError('Changed scale')
        allowed = set(range(2009, 2027)) if inference else DEV
        whitelist = config['history_only_files']['2015']['allowed_dates'] if 'history_only_files' in config else []
        self.raw = Raw(config, allowed, None if inference else config.get('history_only_files'))
        kept, self.dropped = windows(role, arm, horizon, whitelist)
        self.days = kept

    def __len__(self):
        return len(self.days)

    def history(self, day):
        cutoff = stamp(day) - 7200; parts = []
        for lag in LAGS[self.arm]:
            if not np.all(history_times(day, lag) <= cutoff): raise ValueError('Noncausal history')
            parts.append(self.raw.day(str(np.datetime64(day) - np.timedelta64(lag, 'D'))) / np.float32(self.scale))
        if self.arm == 'H-GM':
            x = np.concatenate((np.stack(parts, axis=-1), f15.geometry(f15.history_times(day))), axis=-1)
        else:
            x = np.array(parts[0], np.float32)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite history')
        return x

    def __getitem__(self, index):
        day = self.days[index]
        out = dict(history=torch.from_numpy(self.history(day)), origin=stamp(day), index=index)
        if not self.inference:
            y = np.concatenate([restore(self.raw.day(str(np.datetime64(day) + np.timedelta64(j, 'D'))))
                                for j in range(self.horizon // 12)], axis=0)
            out.update(target=torch.from_numpy(np.array(y, np.float32)), valid=torch.from_numpy(np.isfinite(y)))
        return out


def forward(self, history, restore=True):
    b = history.shape[0]
    if self.input_channels == 1:
        if history.ndim != 4 or tuple(history.shape[1:]) != (12, 71, 72): raise ValueError('history must be [B,12,71,72]')
    elif history.ndim != 5 or tuple(history.shape[1:]) != (12, 71, 72, self.input_channels): raise ValueError('Wrong member input')
    p = self.gwn(history.reshape(b, 12, 5112, self.input_channels)).reshape(b, self.horizon, 71, 72)
    return torch.cat((p, p[..., :1]), dim=-1) if restore else p


def build_model(architecture, device, arm, horizon):
    """Original GlobalGWN constructor; only GWNET dim_in and output_window are passed through."""
    channels, original = CHANNELS[arm], architecture.GWNET
    def constructor(args, device, dim_in, dim_out):
        args.output_window = horizon
        return original(args, device, dim_in=channels, dim_out=dim_out)
    try:
        architecture.GWNET = constructor
        model = architecture.GlobalGWN(device)
    finally:
        architecture.GWNET = original
    model.input_channels, model.horizon = channels, horizon
    model.forward = types.MethodType(forward, model)
    return model


def planar_bits(prediction):
    """Lossless storage of the bf16 network output on the 72 unique columns."""
    if prediction.dtype != torch.bfloat16: raise TypeError('Expected bf16 network output')
    u = prediction[..., :72].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    return (u >> 8).astype(np.uint8), (u & 255).astype(np.uint8)


def tecu_from_planar(hi, lo):
    """Exact float32 TECU = float32(bf16) * float32(147.1), endpoint column restored."""
    bits = (np.asarray(hi).astype(np.uint32) << 24) | (np.asarray(lo).astype(np.uint32) << 16)
    value = bits.view(np.float32) * np.float32(SCALE)
    return np.concatenate((value, value[..., :1]), axis=-1)


class PIAuxiliary:
    """exp0013 PI auxiliary (six frozen terms, Dst caps) over all horizon frames."""
    def __init__(self, physics_config, device, horizon):
        self.phys = base.load_module('_exp0027_physics', SRC / 'objectives/penalty.py')
        self.horizon = horizon
        self.drivers = self.phys.Drivers(physics_config['driver'], device)
        self.physics = self.phys.build(physics_config, device)
        self.assign_gradients = self.phys.assign_gradients

    def driver_slice(self, origins):
        query = np.asarray(origins, np.int64)[:, None] + 7200 * np.arange(self.horizon, dtype=np.int64)
        years = set((query.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970).ravel().tolist())
        if not years <= set(TRAIN_YEARS): raise ValueError('Auxiliary drivers are training-only')
        store = self.drivers.store; ix = np.searchsorted(store.time, query)
        if np.any(ix >= len(store.time)) or not np.array_equal(store.time[ix], query): raise ValueError('Target driver times disagree')
        return store.slice_batched(ix[:, 0], self.horizon, 5183, fields=('F107', 'AE', 'Dst'),
                                   require_cos_sza=True, include_grid_shape=False)

    def __call__(self, prediction, origins):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pred = prediction.float()
            _, terms = self.physics(pred.reshape(len(pred), self.horizon, 5183, 1), None, drivers=self.driver_slice(origins))
            loss = .5 * sum(terms.values())
            counts = self.physics.residual_counts.copy()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite auxiliary loss')
        return loss, {k: float(v.detach()) for k, v in terms.items()}, counts


def sha(path):
    return base.sha(path)


def scientific_config(config):
    """Remove only path locations; every scientific field and hash remains."""
    result = json.loads(json.dumps(config, allow_nan=False))
    for key in ('reference_root', 'asset_root'): result.pop(key, None)
    for group in ('data_files', 'history_only_files', 'reference_epochs'):
        for row in result.get(group, {}).values(): row.pop('path', None)
    for key in ('driver', 'adjacency'): result.get('physics', {}).pop(key, None)
    for row in result.get('candidates', []):
        for key in ('checkpoint',): row.pop(key, None)
    return result


def validate(config, *, stage):
    if config.get('schema') != 'exp0027_h96_v1' or config.get('experiment') != 'exp0027': raise ValueError('Unregistered configuration')
    if stage not in ('train', 'checks', 'inference'): raise ValueError('Unknown stage')
    if stage == 'train':
        if config['arm'] not in ARMS or config['seed'] not in SEEDS or config['horizon'] != HORIZON: raise ValueError('Changed arm/seed/horizon')
        if config['loss'] != LOSS[config['arm']] or config['rho'] != RHO[config['arm']]: raise ValueError('Changed loss/rho')
        if (config['arm'] == 'H-PI') != ('physics' in config): raise ValueError('Physics section mismatch')
    expected = set(range(2009, 2027)) if stage == 'inference' else DEV
    if {int(y) for y in config['data_files']} != expected: raise ValueError('Changed role access')
    if stage != 'inference':
        days = [str(x) for x in np.arange(np.datetime64('2015-12-06'), np.datetime64('2016-01-01'))]
        if set(config['history_only_files']) != {'2015'} or config['history_only_files']['2015']['allowed_dates'] != days:
            raise ValueError('History whitelist changed')
    if config['reference_sha256'] != base.REFERENCE_HASHES or set(config['asset_sha256']) != base.ASSET_NAMES:
        raise ValueError('Changed reference identity')
    for name, digest in config['reference_sha256'].items():
        if sha(Path(config['reference_root']) / name) != digest: raise ValueError('Changed reference ' + name)
    for name, digest in config['asset_sha256'].items():
        if sha(Path(config['asset_root']) / name) != digest: raise ValueError('Changed asset ' + name)
    for group in ('data_files', 'history_only_files', 'reference_epochs'):
        for row in config.get(group, {}).values():
            if sha(row['path']) != row['sha256']: raise ValueError('Changed explicit input ' + str(row['path']))
    if 'physics' in config:
        for key in ('driver', 'adjacency'):
            if sha(config['physics'][key]) != config['physics'][key + '_sha256']: raise ValueError('Changed ' + key)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(path, value)


def digest_order(indices):
    return hashlib.sha256(np.asarray(indices).astype('<i8').tobytes()).hexdigest()
