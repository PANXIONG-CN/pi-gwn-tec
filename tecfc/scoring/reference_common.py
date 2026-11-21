"""Shared helpers for exp0029: periods, Reader wiring, scoring loop.

All error statistics use the repository scoring entry (src/confirm2026): score.sufficient/summary,
final_inputs.Reader and final_score.samples. Nothing here fits, trains or infers.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
WORK = ROOT
EXP = ROOT / 'experiments/exp0029'
ART = EXP / 'artifacts'
from tecfc.scoring import confirm_score as cscore  # noqa: E402  sufficient, summary, bootstrap
from tecfc.scoring import confirm_inputs as fin  # noqa: E402  Reader
from tecfc.scoring import confirm_final as fsc  # noqa: E402  samples

CFG = json.loads(Path(os.environ.get('TECFC_REFERENCE_CONFIG', EXP / 'config.json')).read_text())
YEARS = CFG['global_arrays']
SEEDS = [42, 1024, 2025, 3407, 5000]
LEADS = [str(2 * (k + 1)) for k in range(12)]

PERIOD_YEARS = dict(validation=[2013, 2017], old_evaluation=[2015, 2019, 2020, 2021, 2022],
                    y2023=[2023], y2024=[2024], y2025=[2025], extension_2023_2025=[2023, 2024, 2025])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def day_range(first, last):
    return np.arange(np.datetime64(first), np.datetime64(last) + 1, dtype='datetime64[D]')


def period_dates(name):
    if name == 'descriptive_2026':
        return day_range('2026-01-01', '2026-10-31')
    out = [day_range(f'{y}-01-01', f'{y}-12-31') for y in PERIOD_YEARS[name]]
    return np.concatenate(out)


def to_seconds(dates):
    return dates.astype('datetime64[s]').astype(np.int64)


def origins_spec(tag, dates):
    """Write the origin list (Unix seconds, int64) and return a Reader array spec."""
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / f'origins_{tag}.npy'
    arr = to_seconds(np.asarray(dates, 'datetime64[D]'))
    if path.exists():
        assert np.array_equal(np.load(path), arr)
    else:
        np.save(path, arr)
    return dict(path=str(path), sha256=sha(path))


class Reader(fin.Reader):
    """Adds kinds: global_same_day (yearly arrays, same day) on top of the repository Reader."""

    def prediction(self, name, i):
        d = self.desc['methods'][name]
        if d.get('kind') == 'global_same_day':
            return self.raw(d['years'], self.dates[i])
        if d.get('kind') == 'global_prior_last_frame':
            last = self.raw(d['years'], self.dates[i] - np.timedelta64(1, 'D'))[-1:]
            return np.repeat(last, 12, axis=0)
        if d.get('kind') == 'flat_array':
            # (N,12,71*73[,1]) arrays, reshaped to the common grid; origins aligned by the source origin list
            key = d['source_origins']['path']
            if key not in self.lookups:
                self.lookups[key] = {int(t): j for j, t in enumerate(self.array(d['source_origins']))}
            j = self.lookups[key][int(self.ts[i])]
            arrays = [self.array(a) for a in d['arrays']]
            value = np.asarray(arrays[0][j], np.float64) if len(arrays) == 1 else np.mean([np.asarray(a[j], np.float64) for a in arrays], axis=0)
            return value.reshape(12, 71, 73)
        return super().prediction(name, i)


def make_desc(tag, dates, methods):
    return dict(domain='global', period=tag, max_month=12, origins=origins_spec(tag, dates),
                target=dict(kind='global_raw', years=YEARS), methods=methods,
                main_methods=list(methods))


def persistence_method():
    return dict(kind='global_prior', years=YEARS)


def stats_for(desc, names=None, mask_fn=None, progress=0):
    """(len(names), n, 12, 4) sufficient statistics; mask_fn(valid)->valid for diagnostics only."""
    reader = Reader(desc)
    names = list(desc['methods']) if names is None else list(names)
    n = len(reader.selected)
    out = np.zeros((len(names), n, 12, 4), np.float64)
    for k, i in enumerate(reader.selected):
        y, v = reader.label(i)
        if mask_fn is not None:
            v = mask_fn(v)
        for j, name in enumerate(names):
            out[j, k] = cscore.sufficient(reader.prediction(name, i), y, v)
        if progress and k % progress == 0:
            print(json.dumps(dict(tag=desc['period'], k=k, n=n)), flush=True)
    return reader.ts[reader.selected], names, out


def block_samples(daily_stats, ts):
    """daily_stats: (methods, days, 4). Returns (RMSE draws (5000, methods), digest)."""
    return fsc.samples(daily_stats, ts)


def rmse_of(stats):
    return cscore.summary(stats)


def fnum(x):
    return None if x is None else float(x)
