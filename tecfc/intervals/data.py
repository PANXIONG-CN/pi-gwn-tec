"""exp0028 data access.

Labels, validity masks and frozen predictions are read only through confirm2026.score.load_inputs
(SHA-256 verified, hard failure on mismatch). Descriptors are the committed exp0025 descriptors
(DEVELOPMENT_CHECK_INPUTS.json for 2023-2025, SCORING_INPUTS.json for 2026) plus, for the global
calibration years 2015 and 2019-2022, the exp0019 frozen S2 manifest and the exp0013 inference config
that records the label files and their SHA-256.
"""
import os
import sys
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
from tecfc.scoring import confirm_score as cscore  # reused: load_inputs, sufficient, summary, evaluate
from tecfc.scoring.confirm_records import read, sha  # reused: sha256, json reader
from tecfc.intervals import spec as S


def _abs(x):
    if isinstance(x, dict):
        return {k: (str(REPO / v) if k == 'path' and isinstance(v, str) and not Path(v).is_absolute() else _abs(v))
                for k, v in x.items()}
    if isinstance(x, list):
        return [_abs(v) for v in x]
    return x


def _trim(d, keep):
    out = {k: d[k] for k in ('origins', 'target', 'valid') if k in d}
    out['methods'] = {k: d['methods'][k] for k in keep}
    return out


def part_descriptors():
    """region -> ordered list of (part name, descriptor). Part names: old, ext (2023-2025), y2026."""
    dev = read(REPO / 'experiments/exp0025/DEVELOPMENT_CHECK_INPUTS.json')['domains']
    sc = read(REPO / 'experiments/exp0025/SCORING_INPUTS.json')['datasets']
    fused = read(REPO / 'experiments/exp0019/FUSED_PREDICTIONS.json')
    files = fused['files_sha256']
    cfg = read(REPO / 'experiments/exp0013/configs/inference.json')['data_files']
    sp = fused['systems']['S2']['splits']['original_evaluation']
    old = dict(origins=dict(path=sp['origins'], sha256=files[sp['origins']]),
               target=dict(kind='global_raw', years={str(y): dict(path=cfg[str(y)]['path'], sha256=cfg[str(y)]['sha256'])
                                                      for y in (2015, 2019, 2020, 2021, 2022)}),
               methods=dict(S2=dict(arrays=[dict(path=sp['predictions'], sha256=files[sp['predictions']])])))
    reg = ['final', 'anchor']
    return {
        'global': [('old', old), ('ext', _trim(dev['global_'], ['S2'])), ('y2026', _trim(sc['global_2026'], ['S2']))],
        'china': [('ext', _trim(dev['china'], reg)), ('y2026', _trim(sc['china_2026'], reg))],
        'japan': [('ext', _trim(dev['japan'], reg)), ('y2026', _trim(sc['japan_2026'], reg))],
    }


def system_key(region, method):
    return 'global_S2' if region == 'global' else '%s_%s' % (region, method)


def load_region(region):
    """[(part name, (ts, label, methods))] through the reused reader; every input hash is verified."""
    return [(name, cscore.load_inputs(_abs(desc))) for name, desc in part_descriptors()[region]]


class Series:
    """Chronological series of forecast days of one system; get(g) -> (prediction, label, valid)."""

    def __init__(self, loaded, method):
        self.names = [n for n, _ in loaded]
        self.parts = []
        for _, (ts, label, methods) in loaded:
            assert len(methods[method]) == 1
            self.parts.append((np.asarray(ts, np.int64), label, methods[method][0]))
        self.ts = np.concatenate([p[0] for p in self.parts])
        assert np.all(np.diff(self.ts) > 0) and np.all(self.ts % S.DAY == 0)
        sizes = [len(p[0]) for p in self.parts]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.part_of = np.repeat(np.arange(len(sizes)), sizes)
        self.dates = self.ts.astype('datetime64[s]').astype('datetime64[D]')
        self.year = self.dates.astype('datetime64[Y]').astype(int) + 1970

    def get(self, g):
        k = int(self.part_of[g])
        i = int(g - self.offsets[k])
        _, label, array = self.parts[k]
        y, v = label(i)
        return array[i], y, v


def build_scores(series):
    """s = |y - yhat| at every valid grid point and target time; nan where the label is invalid."""
    G = len(series.ts)
    p, _, _ = series.get(0)
    H = np.asarray(p).shape[0]
    assert H == S.HORIZONS
    C = int(np.prod(np.asarray(p).shape[1:]))
    scores = np.full((H, G, C), np.nan)
    for g in range(G):
        p, y, v = series.get(g)
        p2 = np.asarray(p, np.float64).reshape(H, C)
        y2 = np.asarray(y, np.float64).reshape(H, C)
        v2 = np.asarray(v).reshape(H, C)
        assert v2.dtype == bool and np.isfinite(p2).all() and np.isfinite(y2[v2]).all()
        s = np.abs(p2 - y2)
        s[~v2] = np.nan
        scores[:, g, :] = s
    return scores


def extra_fn(series):
    """extra(g) -> (yhat[H, C], high[H, C]) with high = valid & (y >= 60 TECU)."""
    def extra(g):
        p, y, v = series.get(g)
        yhat = np.asarray(p, np.float64).reshape(S.HORIZONS, -1)
        y2 = np.asarray(y, np.float64).reshape(S.HORIZONS, -1)
        v2 = np.asarray(v).reshape(S.HORIZONS, -1)
        with np.errstate(invalid='ignore'):
            high = v2 & (y2 >= S.HIGH_TEC)
        return yhat, high
    return extra


def index_sets(series, system):
    cfg = S.SYSTEMS[system]
    eval_idx = np.where(np.isin(series.year, cfg['eval_years']))[0]
    if cfg['static'] == 'old_evaluation':
        cal_idx = np.where(series.part_of == series.names.index('old'))[0]
        assert set(series.year[cal_idx].tolist()) == {2015, 2019, 2020, 2021, 2022}
    else:
        cal_idx = np.where(series.year == 2023)[0]
        # regional finals were trained on 2014, 2015 and 2019-2022: nothing earlier than 2023 may be present
        assert series.year.min() == 2023
    assert not set(cal_idx.tolist()) & set(eval_idx.tolist())
    return cal_idx, eval_idx
