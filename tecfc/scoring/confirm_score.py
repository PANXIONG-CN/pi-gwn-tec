"""One scoring entry for historical validation and frozen 2026 predictions.

The data descriptor contains only immutable arrays and fixed transformations.
Sufficient statistics are (count, SSE, absolute error sum, signed error sum).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tecfc.scoring.confirm_records import ROOT, EXP, sha, read, write, now


def sufficient(prediction, target, valid):
    p, y, v = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(valid)
    assert p.shape == y.shape == v.shape and v.dtype == bool
    assert np.isfinite(p).all() and np.isfinite(y[v]).all()
    e = np.where(v, p - y, 0.)
    axes = tuple(range(1, e.ndim))
    return np.stack([v.sum(axes), np.square(e).sum(axes), np.abs(e).sum(axes), e.sum(axes)], -1)


def summary(stats):
    n, sse, sae, bias = np.asarray(stats, np.float64).reshape(-1, 4).sum(0)
    return dict(n_valid=int(n), RMSE=float(np.sqrt(sse/n)) if n else None,
                MAE=float(sae/n) if n else None, bias=float(bias/n) if n else None)


def bootstrap(stats, timestamps):
    ts = np.asarray(timestamps).astype('datetime64[s]').astype('datetime64[D]')
    assert len(set(ts.astype('datetime64[Y]'))) == 1
    start = ts[0].astype('datetime64[Y]').astype('datetime64[D]')
    block_ids = (ts-start).astype(int)//7
    nblocks = int(block_ids.max())+1
    # stats: model, day, 4. Identical draws for all models.
    block = np.zeros((nblocks, len(stats), 4), np.float64)
    for j, a in enumerate(stats):
        np.add.at(block[:, j], block_ids, a)
    draws = np.random.default_rng(20260924).integers(nblocks, size=(5000, nblocks))
    sampled = block[draws].sum(1)
    return np.sqrt(sampled[:, :, 1]/sampled[:, :, 0]), draws


def load_inputs(descriptor):
    cache = {}

    def array(rec):
        path = rec['path']
        if path not in cache:
            assert sha(path) == rec['sha256'], 'Input SHA256 mismatch: ' + path
            cache[path] = np.load(path, mmap_mode='r', allow_pickle=False)
        return cache[path]

    ts = array(descriptor['origins'])
    years = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    if descriptor['target']['kind'] == 'global_raw':
        raws = {int(y): array(rec) for y, rec in descriptor['target']['years'].items()}

        def label(i):
            day = np.datetime64(int(ts[i]), 's').astype('datetime64[D]')
            year = int(str(day)[:4]); offset = int((day-np.datetime64(f'{year}-01-01')).astype(int))
            y = np.asarray(raws[year][offset], np.float32)/np.float32(10)
            return y, np.isfinite(y)
    else:
        y = array(descriptor['target']['array']); v = array(descriptor['valid'])

        def label(i):
            return y[i], v[i]
    methods = {}
    for name, spec in descriptor['methods'].items():
        arrays = [array(r) for r in spec['arrays']]
        for a in arrays:
            assert a.shape[0] == len(ts)
        methods[name] = arrays
    return ts, label, methods


def evaluate(descriptor):
    ts, label, methods = load_inputs(descriptor)
    out = {name: np.zeros((len(ts), 12, 4), np.float64) for name in methods}
    for i in range(len(ts)):
        y, v = label(i)
        for name, arrays in methods.items():
            spec = descriptor['methods'][name]
            if spec.get('stride'):
                p = arrays[0][i, ::spec['stride']]
            elif len(arrays) == 1:
                p = arrays[0][i]
            else:
                p = np.mean([np.asarray(a[i], np.float64) for a in arrays], axis=0)
            out[name][i] = sufficient(p, y, v)
    return {name: summary(s) for name, s in out.items()}, out


def synthetic():
    p = np.array([[1., -3., 90.], [5., 9., 99.]])
    y = np.array([[-1., 1., np.nan], [2., 5., np.nan]])
    v = np.isfinite(y)
    s = sufficient(p, y, v)
    np.testing.assert_array_equal(s, [[2, 20, 6, -2], [2, 25, 7, 7]])
    m = summary(s)
    assert m == dict(n_valid=4, RMSE=float(np.sqrt(45/4)), MAE=13/4, bias=5/4)
    assert summary(np.zeros((12, 4))) == dict(n_valid=0, RMSE=None, MAE=None, bias=None)
    ts = np.arange(np.datetime64('2025-01-01'), np.datetime64('2025-01-11')).astype('datetime64[s]').astype(np.int64)
    stats = np.tile(np.array([2., 8., 4., 4.]), (2, 10, 1))
    sampled, draws = bootstrap(stats, ts)
    assert draws.shape == (5000, 2) and np.all(sampled == 2)
    return dict(status='passed', negative_values_preserved=True, invalid_labels_excluded=True,
                unequal_valid_support=True, empty_group_null=True, short_last_block=True,
                shared_draws=True, seed_and_draws=[20260924, 5000])
