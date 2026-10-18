"""Natural-unit, explicit-mask statistics for the revised manuscript."""
import numpy as np


def masked_statistics(pred, truth, valid):
    pred, truth, valid = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64), np.asarray(valid, dtype=bool)
    if pred.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError('prediction, truth and validity mask must have identical shapes')
    if not np.isfinite(pred[valid]).all() or not np.isfinite(truth[valid]).all():
        raise ValueError('nonfinite value under the evaluation mask')
    error = pred[valid] - truth[valid]
    n = int(error.size)
    if not n:
        raise ValueError('empty evaluation mask')
    sse, sae = float(np.square(error).sum()), float(np.abs(error).sum())
    return dict(sse=sse, sae=sae, n_valid=n, RMSE=float(np.sqrt(sse/n)), MAE=sae/n)


def horizon_statistics(pred, truth, valid):
    """Endpoints refer to one lead; day means pool twelve two-hour leads."""
    h = pred.shape[1]
    selections = {'all': slice(0, h)}
    selections.update({f'lead_{2*(i+1)}h': slice(i, i+1) for i in range(h)})
    selections.update({f'day_{i//12+1}': slice(i, i+12) for i in range(0, h, 12) if i+12 <= h})
    return {key: masked_statistics(pred[:, ix], truth[:, ix], valid[:, ix]) for key, ix in selections.items()}


def summarize_seeds(frame, groups, metrics=('RMSE', 'MAE')):
    """Arithmetic mean and sample SD; retain all prescribed seed rows."""
    return frame.groupby(groups, dropna=False)[list(metrics)].agg(['count', 'mean', 'std'])


def paired_block_interval(paired_block_differences, confidence=.95, repeats=10000, seed=42):
    """Equal-block mean and percentile interval, conditional on saved models.

    Input is one seed-averaged paired difference per precomputed temporal block.
    It is not the pooled full-period RMSE difference or a bootstrap over seeds.
    """
    values = np.asarray(paired_block_differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('one finite paired difference per nonempty block is required')
    if not 0 < confidence < 1 or repeats < 1:
        raise ValueError('confidence in (0, 1) and positive repeats are required')
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean()
                      for _ in range(repeats)])
    lo, hi = np.quantile(means, [(1-confidence)/2, (1+confidence)/2])
    return dict(mean_paired_block_delta=float(values.mean()), low=float(lo), high=float(hi),
                confidence=confidence, n_blocks=len(values), repeats=repeats, seed=seed)


def paired_pooled_rmse_interval(blocks, arm, reference, confidence=.95,
                               repeats=5000, seed=20260924):
    """Year-stratified paired pooled RMSE interval conditional on fixed seeds.

    Rows contain variant, seed, start/end dates, SSE and n_valid. Each year
    retains its block count, including partial tail blocks. All models and
    training seeds share each draw. Blocks must use identical evaluation masks
    upstream; matching counts are necessary but cannot prove mask identity.
    RNG ordering matches the retained 24-hour study (replicate, then year).
    """
    import pandas as pd
    if not 0 < confidence < 1 or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('confidence in (0, 1) and positive integer repeats required')
    f = blocks.loc[blocks.variant.isin([arm, reference])].copy()
    if arm == reference or set(f.variant) != {arm, reference}:
        raise ValueError('two distinct models with sufficient statistics required')
    f['start'] = pd.to_datetime(f.start)
    f['end'] = pd.to_datetime(f.end)
    if f[['start', 'end']].isna().any().any() or (f.end < f.start).any():
        raise ValueError('invalid block dates')
    f['year'] = f.start.dt.year
    if (f.year != f.end.dt.year).any():
        raise ValueError('blocks cannot cross year boundaries')
    keys = ['year', 'start', 'end']
    if f.duplicated(['variant', 'seed'] + keys).any():
        raise ValueError('duplicate model/seed/block row')
    support = f[keys].drop_duplicates().sort_values(keys)
    index = pd.MultiIndex.from_frame(support)
    seeds = sorted(f.seed.unique())
    arrays = []
    for model in [arm, reference]:
        values = []
        for training_seed in seeds:
            z = f[(f.variant == model) & (f.seed == training_seed)].set_index(keys)
            z = z.reindex(index)[['sse', 'n_valid']].to_numpy(dtype=np.float64)
            if not np.isfinite(z).all() or (z[:, 0] < 0).any() or (z[:, 1] <= 0).any() or (z[:, 1] != np.floor(z[:, 1])).any():
                raise ValueError('missing or invalid model/seed/block sufficient statistics')
            values.append(z)
        arrays.append(np.asarray(values))
    a, b = arrays
    if not np.array_equal(a[:, :, 1], b[:, :, 1]) or not np.all(a[:, :, 1] == a[0, :, 1]):
        raise ValueError('models and fixed seeds must share evaluation support')
    years = support.year.to_numpy()
    strata = [np.flatnonzero(years == y) for y in np.unique(years)]
    def delta(ix):
        aa, bb = a[:, ix].sum(1), b[:, ix].sum(1)
        return float(np.mean(np.sqrt(aa[:, 0]/aa[:, 1]) - np.sqrt(bb[:, 0]/bb[:, 1])))
    rng = np.random.default_rng(seed)
    samples = [delta(np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                                    for ix in strata])) for _ in range(repeats)]
    lo, hi = np.quantile(samples, [(1-confidence)/2, (1+confidence)/2], method='linear')
    return dict(mean_pooled_rmse_delta=delta(np.arange(len(support))),
                low=float(lo), high=float(hi), confidence=confidence,
                n_blocks=len(support), training_seeds=[int(s) for s in seeds], repeats=repeats, seed=seed)


def retained_extension_intervals(source):
    """Run the original 98.33% study function on its saved block statistics.

    This uses the byte-identical original statistics.py, not the original-year
    95% estimator above. analyze.py is provenance for block aggregation only and
    is not part of the published Source Data; when present, its hash is checked,
    but it is never imported or executed here.
    """
    import csv
    import hashlib
    import json
    import runpy
    from pathlib import Path
    source = Path(source)
    saved = json.loads((source/'RESULTS.json').read_text())
    for name, expected in saved['analysis_source_sha256'].items():
        if name != 'statistics.py' and not (source/name).exists():
            continue
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original extension source hash mismatch: {name}')
    original = runpy.run_path(str(source/'statistics.py'))
    draws = original['indices']()  # also checks the original index-byte digest
    freeze = json.loads((source/'FREEZE.json').read_text())
    blocks = list(csv.DictReader((source/'BLOCKS.csv').open()))
    if len(blocks) != 78 or freeze['scope']['seeds'] != original['SEEDS']:
        raise ValueError('Original extension block/seed support mismatch')
    for row, frozen in zip(blocks, freeze['statistics']['blocks']):
        if any(row[k] != str(v) for k, v in frozen.items()):
            raise ValueError('Block CSV differs from frozen construction')
    expected_counts = np.array([int(b['windows'])*12*5183 for b in blocks])
    results = {}
    for name in ['P2', 'P3']:
        with np.load(source/f'{name}_block_statistics.npz', allow_pickle=False) as stats:
            a, b = stats['A'], stats['B']
        if (a.shape != (5, 78, 3) or b.shape != a.shape or
                not all(v.dtype == np.float64 and np.isfinite(v).all() and
                        np.all(v >= 0) and np.all(v[..., 2] == expected_counts)
                        for v in [a, b])):
            raise ValueError(f'Invalid original extension sufficient statistics: {name}')
        results[name], _ = original['comparison'](a, b, draws)
    return results


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Recompute retained extension intervals; no models or training.')
    parser.add_argument('--extension-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(retained_extension_intervals(args.extension_dir), indent=2))
