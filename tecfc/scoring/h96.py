"""One registered CPU scoring pass over committed frozen predictions (exp0027).

Sufficient statistics per window and lead: (valid count, SSE, absolute-error sum,
signed-error sum). Pooled metrics, five-seed means/SD, five-seed-mean-prediction
metrics, persistence improvement and 7-day block intervals (year-stratified,
5000 draws, RNG 20260924, shared across models, lead groups and comparisons).
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.dont_write_bytecode = True
from tecfc.features.h96_inputs import tecu_from_planar, ROLE_YEARS, LABEL_END, ARMS, SEEDS  # noqa: E402

ROOT = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve(); EXP = ROOT
PERIODS = ('validation', 'original_evaluation', 'extension_evaluation', 'evaluation_2026')
GROUPS = {'day1': slice(0, 12), 'day2': slice(12, 24), 'day3': slice(24, 36), 'day4': slice(36, 48), 'all48': slice(0, 48)}
COMPARISONS = (('H-PI', 'H-G'), ('H-GM', 'H-G'))
REFERENCES = ('G', 'G-plus-M')
DRAWS, RNG = 5000, 20260924


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + '.partial')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


class Labels:
    """Original product arrays; truth for D..D+3 and the persistence reference."""
    def __init__(self, files):
        self.files, self.cache = files, {}

    def day(self, day):
        d = np.datetime64(day, 'D'); year = str(d)[:4]
        if year == '2026' and str(d) > LABEL_END['evaluation_2026']: raise ValueError('Outside authorized 2026 range')
        if year not in self.cache:
            row = self.files[year]
            if sha(row['path']) != row['sha256']: raise ValueError('Changed label source ' + year)
            self.cache[year] = np.load(row['path'], mmap_mode='r', allow_pickle=False)
        return np.asarray(self.cache[year][int((d - np.datetime64(year + '-01-01')).astype(int))], np.float32) / np.float32(10)

    def truth(self, day):
        return np.concatenate([self.day(np.datetime64(day, 'D') + np.timedelta64(j, 'D')) for j in range(4)])

    def persistence(self, day):
        """Previous day's 12 maps repeated for the four target days."""
        return np.tile(self.day(np.datetime64(day, 'D') - np.timedelta64(1, 'D')), (4, 1, 1))


def sufficient(prediction, truth, valid):
    p, y = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    if p.shape != y.shape or valid.shape != y.shape or not np.isfinite(p).all(): raise ValueError('Prediction integrity')
    e = np.where(valid, p - y, 0.)
    return np.stack([valid.sum((1, 2)), np.square(e).sum((1, 2)), np.abs(e).sum((1, 2)), e.sum((1, 2))], -1)


def metrics(stats):
    n, sse, sae, s = np.asarray(stats, np.float64).reshape(-1, 4).sum(0)
    return dict(n_valid=int(n), RMSE=float(np.sqrt(sse / n)), MAE=float(sae / n), bias=float(s / n))


def draws(origins):
    """Calendar 7-day blocks from 1 January, resampled within each year (ascending years)."""
    dates = np.asarray(origins).astype('datetime64[s]').astype('datetime64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970; rng = np.random.default_rng(RNG); out = []
    for year in sorted(set(years.tolist())):
        sel = np.flatnonzero(years == year); ids = (dates[sel] - np.datetime64(f'{year}-01-01')).astype(int) // 7
        n = int(ids.max()) + 1; index = rng.integers(n, size=(DRAWS, n))
        weights = np.zeros((DRAWS, n), np.int64); np.add.at(weights, (np.arange(DRAWS)[:, None], index), 1)
        out.append((sel, ids, n, weights))
    return out


def resampled_rmse(window_stats, shared):
    """window_stats [N,4] -> RMSE for each of 5000 shared draws."""
    total = np.zeros((DRAWS, 4))
    for sel, ids, n, weights in shared:
        block = np.zeros((n, 4)); np.add.at(block, ids, window_stats[sel]); total += weights @ block
    return np.sqrt(total[:, 1] / total[:, 0])


def classify(delta, ci):
    if delta < 0 and ci[1] < 0: return '改善'
    if delta > 0 and ci[0] > 0: return '变差'
    return '无差别'


def table(stats, persistence, leads):
    """stats {seed: [N,48,4]}; five-seed mean and SD (ddof=1) of single-model metrics."""
    rows = [dict(seed=s, **metrics(stats[s][:, leads])) for s in sorted(stats, key=SEEDS.index)]
    out = dict(by_seed=rows, n_valid=rows[0]['n_valid'])
    for k in ('RMSE', 'MAE', 'bias'):
        values = [r[k] for r in rows]; out[k + '_mean'] = float(np.mean(values)); out[k + '_sd'] = float(np.std(values, ddof=1))
    out['RMSE_improvement_vs_persistence_percent'] = 100 * (1 - out['RMSE_mean'] / persistence['RMSE'])
    return out


def ensemble_row(stats, persistence, leads):
    row = metrics(stats[:, leads]); row['RMSE_improvement_vs_persistence_percent'] = 100 * (1 - row['RMSE'] / persistence['RMSE'])
    return row


def collect(period, frozen, labels):
    """Read every frozen prediction once; sufficient statistics per window and lead."""
    arms = {a: frozen['arms'][a] for a in ARMS if a in frozen['arms']}
    origins = None; arrays = {}
    for arm, members in arms.items():
        for row in members:
            ts = np.load(row['periods'][period]['origins'], allow_pickle=False)
            if origins is None: origins = ts
            elif not np.array_equal(origins, ts): raise ValueError('Origins differ across members')
            arrays[arm, row['seed']] = np.load(row['periods'][period]['predictions'], mmap_mode='r', allow_pickle=False)
    refs = {}; ref_index = {}
    for family in REFERENCES:
        for row in frozen['day1_references'][family]:
            spec = row['periods'][period]; ts = np.load(spec['origins'], allow_pickle=False)
            ref_index[family, row['seed']] = {int(t): j for j, t in enumerate(ts)}
            refs[family, row['seed']] = np.load(spec['predictions'], mmap_mode='r', allow_pickle=False)
    n = len(origins); stats = {k: np.zeros((n, 48, 4)) for k in arrays}; ensemble = {a: np.zeros((n, 48, 4)) for a in arms}
    persistence = np.zeros((n, 48, 4)); ref_stats = {k: np.zeros((n, 12, 4)) for k in refs}
    ref_ensemble = {f: np.zeros((n, 12, 4)) for f in REFERENCES}
    for i, t in enumerate(origins):
        day = np.datetime64(int(t), 's').astype('datetime64[D]'); y = labels.truth(day); v = np.isfinite(y)
        for arm in arms:
            preds = [tecu_from_planar(arrays[arm, s][0, i], arrays[arm, s][1, i]).astype(np.float64) for s in SEEDS]
            for s, p in zip(SEEDS, preds): stats[arm, s][i] = sufficient(p, y, v)
            ensemble[arm][i] = sufficient(np.mean(preds, axis=0), y, v)
        persistence[i] = sufficient(labels.persistence(day), y, v)
        for family in REFERENCES:
            preds = []
            for s in SEEDS:
                p = np.asarray(refs[family, s][ref_index[family, s][int(t)]], np.float64)
                ref_stats[family, s][i] = sufficient(p, y[:12], v[:12]); preds.append(p)
            ref_ensemble[family][i] = sufficient(np.mean(preds, axis=0), y[:12], v[:12])
        if i % 200 == 0: print(json.dumps(dict(period=period, window=i, of=n)), flush=True)
    return origins, stats, ensemble, persistence, ref_stats, ref_ensemble


def score_period(period, frozen, labels):
    origins, stats, ensemble, persistence, ref_stats, ref_ensemble = collect(period, frozen, labels)
    arms = list(ensemble); shared = draws(origins)
    np.savez_compressed(EXP / f'artifacts/scores/{period}_window_lead_stats.npz', origins=origins,
                        **{f'{a}_{s}': stats[a, s] for a, s in stats}, **{f'{a}5': ensemble[a] for a in arms}, persistence=persistence,
                        **{f'{f}_{s}_day1': ref_stats[f, s] for f, s in ref_stats}, **{f'{f}5_day1': ref_ensemble[f] for f in REFERENCES})
    out = dict(windows=int(len(origins)), first_origin=str(np.datetime64(int(origins[0]), 's'))[:10],
               last_origin=str(np.datetime64(int(origins[-1]), 's'))[:10], years=sorted(set(ROLE_YEARS[period])),
               persistence={}, single_model={a: {} for a in arms}, five_seed_mean_prediction={a: {} for a in arms}, comparisons={})
    for g, leads in GROUPS.items():
        p = metrics(persistence[:, leads]); out['persistence'][g] = p
        for a in arms:
            out['single_model'][a][g] = table({s: stats[a, s] for s in SEEDS}, p, leads)
            out['five_seed_mean_prediction'][a][g] = ensemble_row(ensemble[a], p, leads)
    p1 = out['persistence']['day1']
    out['day1_vs_24h_models'] = dict(
        common_windows=int(len(origins)),
        **{f'{a} (96h, day 1)': dict(single_model=out['single_model'][a]['day1'], five_seed_mean_prediction=out['five_seed_mean_prediction'][a]['day1']) for a in arms},
        **{f'{f} (24h)': dict(single_model=table({s: ref_stats[f, s] for s in SEEDS}, p1, slice(0, 12)),
                              five_seed_mean_prediction=ensemble_row(ref_ensemble[f], p1, slice(0, 12))) for f in REFERENCES},
        persistence=p1)
    for a, b in COMPARISONS:
        key = f'{a} - {b}'
        if a not in arms or b not in arms: out['comparisons'][key] = '未做'; continue
        out['comparisons'][key] = {}
        for g, leads in GROUPS.items():
            seeds = [metrics(stats[a, s][:, leads])['RMSE'] - metrics(stats[b, s][:, leads])['RMSE'] for s in SEEDS]
            ea, eb = metrics(ensemble[a][:, leads])['RMSE'], metrics(ensemble[b][:, leads])['RMSE']
            boot = resampled_rmse(ensemble[a][:, leads].sum(1), shared) - resampled_rmse(ensemble[b][:, leads].sum(1), shared)
            ci = np.quantile(boot, [.025, .975]).tolist()
            out['comparisons'][key][g] = dict(paired_seed_RMSE_differences=dict(zip(map(str, SEEDS), seeds)),
                                              improving_seeds=int(sum(d < 0 for d in seeds)), mean_seed_difference=float(np.mean(seeds)),
                                              ensemble_RMSE_difference=ea - eb, ensemble_ci95=ci, classification=classify(ea - eb, ci))
    out['bootstrap'] = dict(block_days=7, stratified_by_year=True, draws=DRAWS, rng=RNG, shared_across_models_leads_and_comparisons=True,
                            blocks_per_year={str(int(np.datetime64(int(origins[sel[0]]), 's').astype('datetime64[Y]').astype(int) + 1970)): n
                                             for sel, ids, n, w in shared})
    return out


def main():
    frozen = read(EXP / 'FROZEN_PREDICTIONS.json')
    for path, digest in frozen['files_sha256'].items():
        if sha(path) != digest: raise ValueError('Changed frozen file ' + path)
    (EXP / 'artifacts/scores').mkdir(parents=True, exist_ok=True)
    labels = Labels(frozen['label_files']); periods = {}
    for period in PERIODS:
        periods[period] = score_period(period, frozen, labels)
        if period == 'evaluation_2026': periods[period]['label'] = '未预注册的描述性结果（2026-01-01 至 10-31 标签，评分一次，不据此做任何选择）'
        write(EXP / 'SCORING_PROGRESS.json', dict(completed=list(periods)))
    reading = {}
    for g in ('day2', 'day3', 'day4'):
        for period in ('validation', 'original_evaluation', 'extension_evaluation'):
            c = periods[period]['comparisons']['H-PI - H-G']
            reading[f'{g}/{period}'] = c if c == '未做' else c[g]['classification']
    benefit = all(v == '改善' for v in reading.values())
    result = dict(experiment='exp0027', status='scored_once', source_commit=frozen['source_commit'], scorer_sha256=sha(__file__),
                  prediction_freeze_sha256=sha(EXP / 'FROZEN_PREDICTIONS.json'), checkpoint_freeze_sha256=frozen['checkpoint_freeze_sha256'],
                  arms_scored=[a for a in ARMS if a in frozen['arms']], arms_not_done=[a for a in ARMS if a not in frozen['arms']],
                  periods=periods,
                  preregistered_reading=dict(rule='H-PI has a multi-day benefit only if the five-seed-mean-prediction comparison H-PI - H-G is 改善 on each of days 2, 3 and 4 in validation, old evaluation and 2023-2025 (9 comparisons).',
                                             conditions=reading, H_PI_multi_day_benefit=benefit,
                                             statement='在多日时效有收益' if benefit else '不记为在多日时效有收益'),
                  caveats=['2023–2025 participated in earlier development; not an independent confirmation.',
                           '2026 numbers are unregistered descriptive results; no selection is based on them.',
                           'Intervals describe time variation for fixed trained models, not seed variation.'])
    write(EXP / 'result.json', result)
    print(json.dumps(dict(status='scored_once', reading=reading, benefit=benefit), ensure_ascii=False))


if __name__ == '__main__':
    main()
