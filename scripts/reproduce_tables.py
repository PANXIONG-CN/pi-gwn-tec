"""Recompute the numbers of records/EVIDENCE.csv from the published daily statistics.

    python scripts/reproduce_tables.py --sd1 /path/to/Supplementary_Data_1
    python scripts/reproduce_tables.py --experiment-repo /path/to/experiment/repository   (authors' layout)

Every table of the article is built from per-day sufficient statistics
(valid count, squared-error sum, absolute-error sum, signed-error sum; per lead
time and, where used, per stratum).  This script loads those statistics, runs
the original table and interval functions of tecfc/scoring and tecfc/intervals
on them, and compares every reproducible row of EVIDENCE.csv (groups K, G, R,
P, I, S, H, C, B) with the recomputed value.  This includes the supplementary
experiments: activity classes and storm events (from the class and event
records in activity/ of Supplementary Data 1), single penalty terms and graph
support, and conditioning in the strong configuration.  Rows that are not
functions of the daily statistics (training-time diagnostics, timings,
parameter counts, thresholds, file counts, the night-gate table, the rapid and
forecast-time input maps, values of the earlier article and of the
literature) are listed as 'record' and checked against the records they cite.
CPU only; a few minutes.

Statistics and records are addressed by their location in the experiment
repository.  With --sd1 the MANIFEST.csv of Supplementary Data 1 maps those
locations to its own file names (see Statistics), so Supplementary Data 1
must include its per-day statistics (statistics/*.npz, available on request).

Exit status 0 only if every recomputed row agrees (absolute difference
<= 1e-6; values reported with three decimals <= 5e-4).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RECORDS = REPO / 'records'
GROUPS = set('KGRPISHCB')


# ---------------------------------------------------------------- statistics files
SPLIT_TOKENS = ('validation', 'original_evaluation', 'extension_evaluation', '2026')
SHARED_ARRAYS = ('origins', 'dates', 'fields', 'description')
INTERVAL_FIELDS = {'half_width_TECU': 'q', 'n_covered': 'cov', 'n_covered_subgroup_Y_ge60': 'sub_cov', 'n_negative_lower_bound': 'neg'}
INTERVAL_ARRAYS = {'n_valid': 'nv', 'n_subgroup_Y_ge60': 'ns', 'trailing_calibration_available': 'T_ok'}


class Arrays:
    """Arrays of a Supplementary Data 1 file under the names of the experiment file they were copied from."""

    def __init__(self, archive, names):
        self.archive, self.names = archive, names  # name in the experiment file -> name in Supplementary Data 1

    @property
    def files(self):
        return list(self.names)

    def __contains__(self, key):
        return key in self.names

    def __getitem__(self, key):
        return self.archive[self.names[key]]


def merged_system(internal):
    """System of a statistics file that Supplementary Data 1 merged with others: '<period>_<system>[_<support>]' or
    '<system>_<period>'; its arrays are stored as '<system>__<array>'."""
    stem = re.sub(r'_(global|S1)$', '', Path(internal).stem)
    for token in SPLIT_TOKENS:
        if stem.startswith(token + '_'):
            return stem[len(token) + 1:]
        if stem.endswith('_' + token):
            return stem[:-len(token) - 1]
    raise ValueError(f'{internal}: no period in the file name')


def interval_array(name):
    """Name in the interval-calibration statistics of an array of intervals/*.npz of Supplementary Data 1."""
    m = re.fullmatch(r'(static|trailing)_alpha(0\.\d+)_(.+)', name)
    if m and m.group(3) in INTERVAL_FIELDS:
        return f"{'S' if m.group(1) == 'static' else 'T'}_{m.group(2)}_{INTERVAL_FIELDS[m.group(3)]}"
    return INTERVAL_ARRAYS.get(name, name)


class Statistics:
    """Resolve a statistics file or record by its experiment-repository location.

    With --sd1, MANIFEST.csv of Supplementary Data 1 maps every location (column internal_source, relative to the
    authors' project directory; ' ; '-separated where one file merges several) to the file that holds it, and npz()
    returns the arrays under the names of the experiment file: merged files store them as '<system>__<array>', the
    interval statistics (intervals/) under descriptive names. Supplementary Data 1 keeps only the all-cell subset of
    the global member experiments (exp0013, exp0015, exp0016, exp0018); the other subsets are absent."""

    def __init__(self, experiment_repo=None, sd1=None):
        self.repo = None if experiment_repo is None else Path(experiment_repo)
        self.sd1 = None if sd1 is None else Path(sd1)
        self.manifest = {}  # location -> [(file of Supplementary Data 1, number of locations merged into it)]
        if self.sd1 is not None:
            with (self.sd1 / 'MANIFEST.csv').open(newline='') as handle:
                for row in csv.DictReader(handle):
                    sources = [s.strip() for s in row.get('internal_source', row.get('source', '')).split(' ; ')]
                    for source in sources:
                        m = re.fullmatch(r'(?:[^/]+/)?(experiments/.+)', source)  # drop the experiment-repository folder
                        self.manifest.setdefault(m.group(1) if m else source, []).append((self.sd1 / row['file'], len(sources)))

    def _entry(self, internal):
        entries = [e for e in self.manifest.get(internal, []) if e[0].suffix == Path(internal).suffix]
        if len(entries) != 1:
            raise FileNotFoundError(f'{internal} is not in Supplementary Data 1 (MANIFEST.csv)' if not entries
                                    else f'{internal}: several files of Supplementary Data 1')
        return entries[0]

    def path(self, internal):
        if self.repo is not None:
            return self.repo / internal
        return self._entry(internal)[0]

    def npz(self, internal):
        if self.repo is not None:
            return np.load(self.repo / internal, allow_pickle=False)
        path, merged = self._entry(internal)
        archive = np.load(path, allow_pickle=False)
        if merged > 1:
            prefix = merged_system(internal) + '__'
            names = {k[len(prefix):]: k for k in archive.files if k.startswith(prefix)}
            if not names:
                raise KeyError(f'{path.name} has no arrays {prefix}* for {internal}')
            names.update({k: k for k in SHARED_ARRAYS if k in archive.files})
            return Arrays(archive, names)
        if path.parent.name == 'intervals':
            return Arrays(archive, {interval_array(k): k for k in archive.files})
        return archive

    def json(self, internal):
        return json.loads(self.path(internal).read_text())

    def rows(self, internal):
        with self.path(internal).open(newline='') as handle:
            return list(csv.DictReader(handle))


def record(name):
    """Values that the article cites but that are not functions of the daily statistics."""
    return json.loads((RECORDS / 'scoring' / name).read_text())


# ---------------------------------------------------------------- 2026 confirmation (groups K, G, R)
K_GLOBAL = [('K1', 'S2-G5'), ('K2', 'G-plus-M5-G5'), ('K3', 'G-geo5-G5'), ('K4', 'PI5-G5')]
K_REGIONAL = [('K5', 'anchor-persistence'), ('K6', 'final-anchor'), ('K7', 'final-persistence')]


def confirmation(stats):
    """Tables of the 2026 confirmation and the matched-month 2023-2025 comparison.

    As in the original one-time scoring entry: table() over the saved statistics
    of each dataset, January-June for the global datasets, then the
    preregistered criteria K1-K7 (classify())."""
    from tecfc.scoring.confirm_final import table, classify
    descriptors = json.loads((RECORDS / 'scoring/confirmation_datasets.json').read_text())['datasets']
    datasets = {}
    for key, desc in descriptors.items():
        z = stats.npz(f'experiments/exp0025/artifacts/scores/{key}.npz')
        names = [str(x) for x in z['names']]
        if names != desc['methods']:
            raise ValueError(f'{key}: statistics do not follow the scoring descriptor')
        # Supplementary Data 1 omits the empty subset arrays (index-sensitivity dataset).
        ts = z['origins']; groups = z['groups'] if 'groups' in z.files and z['groups'].size else None
        group_names = [str(x) for x in z['group_names']] if 'group_names' in z.files else []
        domain = desc['domain']
        result = dict(full=table(z['stats'], groups, names, ts, desc['main_methods'], domain, group_names))
        if domain == 'global' and not key.startswith('S1'):
            months = ts.astype('datetime64[s]').astype('datetime64[D]').astype('datetime64[M]').astype(int) % 12 + 1
            result['January_June'] = table(z['stats'], groups, names, ts, desc['main_methods'], domain, group_names, months <= 6)
        if domain != 'global' and desc['period'] == '2026':
            # As in the original evaluate(): bias change in the persistent-high stratum.
            rows = result['full']['methods']; f = rows['final']['groups']['persistent_high']['bias']; a = rows['anchor']['groups']['persistent_high']['bias']
            result['persistent_high_bias'] = dict(final=f, anchor=a, signed_change=f - a if f is not None and a is not None else None,
                                                  absolute_bias_change=abs(f) - abs(a) if f is not None and a is not None else None)
        datasets[key] = result
    g = datasets['global_2026']['full']['comparisons']
    criteria = {k: classify(g[p], k == 'K4') for k, p in K_GLOBAL}
    for region in ('china', 'japan'):
        pairs = datasets[region + '_2026']['full']['comparisons']
        criteria[region] = {k: classify(pairs[p]) for k, p in K_REGIONAL}
    return dict(datasets=datasets, criteria=criteria), record('confirmation_records.json')


# ---------------------------------------------------------------- regional development 2023-2025 (group R)
def regional_development(stats):
    """Regional 2023-2025 tables: the table part of the original evaluate() and its intervals()."""
    from tecfc.regional import development_score as score
    old, GROUPS, SEEDS = score.old, score.GROUPS, score.SEEDS
    regions = {}
    for region in ('china', 'japan'):
        internal = f'experiments/exp0023/artifacts/{region}_2023-2025_daily_statistics.npz'
        z = stats.npz(internal)
        ts, names, st, groups = z['origins'], [str(x) for x in z['names']], z['stats'], z['groups']
        descriptors = names[:names.index('anchor')]
        methods = descriptors + ['anchor', 'persistence']
        seed_names = [f'{name}::seed{s}' for name in descriptors for s in SEEDS]
        pi = names.index('persistence')
        # From here on: the original evaluate() after the statistics were computed.
        tables = {name: old.table(st[j], groups[j], ts, st[pi], groups[pi]) for j, name in enumerate(names)}
        years = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
        for j, name in enumerate(names):
            tables[name]['group_yearly'] = {str(year): {g: old.summary(groups[j, years == year, k], groups[pi, years == year, k]) for k, g in enumerate(GROUPS)} for year in sorted(set(years))}
        seed_metrics = {}
        for name in descriptors:
            rows = [tables[f'{name}::seed{s}'] for s in SEEDS]
            seed_metrics[name] = old.seed_summary([{k: v for k, v in row.items() if k != 'group_yearly'} for row in rows])
            seed_metrics[name]['group_yearly'] = {year: {group: old.aggregate([row['group_yearly'][year][group] for row in rows]) for group in GROUPS} for year in rows[0]['group_yearly']}
        result = dict(days=len(ts), methods={name: dict(**tables[name], seed_sd=None) for name in methods},
                      seed_metrics=seed_metrics, individual_seeds={name: tables[name] for name in seed_names},
                      sufficient_statistics=dict(path=str(stats.path(internal))))
        result['time_block_intervals'] = score.intervals(result)
        regions[region] = {'2023-2025': result}
    return dict(regions=regions), record('regional_development_records.json')


# ---------------------------------------------------------------- interval calibration (group C)
def interval_calibration(stats):
    """Coverage, widths and their time-block intervals from the per-day coverage statistics.

    The metric part of the original summarize(): K.metrics(), K.strata_for(),
    K.replicates() and S.wording() on the saved per-day counts and quantiles.
    Calibration-set sizes and quantile ranks are records."""
    from tecfc.intervals import spec as S, calibration as K
    records = record('interval_calibration_records.json')
    systems = {}
    for system, cfg in S.SYSTEMS.items():
        z = stats.npz(f'experiments/exp0028/artifacts/day_stats_{system}.npz')
        ts = z['origins']; dates = ts.astype('datetime64[s]').astype('datetime64[D]')
        years = dates.astype('datetime64[Y]').astype(int) + 1970
        nv, ns, ok = z['nv'], z['ns'], z['T_ok']
        tabs = {m: {k: {f: z[f'{m}_{k}_{f}'] for f in ('q', 'cov', 'sub_cov', 'neg')} for k in S.ALPHAS} for m in 'ST'}
        inc = {'S': np.ones(len(ts), bool), 'T': ok}
        scope = {'all': np.ones(len(ts), bool), 'P1': np.isin(years, cfg['P1']), 'P2': np.isin(years, cfg['P2'])}
        for y in cfg['eval_years']:
            scope['Y%d' % y] = years == y
        keep = {m: {name: mask & inc[m] for name, mask in scope.items()} for m in 'ST'}
        frames = {}

        def frame(m, name):
            key = (m, name) if m == 'T' else ('S', name)
            if key not in frames:
                frames[key] = K.strata_for(dates, keep[m][name])
            return frames[key]
        n_days = records['systems'][system]['static']['n_calibration_days']
        out = dict(static=dict(per_alpha={
            k: dict(m=S.level_numerator(n_days, a), level_p=S.level(n_days, a), q_by_horizon=tabs['S'][k]['q'][0].tolist())
            for k, a in S.ALPHAS.items()}))
        out['rolling'] = dict(required_days=S.ROLL_MIN, targets=int(len(ok)), retained=int(ok.sum()), excluded=int((~ok).sum()),
                              excluded_by_year={str(y): int(((~ok) & (years == y)).sum()) for y in cfg['eval_years']},
                              retained_by_year={str(y): int((ok & (years == y)).sum()) for y in cfg['eval_years']},
                              mean_q_by_horizon={k: (tabs['T'][k]['q'][ok].mean(axis=0).tolist() if ok.any() else None) for k in S.ALPHAS})
        out['eval_days'] = dict(S={str(y): int((years == y).sum()) for y in cfg['eval_years']},
                                T={str(y): int((ok & (years == y)).sum()) for y in cfg['eval_years']})
        out['metrics'] = {}; out['T_minus_S'] = {}; tot = {}
        for k in S.ALPHAS:
            tot[k] = {m: K.day_totals(tabs[m][k], nv) for m in 'ST'}
            out['metrics'][k] = {}
            for m in 'ST':
                rows = {}
                for name in scope:
                    mask = keep[m][name]
                    row = K.metrics(tabs[m][k], nv, ns, mask)
                    if row is None:
                        rows[name] = None; continue
                    num, den = tot[k][m]
                    row['coverage_ci95'] = K.interval95(K.replicates(frame(m, name), num, den))
                    row['wording'] = S.wording(row['coverage'], k)
                    row['n_days'] = int(mask.sum())
                    rows[name] = row
                hz = {}
                for l in range(S.HORIZONS):
                    r = K.metrics(tabs[m][k], nv, ns, keep[m]['all'], slice(l, l + 1))
                    hz[str(S.LEADS[l])] = None if r is None else dict(coverage=r['coverage'], mean_width=r['mean_width'], n_valid=r['n_valid'],
                                                                       wording=S.wording(r['coverage'], k))
                out['metrics'][k][m] = dict(scopes=rows, by_lead_hours=hz)
            diffs = {}
            for name in scope:
                mask = keep['T'][name]
                if not mask.any():
                    diffs[name] = None; continue
                rt = K.metrics(tabs['T'][k], nv, ns, mask); rs = K.metrics(tabs['S'][k], nv, ns, mask)
                fr = frame('T', name)
                repT = K.replicates(fr, *tot[k]['T']); repS = K.replicates(fr, *tot[k]['S'])
                diffs[name] = dict(n_days_matched=int(mask.sum()), coverage_T=rt['coverage'], coverage_S_matched=rs['coverage'],
                                   coverage_difference=rt['coverage'] - rs['coverage'], coverage_difference_ci95=K.interval95(repT - repS),
                                   width_T=rt['mean_width'], width_S_matched=rs['mean_width'], width_difference=rt['mean_width'] - rs['mean_width'])
            out['T_minus_S'][k] = diffs
        systems[system] = out
    return dict(systems=systems), records


# ---------------------------------------------------------------- 96-hour forecasts (group H)
def h96_forecasts(stats):
    """The original score_period() with its prediction reader replaced by the saved window statistics."""
    import tempfile
    from tecfc.scoring import h96 as H
    periods = {}
    original_collect, original_exp = H.collect, H.EXP
    try:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'artifacts/scores').mkdir(parents=True)
            H.EXP = Path(tmp)  # score_period() saves its statistics again; keep them out of the repository
            for period in H.PERIODS:
                z = stats.npz(f'experiments/exp0027/artifacts/scores/{period}_window_lead_stats.npz')
                arms = [a for a in H.ARMS if f'{a}5' in z.files]
                loaded = (z['origins'], {(a, s): z[f'{a}_{s}'] for a in arms for s in H.SEEDS}, {a: z[f'{a}5'] for a in arms},
                          z['persistence'], {(f, s): z[f'{f}_{s}_day1'] for f in H.REFERENCES for s in H.SEEDS},
                          {f: z[f'{f}5_day1'] for f in H.REFERENCES})
                H.collect = lambda period, frozen, labels, loaded=loaded: loaded
                periods[period] = H.score_period(period, None, None)
    finally:
        H.collect, H.EXP = original_collect, original_exp
    # As in the original main(): the preregistered reading of the penalty arm over days 2-4.
    reading = {}
    for g in ('day2', 'day3', 'day4'):
        for period in ('validation', 'original_evaluation', 'extension_evaluation'):
            c = periods[period]['comparisons']['H-PI - H-G']
            reading[f'{g}/{period}'] = c if c == '未做' else c[g]['classification']
    benefit = all(v == '改善' for v in reading.values())
    return dict(periods=periods, preregistered_reading=dict(conditions=reading, H_PI_multi_day_benefit=benefit)), {}


# ---------------------------------------------------------------- penalty strength, index ablation, longitude wrap (groups P, I, S)
def ablations(stats):
    """The loops of the original evaluate() over the saved per-day statistics of every arm."""
    from tecfc.scoring.ablation_evaluate import statistics, shared_draws, comparison, SPLITS, ARMS
    import glob as _glob
    folder = 'experiments/exp0026/artifacts/daily'
    tables, ensembles, pairs = {}, {}, {}
    available = {}
    for split in SPLITS:
        for arm in ('G', 'PI', 'EMP', 'G-DE', *ARMS):
            for support in ('global', 'S1'):
                internal = f'{folder}/{split}_{arm}_{support}.npz'
                try:
                    z = stats.npz(internal)
                except FileNotFoundError:
                    continue
                available.setdefault((split, arm), {})[tuple(z['origins'].tolist())] = (z['origins'], z['single'], z['ensemble'], support)

    def own(split, arm):
        files = available.get((split, arm), {})
        rows = sorted(files.values(), key=lambda r: r[3] != 'global')
        return rows[0] if rows else None

    def get(split, arm, origins):
        return available[(split, arm)][tuple(np.asarray(origins).tolist())][1:3]
    for split in SPLITS:
        tables[split], ensembles[split], pairs[split] = {}, {}, {}
        for arm in ('G', 'PI', 'EMP', 'G-DE', *ARMS):
            row = own(split, arm)
            if row is None or len(row[1]) != 5:
                continue
            origins, s, e, _ = row
            tables[split][arm], ensembles[split][arm] = statistics(s), statistics(e)
            tables[split][arm]['windows'] = len(origins)
            ensembles[split][arm]['windows'] = len(origins)
        draw_cache = {}
        for arm in ARMS:
            if arm not in tables[split]:
                continue
            for baseline in (('G', 'PI') if arm.startswith('P-') else ('G-DE', 'G') if arm.startswith('D-') else ('G',)):
                ts_a, ts_b = own(split, arm)[0], own(split, baseline)[0]
                origins = np.intersect1d(ts_a, ts_b)
                if len(origins) != min(len(ts_a), len(ts_b)):
                    raise ValueError('Unexpected comparison support')
                a, ae = get(split, arm, origins)
                b, be = get(split, baseline, origins)
                key = tuple(origins)
                if key not in draw_cache:
                    draw_cache[key] = shared_draws(origins)
                paired = comparison(a, b, ae, be, draw_cache[key])
                paired['windows'] = len(origins)
                pairs[split][arm + '-' + baseline] = paired
    decision = {}
    for arm in ('P-0.1', 'P-1.0', 'P-adapt'):
        comparisons = [pairs[split].get(arm + '-G') for split in SPLITS[:3]]
        decision[arm] = '未完成' if not all(row is not None for row in comparisons) else (
            '该强度有收益' if all(row['classification'] == '改善' for row in comparisons) else '无收益')
    return dict(main_table=tables, ensemble_table=ensembles, paired=pairs, P_decision=decision), record('ablation_records.json')


# ---------------------------------------------------------------- 24-hour members on the strong recipe (groups G, P, I)
def _split_daily(stats, internal, arms, groups):
    z = stats.npz(internal)
    groups = [g for g in groups if f'{arms[0]}_{g}' in z.files]  # Supplementary Data 1: the all-cell subset only
    return z['origins'], {a: {g: z[f'{a}_{g}'] for g in groups} for a in arms}, {a: {g: z[f'{a}5_{g}'] for g in groups} for a in arms}


def penalty_members(stats):
    """Penalty (1%) and empirical-tendency arms: the table loops of the original scoring entry."""
    from tecfc.scoring import pooled as ref
    main_table, ensemble_table, pairs, ensemble_pairs = {}, {}, {}, {}
    for split in ref.SPLITS:
        ts, s, e = _split_daily(stats, f'experiments/exp0013/artifacts/{split}_daily.npz', ('G', 'DO', 'PI', 'EMP'), ref.GROUPS)
        main_table[split] = {arm: ref.tables(s[arm], ts, ref.SEEDS) for arm in ('DO', 'PI', 'EMP', 'G')}
        ensemble_table[split] = {arm: ref.tables(e[arm], ts) for arm in ('DO', 'PI', 'EMP', 'G')}
        pairs[split] = {group: {f'{a}-{b}': ref.paired(s[a][group], s[b][group], ts)
                        for a, b in [('PI', 'DO'), ('EMP', 'DO'), ('PI', 'EMP')]} for group in s['G']}
        ensemble_pairs[split] = {f'{a}5-{b}5': ref.paired(e[a]['all'], e[b]['all'], ts)
                                 for a, b in [('PI', 'G'), ('EMP', 'G'), ('PI', 'DO'), ('EMP', 'DO')]}
    result = dict(main_table=main_table, ensemble_table=ensemble_table, paired=pairs, ensemble_paired=ensemble_pairs)
    # The decision rule reads the high-TEC and disturbed-day subsets, which Supplementary Data 1 does not include.
    if all(set(ref.GROUPS) <= set(pairs[split]) for split in ref.SPLITS):
        result['decision'] = ref.decisions(pairs, main_table)
    return result, record('penalty_members_records.json')


def information_members(stats):
    """Index, history and geometry members: the table loops of the original scoring entry."""
    from tecfc.scoring import pooled as ref
    from tecfc.scoring.members_score import with_sds
    arms = ['G-DE', 'G-long', 'G-geo']
    main_table, ensemble_table, pairs, ensemble_pairs = {}, {}, {}, {}
    for split in ref.SPLITS:
        ts, s, e = _split_daily(stats, f'experiments/exp0015/artifacts/{split}_daily.npz', ['G'] + arms, ref.GROUPS)
        main_table[split] = {arm: with_sds(ref.tables(s[arm], ts, ref.SEEDS)) for arm in ['G'] + arms}
        ensemble_table[split] = {arm: ref.tables(e[arm], ts) for arm in ['G'] + arms}
        pairs[split] = {group: {arm + '-G': ref.paired(s[arm][group], s['G'][group], ts) for arm in arms} for group in s['G']}
        ensemble_pairs[split] = {arm + '5-G5': ref.paired(e[arm]['all'], e['G']['all'], ts) for arm in arms}
    return dict(main_table=main_table, ensemble_table=ensemble_table, paired=pairs, ensemble_paired=ensemble_pairs), {}


def _load_statistics(stats, folder, arm, split, groups):
    z = stats.npz(f'{folder}/{arm}_{split}.npz')
    groups = [k for k in groups if k in z.files]  # Supplementary Data 1: the all-cell subset only
    return z['origins'], {k: z[k] for k in groups}, {k: z['ensemble_' + k] for k in groups}


def _lead_hours(ref, a, b):
    out = {}
    for h in range(12):
        aa, bb = ref.summary(a[:, :, h]), ref.summary(b[:, :, h])
        delta = [x['RMSE'] - y['RMSE'] if x['RMSE'] is not None else None for x, y in zip(aa['by_seed'], bb['by_seed'])]
        out[str(2 * (h + 1))] = dict(by_seed_delta=delta, mean_delta=float(np.mean(delta)) if delta[0] is not None else None)
    return out


def mae_and_climatology(stats):
    """GWN (MAE) and SimVP-clim: the loops of the original score()."""
    from tecfc.scoring import pooled as ref
    from tecfc.scoring.clim_score import enrich, SEEDS, SPLITS
    folder = 'experiments/exp0016/artifacts/statistics'
    main_table, ensemble_table, paired, ensemble_pairs = {}, {}, {}, {}
    for split in SPLITS:
        ts, g, ge = _load_statistics(stats, folder, 'G', split, ref.GROUPS)
        main_table[split] = {'G': enrich(ref.tables(g, ts, SEEDS))}
        ensemble_table[split] = {'G': enrich(ref.tables(ge, ts), True)}
        paired[split] = {}; ensemble_pairs[split] = {}
        for arm in ('G-MAE', 'C-in'):
            _, s, e = _load_statistics(stats, folder, arm, split, ref.GROUPS)
            main_table[split][arm] = enrich(ref.tables(s, ts, SEEDS)); ensemble_table[split][arm] = enrich(ref.tables(e, ts), True)
            paired[split][arm] = {group: ref.paired(s[group], g[group], ts) for group in g}
            for group in g:
                paired[split][arm][group]['lead_hours'] = _lead_hours(ref, s[group], g[group])
            ensemble_pairs[split][arm] = {}
            for group in g:
                row = ref.paired(e[group], ge[group], ts)
                ensemble_pairs[split][arm][group] = {k: row[k] for k in ('mean_delta', 'ci95') if k in row}
    return dict(main_table=main_table, ensemble_table=ensemble_table, paired_vs_G=paired, ensemble_paired_vs_G5=ensemble_pairs), {}


def combined_members(stats):
    """GWN-geo-hist and SimVP-clim-hist: the loops of the original score(), with its additivity table."""
    from tecfc.scoring import pooled as ref
    from tecfc.scoring.combined_score import enrich, SEEDS, SPLITS, ARMS, BASES, COMPARISONS
    folder = 'experiments/exp0018/artifacts/statistics'
    main, ensemble, paired, additivity = {}, {}, {}, {}
    for split in SPLITS:
        main[split] = {}; ensemble[split] = {}; paired[split] = {}; allstats = {}
        for arm in BASES + ARMS:
            ts, s, e = _load_statistics(stats, folder, arm, split, ref.GROUPS)
            allstats[arm] = s
            main[split][arm] = enrich(ref.tables(s, ts, SEEDS)); ensemble[split][arm] = enrich(ref.tables(e, ts), True)
        for arm, controls in COMPARISONS.items():
            for control in controls:
                key = arm + '_minus_' + control; paired[split][key] = {}
                for group in allstats[arm]:
                    a, b = allstats[arm][group], allstats[control][group]
                    value = ref.paired(a, b, ts)
                    value['lead_hours'] = _lead_hours(ref, a, b)
                    paired[split][key][group] = value
        rmse = lambda a: main[split][a]['all']['RMSE']
        improvement = {a: rmse('G') - rmse(a) for a in ('G-long', 'G-geo', 'G-MAE')}
        additivity[split] = {}
        for arm, components in [('G-plus-M', ('G-long', 'G-geo', 'G-MAE')), ('G-plus-S', ('G-long', 'G-geo'))]:
            actual = rmse('G') - rmse(arm); expected = sum(improvement[a] for a in components)
            additivity[split][arm] = dict(reference='G', actual_improvement=actual, single_component_improvements={a: improvement[a] for a in components},
                                          sum_of_component_improvements=expected, actual_minus_sum=actual - expected)
        actual = rmse('C') - rmse('C-plus'); proxy = rmse('C') - rmse('C-in') + improvement['G-long']
        additivity[split]['C-plus'] = dict(reference='C', actual_improvement=actual, sum_of_component_improvements=None,
                                           cross_backbone_proxy_sum=proxy, actual_minus_proxy=actual - proxy,
                                           proxy_components={'C_minus_Cin': rmse('C') - rmse('C-in'), 'G_minus_Glong': improvement['G-long']})
    return dict(main_table=main, ensemble_table=ensemble, paired=paired, additivity=additivity), {}


# ---------------------------------------------------------------- multi-family ensemble, full periods (group G)
def ensemble_tables(stats):
    """Main table of the ensemble experiment: pooled tables() per method with seed labels stripped,
    and the five-seed single-model GWN table, from the saved formal statistics."""
    from tecfc.scoring import pooled as score
    groups = ['all', 'unique_longitude', 'high_TEC', 'disturbed_day']

    def strip_seed_labels(value):  # as in the original table builder
        if isinstance(value, dict):
            return {k: strip_seed_labels(v) for k, v in value.items() if k not in ('by_seed', 'RMSE_seed_sd')}
        return value
    main, single = {}, {}
    for split in score.SPLITS:
        data = stats.npz(f'experiments/exp0019/artifacts/{split}_formal.npz')
        main[split] = {name: strip_seed_labels(score.tables({g: data['stats'][j:j + 1, :, h] for h, g in enumerate(groups)}, data['origins']))
                       for j, name in enumerate(data['methods'].tolist())}
        single[split] = score.tables({g: data['single_G'][:, :, h] for h, g in enumerate(groups)}, data['origins'], score.SEEDS)
    return dict(main_table=main, single_model_five_seed_mean=single), {}


# ---------------------------------------------------------------- zero-shot mapping and product difference, 2023-2025 (group R)
def _definitions(path, names):
    """Execute selected function definitions of a module without running its module-level code."""
    import ast
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def zero_shot_mapping(stats):
    """Regional scores of the mapped global forecasts and of the mapped IGS labels (summary() of the original scorer)."""
    summary = _definitions(REPO / 'tecfc/regional/mapping_score.py', ('summary',))['summary']
    regions = {}
    for region in ('CN', 'JP'):
        z = stats.npz(f'experiments/exp0011/artifacts/{region}_evaluation_daily.npz')
        groups = [str(g) for g in z['groups']]
        baseline = summary(z['P__all'])['RMSE']
        methods = {}
        for k in ('G5', 'product', 'P'):
            a = dict(all=z[k + '__all'], groups=z[k + '__groups'])
            row = summary(a['all']); row['improvement_percent'] = 100 * (1 - row['RMSE'] / baseline)
            row['groups'] = {g: summary(a['groups'][:, :, h]) for h, g in enumerate(groups)}
            methods[k] = row
        regions[region] = dict(evaluation=dict(methods=methods))
    return dict(regions=regions), {}


# ---------------------------------------------------------------- activity classes and storm events, 2023-2026 (groups G, R)
ACTIVITY_CLASSES = ('quiet', 'moderate', 'storm')
ACTIVITY_SYSTEMS = ('S2', 'G-plus-M5', 'C-plus5', 'C-in5', 'G-geo5', 'C5', 'G5', 'PI5', 'EMP5', 'persistence')
REGIONAL_ACTIVITY_SYSTEMS = ('final', 'anchor', 'persistence')


def _dates(origins):
    return [str(d) for d in np.asarray(origins).astype('datetime64[s]').astype('datetime64[D]')]


def _pooled(a):
    """Pooled metrics of summed (n, SSE, SAE, error sum) statistics over all given days and lead times."""
    t = np.asarray(a, dtype=np.float64).reshape(-1, 4).sum(0)
    if t[0] == 0:
        return dict(n=0, RMSE=None, MAE=None, bias=None)
    return dict(n=int(t[0]), RMSE=float(np.sqrt(t[1] / t[0])), MAE=float(t[2] / t[0]), bias=float(t[3] / t[0]))


def _event_summary(events):
    if not events:
        return dict(events=0)
    a = np.array([e['active_hours'] for e in events]); s = np.array([e['span_hours'] for e in events])
    q = lambda x, p: float(np.quantile(x, p, method='linear'))
    return dict(events=len(events), active_hours=dict(min=int(a.min()), q25=q(a, .25), median=q(a, .5), q75=q(a, .75), max=int(a.max()), total=int(a.sum())),
                span_hours=dict(min=int(s.min()), q25=q(s, .25), median=q(s, .5), q75=q(s, .75), max=int(s.max())),
                active_hours_bins={'6-11': int(((a >= 6) & (a < 12)).sum()), '12-23': int(((a >= 12) & (a < 24)).sum()),
                                   '24-47': int(((a >= 24) & (a < 48)).sum()), '>=48': int((a >= 48).sum())},
                strong_events=sum(1 for e in events if e['kp_max'] >= 7.0 or e['dst_min'] <= -100.0))


def activity_classes(stats):
    """Activity classes, storm events and event-level errors: the descriptive aggregation of the original analysis,
    sums of the published per-day statistics over the days of each class or event.

    The class of every target day is re-derived from its maximum Kp and minimum Dst in the class record (all 24 hourly
    OMNI2 values; storm: Kp >= 5 or Dst <= -50 nT; quiet: Kp < 3 and Dst > -50 nT; moderate: the other complete days;
    strong storm, counted only: Kp >= 7 or Dst <= -100 nT). Storm events (at least six active hours, gaps of at most 12 h
    merged) and the target days they touch are read from the event record."""
    days = stats.rows('experiments/exp0030/DAY_CLASSES.csv')
    day_class, complete, strong = {}, {}, {}
    for row in days:
        ok = row['complete_24h'] == '1'
        kp, dst = (float(row['max_kp']), float(row['min_dst_nT'])) if ok else (math.nan, math.nan)
        storm = ok and (kp >= 5.0 or dst <= -50.0)
        quiet = ok and kp < 3.0 and dst > -50.0 and not storm
        day_class[row['date']] = 'storm' if storm else 'quiet' if quiet else 'moderate' if ok else 'excluded'
        complete[row['date']] = ok
        strong[row['date']] = ok and (kp >= 7.0 or dst <= -100.0)
        if day_class[row['date']] != row['class'] or int(strong[row['date']]) != int(row['strong']):
            raise ValueError(f"{row['date']}: the class record does not follow the class rule")

    def counts(selected):
        return dict(calendar_days=len(selected), complete=sum(complete[d] for d in selected), excluded=sum(not complete[d] for d in selected),
                    **{c: sum(day_class[d] == c for d in selected) for c in ACTIVITY_CLASSES}, strong=sum(strong[d] for d in selected))
    dates = sorted(day_class)
    calendar = {f'y{y}': counts([d for d in dates if int(d[:4]) == y]) for y in (2023, 2024, 2025)}
    calendar['y2026_jan_oct'] = counts([d for d in dates if d[:4] == '2026'])
    calendar['y2026_jan_jun'] = counts([d for d in dates if d[:4] == '2026' and int(d[5:7]) <= 6])
    calendar['dev_2023_2025'] = counts([d for d in dates if int(d[:4]) <= 2025])

    events = stats.rows('experiments/exp0030/EVENTS.csv')
    for e in events:
        e.update(active_hours=int(e['active_hours']), span_hours=int(e['span_hours']), kp_max=float(e['kp_max']),
                 dst_min=float(e['dst_min']), start_year=int(e['start_year']), touched_days=e['touched_days'].split(';'))
    in_dev = lambda e: e['start_year'] <= 2025
    in_26 = lambda e: e['start_year'] == 2026
    in_26h1 = lambda e: e['start_year'] == 2026 and int(e['start_utc'][5:7]) <= 6
    by_period = {f'y{y}': _event_summary([e for e in events if e['start_year'] == y]) for y in (2023, 2024, 2025)}
    by_period['dev_2023_2025'] = _event_summary([e for e in events if in_dev(e)])
    by_period['y2026_jan_oct'] = _event_summary([e for e in events if in_26(e)])
    by_period['y2026_jan_jun'] = _event_summary([e for e in events if in_26h1(e)])

    # Global 2023-2025: the days shared by all ten systems (the 24-h persistence statistics end on 2025-12-28).
    mfe = stats.npz('experiments/exp0019/artifacts/extension_evaluation_formal.npz')
    methods, subset = [str(m) for m in mfe['methods']], [str(g) for g in mfe['groups']].index('all')
    pen = stats.npz('experiments/exp0013/artifacts/extension_evaluation_daily.npz')
    dev_dates = _dates(mfe['origins'])
    family = {}
    for arm in ('G-plus-M', 'C-plus', 'C-in', 'G-geo', 'C'):
        z = stats.npz(f'experiments/exp0018/artifacts/statistics/{arm}_extension_evaluation.npz')
        family[arm + '5'] = z['ensemble_all'][0]
        if _dates(z['origins']) != dev_dates:
            raise ValueError(f'{arm}: other days than the multi-family ensemble')
    if _dates(pen['origins']) != dev_dates:
        raise ValueError('penalty (1%): other days than the multi-family ensemble')
    development = {'S2': mfe['stats'][methods.index('S2'), :, subset], **family, 'G5': mfe['stats'][methods.index('G5'), :, subset],
                   'PI5': pen['PI5_all'][0], 'EMP5': pen['EMP5_all'][0]}
    h96 = stats.npz('experiments/exp0027/artifacts/scores/extension_evaluation_window_lead_stats.npz')
    position = {d: i for i, d in enumerate(_dates(h96['origins']))}
    common = [d for d in dev_dates if d in position]
    index = [i for i, d in enumerate(dev_dates) if d in position]
    development = {k: development[k][index] for k in ACTIVITY_SYSTEMS if k != 'persistence'}
    development['persistence'] = h96['persistence'][[position[d] for d in common], :12]
    z = stats.npz('experiments/exp0025/artifacts/scores/global_2026.npz')
    names = [str(n) for n in z['names']]
    domains = {('global', 'dev'): (common, development, in_dev),
               ('global', 'conf26'): (_dates(z['origins']), {k: z['stats'][names.index(k)] for k in ACTIVITY_SYSTEMS}, in_26)}
    for region in ('china', 'japan'):
        for period, internal in (('dev', f'experiments/exp0023/artifacts/{region}_2023-2025_daily_statistics.npz'),
                                 ('conf26', f'experiments/exp0025/artifacts/scores/{region}_2026.npz')):
            z = stats.npz(internal)
            names = [str(n) for n in z['names']]
            domains[(region, period)] = (_dates(z['origins']), {k: z['stats'][names.index(k)] for k in REGIONAL_ACTIVITY_SYSTEMS},
                                         in_dev if period == 'dev' else in_26h1)

    strata, event_level = {}, {}
    for (domain, period), (dates_, data, test) in domains.items():
        classes = np.array([day_class[d] for d in dates_])
        out = {}
        for k, a in data.items():
            out[k] = {c: dict(days=int((classes == c).sum()), **_pooled(a[classes == c])) for c in ACTIVITY_CLASSES}
            out[k]['excluded_days'] = int((classes == 'excluded').sum())
            out[k]['all'] = dict(days=len(dates_), **_pooled(a))
        strata.setdefault(domain, {})[period] = out
        position = {d: i for i, d in enumerate(dates_)}
        per_event = {}
        for e in events:
            selected = [position[d] for d in e['touched_days'] if d in position]
            if test(e) and selected:
                per_event[e['event_id']] = {k: _pooled(a[selected]) for k, a in data.items()}
        pairs = [('S2', 'persistence'), ('S2', 'G5')] if domain == 'global' else [('final', 'persistence'), ('anchor', 'persistence'), ('final', 'anchor')]
        summary = {}
        for a, b in pairs:
            d = np.array([row[a]['RMSE'] - row[b]['RMSE'] for row in per_event.values()])
            summary[f'{a}-{b}'] = dict(events=int(len(d)), lower_rmse_events=int((d < 0).sum()),
                                       median_delta_RMSE=float(np.median(d)) if len(d) else None,
                                       q25_delta_RMSE=float(np.quantile(d, .25)) if len(d) else None,
                                       q75_delta_RMSE=float(np.quantile(d, .75)) if len(d) else None)
        event_level.setdefault(domain, {})[period] = summary
    return dict(calendar=calendar, events=by_period, strata=strata, event_level=event_level), {}


# ---------------------------------------------------------------- penalty terms, graph support, conditioning (groups P, B)
SUPPLEMENTARY_SPLITS = ('validation', 'original_evaluation', 'extension_evaluation', '2026')


def _daily(stats, experiment, split, arms):
    out = {}
    for arm in arms:
        z = stats.npz(f'experiments/{experiment}/artifacts/daily/{split}_{arm}.npz')
        out[arm] = (np.asarray(z['origins']), z['single'], z['ensemble'])
    return out


def _interaction(pa, pb, qa, qb, draws, ref):
    """(pa - pb) - (qa - qb): ensemble level, same-seed pairs and shared time-block draws, as in the original scoring pass."""
    (a, ae), (b, be), (c, ce), (d, de) = pa, pb, qa, qb
    seeds = [(x['RMSE'] - y['RMSE']) - (z['RMSE'] - w['RMSE']) for x, y, z, w in
             zip(ref.summary(a)['by_seed'], ref.summary(b)['by_seed'], ref.summary(c)['by_seed'], ref.summary(d)['by_seed'])]
    value = (ref.summary(ae)['RMSE'] - ref.summary(be)['RMSE']) - (ref.summary(ce)['RMSE'] - ref.summary(de)['RMSE'])
    sampled = (ref.time_rmse(ae, draws) - ref.time_rmse(be, draws)) - (ref.time_rmse(ce, draws) - ref.time_rmse(de, draws))
    low, high = np.quantile(sampled, [.025, .975]).tolist()
    label = '改善' if value < 0 and high < 0 else '变差' if value > 0 and low > 0 else '无差别'
    return dict(by_seed=seeds, seed_mean=float(np.mean(seeds)), ensemble=value, ensemble_ci95=[low, high], classification=label,
                sign='negative = the penalty helps more than with full support')


def penalty_variants(stats):
    """Single penalty terms, gated nighttime term, graph support x penalty and SimVP + penalty (10% gradient ratio, strong
    configuration): the table, comparison and interaction loops of the original single scoring pass over its saved
    per-day statistics. statistics(), shared_draws() and comparison() are those of the penalty-strength experiment; the
    scoring pass used copies of them that differ only in fields that are not cited."""
    from tecfc.scoring.ablation_evaluate import statistics, shared_draws, comparison
    from tecfc.scoring.ablation_score import ref
    terms = ('T-nonneg', 'T-rot', 'T-roti', 'T-diffusion', 'T-chemistry', 'T-night')
    arms = ('G', 'P-0.1', 'S') + terms + ('N-gate', 'P-gate', 'F-local', 'F-local-P', 'F-adapt', 'F-adapt-P', 'F-none', 'F-none-P', 'S-P')
    comparisons = [(arm, 'G') for arm in terms] + [(arm, 'P-0.1') for arm in terms] + [
        ('N-gate', 'T-night'), ('N-gate', 'G'), ('P-gate', 'P-0.1'), ('P-gate', 'G'),
        ('P-0.1', 'G'), ('F-local-P', 'F-local'), ('F-adapt-P', 'F-adapt'), ('F-none-P', 'F-none'), ('S-P', 'S'),
        ('F-local', 'G'), ('F-adapt', 'G'), ('F-none', 'G')]
    effects = {'full': ('P-0.1', 'G'), 'local': ('F-local-P', 'F-local'), 'adapt': ('F-adapt-P', 'F-adapt'),
               'none': ('F-none-P', 'F-none'), 'simvp': ('S-P', 'S')}
    tables, ensembles, pairs, penalty_effects, interactions = {}, {}, {}, {}, {}
    for split in SUPPLEMENTARY_SPLITS:
        data = _daily(stats, 'exp0031', split, arms)
        draw_cache = {}

        def get(arm, origins):
            ts, s, e = data[arm]
            index = np.searchsorted(ts, origins)
            if np.any(index >= len(ts)) or not np.array_equal(ts[index], origins):
                raise ValueError(f'{arm}/{split}: missing common windows')
            return s[:, index], e[:, index]

        def draws(origins):
            if origins.tobytes() not in draw_cache:
                draw_cache[origins.tobytes()] = shared_draws(origins)
            return draw_cache[origins.tobytes()]
        tables[split] = {arm: dict(statistics(s), windows=len(ts)) for arm, (ts, s, e) in data.items()}
        ensembles[split] = {arm: dict(statistics(e), windows=len(ts)) for arm, (ts, s, e) in data.items()}
        pairs[split] = {}
        for a, b in comparisons:
            origins = np.intersect1d(data[a][0], data[b][0])
            (sa, ea), (sb, eb) = get(a, origins), get(b, origins)
            pairs[split][a + '-' + b] = dict(comparison(sa, sb, ea, eb, draws(origins)), windows=len(origins))
        penalty_effects[split] = {name: pairs[split][x + '-' + y] for name, (x, y) in effects.items()}
        interactions[split] = {}
        for name in ('local', 'adapt', 'none', 'simvp'):
            group = effects[name] + effects['full']
            origins = data[group[0]][0]
            for arm in group[1:]:
                origins = np.intersect1d(origins, data[arm][0])
            interactions[split][name + '_minus_full'] = dict(_interaction(*(get(arm, origins) for arm in group), draws(origins), ref),
                                                             windows=len(origins))
    return dict(main_table=tables, ensemble_table=ensembles, paired=pairs, penalty_effects=penalty_effects, interactions=interactions), {}


def conditioning(stats):
    """Gated, FiLM and cross-attention conditioning and the graph-Transformer adaptation (strong configuration) against
    their references: the table and comparison loops of the original single scoring pass over its saved per-day
    statistics (2026: the 292 windows with complete index inputs). Functions as in penalty_variants()."""
    from tecfc.scoring.ablation_evaluate import statistics, shared_draws, comparison
    arms, references = ('C-gate', 'C-film', 'C-xattn', 'GT'), ('G-DE', 'G', 'D-noIndex')
    tables, ensembles, pairs = {}, {}, {}
    for split in SUPPLEMENTARY_SPLITS:
        data = _daily(stats, 'exp0032', split, arms + references)
        support = data[arms[0]][0]
        if not all(np.array_equal(ts, support) for ts, _, _ in data.values()):
            raise ValueError(f'{split}: systems scored on different windows')
        tables[split] = {arm: dict(statistics(s), windows=len(support)) for arm, (_, s, _) in data.items()}
        ensembles[split] = {arm: dict(statistics(e), windows=len(support)) for arm, (_, _, e) in data.items()}
        draws = shared_draws(support)
        pairs[split] = {f'{arm}-{reference}': dict(comparison(data[arm][1], data[reference][1], data[arm][2], data[reference][2], draws),
                                                   windows=len(support)) for arm in arms for reference in references}
    return dict(main_table=tables, ensemble_table=ensembles, paired=pairs), {}


# ---------------------------------------------------------------- baselines, CODE forecast, cost (group B) and earlier configuration (group P)
def _csv(path, key):
    with Path(path).open() as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def baseline_table(stats, documents):
    """Rows of the new systems are re-derived from the rebuilt documents through the table's provenance
    pointers; persistence, CODE forecast, earlier-configuration and manuscript rows are records."""
    table = _csv(RECORDS / 'baselines/BASELINE_TABLE.csv', 'row_id')
    computed = {}
    for cell in json.loads((RECORDS / 'baselines/BASELINE_PROVENANCE.json').read_text()):
        source = cell['file']
        if source not in documents:
            continue
        try:
            value = documents[source][0]
            for token in cell['pointer']:
                value = value[int(token)] if isinstance(value, list) else value[token]
        except (KeyError, IndexError, TypeError):
            continue
        computed.setdefault(cell['row'], {})[cell['column']] = value
    return computed, table


RECORD_SOURCES = {
    'experiments/exp0029/COST_TABLE.csv': lambda stats: ({}, _csv(RECORDS / 'baselines/COST_TABLE.csv', 'object')),
    'experiments/exp0029/C1PG_RELEASE_TIME.json': lambda stats: ({}, json.loads((RECORDS / 'baselines/C1PG_RELEASE_TIME.json').read_text())),
    'experiments/exp0029/C1PG_SCORES.json': lambda stats: ({}, json.loads((RECORDS / 'baselines/C1PG_SCORES.json').read_text())),
    'experiments/exp0029/PERSISTENCE_PERIODS.json': lambda stats: ({}, json.loads((RECORDS / 'baselines/PERSISTENCE_PERIODS.json').read_text())),
    '../manuscript/revisions-1009/revised/Supplementary_Data_1/global_matched/FINAL_ANALYSIS.json':
        lambda stats: ({}, json.loads((RECORDS / 'earlier_configuration/global_matched_FINAL_ANALYSIS.json').read_text())),
    '../manuscript/revisions-1009/revised/Supplementary_Data_1/extension/RESULTS.json':
        lambda stats: ({}, json.loads((RECORDS / 'earlier_configuration/extension_RESULTS.json').read_text())),
    # cost of the external Mixer-SimVP model as reported by its authors (Liu et al. 2024, Table 2)
    'experiments/exp0030/LITERATURE_COST.csv': lambda stats: ({}, _csv(RECORDS / 'baselines/LITERATURE_COST.csv', 'row_id')),
    # records of Supplementary Data 1 (records/supplementary_experiments/): the night-gate table is derived from the
    # training maps, and no per-day statistics are published for the rapid and forecast-time input maps
    'experiments/exp0031/NIGHT_GATE.json': lambda stats: ({}, stats.json('experiments/exp0031/NIGHT_GATE.json')),
    'experiments/exp0033/result.json': lambda stats: ({}, stats.json('experiments/exp0033/result.json')),
}


BUILDERS = {
    'experiments/exp0025/result.json': confirmation,
    'experiments/exp0023/result.json': regional_development,
    'experiments/exp0028/result.json': interval_calibration,
    'experiments/exp0027/result.json': h96_forecasts,
    'experiments/exp0026/result.json': ablations,
    'experiments/exp0013/result.json': penalty_members,
    'experiments/exp0015/result.json': information_members,
    'experiments/exp0016/result.json': mae_and_climatology,
    'experiments/exp0018/result.json': combined_members,
    'experiments/exp0019/result.json': ensemble_tables,
    'experiments/exp0011/result.json': zero_shot_mapping,
    'experiments/exp0030/result.json': activity_classes,
    'experiments/exp0031/result.json': penalty_variants,
    'experiments/exp0032/result.json': conditioning,
}


# ---------------------------------------------------------------- EVIDENCE.csv expressions
def pointer(document, path):
    value = document
    for token in path.lstrip('/').split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        m = re.fullmatch(r'([^\[\]]+)\[(\d+)\]\.(.+)', token)  # 'list[i].key' (night-gate record)
        if m and not (isinstance(value, dict) and token in value):
            value = value[m.group(1)][int(m.group(2))][m.group(3)]
            continue
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def evaluate(expression, document):
    """Value of a json_path expression of EVIDENCE.csv in a regenerated document."""
    expression = expression.strip()
    m = re.fullmatch(r'csv:([^=]+)=([^/]+)/(\S+)(?: / (\d+))?', expression)
    if m:
        key, row, column, divisor = m.groups()
        value = document[row][column]
        if column == 'recomputed':  # 'yes: ...' / 'no: ...' -> 1 / 0, as in EVIDENCE.csv
            return 1 if str(value).lower().startswith('yes') else 0
        value = float(value) if not isinstance(value, (int, float)) else value
        return value / float(divisor) if divisor else value
    m = re.fullmatch(r'100 x \((/.+) - (/.+)\) / (/.+)', expression)
    if m:
        a, b, c = (pointer(document, x) for x in m.groups())
        return 100 * (a - b) / c
    if expression.startswith('100 x '):
        a, b = expression[len('100 x '):].split(' / ')
        return 100 * pointer(document, a) / pointer(document, b)
    return pointer(document, expression)


def tolerance(row):
    unit, value = row['unit'], float(row['value'])
    return 5e-4 if 'three decimals' in row.get('note', '') else 1e-6


def compare(rows, documents):
    report = []
    for row in rows:
        source = row['source_file']
        entry = dict(id=row['id'], group=row['group'], source=source)
        if source not in documents:
            entry['status'] = 'record'; report.append(entry); continue
        expression, _, ci = row['json_path'].partition('; ci: ')
        computed, records = documents[source]
        status = 'match'
        try:
            value = evaluate(expression, computed)
            interval = pointer(computed, ci) if ci else None
        except (KeyError, IndexError, TypeError) as error:
            try:
                value = evaluate(expression, records)
                interval = pointer(records, ci) if ci else None
                status = 'record'
            except (KeyError, IndexError, TypeError):
                entry.update(status='missing', error=repr(error)); report.append(entry); continue
        expected = float(row['value'])
        if isinstance(value, list) and row['unit'] == 'count':
            value = len(value)  # e.g. number of excluded dates
        if isinstance(value, dict) and row['quantity'] == 'preregistered_comparisons':
            value = len(value)
        if isinstance(value, dict) and row['quantity'] == 'improved_preregistered_comparisons':
            value = sum(v == '改善' for v in value.values())
        if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
            entry.update(status='mismatch', recomputed=repr(value)[:80], expected=expected); report.append(entry); continue

        def difference(scale):
            d = abs(scale * float(value) - expected)
            if interval is not None and row['ci_low'] != '':
                d = max(d, abs(scale * interval[0] - float(row['ci_low'])), abs(scale * interval[1] - float(row['ci_high'])))
            return d
        scale = 1.
        diff = difference(1.)
        # EVIDENCE.csv states fractions of the source files as percentages or percentage points.
        if diff > tolerance(row) and row['unit'] in ('%', 'pp') and difference(100.) <= tolerance(row):
            scale, diff = 100., difference(100.)
        entry.update(recomputed=scale * float(value), expected=expected, abs_difference=diff,
                     status=status if diff <= tolerance(row) else 'mismatch')
        report.append(entry)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sd1'); source.add_argument('--experiment-repo')
    parser.add_argument('--evidence', default=str(RECORDS / 'EVIDENCE.csv'))
    parser.add_argument('--only', nargs='*', help='source files to rebuild (default: all)')
    parser.add_argument('--report', default='reproduce_report.json')
    args = parser.parse_args(argv)
    stats = Statistics(args.experiment_repo, args.sd1)
    rows = [r for r in csv.DictReader(open(args.evidence)) if r['group'] in GROUPS]
    documents = {}
    for source, builder in BUILDERS.items():
        if args.only and source not in args.only:
            continue
        documents[source] = builder(stats)
        print(json.dumps(dict(rebuilt=source)), flush=True)
    if not args.only or 'experiments/exp0029/BASELINE_TABLE.csv' in args.only:
        documents['experiments/exp0029/BASELINE_TABLE.csv'] = baseline_table(stats, documents)
    for source, loader in RECORD_SOURCES.items():
        if not args.only or source in args.only:
            documents[source] = loader(stats)
    if args.only:
        rows = [r for r in rows if r['source_file'] in documents]
    report = compare(rows, documents)
    summary = {}
    for entry in report:
        g = summary.setdefault(entry['group'], dict(match=0, mismatch=0, missing=0, record=0))
        g[entry['status']] += 1
    Path(args.report).write_text(json.dumps(dict(summary=summary, rows=report), indent=1) + '\n')
    for group in sorted(summary):
        print(group, json.dumps(summary[group]))
    bad = [e for e in report if e['status'] in ('mismatch', 'missing')]
    for e in bad[:20]:
        print('NOT REPRODUCED', json.dumps(e))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
