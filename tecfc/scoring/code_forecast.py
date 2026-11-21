"""Score C1PG against the same IGS final targets and compare with frozen S2, G5 and 24-h persistence.

Common dates = origin days where the C1PG day is valid (audited) and frozen S2/G5 predictions exist.
Intervals: 7-day blocks, years stratified, 5000 draws, RNG 20260924, shared by all methods
(final_score.samples). Frozen predictions are only read, never regenerated.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tecfc.scoring import reference_common as e29  # noqa: E402

SEEDS = e29.SEEDS


def committed_json(rel):
    p = e29.ROOT / rel
    assert e29.sha(p) == e29.CFG['frozen_record_sha256'][rel], 'frozen record changed: ' + rel
    return json.loads(p.read_text())


def frozen_specs():
    fused = committed_json('experiments/exp0019/FUSED_PREDICTIONS.json')
    old = committed_json('experiments/exp0019/FROZEN_PREDICTIONS.json')
    f26 = committed_json('experiments/exp0025/FROZEN_PREDICTIONS_2026.json')
    out = {}
    for split in ['original_evaluation', 'extension_evaluation']:
        sp = fused['systems']['S2']['splits'][split]
        origins = dict(path=sp['origins'], sha256=fused['files_sha256'][sp['origins']])
        s2 = dict(arrays=[dict(path=sp['predictions'], sha256=fused['files_sha256'][sp['predictions']])], source_origins=origins)
        arrays = []
        for r in old['arms']['G']:
            q = r['splits'][split]
            o = np.load(q['origins']); assert np.array_equal(o, np.load(origins['path']))
            arrays.append(dict(path=q['predictions'], sha256=old['files_sha256'][q['predictions']]))
        out[split] = dict(S2=s2, G5=dict(arrays=arrays, source_origins=origins))
    g = f26['global_']
    out['descriptive_2026'] = dict(S2=dict(arrays=[g['methods']['S2']], source_origins=g['origins']),
                                   G5=dict(arrays=[g['methods']['G5']], source_origins=g['origins']))
    return out


def stats_row(stats):
    s = e29.rmse_of(stats)
    leads = {h: e29.rmse_of(stats[:, k])['RMSE'] for k, h in enumerate(e29.LEADS)}
    return dict(windows=int(stats.shape[0]), n_valid=s['n_valid'], RMSE=s['RMSE'], MAE=s['MAE'], bias=s['bias'], lead_RMSE=leads)


def period_block(tag, dates, c1pg_years, frozen, split, extra_masks=None):
    methods = dict(S2=frozen['S2'], G5=frozen['G5'], persistence=e29.persistence_method(), C1PG=dict(kind='global_same_day', years=c1pg_years))
    desc = e29.make_desc(tag, dates, methods)
    ts, names, st = e29.stats_for(desc, progress=500)
    return ts, names, st


def summarize_group(ts, names, st, label, subsets):
    """subsets: {name: boolean mask over windows}. Returns dict with rows and intervals per subset."""
    out = {}
    for sname, mask in subsets.items():
        sts = ts[mask]; s = st[:, mask]
        rows = {n: stats_row(s[j]) for j, n in enumerate(names)}
        daily = s.sum(2)
        draws, digest = e29.block_samples(daily, sts)
        ci = lambda v: np.quantile(v, [.025, .975]).tolist()  # noqa: E731
        for j, n in enumerate(names):
            rows[n]['RMSE_ci95_time_blocks'] = ci(draws[:, j])
        pairs = {}
        for a, b in [('S2', 'C1PG'), ('G5', 'C1PG'), ('persistence', 'C1PG'), ('S2', 'G5')]:
            i, j = names.index(a), names.index(b)
            pairs[f'{a}-{b}'] = dict(delta_RMSE=rows[a]['RMSE'] - rows[b]['RMSE'], ci95=ci(draws[:, i] - draws[:, j]),
                                     S2_better_in_all_draws=bool(np.all(draws[:, i] < draws[:, j])) if a == 'S2' and b == 'C1PG' else None)
        years = sts.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
        out[sname] = dict(label=label, windows=int(mask.sum()), date_first=str(sts.astype('datetime64[s]').astype('datetime64[D]')[0]),
                          date_last=str(sts.astype('datetime64[s]').astype('datetime64[D]')[-1]), years=sorted(int(y) for y in set(years)),
                          methods=rows, pairs=pairs,
                          intervals=dict(block_days=7, replicates=5000, rng=20260924, year_stratified=len(set(years)) > 1, ci='percentile 2.5/97.5',
                                         common_draw_sha256=digest, scope='time variation conditional on frozen models/products'))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--converted', type=Path, required=True)
    a = p.parse_args()
    frozen = frozen_specs()
    result = dict(experiment='exp0029', stage='c1pg_scores', at_utc=datetime.now(timezone.utc).isoformat(), periods={},
                  labels=dict(descriptive_2026='未预注册的描述性结果；与 exp0025 的全球窗口相同', all='C1PG 是 CODE 运行产品，生成条件与本研究不同，仅作参考对照'))
    # old evaluation: old arrays (as used in the manuscript, including the 4 carried-forward days)
    old = e29.CFG['old_c1pg']['arrays']
    old_years = {y: dict(path=r['path'], sha256=r['sha256']) for y, r in old.items()}
    dates = e29.period_dates('old_evaluation')
    ts, names, st = period_block('old_evaluation', dates, old_years, frozen['original_evaluation'], 'original_evaluation')
    d = dates
    imputed = np.isin(d, np.arange(np.datetime64('2022-12-28'), np.datetime64('2023-01-01'), dtype='datetime64[D]'))
    result['periods']['old_evaluation'] = summarize_group(ts, names, st, 'old evaluation 2015, 2019-2022', {
        'all_1826_as_in_manuscript': np.ones(len(d), bool), 'excluding_4_carried_forward_days': ~imputed})
    # new products
    valid = set(json.loads((a.converted / 'C1PG_VALID_DAYS.json').read_text()))
    new_years = {str(y): dict(path=str(a.converted / f'C1PG_{y}.npy'), sha256=e29.sha(a.converted / f'C1PG_{y}.npy')) for y in (2023, 2024, 2025, 2026)}
    result['converted_array_sha256'] = {k: v['sha256'] for k, v in new_years.items()}
    for tag, split, per in [('extension_2023_2025', 'extension_evaluation', 'extension_2023_2025'), ('descriptive_2026', 'descriptive_2026', 'descriptive_2026')]:
        dates_all = e29.period_dates(per)
        dates = np.array([x for x in dates_all if str(x) in valid], dtype='datetime64[D]')
        ts, names, st = period_block(tag + '_common', dates, new_years, frozen[split], split)
        yrs = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
        subsets = {'pooled': np.ones(len(ts), bool)}
        if tag == 'extension_2023_2025':
            subsets.update({str(y): yrs == y for y in (2023, 2024, 2025)})
        result['periods'][tag] = summarize_group(ts, names, st, tag, subsets)
        result['periods'][tag]['_common_dates'] = dict(period_days=int(len(dates_all)), c1pg_valid_days=int(len(dates)),
                                                      missing_or_rejected=[str(x) for x in dates_all if str(x) not in valid])
    # consistency of the recomputed frozen predictions with the frozen results
    ref = {}
    r19 = committed_json('experiments/exp0019/result.json'); r25 = committed_json('experiments/exp0025/result.json')
    ref['old_S2'] = r19['main_table']['original_evaluation']['S2']['all']['RMSE']; ref['old_G5'] = r19['main_table']['original_evaluation']['G5']['all']['RMSE']
    ref['ext_S2'] = r19['main_table']['extension_evaluation']['S2']['all']['RMSE']; ref['ext_G5'] = r19['main_table']['extension_evaluation']['G5']['all']['RMSE']
    ref['2026_S2'] = r25['datasets']['global_2026']['full']['methods']['S2']['all']['RMSE']; ref['2026_G5'] = r25['datasets']['global_2026']['full']['methods']['G5']['all']['RMSE']
    P = result['periods']
    cons = {}
    for key, got, windows_full, windows_got in [
            ('old_S2', P['old_evaluation']['all_1826_as_in_manuscript']['methods']['S2']['RMSE'], 1826, P['old_evaluation']['all_1826_as_in_manuscript']['windows']),
            ('old_G5', P['old_evaluation']['all_1826_as_in_manuscript']['methods']['G5']['RMSE'], 1826, P['old_evaluation']['all_1826_as_in_manuscript']['windows']),
            ('ext_S2', P['extension_2023_2025']['pooled']['methods']['S2']['RMSE'], 1096, P['extension_2023_2025']['pooled']['windows']),
            ('ext_G5', P['extension_2023_2025']['pooled']['methods']['G5']['RMSE'], 1096, P['extension_2023_2025']['pooled']['windows']),
            ('2026_S2', P['descriptive_2026']['pooled']['methods']['S2']['RMSE'], 304, P['descriptive_2026']['pooled']['windows']),
            ('2026_G5', P['descriptive_2026']['pooled']['methods']['G5']['RMSE'], 304, P['descriptive_2026']['pooled']['windows'])]:
        cons[key] = dict(frozen=ref[key], here=got, abs_diff=abs(ref[key] - got), full_window_set=bool(windows_got == windows_full), windows=windows_got)
    result['consistency_with_frozen_records'] = cons
    path = e29.EXP / 'C1PG_SCORES.json'
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('written', path)


if __name__ == '__main__':
    main()
