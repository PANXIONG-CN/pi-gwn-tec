"""exp0028: empirical prediction intervals calibrated on frozen forecasts (CPU only).

Stages (run from anywhere):  selftest | checks | compute
  selftest  synthetic check C only, nothing is written
  checks    A (RMSE reproduction), B (causality), C (synthetic exchangeable residuals) -> CHECKS.json
  compute   requires CHECKS.json passed; coverage, width, block bootstrap -> result.json, RESULT_CN.md
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tecfc.intervals import spec as S
from tecfc.intervals import calibration as K
from tecfc.intervals import data as D

REPO = D.REPO
OUT = REPO / 'experiments/exp0028'
ART = OUT / 'artifacts'
NAMES = {'global_S2': '全球 S2', 'china_final': '中国定稿', 'china_anchor': '中国锚(映射S2)',
         'japan_final': '日本定稿', 'japan_anchor': '日本锚(映射S2)'}


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, allow_nan=False) + '\n')


def file_sha(path):
    return D.sha(path)


# ----------------------------------------------------------------------------------------------- check C
def synthetic_check():
    unit = {}
    unit['level_numerator_n99_alpha010'] = S.level_numerator(99, S.ALPHAS['0.10']) == 90
    unit['level_numerator_n1826'] = (S.level_numerator(1826, S.ALPHAS['0.10']), S.level_numerator(1826, S.ALPHAS['0.20'])) == (1645, 1462)
    unit['rank_saturates_at_one'] = S.rank(5, 1000, S.ALPHAS['0.10']) == 1000 and S.level(5, S.ALPHAS['0.10']) == 1.0
    rng = np.random.default_rng(S.SEED)
    ok = True
    for n_days, n_pool in [(50, 777), (365, 10000), (1826, 4001)]:
        v = rng.random(n_pool)
        q, _ = K.pooled_quantiles(v.copy(), n_days)
        srt = np.sort(v)
        for k, a in S.ALPHAS.items():
            ok &= q[k] == srt[S.rank(n_days, n_pool, a) - 1]
    unit['partition_equals_sorted_order_statistic'] = bool(ok)
    ts = np.arange(10) * S.DAY
    a, b = K.window(ts, 9 * S.DAY)
    unit['window_is_D-366_to_D-2_inclusive'] = (ts[a], ts[b - 1]) == (0, 7 * S.DAY)

    H, G, C = S.HORIZONS, 1500, 100
    t0 = int(np.datetime64('2019-01-01T00:00:00', 's').astype(np.int64))
    ts = t0 + np.arange(G, dtype=np.int64) * S.DAY
    sigma = 1.0 + 0.25 * np.arange(H)
    rng = np.random.default_rng(S.SEED)
    scores = np.abs(rng.standard_t(5, size=(H, G, C)) * sigma[:, None, None])
    scores[rng.random((H, G, C)) < 0.05] = np.nan
    yhat = rng.normal(4.0, 3.0, size=(G, H, C))

    def extra(g):
        return yhat[g], np.isfinite(scores[:, g, :]) & (yhat[g] > 6.0)

    cal_idx = np.arange(365)
    eval_idx = np.arange(365, G)
    res = K.run_intervals(scores, ts, cal_idx, eval_idx, extra)
    dates = ts[eval_idx].astype('datetime64[s]').astype('datetime64[D]')
    rows = {}
    passed = all(unit.values())
    for m in 'ST':
        mask = res['included'][m]
        for k in S.ALPHAS:
            row = K.metrics(res['tabs'][m][k], res['nv'], res['ns'], mask)
            by_h = [K.metrics(res['tabs'][m][k], res['nv'], res['ns'], mask, slice(l, l + 1))['coverage'] for l in range(H)]
            num, den = K.day_totals(res['tabs'][m][k], res['nv'])
            ci = K.interval95(K.replicates(K.strata_for(dates, mask), num, den))
            diff = row['coverage'] - S.NOMINAL[k]
            good = abs(diff) <= S.CHECK_C_TOLERANCE
            passed &= bool(good)
            rows['%s/%s' % (m, k)] = dict(n_eval_days=int(mask.sum()), coverage=row['coverage'], nominal=S.NOMINAL[k],
                                         difference=diff, within_tolerance=bool(good), coverage_ci95=ci,
                                         expected_level=S.level(len(cal_idx), S.ALPHAS[k]),
                                         horizon_coverage_min=min(by_h), horizon_coverage_max=max(by_h))
    return dict(status='passed' if passed else 'failed', tolerance_absolute=S.CHECK_C_TOLERANCE,
                design=dict(days=G, cells=C, horizons=H, calibration_days=len(cal_idx), evaluation_days=len(eval_idx),
                            residual='sigma_l * Student-t(5), iid over days and cells, 5%% invalid, seed %d' % S.SEED,
                            same_code_as_real_data=True),
                unit_tests=unit, coverage=rows)


# ----------------------------------------------------------------------------------------------- check A
def check_A():
    rows, info = [], []
    for region, parts in D.part_descriptors().items():
        for name, desc in parts:
            table, _ = D.cscore.evaluate(D._abs(desc))      # the exp0025 scorer itself
            for method, row in table.items():
                system = D.system_key(region, method)
                expected = S.EXPECTED_RMSE.get(system, {}).get(name)
                rec = dict(system=system, part=name, n_valid=row['n_valid'], RMSE=row['RMSE'], expected=expected)
                if expected is None:
                    rec.update(abs_diff=None, passed=None, note='no preregistered expected value; observed only')
                    info.append(rec)
                else:
                    rec.update(abs_diff=abs(row['RMSE'] - expected), passed=bool(abs(row['RMSE'] - expected) < S.CHECK_A_TOLERANCE))
                    rows.append(rec)
                print('A', json.dumps(clean(rec), ensure_ascii=False), flush=True)
    return dict(status='passed' if all(r['passed'] for r in rows) else 'failed',
                tolerance_absolute=S.CHECK_A_TOLERANCE, source='confirm2026.score.evaluate on frozen predictions',
                rows=rows, observed_only=info)


# ----------------------------------------------------------------------------------------------- check B
def check_B():
    out = {}
    passed = True
    for region in ('global', 'china', 'japan'):
        loaded = D.load_region(region)
        for method in (['S2'] if region == 'global' else ['final', 'anchor']):
            system = D.system_key(region, method)
            series = D.Series(loaded, method)
            cal_idx, eval_idx = D.index_sets(series, system)
            dummy = np.zeros((S.HORIZONS, len(series.ts), 1))
            roll = K.rolling_quantiles(dummy, series.ts, eval_idx)      # asserts causality for every target day
            ok = roll['ok']
            t = series.ts[eval_idx]
            margin = (t[ok] - S.DAY - roll['last_target_time'][ok]) / 3600.0
            years = series.year[eval_idx]
            out[system] = dict(
                targets=int(len(eval_idx)), retained=int(ok.sum()), excluded=int((~ok).sum()),
                excluded_by_year={str(y): int(((~ok) & (years == y)).sum()) for y in sorted(set(years.tolist())) if ((~ok) & (years == y)).any()},
                excluded_first_last=([str(series.dates[eval_idx][~ok][0]), str(series.dates[eval_idx][~ok][-1])] if (~ok).any() else None),
                latest_calibration_target_time_never_after_D_minus_1_00UTC=bool(np.all(roll['last_target_time'][ok] <= t[ok] - S.DAY)),
                margin_hours_min=float(margin.min()) if ok.any() else None, margin_hours_max=float(margin.max()) if ok.any() else None,
                first_calibration_day=str(series.dates[0]), static_calibration_days=int(len(cal_idx)),
                earliest_year_in_series=int(series.year.min()))
            passed &= out[system]['latest_calibration_target_time_never_after_D_minus_1_00UTC']
            print('B', system, json.dumps(clean(out[system]), ensure_ascii=False), flush=True)
    ts = np.arange(800) * S.DAY
    g = 500
    a = int(np.searchsorted(ts, ts[g] - S.ROLL_FIRST * S.DAY))
    b_bad = int(np.searchsorted(ts, ts[g] - 1 * S.DAY, side='right'))
    detected = bool(ts[b_bad - 1] + S.LAST_LEAD_OFFSET > ts[g] - S.DAY)
    passed &= detected
    return dict(status='passed' if passed else 'failed', per_system=out,
                negative_control_window_including_D_minus_1_is_detected=detected,
                assertion='inside core28.rolling_quantiles, executed for every retained target day on the real timestamps')


def stage_checks():
    OUT.mkdir(parents=True, exist_ok=True)
    A = check_A()
    B = check_B() if A['status'] == 'passed' else dict(status='not_run')
    C = synthetic_check()
    status = 'passed' if A['status'] == B['status'] == C['status'] == 'passed' else 'failed'
    write_json(OUT / 'CHECKS.json', dict(status=status, at_utc=now(), A=A, B=B, C=C,
                                         code_sha256={f: file_sha(HERE / f) for f in ('spec.py', 'calibration.py', 'data.py', 'run.py')}))
    print('checks:', status, '| A', A['status'], '| B', B['status'], '| C', C['status'])
    if status != 'passed':
        raise SystemExit('STOP: a check failed; see experiments/exp0028/CHECKS.json')


# ----------------------------------------------------------------------------------------------- compute
def summarize(system, series, cal_idx, eval_idx, res, scores):
    cfg = S.SYSTEMS[system]
    years = series.year[eval_idx]
    dates = series.dates[eval_idx]
    nv, ns = res['nv'], res['ns']
    scope = {'all': np.ones(len(eval_idx), bool),
             'P1': np.isin(years, cfg['P1']), 'P2': np.isin(years, cfg['P2'])}
    for y in cfg['eval_years']:
        scope['Y%d' % y] = years == y
    inc = res['included']
    keep = {m: {} for m in 'ST'}
    for m in 'ST':
        for name, mask in scope.items():
            keep[m][name] = mask & inc[m]
    frames = {}

    def frame(m, name):
        key = (m, name) if m == 'T' else ('S', name)
        if key not in frames:
            frames[key] = K.strata_for(dates, keep[m][name])
        return frames[key]

    out = dict(label=NAMES[system], cfg=dict(method=cfg['method'], static_calibration=cfg['static'], eval_years=cfg['eval_years'], P1=cfg['P1'], P2=cfg['P2']))
    # calibration records
    qS = res['S']
    out['static'] = dict(n_calibration_days=res['S']['n_days'],
                         calibration_years=sorted(set(series.year[cal_idx].tolist())),
                         per_alpha={k: dict(m=S.level_numerator(qS['n_days'], a), level_p=S.level(qS['n_days'], a),
                                            rank_by_horizon=[S.rank(qS['n_days'], int(n), a) for n in qS['pool']],
                                            q_by_horizon=qS['q'][k], n_pool_by_horizon=qS['pool'])
                                     for k, a in S.ALPHAS.items()})
    T = res['T']
    ok = T['ok']
    out['rolling'] = dict(window='D-366..D-2', required_days=S.ROLL_MIN, targets=int(len(ok)), retained=int(ok.sum()),
                          excluded=int((~ok).sum()),
                          excluded_by_year={str(y): int(((~ok) & (years == y)).sum()) for y in cfg['eval_years']},
                          retained_by_year={str(y): int((ok & (years == y)).sum()) for y in cfg['eval_years']},
                          excluded_dates=[str(d) for d in dates[~ok]] if (~ok).sum() <= 40 else
                          [str(dates[~ok][0]) + ' ... ' + str(dates[~ok][-1]) + ' (%d days)' % (~ok).sum()],
                          calibration_days_min=int(T['n_cal'][ok].min()) if ok.any() else None,
                          calibration_days_max=int(T['n_cal'].max()),
                          min_margin_hours_before_D_minus_1_00UTC=float(((series.ts[eval_idx][ok] - S.DAY - T['last_target_time'][ok]) / 3600.0).min()) if ok.any() else None,
                          mean_q_by_horizon={k: (np.nanmean(T['q'][k], axis=0) if ok.any() else None) for k in S.ALPHAS})
    out['eval_days'] = dict(S={str(y): int((years == y).sum()) for y in cfg['eval_years']},
                            T={str(y): int((ok & (years == y)).sum()) for y in cfg['eval_years']})
    out['metrics'] = {}
    out['T_minus_S'] = {}
    tot = {}
    for k in S.ALPHAS:
        tot[k] = {m: K.day_totals(res['tabs'][m][k], nv) for m in 'ST'}
        out['metrics'][k] = {}
        for m in 'ST':
            tab = res['tabs'][m][k]
            rows = {}
            for name in scope:
                mask = keep[m][name]
                row = K.metrics(tab, nv, ns, mask)
                if row is None:
                    rows[name] = None
                    continue
                num, den = tot[k][m]
                row['coverage_ci95'] = K.interval95(K.replicates(frame(m, name), num, den))
                row['wording'] = S.wording(row['coverage'], k)
                row['n_days'] = int(mask.sum())
                rows[name] = row
            hz = {}
            for l in range(S.HORIZONS):
                r = K.metrics(tab, nv, ns, keep[m]['all'], slice(l, l + 1))
                hz[str(S.LEADS[l])] = None if r is None else dict(coverage=r['coverage'], mean_width=r['mean_width'], n_valid=r['n_valid'],
                                                                   wording=S.wording(r['coverage'], k))
            out['metrics'][k][m] = dict(scopes=rows, by_lead_hours=hz)
        diffs = {}
        for name in scope:
            mask = keep['T'][name]
            if not mask.any():
                diffs[name] = None
                continue
            rt = K.metrics(res['tabs']['T'][k], nv, ns, mask)
            rs = K.metrics(res['tabs']['S'][k], nv, ns, mask)
            fr = frame('T', name)
            repT = K.replicates(fr, *tot[k]['T'])
            repS = K.replicates(fr, *tot[k]['S'])
            diffs[name] = dict(n_days_matched=int(mask.sum()), coverage_T=rt['coverage'], coverage_S_matched=rs['coverage'],
                               coverage_difference=rt['coverage'] - rs['coverage'], coverage_difference_ci95=K.interval95(repT - repS),
                               width_T=rt['mean_width'], width_S_matched=rs['mean_width'], width_difference=rt['mean_width'] - rs['mean_width'])
        out['T_minus_S'][k] = diffs
    # integrity: RMSE implied by the stored scores equals the scorer's (cross-check of the residual source)
    cross = {}
    for pname in ('ext', 'y2026'):
        idx = np.where(series.part_of == series.names.index(pname))[0]
        s = scores[:, idx, :]
        fin = np.isfinite(s)
        cross[pname] = float(np.sqrt(np.square(s[fin]).sum() / fin.sum()))
    out['rmse_from_scores'] = cross
    return out


def stage_compute():
    chk = json.loads((OUT / 'CHECKS.json').read_text())
    assert chk['status'] == 'passed', 'CHECKS.json has not passed'
    ART.mkdir(parents=True, exist_ok=True)
    result = dict(experiment='exp0028', status='completed_descriptive_calibration', at_utc=now(), systems={})
    for region in ('global', 'china', 'japan'):
        loaded = D.load_region(region)
        for method in (['S2'] if region == 'global' else ['final', 'anchor']):
            system = D.system_key(region, method)
            series = D.Series(loaded, method)
            cal_idx, eval_idx = D.index_sets(series, system)
            scores = D.build_scores(series)
            print('compute', system, 'days', len(series.ts), 'cells', scores.shape[2], 'calibration', len(cal_idx), 'evaluation', len(eval_idx), flush=True)
            res = K.run_intervals(scores, series.ts, cal_idx, eval_idx, D.extra_fn(series))
            result['systems'][system] = summarize(system, series, cal_idx, eval_idx, res, scores)
            np.savez_compressed(ART / ('day_stats_%s.npz' % system), origins=series.ts[eval_idx], nv=res['nv'], ns=res['ns'],
                                T_ok=res['T']['ok'], **{'%s_%s_%s' % (m, k, f): res['tabs'][m][k][f]
                                                         for m in 'ST' for k in S.ALPHAS for f in ('q', 'cov', 'sub_cov', 'neg')})
            del scores, res
    result['spec'] = dict(alphas={k: list(v) for k, v in S.ALPHAS.items()}, replicates=S.REPLICATES, block_days=S.BLOCK_DAYS,
                          seed=S.SEED, high_tec=S.HIGH_TEC, rolling=dict(first=S.ROLL_FIRST, last=S.ROLL_LAST, minimum=S.ROLL_MIN))
    result['code_sha256'] = {f: file_sha(HERE / f) for f in ('spec.py', 'calibration.py', 'data.py', 'run.py')}
    result['preregistration_sha256'] = file_sha(OUT / 'PREREGISTRATION_CN.txt')
    result['checks_sha256'] = file_sha(OUT / 'CHECKS.json')
    result['caveats'] = ['区间是经验校准的结果，不声称有限样本覆盖保证。',
                         '2023–2025 已参与开发；2026 的数字是未预注册的描述性结果。',
                         'JPLD 与 IGS GIM 都是制图产品，不称独立真值。',
                         '“经验子组覆盖率”按真值 ≥60 TECU 取子样本，是对结果变量的选择，不是条件覆盖保证。']
    write_json(OUT / 'result.json', result)
    (OUT / 'RESULT_CN.md').write_text(render(result))
    print('compute: done')


# ----------------------------------------------------------------------------------------------- report
def pct(x, d=2):
    return '—' if x is None else ('%.*f%%' % (d, 100 * x))


def cell(row, ci=True):
    if row is None:
        return '—'
    s = pct(row['coverage'])
    if ci:
        s += ' [%s, %s]' % (pct(row['coverage_ci95'][0], 1), pct(row['coverage_ci95'][1], 1))
    return s


def render(r):
    sysd = r['systems']
    L = ['# exp0028 预测区间的经验校准（CPU，冻结预测，不改点预测）', '',
         '状态：已完成。区间 [ŷ−q, ŷ+q]，q 为各时效绝对残差的经验分位数；办法 S 静态校准，办法 T 滚动校准（D−366 至 D−2，须满 365 个预报日）。'
         '名义水平 90%（主）与 80%（辅）。2023–2025 已参与开发；**2026 为未预注册的描述性结果**。覆盖率是经验值，不含有限样本保证。', '']
    chk = json.loads((OUT / 'CHECKS.json').read_text())
    L += ['检查：A（残差来源复算 RMSE，差 <1e-6）%s；B（因果性逐日断言）%s；C（合成可交换残差，±1 个百分点）%s。' %
          tuple('通过' if chk[x]['status'] == 'passed' else '未通过' for x in 'ABC'), '']
    for k, title in (('0.10', '主表：名义 90%（用语界限 88%–92%）'), ('0.20', '辅表：名义 80%（用语界限 78%–82%）')):
        L += ['## ' + title, '', '| 系统 | 评价期 | S 覆盖率 [95%块区间] | S 宽度 | T 覆盖率 [95%块区间] | T 宽度 | S 用语 / T 用语 |', '|---|---|---|---|---|---|---|']
        for sysname, d in sysd.items():
            for sc, lab in (('P1', '、'.join(str(y) for y in d['cfg']['P1']).replace('、2024、2025', '–2025').replace('2024、2025', '2024–2025')), ('P2', '2026')):
                S_, T_ = d['metrics'][k]['S']['scopes'][sc], d['metrics'][k]['T']['scopes'][sc]
                L.append('| %s | %s | %s | %s | %s | %s | %s / %s |' % (
                    d['label'], lab, cell(S_), '—' if S_ is None else '%.3f' % S_['mean_width'], cell(T_),
                    '—' if T_ is None else '%.3f' % T_['mean_width'], '—' if S_ is None else S_['wording'], '—' if T_ is None else T_['wording']))
        L.append('')
    L += ['宽度单位 TECU（2q 的有效格点时刻平均）。全期、逐年、12 时效与下界为负比例见 result.json。', '',
          '## 办法 T − 办法 S（在 T 保留的共同目标日上）', '',
          '| 系统 | 评价期 | 90% 覆盖率差 [95%块区间] | 90% 宽度差 | 80% 覆盖率差 | 80% 宽度差 | 共同日数 |', '|---|---|---|---|---|---|---|']
    for sysname, d in sysd.items():
        for sc in ('P1', 'P2'):
            a, b = d['T_minus_S']['0.10'][sc], d['T_minus_S']['0.20'][sc]
            if a is None:
                L.append('| %s | %s | — | — | — | — | 0（全部剔除） |' % (d['label'], sc))
                continue
            L.append('| %s | %s | %+.2f pp [%+.1f, %+.1f] | %+.3f | %+.2f pp | %+.3f | %d |' % (
                d['label'], sc, 100 * a['coverage_difference'], 100 * a['coverage_difference_ci95'][0], 100 * a['coverage_difference_ci95'][1],
                a['width_difference'], 100 * b['coverage_difference'], b['width_difference'], a['n_days_matched']))
    L += ['', '## 办法 T 剔除的目标日（不足 365 个有预测的预报日）', '', '| 系统 | 评价日 | 剔除 | 按年剔除 |', '|---|---|---|---|']
    for sysname, d in sysd.items():
        rl = d['rolling']
        L.append('| %s | %d | %d | %s |' % (d['label'], rl['targets'], rl['excluded'], '；'.join('%s：%d' % (y, n) for y, n in rl['excluded_by_year'].items())))
    L += ['', '## 经验子组覆盖率（真值 ≥60 TECU 的格点时刻，名义 90%）', '', '| 系统 | 评价期 | 样本数 | S | T |', '|---|---|---|---|---|']
    for sysname, d in sysd.items():
        for sc in ('P1', 'P2'):
            S_, T_ = d['metrics']['0.10']['S']['scopes'][sc], d['metrics']['0.10']['T']['scopes'][sc]
            L.append('| %s | %s | %s | %s | %s |' % (d['label'], sc, '—' if S_ is None else '{:,}'.format(S_['n_subgroup']),
                                                       '—' if S_ is None or S_['subgroup_coverage'] is None else pct(S_['subgroup_coverage']),
                                                       '—' if T_ is None or T_['subgroup_coverage'] is None else pct(T_['subgroup_coverage'])))
    L += ['', '## 说明', '- ' + '\n- '.join(r['caveats']), '- P1 = 全球 2023–2025、区域 2024–2025；P2 = 2026（全球 1–10 月，区域 1–6 月）。区域 2023 年仅用于校准。',
          '- 本实验不训练、不推理，未改动任何点预测；未做后续实验。', '']
    return '\n'.join(L)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        print(json.dumps(clean(synthetic_check()), ensure_ascii=False, indent=1))
    elif stage == 'checks':
        stage_checks()
    elif stage == 'compute':
        stage_compute()
    else:
        raise SystemExit(__doc__)
