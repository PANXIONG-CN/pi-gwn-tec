"""exp0028 interval machinery. Arrays only: scores[H, G, C] (nan = invalid case) over the series of forecast days.

s = |y - yhat| per valid grid point and target time. Method S: one static calibration set.
Method T: rolling window D-366 .. D-2 with at least 365 forecast days. Quantile rule in spec28.
"""
import numpy as np
from tecfc.intervals import spec as S


def window(ts, t):
    """Index range [a, b) of forecast days d with D-366 <= d <= D-2 (ts ascending, Unix seconds at 00 UTC)."""
    a = int(np.searchsorted(ts, t - S.ROLL_FIRST * S.DAY, side='left'))
    b = int(np.searchsorted(ts, t - S.ROLL_LAST * S.DAY, side='right'))
    return a, b


def pooled_quantiles(values, n_days):
    """Upper-sample-value quantiles of the pooled valid scores at both alphas (rank rule in spec28)."""
    n_pool = int(values.size)
    assert n_pool > 0
    ranks = {k: S.rank(n_days, n_pool, a) for k, a in S.ALPHAS.items()}
    assert all(1 <= r <= n_pool for r in ranks.values())
    part = np.partition(values, sorted({r - 1 for r in ranks.values()}))
    return {k: float(part[r - 1]) for k, r in ranks.items()}, n_pool


def static_quantiles(scores, cal_idx):
    n_days = len(cal_idx)
    H = scores.shape[0]
    q = {k: np.empty(H) for k in S.ALPHAS}
    pool = np.empty(H, np.int64)
    for l in range(H):
        block = scores[l][cal_idx]
        r, pool[l] = pooled_quantiles(block[np.isfinite(block)], n_days)
        for k in q:
            q[k][l] = r[k]
    return q, pool, n_days


def rolling_quantiles(scores, ts, targets):
    """Per target index g: calibration = forecast days in [D-366, D-2]; excluded unless 365 days available.

    The causality assertion runs inside the loop: the latest target time of the calibration residuals
    (last calibration forecast day 00 UTC + 22 h) must not be later than D-1 00 UTC.
    """
    H = scores.shape[0]
    T = len(targets)
    q = {k: np.full((T, H), np.nan) for k in S.ALPHAS}
    pool = np.zeros((T, H), np.int64)
    n_cal = np.zeros(T, np.int64)
    ok = np.zeros(T, bool)
    last = np.full(T, -1, np.int64)
    for j, g in enumerate(targets):
        t = int(ts[g])
        a, b = window(ts, t)
        n_cal[j] = b - a
        if b - a < S.ROLL_MIN:
            continue
        last[j] = int(ts[b - 1]) + S.LAST_LEAD_OFFSET
        assert last[j] <= t - S.DAY, 'causality violated for target %d' % t
        assert int(ts[b - 1]) <= t - S.ROLL_LAST * S.DAY and int(ts[a]) >= t - S.ROLL_FIRST * S.DAY
        ok[j] = True
        for l in range(H):
            block = scores[l, a:b]
            r, pool[j, l] = pooled_quantiles(block[np.isfinite(block)], b - a)
            for k in q:
                q[k][j, l] = r[k]
    return dict(q=q, pool=pool, n_cal=n_cal, ok=ok, last_target_time=last)


def run_intervals(scores, ts, cal_idx, eval_idx, extra):
    """Both methods on the evaluation days. extra(g) -> (yhat[H, C], high[H, C]) with high = valid & y >= 60."""
    H = scores.shape[0]
    E = len(eval_idx)
    qS, poolS, nS = static_quantiles(scores, cal_idx)
    roll = rolling_quantiles(scores, ts, eval_idx)
    nv = np.zeros((E, H), np.int64)
    ns = np.zeros((E, H), np.int64)
    tabs = {m: {k: {f: np.zeros((E, H)) if f == 'q' else np.zeros((E, H), np.int64)
                    for f in ('q', 'cov', 'sub_cov', 'neg')} for k in S.ALPHAS} for m in 'ST'}
    included = {'S': np.ones(E, bool), 'T': roll['ok']}
    for e, g in enumerate(eval_idx):
        s = scores[:, g, :]
        yhat, high = extra(g)
        valid = np.isfinite(s)
        nv[e] = valid.sum(1)
        ns[e] = high.sum(1)
        for m in 'ST':
            if not included[m][e]:
                continue
            for k in S.ALPHAS:
                q = qS[k] if m == 'S' else roll['q'][k][e]
                tab = tabs[m][k]
                tab['q'][e] = q
                with np.errstate(invalid='ignore'):
                    covered = valid & (s <= q[:, None])
                    tab['cov'][e] = covered.sum(1)
                    tab['sub_cov'][e] = (covered & high).sum(1)
                    tab['neg'][e] = (valid & ((yhat - q[:, None]) < 0)).sum(1)
    return dict(S=dict(q=qS, pool=poolS, n_days=nS), T=roll, nv=nv, ns=ns, tabs=tabs, included=included)


def metrics(tab, nv, ns, mask, hsel=slice(None)):
    """Pooled metrics over the evaluated days in mask (bool over evaluation days) and horizon selection."""
    n = nv[mask][:, hsel]
    den = int(n.sum())
    if den == 0:
        return None
    q = tab['q'][mask][:, hsel]
    cov = int(tab['cov'][mask][:, hsel].sum())
    neg = int(tab['neg'][mask][:, hsel].sum())
    sub_n = ns[mask][:, hsel]
    sub_den = int(sub_n.sum())
    sub_cov = int(tab['sub_cov'][mask][:, hsel].sum())
    return dict(n_valid=den, n_covered=cov, coverage=cov / den, mean_width=float((n * 2.0 * q).sum() / den),
                neg_lower_fraction=neg / den, n_subgroup=sub_den,
                subgroup_coverage=(sub_cov / sub_den if sub_den else None),
                subgroup_mean_width=(float((sub_n * 2.0 * q).sum() / sub_den) if sub_den else None))


def day_totals(tab, nv):
    return tab['cov'].sum(1).astype(float), nv.sum(1).astype(float)


def strata_for(dates, mask):
    """Year-stratified 7-day block frame: blocks anchored at 1 January of each year, only blocks that contain
    at least one evaluated day are resampled; one RNG per frame, years ascending."""
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    rng = np.random.default_rng(S.SEED)
    out = []
    for y in sorted(set(years[mask].tolist())):
        sel = mask & (years == y)
        start = np.datetime64('%d-01-01' % y)
        ids = (dates[sel] - start).astype(int) // S.BLOCK_DAYS
        _, inv = np.unique(ids, return_inverse=True)
        n = int(inv.max()) + 1
        draw = rng.integers(n, size=(S.REPLICATES, n))
        w = np.zeros((S.REPLICATES, n))
        np.add.at(w, (np.arange(S.REPLICATES)[:, None], draw), 1.0)
        out.append((sel, inv, n, w))
    return out


def replicates(strata, num_day, den_day):
    num = np.zeros(S.REPLICATES)
    den = np.zeros(S.REPLICATES)
    for sel, inv, n, w in strata:
        num += w @ np.bincount(inv, weights=num_day[sel], minlength=n)
        den += w @ np.bincount(inv, weights=den_day[sel], minlength=n)
    return num / den


def interval95(rep):
    lo, hi = np.quantile(rep, S.CI_QUANTILES)
    return [float(lo), float(hi)]
