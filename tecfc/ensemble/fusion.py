"""Small, bias-inclusive forecast-fusion calculations; no trainer imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STAT_COLUMNS = ["sse", "sae", "error_sum", "n"]
MOMENT_COLUMNS = ["sse_left", "sse_right", "error_cross_sum", "error_sum_left", "error_sum_right", "n"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sufficient(prediction, target, mask):
    """Each row is one lead; invalid labels contribute exactly zero."""
    if not prediction.shape == target.shape == mask.shape or mask.dtype != bool:
        raise ValueError("prediction, target and boolean support must have identical shapes")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("nonfinite value on the declared valid support")
    e = np.zeros(target.shape, dtype=np.float64)
    np.subtract(prediction, target, out=e, where=mask, dtype=np.float64)
    return np.stack(((e * e).sum(-1), np.abs(e).sum(-1), e.sum(-1), mask.sum(-1)), -1)


def moments(left, right, target, mask):
    """Uncentered errors retain bias: sum((left-y)*(right-y))."""
    a = np.zeros(target.shape, np.float64)
    b = np.zeros(target.shape, np.float64)
    if not left.shape == right.shape == target.shape == mask.shape:
        raise ValueError("unaligned error-cross-product inputs")
    if not all(np.isfinite(x[mask]).all() for x in (left, right, target)):
        raise ValueError("nonfinite value on the common support")
    np.subtract(left, target, out=a, where=mask, dtype=np.float64)
    np.subtract(right, target, out=b, where=mask, dtype=np.float64)
    return np.stack(((a*a).sum(-1), (b*b).sum(-1), (a*b).sum(-1), a.sum(-1), b.sum(-1), mask.sum(-1)), -1)


def metrics(stats):
    s = np.asarray(stats, dtype=np.float64).reshape(-1, 4).sum(0)
    n = int(round(s[3]))
    return dict(sse=float(s[0]), sae=float(s[1]), error_sum=float(s[2]), n_valid=n,
                RMSE=float(np.sqrt(s[0]/n)) if n else None,
                MAE=float(s[1]/n) if n else None, bias=float(s[2]/n) if n else None)


def fused_sse(m, w):
    m = np.asarray(m, np.float64)
    # A convex combination of vectors, not centered covariance.
    q = (1-w)**2*m[..., 0] + w*w*m[..., 1] + 2*w*(1-w)*m[..., 2]
    tol = 1e-11*np.maximum(m[..., 0]+m[..., 1], 1.)
    if np.any(q < -tol):
        raise ValueError("cross moments imply a negative squared error")
    return np.maximum(q, 0.)


def weight_objective(by_seed, w):
    """One pooled RMSE per seed, then an equally weighted seed mean."""
    m = np.asarray(by_seed, np.float64)
    if m.ndim != 2 or m.shape[1] != 6 or np.any(m[:, 5] <= 0):
        raise ValueError("need [seed,6] pooled nonempty moments")
    return float(np.sqrt(fused_sse(m, w)/m[:, 5]).mean())


def select_weight(by_seed):
    """Deterministic bounded convex minimization; endpoint ties prefer w=0."""
    low, high = 0., 1.
    phi = (np.sqrt(5.)-1)/2
    a, b = high-phi*(high-low), low+phi*(high-low)
    fa, fb = weight_objective(by_seed, a), weight_objective(by_seed, b)
    for _ in range(100):
        if fa <= fb:
            high, b, fb = b, a, fa
            a = high-phi*(high-low)
            fa = weight_objective(by_seed, a)
        else:
            low, a, fa = a, b, fb
            b = low+phi*(high-low)
            fb = weight_objective(by_seed, b)
    candidates = [0., 1., (low+high)/2]
    values = [weight_objective(by_seed, w) for w in candidates]
    minimum = min(values)
    # Avoid a meaningless tiny nonzero weight on a numerically flat optimum.
    w = min(w for w, v in zip(candidates, values) if v <= minimum+1e-12)
    return dict(weight_right=float(w), mean_seed_pooled_RMSE=weight_objective(by_seed, w),
                endpoint_objectives={"w0": values[0], "w1": values[1]},
                optimizer="bounded golden section, 100 iterations, endpoints included, absolute tie tolerance 1e-12 TECU")


def paired(candidate, reference, origins, block_days=7, reps=5000, rng_seed=20260924):
    """Calendar blocks shared across seeds; fixed-model temporal uncertainty."""
    a, b = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    if not a.shape == b.shape or a.ndim != 4 or a.shape[-1] != 4:
        raise ValueError("paired inputs must be [seed,day,lead,stat]")
    if not np.array_equal(a[..., 3], b[..., 3]):
        raise ValueError("paired target support differs")
    aa, bb = a.sum(2), b.sum(2)
    na, nb = aa[..., 3].sum(1), bb[..., 3].sum(1)
    if np.any(na <= 0) or np.any(nb <= 0):
        return dict(status="empty_stratum")
    delta = np.sqrt(aa[..., 0].sum(1)/na)-np.sqrt(bb[..., 0].sum(1)/nb)
    dates = np.asarray(origins).astype("datetime64[s]").astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int)+1970
    rng = np.random.default_rng(rng_seed)
    sa = np.zeros((reps, len(aa), 2)); sb = sa.copy()
    for year in sorted(set(years.tolist())):
        indices = np.flatnonzero(years == year)
        start, end = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year+1}-01-01")
        count = (int((end-start).astype(int))+block_days-1)//block_days
        blocks = ((dates[indices]-start).astype(int)//block_days).astype(int)
        ba = np.zeros((count, len(aa), 2)); bc = ba.copy()
        for index, block in zip(indices, blocks):
            ba[block, :, 0] += aa[:, index, 0]; ba[block, :, 1] += aa[:, index, 3]
            bc[block, :, 0] += bb[:, index, 0]; bc[block, :, 1] += bb[:, index, 3]
        draws = rng.integers(0, count, (reps, count))
        sa += ba[draws].sum(1); sb += bc[draws].sum(1)
    good = np.all((sa[..., 1] > 0) & (sb[..., 1] > 0), axis=1)
    boot = (np.sqrt(sa[good, :, 0]/sa[good, :, 1])-np.sqrt(sb[good, :, 0]/sb[good, :, 1])).mean(1)
    leave = [float(np.delete(delta, i).mean()) for i in range(len(delta))] if len(delta)>1 else []
    ci = np.quantile(boot, [.025, .975]).tolist() if len(boot) else None
    harm = bool(len(delta) == 5 and delta.mean()>0 and (delta>0).sum()>=4 and all(x>0 for x in leave) and ci and ci[0]>0)
    return dict(delta_rmse_by_seed=delta.tolist(), delta_rmse_mean=float(delta.mean()),
                delta_rmse_sample_sd=float(delta.std(ddof=1)) if len(delta)>1 else None,
                improved_seeds=int((delta<0).sum()), worsened_seeds=int((delta>0).sum()),
                leave_one_seed_means=leave, conditional_time_ci95=ci, block_days=block_days,
                usable_replicates=int(good.sum()), replicates=reps, rng=rng_seed,
                automatic_repeated_harm=harm,
                scheme="year-stratified non-overlapping calendar blocks; shared draws across seed pairs",
                condition="fixed trained model pairs; reused control members are not independent training replicates")
