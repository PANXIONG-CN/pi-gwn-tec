from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np


def split_chronological_windows(n_windows: int, calibration_frac: float) -> Tuple[slice, slice]:
    n_windows = int(n_windows)
    calibration_frac = float(calibration_frac)
    if n_windows <= 1:
        raise ValueError(f"n_windows must be > 1, got {n_windows}")
    if not (0.0 < calibration_frac < 1.0):
        raise ValueError(f"calibration_frac must be in (0, 1), got {calibration_frac}")

    n_calib = int(math.floor(n_windows * calibration_frac))
    if n_calib <= 0 or n_calib >= n_windows:
        raise ValueError(f"invalid calibration split: n_windows={n_windows}, n_calib={n_calib}")
    return slice(0, n_calib), slice(n_calib, n_windows)


def _quantile_higher(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def conformal_qhat_from_residuals(abs_residuals: np.ndarray, alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    arr = np.asarray(abs_residuals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        raise ValueError("abs_residuals has no finite values")

    n = int(arr.size)
    q = math.ceil((n + 1) * (1.0 - alpha)) / float(n)
    q = min(max(q, 0.0), 1.0)
    return _quantile_higher(arr, q)


def symmetric_interval_from_point(pred: np.ndarray, qhat: float) -> Tuple[np.ndarray, np.ndarray]:
    qhat = float(qhat)
    if not math.isfinite(qhat) or qhat < 0.0:
        raise ValueError(f"qhat must be finite and >= 0, got {qhat}")

    pred_arr = np.asarray(pred, dtype=np.float32)
    delta = np.asarray(qhat, dtype=np.float32)
    return pred_arr - delta, pred_arr + delta


def compute_interval_metrics(pred: np.ndarray, true: np.ndarray, qhat: float) -> Dict[str, float]:
    lower, upper = symmetric_interval_from_point(pred, qhat)
    truth = np.asarray(true, dtype=np.float32)
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(truth)
    count = int(np.sum(valid))
    if count <= 0:
        return {"coverage": float("nan"), "mean_width": float("nan"), "count": 0}

    covered = (truth >= lower) & (truth <= upper)
    widths = upper - lower
    return {
        "coverage": float(np.mean(covered[valid])),
        "mean_width": float(np.mean(widths[valid])),
        "count": count,
    }


def summarize_grouped_interval_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    qhat: float,
    groups: Iterable[str],
) -> List[Dict[str, float]]:
    pred_arr = np.asarray(pred, dtype=np.float32)
    true_arr = np.asarray(true, dtype=np.float32)
    groups_arr = np.asarray(list(groups), dtype=object)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"shape mismatch: pred={pred_arr.shape}, true={true_arr.shape}")
    if pred_arr.ndim < 1:
        raise ValueError(f"pred must have at least 1 dimension, got {pred_arr.shape}")
    if int(pred_arr.shape[0]) != int(groups_arr.shape[0]):
        raise ValueError(f"group length mismatch: windows={pred_arr.shape[0]}, groups={groups_arr.shape[0]}")

    lower, upper = symmetric_interval_from_point(pred_arr, qhat)
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(true_arr)
    covered = (true_arr >= lower) & (true_arr <= upper)
    widths = upper - lower

    rows: List[Dict[str, float]] = []
    for group in sorted({str(x) for x in groups_arr.tolist()}):
        mask = groups_arr == group
        valid_group = valid[mask]
        count = int(np.sum(valid_group))
        if count <= 0:
            coverage = float("nan")
            mean_width = float("nan")
        else:
            coverage = float(np.mean(covered[mask][valid_group]))
            mean_width = float(np.mean(widths[mask][valid_group]))
        rows.append(
            {
                "group": str(group),
                "count": count,
                "coverage": coverage,
                "mean_width": mean_width,
            }
        )
    return rows


def china_calendar_roles(dates):
    dates=np.asarray(dates,dtype="datetime64[D]")
    if len(np.unique(dates))!=len(dates):
        raise ValueError("duplicate dates")
    return {"calibration_2015_h1":(dates>=np.datetime64("2015-01-01"))&(dates<=np.datetime64("2015-06-30")),"development_2015_h2":(dates>=np.datetime64("2015-07-01"))&(dates<=np.datetime64("2015-12-31")),"retrospective_2019":dates.astype("datetime64[Y]")==np.datetime64("2019"),"retrospective_2024":dates.astype("datetime64[Y]")==np.datetime64("2024")}


def _residuals(pred,true):
    pred,true=np.asarray(pred,dtype=np.float64),np.asarray(true,dtype=np.float64)
    if pred.shape!=true.shape or pred.ndim!=3:
        raise ValueError("predictions/truth must have identical [day,lead,node] shapes")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("finite all-valid product required")
    return np.abs(pred-true)


def horizon_qhat(pred,true,alpha=0.05):
    if not 0<alpha<1:
        raise ValueError("alpha must be in (0,1)")
    scores=_residuals(pred,true)
    if scores.shape[0]==0 or scores.shape[2]==0:
        raise ValueError("empty calibration set")
    m=scores.shape[0]*scores.shape[2]
    # Match the author-frozen q_h and numpy higher empirical quantile exactly.
    q=min(1.0,np.ceil((m+1)*(1-alpha))/m)
    qhat=np.array([np.quantile(scores[:,h,:].ravel(),q,method="higher") for h in range(scores.shape[1])])
    return qhat,{"alpha":alpha,"scores_per_lead":m,"quantile_level":float(q),"quantile_method":"higher","calibration_days":scores.shape[0],"independent_sample_count_claimed":False}


def horizon_interval_metrics(pred,true,qhat,alpha=0.05):
    residuals=_residuals(pred,true)
    qhat=np.asarray(qhat,dtype=np.float64)
    if qhat.shape!=(residuals.shape[1],) or not np.isfinite(qhat).all() or np.any(qhat<0) or not 0<alpha<1:
        raise ValueError("one finite nonnegative qhat per lead and alpha in (0,1) required")
    if residuals.shape[0]==0 or residuals.shape[2]==0:
        raise ValueError("empty evaluation role")
    rows=[]
    for h,width in enumerate(qhat):
        r=residuals[:,h,:]
        covered=r<=width
        score=2*width+(2/alpha)*np.maximum(r-width,0)
        rows.append({"lead":h+1,"lead_hours":2*(h+1),"n_days":r.shape[0],"n_element_scores":r.size,"qhat_tecu":float(width),"empirical_coverage":float(covered.mean()),"mean_width_tecu":float(2*width),"mean_interval_score_tecu":float(score.mean())})
    return rows


def calibration_main(argv=None):
    """Calibrate a saved regional run by calendar date and forecast lead."""
    import argparse
    import csv
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=calibration_main.__doc__)
    parser.add_argument('--calibrate-predictions', required=True,
                        help='Evaluation directory with predictions.npy, truth.npy and origins.npy in TECU')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--alpha', type=float, default=.05)
    args = parser.parse_args(argv)
    root, out = Path(args.calibrate_predictions), Path(args.out_dir)
    pred = np.load(root/'predictions.npy', mmap_mode='r', allow_pickle=False)
    truth = np.load(root/'truth.npy', mmap_mode='r', allow_pickle=False)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred, truth = pred[..., 0], truth[..., 0]
    origins = np.load(root/'origins.npy', allow_pickle=False)
    dates = origins.astype('datetime64[s]').astype('datetime64[D]')
    if len(dates) != len(pred):
        raise ValueError('one UTC origin per daily prediction is required')
    roles = china_calendar_roles(dates)
    cal = roles.pop('calibration_2015_h1')
    if cal.sum() != 181:
        raise ValueError('the recorded protocol requires all 181 calibration dates')
    qhat, metadata = horizon_qhat(pred[cal], truth[cal], args.alpha)
    rows = []
    for role, selected in roles.items():
        if selected.any():
            rows.extend(dict(role=role, **row) for row in horizon_interval_metrics(
                pred[selected], truth[selected], qhat, args.alpha))
    if not rows:
        raise ValueError('no evaluation dates from 2015 H2, 2019 or 2024')
    out.mkdir(parents=True, exist_ok=False)
    with (out/'leadwise_intervals.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out/'calibration.json').write_text(json.dumps(dict(
        **metadata, qhat_tecu=qhat.tolist(), protocol='regional_calendar_leadwise_v1'), indent=2)+'\n')
