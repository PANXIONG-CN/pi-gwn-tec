from __future__ import annotations

import sys
if '--calibrate-predictions' in sys.argv:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lib.uq.conformal import calibration_main
    calibration_main()
    raise SystemExit(0)
if '--revised-sources' in sys.argv:
    from revised_materials import main
    main()
    raise SystemExit(0)

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from paper_common import assert_out_dir_unlocked


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def _bootstrap_ci_mean(values: Sequence[float], b: int, rng: np.random.Generator) -> Tuple[float, float]:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        m = float(v[0])
        return m, m
    idx = rng.integers(0, v.size, size=(int(b), v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5]).astype(np.float64)
    return float(lo), float(hi)


def _p_to_stars(p: float) -> str:
    if not _is_finite(p):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _normalize_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    return bool(default)


def _normalize_split_profile(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return "default"
    if s in ("d", "batchd", "d-split"):
        return "d"
    if s in ("default", "batchc", "c"):
        return "default"
    return s


def _split_csv(s: str) -> List[str]:
    s = str(s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in re.split(r"[,\s]+", s):
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception as e:
            raise ValueError(f"Invalid integer token: {tok}") from e
    return out


def _load_seed_whitelist(path: Path) -> List[int]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    # Try JSON list first
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except Exception:
        pass
    # Fallback: treat as delimited text
    return _parse_int_list(raw)


def _apply_run_filters(
    runs_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    stepwise_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    *,
    dataset_slugs: Sequence[str],
    paper_slugs: Sequence[str],
    paper_slug_prefixes: Sequence[str],
    split_profiles: Sequence[str],
    seed_whitelist: Sequence[int],
    seed_whitelist_slugs: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if runs_df.empty:
        return runs_df, metrics_df, stepwise_df, phys_df, cost_df

    df = runs_df.copy()

    if dataset_slugs:
        df = df[df["dataset_slug"].isin(list(dataset_slugs))].copy()
    if paper_slugs:
        df = df[df["paper_slug"].isin(list(paper_slugs))].copy()
    if paper_slug_prefixes:
        mask = pd.Series(False, index=df.index)
        for prefix in paper_slug_prefixes:
            mask = mask | df["paper_slug"].astype(str).str.startswith(str(prefix))
        df = df[mask].copy()
    if split_profiles:
        df = df[df["split_profile"].isin(list(split_profiles))].copy()
    if seed_whitelist:
        if seed_whitelist_slugs:
            mask = (~df["paper_slug"].isin(list(seed_whitelist_slugs))) | df["seed"].isin(list(seed_whitelist))
            df = df[mask].copy()
        else:
            df = df[df["seed"].isin(list(seed_whitelist))].copy()

    keep = set(df["run_dir"].astype(str).tolist())
    if not metrics_df.empty:
        metrics_df = metrics_df[metrics_df["run_dir"].isin(keep)].copy()
    if not stepwise_df.empty:
        stepwise_df = stepwise_df[stepwise_df["run_dir"].isin(keep)].copy()
    if not phys_df.empty:
        phys_df = phys_df[phys_df["run_dir"].isin(keep)].copy()
    if not cost_df.empty:
        cost_df = cost_df[cost_df["run_dir"].isin(keep)].copy()
    return df, metrics_df, stepwise_df, phys_df, cost_df


def _derive_paper_slug(manifest: dict) -> str:
    args = manifest.get("args") or {}
    slug = ""

    if str(manifest.get("paper_slug") or "").strip():
        slug = str(manifest["paper_slug"]).strip()
    elif str(args.get("paper_slug") or "").strip():
        slug = str(args["paper_slug"]).strip()

    # Fallback: derive from cmdline when argparse on remote side ignored -paper_slug.
    # Example: "... -paper_slug a1_pinn_pilot_pi_static_lp0p02 -lambda_phys 0.02 ..."
    cmd = str(args.get("cmdline") or "").strip()
    if cmd:
        m = re.search(
            r'(?:^|\s)-{1,2}paper_slug(?:=|\s+)(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))',
            cmd,
        )
        if m:
            cand = next((g for g in m.groups() if g), "")
            if str(cand or "").strip():
                slug = str(cand).strip()

    if slug:
        holdout_mode = str(args.get("holdout_mode") or "off").strip().lower()
        if holdout_mode not in ("", "off", "none", "0", "false") and str(args.get("holdout_mask_path") or "").strip():
            if not slug.endswith("__loo"):
                slug = f"{slug}__loo"
        return slug
    model_lower = str(manifest.get("model") or "").strip().lower()

    # Batch D: explicit slugs
    if str(args.get("hybrid_bg_path") or "").strip() or str(args.get("out_model_slug") or "").strip().lower() == "hybrid_model":
        slug = "hybrid"
    elif model_lower == "zeroshot_global_bg":
        slug = "global_bg"
    # Default mapping (fix_docs/Paper_Experiments_Master_Plan.md §9.2)
    elif model_lower == "gwn":
        use_pinn = _normalize_bool(args.get("use_pinn"), default=False)
        use_drivers = _normalize_bool(args.get("use_drivers"), default=False)
        use_diffusion = _normalize_bool(args.get("use_diffusion"), default=True)
        nochem = _normalize_bool(args.get("nochem"), default=False)
        if use_pinn and (not use_diffusion):
            slug = "pi_gwn-nodiff"
        elif use_pinn and nochem:
            slug = "pi_gwn-nochem"
        elif (not use_pinn) and (not use_drivers):
            slug = "do_gwn"
        elif (not use_pinn) and use_drivers:
            slug = "de_gwn"
        elif use_pinn and (not use_drivers):
            slug = "pi_gwn_static"
        elif use_pinn and use_drivers:
            slug = "pi_gwn"
        else:
            slug = "gwn"
    elif model_lower == "tec_mollm":
        slug = "st_gpt"
    elif model_lower == "optimizedcsa_wtconvlstm":
        slug = "csa_wtconvlstm"
    elif model_lower in ("convlstm", "predrnn"):
        slug = model_lower
    else:
        slug = model_lower or "unknown"

    # Batch D: LOO holdout suffix (does not change base model family)
    holdout_mode = str(args.get("holdout_mode") or "off").strip().lower()
    if holdout_mode not in ("", "off", "none", "0", "false") and str(args.get("holdout_mask_path") or "").strip():
        slug = f"{slug}__loo"

    return slug


def _raw_china_hires_data_root(manifest: dict) -> str:
    args = manifest.get("args") or {}
    return str(args.get("china_hires_data_root") or "").strip()


def _protocol_signature(manifest: dict) -> str:
    proto = manifest.get("protocol") or {}
    lag = int(proto.get("lag", 0) or 0)
    horizon = int(proto.get("horizon", 0) or 0)
    stride = int(proto.get("stride", 0) or 0)
    prefix = _normalize_bool(proto.get("prefix_boundary"), default=False)
    year_split = _normalize_bool(proto.get("year_split"), default=False)
    normalizer = str(proto.get("normalizer") or "")
    split_profile = str(proto.get("split_profile") or "").strip().lower()
    return f"lag{lag}_hor{horizon}_str{stride}_pb{int(prefix)}_ys{int(year_split)}_norm{normalizer}_sp{split_profile}"


def _is_paper_protocol(manifest: dict) -> bool:
    proto = manifest.get("protocol") or {}
    lag = int(proto.get("lag", 0) or 0)
    horizon = int(proto.get("horizon", 0) or 0)
    stride = int(proto.get("stride", 0) or 0)
    prefix = _normalize_bool(proto.get("prefix_boundary"), default=False)
    year_split = _normalize_bool(proto.get("year_split"), default=False)
    return (lag == 12) and (horizon == 12) and (stride == 12) and prefix and year_split


def _parse_metrics_json(metrics: dict) -> List[dict]:
    rows: List[dict] = []
    splits = metrics.get("splits")
    if isinstance(splits, dict) and splits:
        for split_name, sm in splits.items():
            if not isinstance(sm, dict):
                continue
            overall = sm.get("overall")
            if not isinstance(overall, dict):
                continue
            rows.append({"split": str(split_name), **overall})
        return rows
    overall = metrics.get("overall")
    if isinstance(overall, dict):
        rows.append({"split": "test", **overall})
    return rows


def _load_stepwise_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" not in df.columns:
        df["split"] = "test"
    return df


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    dataset_slug: str
    dataset: str
    model: str
    seed: int
    run_tag: str
    paper_slug: str
    config: str
    debug: bool
    epochs: int
    split_profile: str
    china_hires_data_root: str
    protocol_sig: str
    protocol_is_paper: bool
    complete_score: int
    mtime: float


def _compute_complete_score(run_dir: Path) -> int:
    score = 0
    if (run_dir / "metrics.json").exists():
        score += 10
    if (run_dir / "stepwise_rmse.csv").exists():
        score += 1
    if (run_dir / "compute_cost.json").exists():
        score += 1
    if (run_dir / "physics_metrics.json").exists():
        score += 1
    # Paper runs may upload checkpoints to OSS and keep only an *.ossref marker locally.
    if (run_dir / "best_model.pth").exists() or (run_dir / "best_model.ossref").exists():
        score += 1
    # Paper runs should upload large predictions to OSS and leave an *.ossref marker.
    # Do NOT prefer local *_test_preds.npy over OSS refs, otherwise re-test can accidentally
    # promote debug/smoke runs by virtue of having local arrays.
    if (run_dir / "predictions.ossref").exists():
        score += 1
    return int(score)


def _load_run(manifest_path: Path) -> Tuple[RunRecord, List[dict], pd.DataFrame, Optional[dict], Optional[dict]]:
    run_dir = manifest_path.parent
    man = _read_json(manifest_path)

    dataset_slug = str(man.get("dataset_slug") or "")
    dataset = str(man.get("dataset") or "")
    model = str(man.get("model") or "")
    seed = int(man.get("seed") or 0)
    run_tag = str(man.get("run_tag") or run_dir.name)
    paper_slug = _derive_paper_slug(man)
    proto_sig = _protocol_signature(man)
    protocol_is_paper = _is_paper_protocol(man)
    args = man.get("args") or {}
    config = str(args.get("config") or "").strip()
    debug = _normalize_bool(args.get("debug"), default=False)
    try:
        epochs = int(args.get("epochs") or 0)
    except Exception:
        epochs = 0

    split_profile = str((man.get("protocol") or {}).get("split_profile") or "").strip().lower()
    if not split_profile:
        split_profile = str(args.get("split_profile") or "default").strip().lower() or "default"
    china_hires_data_root = _raw_china_hires_data_root(man)

    metrics_rows: List[dict] = []
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = _read_json(metrics_path)
            for r in _parse_metrics_json(metrics):
                metrics_rows.append(r)
        except Exception:
            metrics_rows = []

    stepwise_df = pd.DataFrame()
    stepwise_path = run_dir / "stepwise_rmse.csv"
    if stepwise_path.exists():
        try:
            stepwise_df = _load_stepwise_csv(stepwise_path)
        except Exception:
            stepwise_df = pd.DataFrame()

    phys = None
    phys_path = run_dir / "physics_metrics.json"
    if phys_path.exists():
        try:
            phys = _read_json(phys_path)
        except Exception:
            phys = None

    cost = None
    cost_path = run_dir / "compute_cost.json"
    if cost_path.exists():
        try:
            cost = _read_json(cost_path)
        except Exception:
            cost = None

    complete_score = _compute_complete_score(run_dir)
    try:
        mtime = float(manifest_path.stat().st_mtime)
    except Exception:
        mtime = 0.0

    rec = RunRecord(
        run_dir=run_dir,
        dataset_slug=dataset_slug,
        dataset=dataset,
        model=model,
        seed=seed,
        run_tag=run_tag,
        paper_slug=paper_slug,
        config=config,
        debug=bool(debug),
        epochs=int(epochs),
        split_profile=split_profile,
        china_hires_data_root=china_hires_data_root,
        protocol_sig=proto_sig,
        protocol_is_paper=bool(protocol_is_paper),
        complete_score=complete_score,
        mtime=mtime,
    )
    return rec, metrics_rows, stepwise_df, phys, cost


def _discover_manifests(outputs_root: Path) -> List[Path]:
    # Use glob over the canonical outputs layout; avoid rglob over large arrays.
    return sorted(outputs_root.glob("*/*/seed_*/*/manifest.json"))


def _select_latest_complete(runs: List[RunRecord]) -> List[RunRecord]:
    df = pd.DataFrame([r.__dict__ for r in runs])
    if df.empty:
        return []
    # Prefer "paper" runs when multiple runs collide:
    # - non-debug > debug
    # - conf/paper/* config > ad-hoc CLI runs (helps avoid debug/smoke runs getting picked after re-test mtime updates)
    # - larger epochs > smaller epochs
    df["debug"] = df["debug"].fillna(False).astype(bool)
    df["epochs"] = pd.to_numeric(df["epochs"], errors="coerce").fillna(0).astype(int)
    df["config"] = df["config"].fillna("").astype(str)
    df["is_paper_config"] = df["config"].str.startswith("conf/paper/")
    df = df.sort_values(
        ["protocol_is_paper", "is_paper_config", "debug", "epochs", "complete_score", "mtime"],
        ascending=[False, False, True, False, False, False],
    )
    group_cols = ["dataset_slug", "paper_slug", "seed", "split_profile", "protocol_sig"]
    picked = df.drop_duplicates(subset=group_cols, keep="first")
    out: List[RunRecord] = []
    for row in picked.to_dict(orient="records"):
        row.pop("is_paper_config", None)
        out.append(RunRecord(**row))
    return out


def _build_stats(
    metrics_df: pd.DataFrame,
    *,
    b: int,
    rng: np.random.Generator,
    group_cols: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for keys, g in metrics_df.groupby(list(group_cols), dropna=False):
        vals = [float(x) for x in g[value_col].tolist() if _is_finite(x)]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        lo, hi = _bootstrap_ci_mean(vals, b=b, rng=rng)
        row = {k: v for k, v in zip(group_cols, keys)}
        row.update({"n": int(len(vals)), "mean": mean, "std": std, "ci95_lo": lo, "ci95_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_wilcoxon(
    metrics_df: pd.DataFrame,
    *,
    dataset_slug: str,
    split: str,
    metric: str,
    a: str,
    b: str,
) -> dict:
    sub = metrics_df[
        (metrics_df["dataset_slug"] == dataset_slug)
        & (metrics_df["split"] == split)
        & (metrics_df["metric"] == metric)
        & (metrics_df["paper_slug"].isin([a, b]))
    ].copy()
    if sub.empty:
        return {"ok": False, "n": 0, "p": float("nan"), "stars": ""}
    pv = sub.pivot_table(index="seed", columns="paper_slug", values="value", aggfunc="first")
    if (a not in pv.columns) or (b not in pv.columns):
        return {"ok": False, "n": 0, "p": float("nan"), "stars": ""}
    pv = pv[[a, b]].dropna()
    if pv.shape[0] < 2:
        return {"ok": False, "n": int(pv.shape[0]), "p": float("nan"), "stars": ""}
    try:
        res = wilcoxon(pv[a].astype(float), pv[b].astype(float))
        p = float(res.pvalue)
        return {"ok": True, "n": int(pv.shape[0]), "p": p, "stars": _p_to_stars(p)}
    except Exception:
        return {"ok": False, "n": int(pv.shape[0]), "p": float("nan"), "stars": ""}


def _pick_first_present(candidates: Sequence[str], available: Sequence[str]) -> Optional[str]:
    avail = set(str(x) for x in (available or []))
    for cand in candidates:
        if str(cand) in avail:
            return str(cand)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect paper-facing results from outputs/ and export tables/figure data sources.")
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--out_dir", default="paper_artifacts")
    ap.add_argument("--force", action="store_true", help="allow overwriting locked out_dir (finial)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--include_nonpaper", action="store_true", help="include non-paper protocol runs")
    ap.add_argument("--dataset_slugs", default="", help="comma-separated dataset_slug whitelist (e.g., gim,china_hires)")
    ap.add_argument("--paper_slugs", default="", help="comma-separated paper_slug whitelist")
    ap.add_argument("--paper_slug_prefixes", default="", help="comma-separated paper_slug prefix whitelist")
    ap.add_argument(
        "--split_profiles",
        default="",
        help="comma-separated split_profile whitelist (e.g., default,d). For China D-split, pass d (case-insensitive).",
    )
    ap.add_argument("--seed_whitelist", default="", help="comma/space-separated seed whitelist")
    ap.add_argument("--seed_whitelist_file", default="", help="path to seed whitelist file (json list or delimited text)")
    ap.add_argument(
        "--seed_whitelist_slugs",
        default="",
        help="if set, apply seed whitelist only to these paper_slug(s) (comma-separated)",
    )
    args = ap.parse_args()

    repo = _repo_root()
    outputs_root = Path(args.outputs_root).expanduser()
    if not outputs_root.is_absolute():
        outputs_root = (repo / outputs_root).resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()

    assert_out_dir_unlocked(out_dir, force=bool(args.force))

    dataset_slugs = _split_csv(args.dataset_slugs)
    paper_slugs = _split_csv(args.paper_slugs)
    paper_slug_prefixes = _split_csv(args.paper_slug_prefixes)
    split_profiles = sorted({_normalize_split_profile(x) for x in _split_csv(args.split_profiles)})
    seed_whitelist: List[int] = []
    try:
        seed_whitelist.extend(_parse_int_list(args.seed_whitelist))
    except Exception as e:
        raise ValueError(f"--seed_whitelist parse error: {e}") from e
    if str(args.seed_whitelist_file or "").strip():
        sp = Path(str(args.seed_whitelist_file)).expanduser()
        if not sp.is_absolute():
            sp = (repo / sp).resolve()
        seed_whitelist.extend(_load_seed_whitelist(sp))
    seed_whitelist = sorted(set(int(x) for x in seed_whitelist))
    seed_whitelist_slugs = _split_csv(args.seed_whitelist_slugs)

    manifests = _discover_manifests(outputs_root)
    runs: List[RunRecord] = []
    metrics_rows: List[dict] = []
    stepwise_rows: List[pd.DataFrame] = []
    phys_rows: List[dict] = []
    cost_rows: List[dict] = []

    for mp in manifests:
        rec, mrows, step_df, phys, cost = _load_run(mp)
        runs.append(rec)
        for r in mrows:
            metrics_rows.append(
                {
                    "run_dir": str(rec.run_dir),
                    "dataset_slug": rec.dataset_slug,
                    "paper_slug": rec.paper_slug,
                    "seed": rec.seed,
                    "split_profile": rec.split_profile,
                    "protocol_sig": rec.protocol_sig,
                    "protocol_is_paper": rec.protocol_is_paper,
                    "split": str(r.get("split", "test")),
                    **{k: r.get(k) for k in ("MAE", "RMSE", "R2", "CORR")},
                }
            )
        if not step_df.empty:
            df = step_df.copy()
            df["run_dir"] = str(rec.run_dir)
            df["dataset_slug"] = rec.dataset_slug
            df["paper_slug"] = rec.paper_slug
            df["seed"] = rec.seed
            df["split_profile"] = rec.split_profile
            df["protocol_sig"] = rec.protocol_sig
            df["protocol_is_paper"] = rec.protocol_is_paper
            stepwise_rows.append(df)
        if isinstance(phys, dict):
            phys_rows.append(
                {
                    "run_dir": str(rec.run_dir),
                    "dataset_slug": rec.dataset_slug,
                    "paper_slug": rec.paper_slug,
                    "seed": rec.seed,
                    "split_profile": rec.split_profile,
                    "protocol_sig": rec.protocol_sig,
                    "protocol_is_paper": rec.protocol_is_paper,
                    **phys,
                }
            )
        if isinstance(cost, dict):
            cost_rows.append(
                {
                    "run_dir": str(rec.run_dir),
                    "dataset_slug": rec.dataset_slug,
                    "paper_slug": rec.paper_slug,
                    "seed": rec.seed,
                    "split_profile": rec.split_profile,
                    "protocol_sig": rec.protocol_sig,
                    "protocol_is_paper": rec.protocol_is_paper,
                    **cost,
                }
            )

    selected = _select_latest_complete(runs)
    selected_set = {str(r.run_dir) for r in selected}

    runs_df = pd.DataFrame([r.__dict__ for r in selected]).sort_values(["dataset_slug", "paper_slug", "seed"])
    metrics_df_raw = pd.DataFrame(metrics_rows)
    stepwise_df_raw = pd.concat(stepwise_rows, ignore_index=True) if stepwise_rows else pd.DataFrame()
    phys_df_raw = pd.DataFrame(phys_rows)
    cost_df_raw = pd.DataFrame(cost_rows)

    # Keep only selected runs
    metrics_df = metrics_df_raw[metrics_df_raw["run_dir"].isin(selected_set)].copy() if not metrics_df_raw.empty else pd.DataFrame()
    stepwise_df = stepwise_df_raw[stepwise_df_raw["run_dir"].isin(selected_set)].copy() if not stepwise_df_raw.empty else pd.DataFrame()
    phys_df = phys_df_raw[phys_df_raw["run_dir"].isin(selected_set)].copy() if not phys_df_raw.empty else pd.DataFrame()
    cost_df = cost_df_raw[cost_df_raw["run_dir"].isin(selected_set)].copy() if not cost_df_raw.empty else pd.DataFrame()

    if not bool(args.include_nonpaper):
        if not runs_df.empty:
            runs_df = runs_df[runs_df["protocol_is_paper"]].copy()
            keep = set(runs_df["run_dir"].astype(str).tolist())
            if not metrics_df.empty:
                metrics_df = metrics_df[metrics_df["run_dir"].isin(keep)].copy()
            if not stepwise_df.empty:
                stepwise_df = stepwise_df[stepwise_df["run_dir"].isin(keep)].copy()
            if not phys_df.empty:
                phys_df = phys_df[phys_df["run_dir"].isin(keep)].copy()
            if not cost_df.empty:
                cost_df = cost_df[cost_df["run_dir"].isin(keep)].copy()

    # Apply user filters (dataset/paper_slug/seed)
    runs_df, metrics_df, stepwise_df, phys_df, cost_df = _apply_run_filters(
        runs_df,
        metrics_df,
        stepwise_df,
        phys_df,
        cost_df,
        dataset_slugs=dataset_slugs,
        paper_slugs=paper_slugs,
        paper_slug_prefixes=paper_slug_prefixes,
        split_profiles=split_profiles,
        seed_whitelist=seed_whitelist,
        seed_whitelist_slugs=seed_whitelist_slugs,
    )

    # Export raw tidy data (truth source)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(out_dir / "raw" / "runs.csv", index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(out_dir / "raw" / "metrics_overall.csv", index=False)
    if not stepwise_df.empty:
        stepwise_df.to_csv(out_dir / "raw" / "stepwise_rmse.csv", index=False)
    if not phys_df.empty:
        phys_df.to_csv(out_dir / "raw" / "physics_metrics.csv", index=False)
    if not cost_df.empty:
        cost_df.to_csv(out_dir / "raw" / "compute_cost.csv", index=False)

    # Stats (overall metrics)
    if not metrics_df.empty:
        melted = metrics_df.melt(
            id_vars=["dataset_slug", "paper_slug", "seed", "split_profile", "split"],
            value_vars=["RMSE", "MAE", "R2", "CORR"],
            var_name="metric",
            value_name="value",
        )
    else:
        melted = pd.DataFrame()

    rng = np.random.default_rng(0)
    if not melted.empty:
        stats_df = _build_stats(
            melted,
            b=int(args.bootstrap),
            rng=rng,
            group_cols=["dataset_slug", "split_profile", "split", "paper_slug", "metric"],
            value_col="value",
        )
    else:
        stats_df = pd.DataFrame()

    (out_dir / "stats").mkdir(parents=True, exist_ok=True)
    if not stats_df.empty:
        stats_df.to_csv(out_dir / "stats" / "overall_metrics_stats.csv", index=False)

    # Figure data sources: stepwise RMSE aggregated across seeds
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    if not stepwise_df.empty:
        stepwise_df2 = stepwise_df.copy()
        if "split" not in stepwise_df2.columns:
            stepwise_df2["split"] = "test"
        # Some writers use dataset name; normalize to dataset_slug.
        if "dataset" in stepwise_df2.columns:
            stepwise_df2["dataset"] = stepwise_df2["dataset_slug"]
        gcols = ["dataset_slug", "paper_slug", "split_profile", "split", "horizon", "step"]
        agg = stepwise_df2.groupby(gcols, dropna=False)["rmse"].agg(["count", "mean", "std"]).reset_index()
        agg = agg.rename(columns={"count": "n", "mean": "rmse_mean", "std": "rmse_std"})
        agg.to_csv(out_dir / "figures" / "stepwise_rmse_summary.csv", index=False)

    # Figure data sources: physics posterior metrics (test only)
    if not phys_df.empty:
        phys_keys = [
            "nonneg_rate",
            "rot_validity",
            "roti_validity",
            "night_monotonicity",
            "rot_validity_dynamic",
            "roti_validity_dynamic",
        ]
        p = phys_df.copy()
        for k in phys_keys:
            if k not in p.columns:
                p[k] = np.nan
            p[k] = pd.to_numeric(p[k], errors="coerce")
        gcols = ["dataset_slug", "paper_slug", "split_profile"]
        agg = p.groupby(gcols, dropna=False).agg(
            n=("seed", "count"),
            **{f"{k}_mean": (k, "mean") for k in phys_keys},
            **{f"{k}_std": (k, "std") for k in phys_keys},
        ).reset_index()
        agg.to_csv(out_dir / "figures" / "physics_metrics_summary.csv", index=False)

    # Table 1 (GIM, test)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    if not stats_df.empty:
        t1 = stats_df[(stats_df["dataset_slug"] == "gim") & (stats_df["split"] == "test")].copy()
        if not t1.empty:
            t1w = t1.pivot_table(index="paper_slug", columns="metric", values=["mean", "std", "ci95_lo", "ci95_hi", "n"], aggfunc="first")
            t1w.columns = [f"{a}_{b}" for a, b in t1w.columns]
            t1w = t1w.reset_index()
            t1w.to_csv(out_dir / "tables" / "table1_global_gim.csv", index=False)

            # paired Wilcoxon (paper plan focus)
            tests: List[dict] = []
            pi_slug = _pick_first_present(
                [
                    "a1_pinn_confirm_pi_pinn_drvonly_adv_lp0p10",
                    "a1_pinn_confirm_pi_static_lp0p10",
                    "pi_gwn",
                ],
                t1["paper_slug"].astype(str).tolist(),
            )
            for metric in ["RMSE", "MAE", "R2", "CORR"]:
                if pi_slug:
                    tests.append({
                        "dataset_slug": "gim",
                        "split": "test",
                        "metric": metric,
                        "a": pi_slug,
                        "b": "do_gwn",
                        **_paired_wilcoxon(melted, dataset_slug="gim", split="test", metric=metric, a=pi_slug, b="do_gwn"),
                    })
                    tests.append({
                        "dataset_slug": "gim",
                        "split": "test",
                        "metric": metric,
                        "a": pi_slug,
                        "b": "de_gwn",
                        **_paired_wilcoxon(melted, dataset_slug="gim", split="test", metric=metric, a=pi_slug, b="de_gwn"),
                    })

            # A1 PINN ablation (confirm): compare each confirm variant vs DO baseline.
            # 约定命名：a1_pinn_confirm_*（详见 fix_docs/A1_GIM_GWN_PINN_Ablation_Sweeps_Execution_Plan.md）
            confirm_slugs = sorted({s for s in t1["paper_slug"].astype(str).tolist() if s.startswith("a1_pinn_confirm_")})
            for slug in confirm_slugs:
                if pi_slug and slug == pi_slug:
                    continue
                for metric in ["RMSE", "MAE", "R2", "CORR"]:
                    tests.append({
                        "dataset_slug": "gim",
                        "split": "test",
                        "metric": metric,
                        "a": slug,
                        "b": "do_gwn",
                        **_paired_wilcoxon(melted, dataset_slug="gim", split="test", metric=metric, a=slug, b="do_gwn"),
                    })
            pd.DataFrame(tests).to_csv(out_dir / "tables" / "table1_wilcoxon.csv", index=False)

    # Table 2 (China default split_profile)
    if not stats_df.empty:
        t2 = stats_df[(stats_df["dataset_slug"] == "china_hires") & (stats_df["split_profile"] == "default")].copy()
        if not t2.empty:
            t2.to_csv(out_dir / "tables" / "table2_china_default.csv", index=False)
            for split in sorted(set(t2["split"].astype(str).tolist())):
                sub = t2[t2["split"].astype(str) == split].copy()
                if sub.empty:
                    continue
                w = sub.pivot_table(index="paper_slug", columns="metric", values=["mean", "std", "ci95_lo", "ci95_hi", "n"], aggfunc="first")
                w.columns = [f"{a}_{b}" for a, b in w.columns]
                w = w.reset_index()
                w.to_csv(out_dir / "tables" / f"table2_china_default_{split}.csv", index=False)

    # Table 3 (Batch D: China split_profile=D)
    if not stats_df.empty:
        t3 = stats_df[(stats_df["dataset_slug"] == "china_hires") & (stats_df["split_profile"].isin(["d", "batchd", "d-split"]))].copy()
        if not t3.empty:
            t3.to_csv(out_dir / "tables" / "table3_batchd.csv", index=False)
            for split in sorted(set(t3["split"].astype(str).tolist())):
                sub = t3[t3["split"].astype(str) == split].copy()
                if sub.empty:
                    continue
                w = sub.pivot_table(index="paper_slug", columns="metric", values=["mean", "std", "ci95_lo", "ci95_hi", "n"], aggfunc="first")
                w.columns = [f"{a}_{b}" for a, b in w.columns]
                w = w.reset_index()
                w.to_csv(out_dir / "tables" / f"table3_batchd_{split}.csv", index=False)

    # Table 4 (compute cost; per run, no bootstrap by default)
    if not cost_df.empty:
        cols = [
            "dataset_slug",
            "paper_slug",
            "seed",
            "total_params",
            "trainable_params",
            "train_wall_time_s",
            "train_time_per_epoch_s",
            "epochs_ran",
            "max_memory_train_allocated_MiB",
            "max_memory_infer_allocated_MiB",
            "inference_latency_ms",
            "gpu_name",
        ]
        c = cost_df.copy()
        for k in cols:
            if k not in c.columns:
                c[k] = np.nan
        c = c[cols].copy()
        c.to_csv(out_dir / "tables" / "table4_compute_cost_raw.csv", index=False)
        # Aggregate over seeds (mean)
        agg_cols = ["dataset_slug", "paper_slug"]
        c_num = c.copy()
        for col in cols:
            if col in ("dataset_slug", "paper_slug", "seed", "gpu_name"):
                continue
            c_num[col] = pd.to_numeric(c_num[col], errors="coerce")
        c_agg = c_num.groupby(agg_cols, dropna=False).agg(
            n=("seed", "count"),
            total_params_mean=("total_params", "mean"),
            trainable_params_mean=("trainable_params", "mean"),
            train_wall_time_s_mean=("train_wall_time_s", "mean"),
            inference_latency_ms_mean=("inference_latency_ms", "mean"),
            max_memory_train_allocated_MiB_mean=("max_memory_train_allocated_MiB", "mean"),
            max_memory_infer_allocated_MiB_mean=("max_memory_infer_allocated_MiB", "mean"),
        ).reset_index()
        c_agg.to_csv(out_dir / "tables" / "table4_compute_cost.csv", index=False)

    _safe_json_dump(
        {
            "schema_version": 1,
            "outputs_root": str(outputs_root),
            "selected_runs": int(len(runs_df)),
            "filters": {
                "dataset_slugs": list(dataset_slugs),
                "paper_slugs": list(paper_slugs),
                "paper_slug_prefixes": list(paper_slug_prefixes),
                "split_profiles": list(split_profiles),
                "seed_whitelist": list(seed_whitelist),
                "seed_whitelist_slugs": list(seed_whitelist_slugs),
            },
            "notes": [
                "All exports under paper_artifacts/ are derived from outputs/* artifacts only.",
                "Default policy: prefer non-debug + conf/paper config + higher epochs, then complete+mtime per (dataset_slug,paper_slug,seed,split_profile,protocol_sig).",
            ],
        },
        out_dir / "meta.json",
    )

    print("[ok] wrote:", str(out_dir))


if __name__ == "__main__":
    main()
