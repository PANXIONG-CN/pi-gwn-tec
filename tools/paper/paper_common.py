from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LOCK_FILENAME = "finial"


def repo_root() -> Path:
    return _REPO_ROOT


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def split_csv(s: str) -> List[str]:
    s = str(s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def normalize_bool(v: Any, default: bool = False) -> bool:
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


def is_finite(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def derive_paper_slug(manifest: dict) -> str:
    if str(manifest.get("paper_slug") or "").strip():
        return str(manifest["paper_slug"]).strip()

    args = manifest.get("args") or {}
    if str(args.get("paper_slug") or "").strip():
        return str(args["paper_slug"]).strip()

    cmd = str(args.get("cmdline") or "").strip()
    if cmd:
        m = re.search(
            r'(?:^|\s)-{1,2}paper_slug(?:=|\s+)(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))',
            cmd,
        )
        if m:
            slug = next((g for g in m.groups() if g), "")
            if str(slug or "").strip():
                return str(slug).strip()

    model_lower = str(manifest.get("model") or "").strip().lower()

    # Batch D: explicit slugs
    if str(args.get("hybrid_bg_path") or "").strip() or str(args.get("out_model_slug") or "").strip().lower() == "hybrid_model":
        slug = "hybrid"
    elif model_lower == "zeroshot_global_bg":
        slug = "global_bg"
    elif model_lower == "gwn":
        use_pinn = normalize_bool(args.get("use_pinn"), default=False)
        use_drivers = normalize_bool(args.get("use_drivers"), default=False)
        use_diffusion = normalize_bool(args.get("use_diffusion"), default=True)
        nochem = normalize_bool(args.get("nochem"), default=False)
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

    holdout_mode = str(args.get("holdout_mode") or "off").strip().lower()
    if holdout_mode not in ("", "off", "none", "0", "false") and str(args.get("holdout_mask_path") or "").strip():
        slug = f"{slug}__loo"

    return slug


def protocol_signature(manifest: dict) -> str:
    proto = manifest.get("protocol") or {}
    lag = int(proto.get("lag", 0) or 0)
    horizon = int(proto.get("horizon", 0) or 0)
    stride = int(proto.get("stride", 0) or 0)
    prefix = normalize_bool(proto.get("prefix_boundary"), default=False)
    year_split = normalize_bool(proto.get("year_split"), default=False)
    normalizer = str(proto.get("normalizer") or "")
    split_profile = str(proto.get("split_profile") or "").strip().lower()
    return f"lag{lag}_hor{horizon}_str{stride}_pb{int(prefix)}_ys{int(year_split)}_norm{normalizer}_sp{split_profile}"


def is_paper_protocol(manifest: dict) -> bool:
    proto = manifest.get("protocol") or {}
    lag = int(proto.get("lag", 0) or 0)
    horizon = int(proto.get("horizon", 0) or 0)
    stride = int(proto.get("stride", 0) or 0)
    prefix = normalize_bool(proto.get("prefix_boundary"), default=False)
    year_split = normalize_bool(proto.get("year_split"), default=False)
    return (lag == 12) and (horizon == 12) and (stride == 12) and prefix and year_split


def compute_complete_score(run_dir: Path) -> int:
    score = 0
    if (run_dir / "metrics.json").exists():
        score += 10
    if (run_dir / "stepwise_rmse.csv").exists():
        score += 1
    if (run_dir / "compute_cost.json").exists():
        score += 1
    if (run_dir / "physics_metrics.json").exists():
        score += 1
    if (run_dir / "best_model.pth").exists() or (run_dir / "best_model.ossref").exists():
        score += 1
    if (run_dir / "predictions.npy").exists() or (run_dir / "predictions.ossref").exists():
        score += 1
    return int(score)


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    dataset_slug: str
    dataset: str
    model: str
    seed: int
    run_tag: str
    paper_slug: str
    split_profile: str
    protocol_sig: str
    protocol_is_paper: bool
    complete_score: int
    mtime: float


def load_run(manifest_path: Path) -> RunRecord:
    run_dir = manifest_path.parent
    man = read_json(manifest_path)

    dataset_slug = str(man.get("dataset_slug") or "")
    dataset = str(man.get("dataset") or "")
    model = str(man.get("model") or "")
    seed = int(man.get("seed") or 0)
    run_tag = str(man.get("run_tag") or run_dir.name)
    paper_slug = derive_paper_slug(man)
    proto_sig = protocol_signature(man)
    protocol_is_paper = is_paper_protocol(man)

    args = man.get("args") or {}
    split_profile = str((man.get("protocol") or {}).get("split_profile") or "").strip().lower()
    if not split_profile:
        split_profile = str(args.get("split_profile") or "default").strip().lower() or "default"

    complete_score = compute_complete_score(run_dir)
    try:
        mtime = float(manifest_path.stat().st_mtime)
    except Exception:
        mtime = 0.0

    return RunRecord(
        run_dir=run_dir,
        dataset_slug=dataset_slug,
        dataset=dataset,
        model=model,
        seed=seed,
        run_tag=run_tag,
        paper_slug=paper_slug,
        split_profile=split_profile,
        protocol_sig=proto_sig,
        protocol_is_paper=bool(protocol_is_paper),
        complete_score=complete_score,
        mtime=mtime,
    )


def discover_manifests(outputs_root: Path) -> List[Path]:
    return sorted(outputs_root.glob("*/*/seed_*/*/manifest.json"))


def select_latest_by(
    runs: Sequence[RunRecord],
    *,
    key_fields: Sequence[str],
    prefer_paper_protocol: bool = True,
) -> List[RunRecord]:
    best: dict[Tuple[Any, ...], RunRecord] = {}
    for r in runs:
        key = tuple(getattr(r, f) for f in key_fields)
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        a = (
            int(prefer_paper_protocol and r.protocol_is_paper),
            int(r.complete_score),
            float(r.mtime),
        )
        b = (
            int(prefer_paper_protocol and prev.protocol_is_paper),
            int(prev.complete_score),
            float(prev.mtime),
        )
        if a > b:
            best[key] = r
    return sorted(best.values(), key=lambda x: tuple(getattr(x, f) for f in key_fields))


def resolve_under_repo(p: str, *, repo: Optional[Path] = None) -> Path:
    repo = repo or repo_root()
    path = Path(str(p)).expanduser()
    if path.is_absolute():
        return path
    return (repo / path).resolve()


def finial_lock_path(out_dir: Path) -> Path:
    return Path(out_dir) / LOCK_FILENAME


def assert_out_dir_unlocked(out_dir: Path, *, force: bool = False) -> None:
    lock_path = finial_lock_path(out_dir)
    if lock_path.exists() and not bool(force):
        raise SystemExit(
            "\n".join(
                [
                    "⚠️ 输出目录已被 finial 锁定，拒绝覆盖写入。",
                    f"- out_dir: {str(Path(out_dir).resolve())}",
                    f"- lock: {str(lock_path.resolve())}",
                    "",
                    "如确需覆盖：请显式传入 `--force`，或先移除锁文件后再运行。",
                ]
            )
        )


def iter_existing(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if p.exists()]
