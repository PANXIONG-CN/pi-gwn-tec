import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class WandbHousekeepingConfig:
    keep_runs: int
    sync_old_clean: bool
    sync_latest_offline: bool
    clean_force: bool
    delete_on_sync_fail: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _wandb_dir() -> Path:
    p = os.getenv('WANDB_DIR', '')
    if p:
        return Path(p).expanduser().resolve()
    return (_repo_root() / 'wandb').resolve()


def _load_cfg() -> WandbHousekeepingConfig:
    def _b(name: str, default: bool) -> bool:
        v = os.getenv(name, None)
        if v is None:
            return default
        return str(v).lower() in ('1', 'true', 'yes')

    try:
        keep = int(os.getenv('WANDB_KEEP_RUNS', '1'))
    except Exception:
        keep = 1
    keep = max(0, int(keep))

    return WandbHousekeepingConfig(
        keep_runs=keep,
        sync_old_clean=_b('WANDB_SYNC_OLD_CLEAN', True),
        sync_latest_offline=_b('WANDB_SYNC_LATEST_OFFLINE', True),
        clean_force=_b('WANDB_SYNC_CLEAN_FORCE', True),
        delete_on_sync_fail=_b('WANDB_DELETE_ON_SYNC_FAIL', False),
    )


def _list_run_dirs(wandb_dir: Path) -> List[Path]:
    if not wandb_dir.exists():
        return []
    out: List[Path] = []
    for p in wandb_dir.iterdir():
        try:
            if p.is_symlink():
                continue
            if not p.is_dir():
                continue
            name = p.name
            if name.startswith('run-') or name.startswith('offline-run-'):
                out.append(p)
        except Exception:
            continue
    try:
        out.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return out


def _wandb_sync(path: Path, clean: bool, clean_force: bool) -> Tuple[bool, str]:
    cmd = ['wandb', 'sync']
    # be explicit for robustness
    cmd += ['--include-online', '--include-offline']
    if clean:
        cmd.append('--clean')
        if clean_force:
            cmd.append('--clean-force')
    cmd.append(str(path))
    try:
        cp = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok = (cp.returncode == 0)
        return ok, (cp.stdout or '')
    except Exception as e:
        return False, str(e)


def _wandb_sync_retry(path: Path, clean: bool, clean_force: bool) -> Tuple[bool, str]:
    """Retry `wandb sync` to tolerate transient DNS/network failures."""
    try:
        retries = int(os.getenv("WANDB_SYNC_RETRIES", "5"))
    except Exception:
        retries = 5
    retries = max(1, int(retries))

    try:
        delay_s = float(os.getenv("WANDB_SYNC_RETRY_DELAY_S", "15"))
    except Exception:
        delay_s = 15.0
    delay_s = max(0.0, float(delay_s))

    last_out = ""
    for i in range(1, retries + 1):
        ok, out = _wandb_sync(path, clean=clean, clean_force=clean_force)
        last_out = out
        if ok:
            return True, out
        if i < retries and delay_s > 0:
            time.sleep(delay_s)
    return False, last_out


def run_housekeeping() -> None:
    """Keep only latest N W&B run dirs under WANDB_DIR.

    Policy:
    - Old run dirs: `wandb sync --clean` first, then delete if still exists.
    - Latest run dir: if WANDB_MODE=offline and enabled, `wandb sync` (no clean) to upload but keep local copy.
    """
    cfg = _load_cfg()
    wandb_dir = _wandb_dir()
    run_dirs = _list_run_dirs(wandb_dir)
    if not run_dirs:
        return

    keep_n = int(cfg.keep_runs)
    keep = run_dirs[:keep_n] if keep_n > 0 else []
    old = run_dirs[keep_n:] if keep_n >= 0 else []

    mode = str(os.getenv('WANDB_MODE', '')).lower()
    if keep and (mode == 'offline') and cfg.sync_latest_offline:
        _wandb_sync_retry(keep[0], clean=False, clean_force=cfg.clean_force)

    if not old:
        return

    if cfg.sync_old_clean:
        for d in old:
            is_offline = d.name.startswith('offline-run-')
            ok, _ = _wandb_sync(d, clean=True, clean_force=cfg.clean_force)
            # Online runs are already synced; delete even if sync command fails.
            # Offline runs are deleted only when sync succeeded (unless explicitly overridden).
            if (not ok) and is_offline and (not cfg.delete_on_sync_fail):
                continue
            if d.exists():
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
    else:
        for d in old:
            try:
                shutil.rmtree(d)
            except Exception:
                pass
