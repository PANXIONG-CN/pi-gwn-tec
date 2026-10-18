from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Optional


def _is_real_wandb(mod: Optional[ModuleType]) -> bool:
    if mod is None:
        return False
    return callable(getattr(mod, "init", None))


def _has_shadow_wandb_dir(path_entry: str) -> bool:
    """判断 sys.path 条目是否会把本地 `wandb/` 运行目录误当成 Python 包。"""
    base = path_entry or os.getcwd()
    try:
        base = os.path.abspath(base)
    except Exception:
        return False
    wdir = os.path.join(base, "wandb")
    if not os.path.isdir(wdir):
        return False
    # 若存在 __init__.py，说明它是一个真正的 Python 包（通常是 site-packages 里的 wandb）
    if os.path.isfile(os.path.join(wdir, "__init__.py")):
        return False
    return True


def import_wandb() -> Optional[ModuleType]:
    """稳健导入 `wandb` Python 包，避免 repo 本地 `wandb/` 目录遮蔽。

    本仓库会在 repo 根目录创建 `wandb/` 产物目录（WANDB_DIR 默认值），该目录通常没有 `__init__.py`，
    在 PEP-420 规则下会被 Python 视为 namespace package，从而把真正的 `wandb` 包“遮蔽”掉。
    这里会检测这种情况并在必要时临时移除遮蔽路径后重试导入。
    """
    try:
        import wandb as wb  # type: ignore
    except Exception:
        wb = None

    if _is_real_wandb(wb):
        return wb

    bad = [p for p in list(sys.path) if _has_shadow_wandb_dir(p)]
    if not bad:
        return None

    orig = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in bad]
        for k in list(sys.modules.keys()):
            if k == "wandb" or k.startswith("wandb."):
                sys.modules.pop(k, None)
        importlib.invalidate_caches()
        try:
            import wandb as wb2  # type: ignore
        except Exception:
            wb2 = None
        return wb2 if _is_real_wandb(wb2) else None
    finally:
        sys.path = orig

