"""
GIMtec 邻接矩阵统一发现/构建/加载（与模型解耦）。

规则：
- 统一目录：data/adj/<DATASET>/
- 文件命名：<DATASET>_adj_<graph_tag>[ _<adj_model> ]_<YYYYMMDD_HHMMSS>.npy
- 优先级：优先按 graph_tag/adj_model 精确选取最新文件；若无，则尝试 data/<DATASET>/<DATASET>.npy；
  若仍无且 DATASET in {GIMtec, TEC, china_hires}，按请求的 graph_tag 构建并保存。
"""
from __future__ import annotations

import glob
import os
import re
import time
from typing import Tuple

import numpy as np

from lib.predifineGraph import constructGraph, get_adjacency_matrix

_TIMESTAMP_RE = re.compile(r"^(?P<body>.+)_(?P<ts>\d{8}_\d{6})$")
_RANDOM_SEED_RE = re.compile(r"(?:^|_)(?:seed)?(?P<seed>\d+)(?:$|_)")
_MAGNETIC_K_RE = re.compile(r"(?:^|_)knn(?P<k>\d+)(?:$|_)")

# Centered-dipole approximation used only for a magnetic-aware counterfactual.
# Constants are stable enough for a fixed protocol and avoid pulling in heavy geomagnetic deps.
_CDIP_NORTH_LAT_DEG = 80.65
_CDIP_NORTH_LON_DEG = -72.68


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _adj_store_dir(dataset: str) -> str:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    return os.path.join(_repo, 'data', 'adj', dataset)


def _legacy_adj_path(dataset: str) -> str:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    return os.path.join(_repo, 'data', dataset, f'{dataset}.npy')


def _grid_shape_for_dataset(dataset: str) -> Tuple[int, int] | None:
    ds = str(dataset).strip().lower()
    if ds in {'gimtec', 'tec'}:
        return 71, 73
    if ds == 'china_hires':
        return 41, 71
    return None


def _adj_body(graph_tag: str | None = None, adj_model: str | None = None) -> str | None:
    tag = str(graph_tag or '').strip()
    model = str(adj_model or '').strip()
    if not tag:
        return None
    return f"{tag}_{model}" if model else tag


def _parse_adj_body(dataset: str, path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = f"{dataset}_adj_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]
    m = _TIMESTAMP_RE.match(suffix)
    if not m:
        return None
    return m.group('body')


def _sorted_adj_candidates(dataset: str) -> list[str]:
    store = _adj_store_dir(dataset)
    if not os.path.isdir(store):
        return []
    return sorted(
        glob.glob(os.path.join(store, f"{dataset}_adj_*.npy")),
        key=lambda p: (os.path.getmtime(p), p),
    )


def latest_adj_path(dataset: str, graph_tag: str | None = None, adj_model: str | None = None) -> str | None:
    """返回 data/adj/<DATASET>/ 中最新的邻接文件；若提供 graph_tag/adj_model，则精确匹配。"""
    cands = _sorted_adj_candidates(dataset)
    if not cands:
        return None
    body = _adj_body(graph_tag=graph_tag, adj_model=adj_model)
    if body is None:
        return cands[-1]
    filtered = [p for p in cands if _parse_adj_body(dataset, p) == body]
    return filtered[-1] if filtered else None


def build_grid8_adj(H: int = 71, W: int = 73) -> np.ndarray:
    """71×73 网格 8 邻接（0/1）。"""
    return constructGraph(H, W).astype(np.float32)


def build_random_permutation_adj(H: int = 71, W: int = 73, seed: int = 3407) -> np.ndarray:
    """对 grid8 邻接做固定节点置换，保持节点/边数与度分布不变。"""
    base = build_grid8_adj(H, W)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(H * W))
    return base[np.ix_(perm, perm)].astype(np.float32, copy=False)


def _grid_lon_lat(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(87.5, -87.5, int(H), dtype=np.float32)
    lon = np.linspace(-180.0, 180.0, int(W), dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid.reshape(-1), lat_grid.reshape(-1)


def _sph_to_cart(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64).reshape(-1))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64).reshape(-1))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)


def _centered_dipole_basis() -> np.ndarray:
    axis = _sph_to_cart(
        np.asarray([_CDIP_NORTH_LAT_DEG], dtype=np.float64),
        np.asarray([_CDIP_NORTH_LON_DEG], dtype=np.float64),
    )[0]
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(ref, axis)
    if np.linalg.norm(x_axis) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(ref, axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.stack([x_axis, y_axis, axis], axis=0)


def build_magnetic_aware_adj(H: int = 71, W: int = 73, k: int = 8) -> np.ndarray:
    """Centered-dipole 近似下的对称 kNN 邻接，并保留自环语义以对齐现有 grid8。"""
    num_nodes = int(H * W)
    if num_nodes <= 1:
        return np.eye(num_nodes, dtype=np.float32)
    k = max(1, min(int(k), num_nodes - 1))
    lon_deg, lat_deg = _grid_lon_lat(H, W)
    geo_xyz = _sph_to_cart(lat_deg, lon_deg)
    basis = _centered_dipole_basis()
    mag_xyz = (geo_xyz @ basis.T).astype(np.float32, copy=False)
    similarity = mag_xyz @ mag_xyz.T
    np.fill_diagonal(similarity, -np.inf)
    nn_idx = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), k)
    cols = nn_idx.reshape(-1)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[rows, cols] = 1.0
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return adj


def _random_seed_from_adj_model(adj_model: str | None, default: int = 3407) -> int:
    model = str(adj_model or '').strip()
    if not model:
        return int(default)
    m = _RANDOM_SEED_RE.search(model)
    return int(m.group('seed')) if m else int(default)


def _magnetic_k_from_adj_model(adj_model: str | None, default: int = 8) -> int:
    model = str(adj_model or '').strip()
    if not model:
        return int(default)
    m = _MAGNETIC_K_RE.search(model)
    return int(m.group('k')) if m else int(default)


def _build_named_adj(dataset: str, graph_tag: str, adj_model: str | None = None) -> np.ndarray:
    shape = _grid_shape_for_dataset(dataset)
    if shape is None:
        raise ValueError(f"Unsupported dataset for managed adjacency: {dataset}")
    H, W = shape
    tag = str(graph_tag or 'grid8').strip().lower()
    if tag == 'grid8':
        return build_grid8_adj(H, W)
    if tag == 'random':
        seed = _random_seed_from_adj_model(adj_model)
        return build_random_permutation_adj(H, W, seed=seed)
    if tag == 'magnetic':
        k = _magnetic_k_from_adj_model(adj_model, default=8)
        return build_magnetic_aware_adj(H, W, k=k)
    raise ValueError(f"Unsupported graph_tag={graph_tag!r} for dataset={dataset!r}")


def _validate_adj_shape(A: np.ndarray, num_nodes: int, dataset: str, path: str) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    if A.shape != (int(num_nodes), int(num_nodes)):
        raise ValueError(
            f"Adjacency shape mismatch for {dataset}: got {A.shape}, "
            f"expected {(int(num_nodes), int(num_nodes))} from {path}"
        )
    return A


def save_adj(dataset: str, A: np.ndarray, graph_tag: str = 'grid8', adj_model: str | None = None) -> str:
    store = _adj_store_dir(dataset)
    _ensure_dir(store)
    ts = time.strftime('%Y%m%d_%H%M%S')
    if adj_model:
        fname = f"{dataset}_adj_{graph_tag}_{adj_model}_{ts}.npy"
    else:
        fname = f"{dataset}_adj_{graph_tag}_{ts}.npy"
    out = os.path.join(store, fname)
    np.save(out, A)
    return out


def load_or_build_adj(dataset: str, num_nodes: int, graph_tag: str = 'grid8', adj_model: str | None = None) -> Tuple[np.ndarray, str]:
    """统一入口：返回 (A, path)。

    优先按 graph_tag/adj_model 精确匹配 data/adj/<DATASET>/；
    若无则尝试 data/<DATASET>/<DATASET>.npy；
    若仍无且数据集受支持，则按 graph_tag 构建并保存到统一目录。
    """
    graph_tag = str(graph_tag or 'grid8').strip()
    adj_model = str(adj_model or '').strip()

    # 精确匹配统一目录文件，避免不同 counterfactual 串线。
    path_latest = latest_adj_path(dataset, graph_tag=graph_tag, adj_model=adj_model)
    if path_latest and os.path.exists(path_latest):
        A = _validate_adj_shape(np.load(path_latest), num_nodes, dataset, path_latest)
        return A, path_latest

    # 兼容旧路径 data/<DATASET>/<DATASET>.npy，仅对默认 grid8 语义放行。
    legacy = _legacy_adj_path(dataset)
    if os.path.exists(legacy):
        if graph_tag not in {'', 'grid8'} or adj_model:
            raise FileNotFoundError(
                f"No exact adjacency found for dataset={dataset}, graph_tag={graph_tag}, adj_model={adj_model}; "
                f"refusing to silently fall back to legacy {legacy}"
            )
        A, _ = get_adjacency_matrix(legacy, num_nodes)
        A = _validate_adj_shape(A, num_nodes, dataset, legacy)
        return A, legacy

    # 规则网格数据集按 graph_tag 构建并保存。
    shape = _grid_shape_for_dataset(dataset)
    if shape is not None:
        A = _build_named_adj(dataset, graph_tag=graph_tag or 'grid8', adj_model=adj_model)
        A = _validate_adj_shape(A, num_nodes, dataset, '<built>')
        out = save_adj(dataset, A, graph_tag=graph_tag or 'grid8', adj_model=adj_model or None)
        return A, out

    # 其他数据集不再静默返回零矩阵，避免实验误跑。
    raise FileNotFoundError(f"Unable to resolve adjacency for dataset={dataset}, graph_tag={graph_tag}, adj_model={adj_model}")
