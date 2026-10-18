import os
import io
import time
import json
import math
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from lib.wandb_import import import_wandb
    wandb = import_wandb()
except Exception:  # pragma: no cover
    wandb = None

# Public release: remote object-store uploads are disabled. The training path
# keeps all artifacts locally under outputs/.
requests = None

# --------------------------- helpers ---------------------------

DATASET_SLUGS = {
    'gimtec': 'gim',
    'tec': 'gim',
    'china_hires': 'china_hires',
    'metr_la': 'metrla',
    'nyc_taxi': 'nyc_taxi',
    'nyc_bike': 'nyc_bike',
    'pems08': 'pems08',
}


def _now_utc_ts() -> str:
    # UTC ISO-like without colons: %Y%m%dT%H%M%SZ
    import datetime as _dt
    return _dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def _now_bj_ts() -> str:
    # Beijing time ISO-like without colons: %Y%m%dT%H%M%S+0800
    import datetime as _dt
    tz = _dt.timezone(_dt.timedelta(hours=8))
    return _dt.datetime.now(tz).strftime('%Y%m%dT%H%M%S+0800')


def _safe_json_dump(obj, path):
    from pathlib import Path as _Path
    path = _Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def _to_builtin(obj):
    """Convert common objects into YAML/JSON-safe builtins."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    return str(obj)


def _safe_yaml_dump(obj, path: Path) -> bool:
    try:
        import yaml as _yaml
    except Exception:
        return False
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        _yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)
    return True


def dataset_slug(name: str) -> str:
    if not name:
        return 'unknown'
    key = name.strip().lower()
    return DATASET_SLUGS.get(key, key)


def wandb_run_name(args, *, run_tag: Optional[str] = None) -> str:
    """Build a human-readable W&B run.name aligned with outputs/<...>/<run_tag>/.

    Format (paper_slug omitted when empty):
      <dataset_slug>-<model_slug>-<mode>-seed<seed>-<paper_slug?>-<run_tag>

    Where model_slug follows outputs layout:
      - out_model_slug (if provided) > pretrain gptst > args.model
    """
    ds = dataset_slug(str(getattr(args, 'dataset', '') or ''))
    mode = str(getattr(args, 'mode', '') or '').strip().lower() or 'run'

    out_model = str(getattr(args, 'out_model_slug', '') or '').strip().lower()
    if out_model:
        model = out_model
    elif mode == 'pretrain':
        model = 'gptst'
    else:
        model = str(getattr(args, 'model', '') or 'model').strip().lower() or 'model'

    try:
        seed = int(getattr(args, 'seed', 0) or 0)
    except Exception:
        seed = 0

    paper_slug = str(getattr(args, 'paper_slug', '') or '').strip()

    tag = str(run_tag or getattr(args, 'run_tag', '') or '').strip()
    if not tag:
        try:
            log_dir = str(getattr(args, 'log_dir', '') or '').strip()
            if log_dir:
                tag = Path(log_dir).resolve().name
        except Exception:
            tag = ''
    if not tag:
        ts = str(getattr(args, 'ts_bj', '') or getattr(args, 'ts_utc', '') or '').strip()
        if not ts:
            try:
                ts = _now_bj_ts()
            except Exception:
                ts = ''
        rid8 = str(getattr(args, 'wandb_run_id', '') or 'local')[:8]
        tag = f"{ts}-{rid8}" if ts else rid8

    parts = [ds, model, mode, f"seed{seed}"]
    if paper_slug:
        parts.append(paper_slug)
    if tag:
        parts.append(tag)
    return "-".join([p for p in parts if str(p).strip()])


def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters()) if model is not None else 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0
    return int(total), int(trainable)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # shapes: [B,T,N,(D)] or broadcastable; compute over all elements
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    # filter NaNs
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float('nan')
    yt = yt[m]
    yp = yp[m]
    ss_res = np.sum((yt - yp) ** 2)
    yt_mean = np.mean(yt)
    ss_tot = np.sum((yt - yt_mean) ** 2)
    if ss_tot <= 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float('nan')
    yt = yt[m]
    yp = yp[m]
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = (np.linalg.norm(yt) * np.linalg.norm(yp))
    if denom == 0:
        return float('nan')
    return float(np.dot(yt, yp) / denom)


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float('nan'), float('nan')
    diff = (yp - yt)[m]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _compute_digest(arr: np.ndarray) -> str:
    # simple md5 of raw bytes for truth package
    h = hashlib.md5()
    h.update(arr.tobytes())
    return 'md5:' + h.hexdigest()


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode('utf-8', errors='ignore').strip()
    except Exception:
        return None


def _oss_bucket_from_env():
    """Object-store uploads are intentionally disabled in the public release."""
    return None, None, None


def _oss_uri(bucket_name: str, key: str) -> str:
    return f"oss://{bucket_name}/{key.lstrip('/')}"


def _oss_upload_file_with_stats(bucket, key: str, local_path: Path) -> Tuple[bool, Dict]:
    """Public release no-op for remote uploads."""
    return False, {'ok': False, 'error': 'remote upload disabled in public release'}


def _oss_upload_file(bucket, key: str, local_path: Path) -> bool:
    """Upload file to OSS using resumable upload, and verify size matches."""
    ok, _ = _oss_upload_file_with_stats(bucket, key, local_path)
    return bool(ok)


def _spawn_async_oss_uploader(repo_root: Path, out_dir: Path, tasks_path: Path) -> bool:
    """Public release no-op for detached remote uploads."""
    return False


# ------------------------ main entrypoints ---------------------

def prepare_outdir(args, run_tag: Optional[str] = None) -> Path:
    dataset_name = getattr(args, 'dataset', 'DATASET')
    ds_slug = dataset_slug(dataset_name)
    out_model = str(getattr(args, 'out_model_slug', '') or '').strip()
    model_slug_val = (out_model if out_model else getattr(args, 'model', 'model')).lower()
    seed = int(getattr(args, 'seed', 0))
    tag = run_tag or getattr(args, 'ts_bj', None) or _now_bj_ts()
    # Resolve outputs directory from repo root instead of CWD so that training
    # behaves the same whether launched from repo/ or repo/model/.
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent
    out_root_env = str(os.getenv('OUTPUTS_ROOT', 'outputs') or 'outputs').strip()
    out_root = Path(out_root_env)
    if not out_root.is_absolute():
        out_root = repo_root / out_root_env
    base = out_root / ds_slug / model_slug_val / f'seed_{seed}' / tag
    ensure_dir(base)
    # propagate back for the rest of the pipeline
    args.log_dir = str(base)
    # record ts_utc + run_tag for manifest use
    try:
        if not getattr(args, 'ts_bj', None):
            args.ts_bj = tag.split('-', 1)[0]
        if not getattr(args, 'ts_utc', None):
            args.ts_utc = _now_utc_ts()
        args.run_tag = tag
    except Exception:
        pass
    # best-effort dataset-level symlink (e.g., outputs/GIMtec -> outputs/gim)
    try:
        ds_upper = str(dataset_name)
        if ds_upper and ds_upper.lower() != ds_slug:
            legacy = out_root / ds_upper
            target = out_root / ds_slug
            if not legacy.exists():
                legacy.symlink_to(target, target_is_directory=True)
    except Exception:
        pass
    return base


def save_config_yaml(out_dir: Path, args, args_predictor=None, overwrite: bool = False) -> Optional[str]:
    """Write effective config snapshot to outputs/<...>/config.yaml."""
    try:
        out_dir = Path(out_dir)
        path = out_dir / 'config.yaml'
        if path.exists() and (not overwrite):
            return str(path)
        repo_root = Path(__file__).resolve().parent.parent
        cfg = {
            'schema_version': 1,
            'meta': {
                'repo_root': str(repo_root),
                'code_entry': 'model/Run.py',
                'git_commit': _git_commit(repo_root),
                'cmdline': str(getattr(args, 'cmdline', '')),
                'dataset': str(getattr(args, 'dataset', '')),
                'model': str(getattr(args, 'model', '')),
                'mode': str(getattr(args, 'mode', '')),
                'seed': int(getattr(args, 'seed', 0)),
                'ts_utc': str(getattr(args, 'ts_utc', '')),
                'ts_bj': str(getattr(args, 'ts_bj', '')),
                'run_tag': str(getattr(args, 'run_tag', '')),
                'wandb_run_id': str(getattr(args, 'wandb_run_id', '')) if getattr(args, 'wandb_run_id', None) else None,
            },
            'args': {k: _to_builtin(getattr(args, k)) for k in sorted(vars(args).keys()) if not k.startswith('_')},
        }
        if args_predictor is not None:
            try:
                cfg['predictor_args'] = {k: _to_builtin(getattr(args_predictor, k))
                                         for k in sorted(vars(args_predictor).keys()) if not k.startswith('_')}
            except Exception:
                cfg['predictor_args'] = _to_builtin(vars(args_predictor))
        ok = _safe_yaml_dump(cfg, path)
        return str(path) if ok else None
    except Exception:
        return None


def _build_truth_if_needed(args, out_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[Dict]]:
    """For GIMtec/TEC only: ensure a centralized truth object exists on OSS and return:

    (truth_ref_rel, truth_abspath, truth_oss_uri, truth_digest, upload_stats)

    truth_ref_rel points to a small local JSON pointer under repo_root/coordinates/truth/.
    """
    try:
        ds_key = str(getattr(args, 'dataset', '')).lower()
        if ds_key not in ['gimtec', 'tec']:
            return None, None, None, None, None
        # construct protocol key
        lag = int(getattr(args, 'lag', 12))
        horizon = int(getattr(args, 'horizon', 12))
        stride = horizon if bool(getattr(args, 'stride_horizon', False)) else 1
        prefix_from = 'lastyear' if bool(getattr(args, 'prefix_boundary', True)) else 'none'
        this_file = Path(__file__).resolve()
        repo_root = this_file.parent.parent
        truth_root = repo_root / 'coordinates' / 'truth'
        ensure_dir(truth_root)
        base = f"gimtec__lag{lag}_hor{horizon}_stride{stride}_prefix-{prefix_from}"
        ptr_path = truth_root / f"{base}.json"
        truth_ref_rel = str(ptr_path.relative_to(repo_root))
        truth_abspath = str(ptr_path.resolve())
        upload_stats = None
        desired_protocol = {
            'lag': lag,
            'horizon': horizon,
            'stride': stride,
            'prefix_boundary': bool(getattr(args, 'prefix_boundary', True)),
            'year_split': bool(getattr(args, 'year_split', False)),
            'interval_min': int(getattr(args, 'interval', 120)),
            'normalizer': str(getattr(args, 'normalizer', '')),
        }

        bucket, bucket_name, endpoint = _oss_bucket_from_env()
        if bucket is None or not bucket_name:
            # OSS not configured: fall back to local-only pointer (no truth payload)
            if not ptr_path.exists():
                _safe_json_dump(
                    {
                        'schema_version': 1,
                        'dataset': getattr(args, 'dataset', ''),
                        'protocol': {
                            'lag': lag,
                            'horizon': horizon,
                            'stride': stride,
                            'prefix_boundary': bool(getattr(args, 'prefix_boundary', True)),
                            'year_split': bool(getattr(args, 'year_split', False)),
                            'interval_min': int(getattr(args, 'interval', 120)),
                            'normalizer': str(getattr(args, 'normalizer', '')),
                        },
                        'oss': None,
                        'digest': None,
                        'note': 'OSS not configured; truth payload not built.',
                    },
                    ptr_path,
                )
            try:
                meta = json.load(open(ptr_path, 'r', encoding='utf-8'))
            except Exception:
                meta = {}
            return truth_ref_rel, truth_abspath, None, meta.get('digest'), None

        # Truth payload format on OSS: compressed NPZ (contains y_true.npy)
        truth_key = f"pi-gwn-tec/coordinates/truth/{base}.npz"
        truth_oss_uri = _oss_uri(bucket_name, truth_key)

        # If pointer already exists, trust it unless format mismatched (avoid repeated expensive truth builds)
        if ptr_path.exists():
            try:
                meta = json.load(open(ptr_path, 'r', encoding='utf-8'))
                oss_meta = meta.get('oss') or {}
                # Retry upload if last attempt left a tmp payload behind.
                tmp_path = meta.get('tmp_path')
                if tmp_path:
                    tp = Path(str(tmp_path))
                    if tp.exists():
                        try:
                            ok, upload_stats = _oss_upload_file_with_stats(bucket, truth_key, tp)
                        except Exception:
                            ok, upload_stats = False, None
                        if ok:
                            meta.update({
                                'schema_version': int(meta.get('schema_version', 1)),
                                'dataset': getattr(args, 'dataset', ''),
                                'format': 'npz',
                                'npz_key': 'y_true',
                                'protocol': {
                                    'lag': lag,
                                    'horizon': horizon,
                                    'stride': stride,
                                    'prefix_boundary': bool(getattr(args, 'prefix_boundary', True)),
                                    'year_split': bool(getattr(args, 'year_split', False)),
                                    'interval_min': int(getattr(args, 'interval', 120)),
                                    'normalizer': str(getattr(args, 'normalizer', '')),
                                },
                                'oss': {
                                    'bucket': bucket_name,
                                    'endpoint': endpoint,
                                    'key': truth_key,
                                    'uri': truth_oss_uri,
                                    'size_bytes': int(tp.stat().st_size) if tp.exists() else None,
                                },
                            })
                            meta.pop('note', None)
                            meta.pop('tmp_path', None)
                            _safe_json_dump(meta, ptr_path)
                            try:
                                tp.unlink()
                            except Exception:
                                pass
                            return truth_ref_rel, truth_abspath, truth_oss_uri, meta.get('digest'), upload_stats
                # If pointer already matches our desired NPZ key, do nothing.
                proto = meta.get('protocol') or {}
                proto_ok = all(proto.get(k) == v for k, v in desired_protocol.items())
                if meta.get('format') == 'npz' and oss_meta.get('key') == truth_key and meta.get('digest') and proto_ok:
                    # Keep pointer consistent with current URI scheme (oss://) without rebuilding payload.
                    try:
                        if isinstance(oss_meta, dict) and oss_meta.get('uri') != truth_oss_uri:
                            oss_meta = dict(oss_meta)
                            oss_meta['uri'] = truth_oss_uri
                            meta['oss'] = oss_meta
                            _safe_json_dump(meta, ptr_path)
                    except Exception:
                        pass
                    return truth_ref_rel, truth_abspath, truth_oss_uri, meta.get('digest'), upload_stats
            except Exception:
                pass

        # Build truth once (streaming) -> upload OSS -> delete local payload -> write pointer JSON
        from lib.dataloader import get_dataloader
        old_debug = getattr(args, 'debug', False)
        try:
            setattr(args, 'debug', False)
        except Exception:
            pass
        try:
            _, _, test_loader, scaler_data, *_ = get_dataloader(
                args,
                normalizer=getattr(args, 'normalizer', 'tec01'),
                single=False,
            )
        finally:
            try:
                setattr(args, 'debug', old_debug)
            except Exception:
                pass
        if test_loader is None:
            return truth_ref_rel, truth_abspath, truth_oss_uri, None, None
        try:
            total = int(len(getattr(test_loader, 'dataset', [])))
        except Exception:
            total = 0
        if total <= 0:
            return truth_ref_rel, truth_abspath, truth_oss_uri, None, None

        # Stream-write NPZ (zip of y_true.npy) to avoid a 5GB local .npy
        import zipfile
        from numpy.lib.format import write_array_header_1_0, dtype_to_descr

        try:
            compresslevel = int(os.getenv('TRUTH_NPZ_COMPRESSLEVEL', '3'))
        except Exception:
            compresslevel = 3
        compresslevel = int(min(9, max(0, compresslevel)))

        tmp_path = truth_root / f"{base}.tmp.npz"
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        h_md5 = hashlib.md5()
        cursor = 0
        shape = None
        dtype = None
        try:
            with zipfile.ZipFile(
                tmp_path,
                mode='w',
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=compresslevel,
                allowZip64=True,
            ) as zf:
                # y_true.npy is >4GB uncompressed; force ZIP64 for this entry.
                with zf.open('y_true.npy', 'w', force_zip64=True) as f:
                    header_written = False
                    for batch in test_loader:
                        if isinstance(batch, (list, tuple)):
                            _, target, _ = batch if len(batch) == 3 else (batch[0], batch[1], None)
                        else:
                            _, target = batch
                        yb = target[..., :1].detach().cpu().numpy()
                        # inverse transform to TECU domain
                        try:
                            yb = scaler_data.inverse_transform(yb)
                        except Exception:
                            pass
                        yb = np.asarray(yb)
                        if not header_written:
                            dtype = yb.dtype
                            shape = (total,) + yb.shape[1:]
                            header = {
                                'descr': dtype_to_descr(np.dtype(dtype)),
                                'fortran_order': False,
                                'shape': shape,
                            }
                            write_array_header_1_0(f, header)
                            header_written = True
                        bsz = int(yb.shape[0])
                        if cursor + bsz > total:
                            yb = yb[: max(0, total - cursor)]
                            bsz = int(yb.shape[0])
                        if bsz <= 0:
                            break
                        yb_c = np.ascontiguousarray(yb)
                        raw = yb_c.tobytes(order='C')
                        f.write(raw)
                        h_md5.update(raw)
                        cursor += bsz
                        if cursor >= total:
                            break
            if cursor != total:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return truth_ref_rel, truth_abspath, truth_oss_uri, None, None
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return truth_ref_rel, truth_abspath, truth_oss_uri, None, None

        digest = 'md5:' + h_md5.hexdigest()

        # Upload + verify
        ok, upload_stats = _oss_upload_file_with_stats(bucket, truth_key, tmp_path)
        if ok:
            meta = {
                'schema_version': 2,
                'dataset': getattr(args, 'dataset', ''),
                'format': 'npz',
                'npz_key': 'y_true',
                'protocol': desired_protocol,
                'shape': [int(x) for x in (shape or ())],
                'dtype': str(dtype),
                'digest': digest,
                'oss': {
                    'bucket': bucket_name,
                    'endpoint': endpoint,
                    'key': truth_key,
                    'uri': truth_oss_uri,
                    'size_bytes': int(tmp_path.stat().st_size) if tmp_path.exists() else None,
                },
            }
            _safe_json_dump(meta, ptr_path)
            try:
                tmp_path.unlink()
            except Exception:
                pass
        else:
            # Keep local tmp file if upload failed (user can retry manually)
            meta = {
                'schema_version': 2,
                'dataset': getattr(args, 'dataset', ''),
                'format': 'npz',
                'npz_key': 'y_true',
                'digest': digest,
                'oss': {
                    'bucket': bucket_name,
                    'endpoint': endpoint,
                    'key': truth_key,
                    'uri': truth_oss_uri,
                },
                'note': 'Truth build completed but OSS upload failed; tmp.npz retained.',
                'tmp_path': str(tmp_path),
            }
            _safe_json_dump(meta, ptr_path)
        return truth_ref_rel, truth_abspath, truth_oss_uri, digest, upload_stats
    except Exception:
        return None, None, None, None, None


def _collect_env() -> Dict:
    env = {}
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda'] = getattr(torch.version, 'cuda', None)
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['gpu_name'] = torch.cuda.get_device_name(0)
            env['capability'] = torch.cuda.get_device_capability(0)
    except Exception:
        pass
    try:
        import numpy
        env['numpy'] = numpy.__version__
    except Exception:
        pass
    try:
        import sys
        env['python'] = sys.version
    except Exception:
        pass
    return env


def _save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_stepwise_csv(path: Path, rows: List[Dict[str, object]]):
    # columns: seed,dataset,model_slug,horizon,step,rmse,(optional)split
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['seed', 'dataset', 'model_slug', 'horizon', 'step', 'rmse']
    try:
        if any(('split' in r) for r in rows):
            cols.append('split')
    except Exception:
        pass
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def _wandb_log_artifact(run, file_path: Path, atype: str = 'file', aliases: Optional[List[str]] = None):
    if wandb is None or run is None:
        return
    if not file_path.exists():
        return
    try:
        art = wandb.Artifact(name=f"{file_path.stem}-{run.id}", type=atype)
        art.add_file(str(file_path))
        run.log_artifact(art, aliases=aliases or ['latest'])
    except Exception:
        pass


def _infer_years_split(args) -> Optional[Dict[str, List[int]]]:
    ds = str(getattr(args, 'dataset', '')).lower()
    if ds in ['gimtec', 'tec']:
        # Matches lib/datasets/gimtec_pretrain.py
        return {
            'train': [2009, 2010, 2011, 2012, 2014, 2016, 2018],
            'val': [2013, 2017],
            'test': [2015, 2019, 2020, 2021, 2022],
        }
    if ds == 'china_hires':
        profile = str(getattr(args, 'split_profile', 'default') or 'default').strip().lower()
        if profile in ('d', 'batchd', 'd-split'):
            # Matches lib/datasets/china_hires.py (Batch D split_profile=D)
            return {
                'train': [2014, 2016, 2018, 2020],
                'val': [2017, 2021],
                'test': [2015],
                'ood': [2019, 2022],
            }
        # Matches lib/datasets/china_hires.py (default split)
        return {
            'train': [2014, 2016, 2018, 2020, 2022, 2023],
            'val': [2017, 2021],
            'test': [2015],
            'ood': [2019, 2024],
        }
    return None


def _infer_grid_shape(args) -> Optional[List[int]]:
    ds = str(getattr(args, 'dataset', '')).lower()
    if ds in ['gimtec', 'tec']:
        return [71, 73]
    h = getattr(args, 'height', None)
    w = getattr(args, 'width', None)
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return [h, w]
    return None


def _parse_split_metrics_from_log(log_path: Path) -> Dict[str, Dict[str, object]]:
    """Parse BasicTrainer logs to derive metrics per tag (e.g., test/ood/test_in_domain).

    Expected lines:
      [test] Horizon 01, MAE: 1.46, RMSE: 2.30, R2:0.9657, CORR:0.9755 (valid=100.0%)
      [test] Average Horizon, MAE: 1.89, RMSE: 2.95, R2:0.9433, CORR:0.9564
    """
    import re
    if not log_path.exists():
        return {}
    per_step: Dict[str, Dict[int, Dict[str, object]]] = {}
    overall: Dict[str, Dict[str, object]] = {}
    # Regex is intentionally permissive on whitespace.
    r_step = re.compile(
        r"\[([^\]]+)\]\s+Horizon\s+(\d+),\s+MAE:\s+([0-9.]+),\s+RMSE:\s+([0-9.]+),\s+R2:([\-0-9.]+),\s+CORR:([\-0-9.]+)"
    )
    r_avg = re.compile(
        r"\[([^\]]+)\]\s+Average\s+Horizon,\s+MAE:\s+([0-9.]+),\s+RMSE:\s+([0-9.]+),\s+R2:([\-0-9.]+),\s+CORR:([\-0-9.]+)"
    )
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = r_step.search(line)
            if m:
                tag = str(m.group(1)).strip().lower()
                step = int(m.group(2))
                per_step.setdefault(tag, {})[step] = {
                    'step': step,
                    'MAE': float(m.group(3)),
                    'RMSE': float(m.group(4)),
                    'R2': float(m.group(5)),
                    'CORR': float(m.group(6)),
                }
                continue
            m = r_avg.search(line)
            if m:
                tag = str(m.group(1)).strip().lower()
                overall[tag] = {
                    'MAE': float(m.group(2)),
                    'RMSE': float(m.group(3)),
                    'R2': float(m.group(4)),
                    'CORR': float(m.group(5)),
                }
    out: Dict[str, Dict[str, object]] = {}
    keys = sorted(set(list(per_step.keys()) + list(overall.keys())))
    for k in keys:
        ps = per_step.get(k) or {}
        ps_list = [ps[i] for i in sorted(ps.keys())] if ps else None
        out[k] = {
            'overall': overall.get(k),
            'per_step': ps_list,
        }
    return out


def _parse_test_metrics_from_log(log_path: Path) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
    """Backward-compatible: return only [test] overall + per_step metrics."""
    try:
        d = _parse_split_metrics_from_log(log_path)
        t = d.get('test') or {}
        return t.get('overall'), t.get('per_step')
    except Exception:
        return None, None


def post_run_collect_and_upload(run, args, model, out_dir: Path) -> None:
    """Unify final result saving and small-file uploading.

    - Create metrics.json, stepwise_rmse.csv, compute_cost.json, manifest.json if missing.
    - Upload small files to W&B (if online).
    - Prepare OSS hints in W&B summary (actual OSS upload handled by shell script).
    """
    out_dir = Path(out_dir)
    # Truth package (GIMtec only, auto-build if missing)
    truth_ref_rel, truth_ref_abs, truth_oss_uri, truth_digest, truth_upload = _build_truth_if_needed(args, out_dir)
    oss_uploads: List[Dict] = []
    if truth_upload:
        try:
            truth_upload = dict(truth_upload)
            truth_upload['name'] = 'truth'
            truth_upload['uri'] = truth_oss_uri
            oss_uploads.append(truth_upload)
        except Exception:
            pass

    # Metrics and arrays paths (may be produced already by Trainer)
    preds_files = list(out_dir.glob("*_test_preds.npy"))
    # choose latest by mtime to avoid picking stale leftovers
    src_preds = None
    if preds_files:
        try:
            preds_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass
        src_preds = preds_files[0]
    preds_path = out_dir / 'predictions.npy'

    ood_files = list(out_dir.glob("*_ood_preds.npy"))
    src_ood_preds = None
    if ood_files:
        try:
            ood_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass
        src_ood_preds = ood_files[0]
    ood_path = out_dir / 'ood_predictions.npy'

    ds_slug = dataset_slug(getattr(args, 'dataset', ''))

    # Optional: upload big predictions to OSS then delete local to save disk
    preds_oss_uri = None
    preds_async_scheduled = False
    ood_oss_uri = None
    best_oss_uri = None
    best_ckpt_local = None
    async_on = False
    direct_on = False
    thr_mb = 100
    # China paper/QC runs: enforce sync + always upload predictions when direct OSS upload is enabled.
    try:
        if str(ds_slug) == 'china_hires':
            async_on = False
            if direct_on:
                thr_mb = 0
    except Exception:
        pass
    try:
        if src_preds is not None and src_preds.exists():
            size_mb = float(src_preds.stat().st_size) / (1024.0 * 1024.0)
            if direct_on and size_mb >= float(thr_mb):
                bucket, bucket_name, _ = _oss_bucket_from_env()
                if bucket is not None and bucket_name:
                    md_slug = str(getattr(args, 'model', 'model')).lower()
                    seed = int(getattr(args, 'seed', 0))
                    run_tag = getattr(args, 'run_tag', '') or getattr(args, 'ts_bj', '') or _now_bj_ts()
                    prefix = f"pi-gwn-tec/{ds_slug}/{md_slug}/seed_{seed}/{run_tag}/"
                    key = prefix + 'predictions.npy'
                    preds_uri = _oss_uri(bucket_name, key)
                    if async_on:
                        preds_oss_uri = preds_uri
                        # Schedule background upload; keep local file until upload succeeds.
                        try:
                            repo_root = Path(__file__).resolve().parent.parent
                            tasks = {
                                'schema_version': 1,
                                'created_ts_bj': _now_bj_ts(),
                                'created_ts_utc': _now_utc_ts(),
                                'tasks': [{
                                    'name': 'predictions',
                                    'out_dir': str(out_dir),
                                    'local_path': str(src_preds),
                                    'key': key,
                                    # Prefer args.wandb_run_id; fallback to run_tag suffix if possible.
                                    'wandb_run_id': getattr(args, 'wandb_run_id', None)
                                                   or (str(run_tag).split('-', 1)[1] if '-' in str(run_tag) else None),
                                }],
                            }
                            tasks_path = out_dir / f"oss_upload_tasks_{_now_bj_ts()}.json"
                            _safe_json_dump(tasks, tasks_path)
                            _spawn_async_oss_uploader(repo_root, out_dir, tasks_path)
                            preds_async_scheduled = True
                            # Mark pending uri for the manifest; final .ossref is written by the uploader on success.
                            try:
                                (out_dir / 'predictions.ossref.pending').write_text(preds_uri + "\n", encoding='utf-8')
                            except Exception:
                                pass
                            # Add a pending record to compute_cost.json; uploader will append a success record later.
                            try:
                                oss_uploads.append({
                                    'ok': False,
                                    'pending': True,
                                    'name': 'predictions',
                                    'key': key,
                                    'uri': preds_uri,
                                    'local_path': str(src_preds),
                                    'bytes': int(src_preds.stat().st_size),
                                    'seconds': None,
                                    'mib_per_sec': None,
                                    'attempts': 0,
                                    'errors': [],
                                    'ts_bj': _now_bj_ts(),
                                    'ts_utc': _now_utc_ts(),
                                })
                            except Exception:
                                pass
                        except Exception:
                            preds_oss_uri = None
                    else:
                        ok, up = _oss_upload_file_with_stats(bucket, key, src_preds)
                        try:
                            up = dict(up)
                            up['name'] = 'predictions'
                            up['uri'] = preds_uri
                            oss_uploads.append(up)
                        except Exception:
                            pass
                        if ok:
                            preds_oss_uri = preds_uri
                            # write pointer for local traceability
                            ptr = out_dir / 'predictions.ossref'
                            try:
                                ptr.write_text(preds_uri + "\n", encoding='utf-8')
                            except Exception:
                                pass
                            try:
                                src_preds.unlink()
                            except Exception:
                                pass
    except Exception:
        preds_oss_uri = None

    # Optional: upload big OOD predictions to OSS then delete local to save disk
    try:
        if src_ood_preds is not None and src_ood_preds.exists():
            size_mb = float(src_ood_preds.stat().st_size) / (1024.0 * 1024.0)
            if direct_on and size_mb >= float(thr_mb):
                bucket, bucket_name, _ = _oss_bucket_from_env()
                if bucket is not None and bucket_name:
                    md_slug = str(getattr(args, 'model', 'model')).lower()
                    seed = int(getattr(args, 'seed', 0))
                    run_tag = getattr(args, 'run_tag', '') or getattr(args, 'ts_bj', '') or _now_bj_ts()
                    prefix = f"pi-gwn-tec/{ds_slug}/{md_slug}/seed_{seed}/{run_tag}/"
                    key = prefix + 'ood_predictions.npy'
                    ood_uri = _oss_uri(bucket_name, key)
                    ok, up = _oss_upload_file_with_stats(bucket, key, src_ood_preds)
                    try:
                        up = dict(up)
                        up['name'] = 'ood_predictions'
                        up['uri'] = ood_uri
                        oss_uploads.append(up)
                    except Exception:
                        pass
                    if ok:
                        ood_oss_uri = ood_uri
                        ptr = out_dir / 'ood_predictions.ossref'
                        try:
                            ptr.write_text(ood_uri + "\n", encoding='utf-8')
                        except Exception:
                            pass
                        try:
                            src_ood_preds.unlink()
                        except Exception:
                            pass
    except Exception:
        ood_oss_uri = None

    # Best checkpoint: upload to OSS (optional) and write best_model.ossref
    try:
        # Prefer existing best_model.ossref if present
        ptr_best = out_dir / 'best_model.ossref'
        if ptr_best.exists():
            try:
                best_oss_uri = ptr_best.read_text(encoding='utf-8', errors='ignore').strip() or None
            except Exception:
                best_oss_uri = None

        # Determine local best checkpoint path (prefer symlink best_model.pth; fallback to latest *.pth)
        best_link = out_dir / 'best_model.pth'
        if best_link.exists():
            best_ckpt_local = best_link
        else:
            pths = list(out_dir.glob("*.pth"))
            if pths:
                try:
                    pths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                except Exception:
                    pass
                best_ckpt_local = pths[0]
                # Create a stable local symlink for convenience
                try:
                    best_link.symlink_to(best_ckpt_local.name)
                    best_ckpt_local = best_link
                except Exception:
                    pass

        # Upload checkpoint when direct OSS upload is enabled
        write_ref = False
        if direct_on and write_ref and (best_ckpt_local is not None) and best_ckpt_local.exists():
            bucket, bucket_name, _ = _oss_bucket_from_env()
            if bucket is not None and bucket_name:
                ds_slug = dataset_slug(getattr(args, 'dataset', ''))
                md_slug = str(getattr(args, 'model', 'model')).lower()
                seed = int(getattr(args, 'seed', 0))
                run_tag = getattr(args, 'run_tag', '') or getattr(args, 'ts_bj', '') or _now_bj_ts()
                prefix = f"pi-gwn-tec/{ds_slug}/{md_slug}/seed_{seed}/{run_tag}/"
                key = prefix + 'best_model.pth'
                # Skip upload if we already have a usable pointer
                if not best_oss_uri:
                    ok, up = _oss_upload_file_with_stats(bucket, key, best_ckpt_local)
                    try:
                        up = dict(up)
                        up['name'] = 'best_model'
                        up['uri'] = _oss_uri(bucket_name, key)
                        oss_uploads.append(up)
                    except Exception:
                        pass
                    if ok:
                        best_oss_uri = _oss_uri(bucket_name, key)
                        try:
                            ptr_best.write_text(best_oss_uri + "\n", encoding='utf-8')
                        except Exception:
                            pass
    except Exception:
        best_oss_uri = best_oss_uri or None

    # If local predictions still exist, prefer a symlink alias instead of copy
    try:
        if (not preds_async_scheduled) and (not preds_path.exists()) and (src_preds is not None) and src_preds.exists():
            preds_path.symlink_to(src_preds.name)
    except Exception:
        pass
    try:
        if (not ood_path.exists()) and (src_ood_preds is not None) and src_ood_preds.exists():
            ood_path.symlink_to(src_ood_preds.name)
    except Exception:
        pass

    # Prepare metrics from arrays if available
    metrics_path = out_dir / 'metrics.json'
    stepwise_path = out_dir / 'stepwise_rmse.csv'
    # Prefer parsing Trainer logs; avoids requiring ground-truth arrays locally.
    # Always refresh (supports repeated -mode test runs into the same log_dir).
    try:
        # Exclude auxiliary logs (e.g., async uploader) and pick the first file that actually contains [test] lines.
        logs = list(out_dir.glob("*_test.log"))
        if not logs:
            logs = [p for p in out_dir.glob("*.log") if not p.name.startswith('oss_')]
        try:
            logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass

        overall, per_step = None, None
        splits = None
        log_path = None
        for p in logs:
            d = _parse_split_metrics_from_log(p)
            if not d:
                continue
            # Keep only paper-facing evaluation tags; ignore val/train noise.
            d = {k: v for k, v in d.items()
                 if (k == 'test') or k.startswith('test_') or (k == 'ood') or k.startswith('ood_')}
            if not d:
                continue
            splits = d
            t = d.get('test') or {}
            overall = t.get('overall')
            per_step = t.get('per_step')
            log_path = p
            break
        if splits:
            multi = any(k != 'test' for k in splits.keys())
            payload = {
                'overall': overall,
                'per_step': per_step,
                'source': {'type': 'log', 'path': str(log_path) if log_path else None},
            }
            if multi:
                payload = {
                    'schema_version': 2,
                    'splits': splits,
                    **payload,
                }
            _safe_json_dump(payload, metrics_path)

            # stepwise_rmse.csv
            ds_slug = dataset_slug(getattr(args, 'dataset', ''))
            model_slug = str(getattr(args, 'model', 'model')).lower()
            seed = int(getattr(args, 'seed', 0))
            H = int(getattr(args, 'horizon', 0)) or 0
            rows = []
            for split_name, sm in splits.items():
                ps = (sm or {}).get('per_step') if isinstance(sm, dict) else None
                if not ps:
                    continue
                h_eff = int(H) or int(max([pp.get('step', 0) for pp in ps]))
                for pp in ps:
                    r = {
                        'seed': seed,
                        'dataset': ds_slug,
                        'model_slug': model_slug,
                        'horizon': int(h_eff),
                        'step': int(pp.get('step', 0)),
                        'rmse': round(float(pp.get('RMSE', float('nan'))), 6),
                    }
                    if multi:
                        r['split'] = str(split_name)
                    rows.append(r)
            if rows:
                _save_stepwise_csv(stepwise_path, rows)
    except Exception:
        pass

    # compute_cost.json
    cc_path = out_dir / 'compute_cost.json'
    cc = {}
    if cc_path.exists():
        try:
            cc = json.load(open(cc_path, 'r', encoding='utf-8'))
        except Exception:
            cc = {}
    if not cc:
        cc = {
            'start_ts': getattr(args, '_run_start_ts', None),
            'start_ts_bj': getattr(args, 'ts_bj', None),
            'start_ts_utc': getattr(args, 'ts_utc', None),
            'end_ts': _now_bj_ts(),
            'end_ts_bj': _now_bj_ts(),
            'end_ts_utc': _now_utc_ts(),
            'total_seconds': None,
            'epochs': int(getattr(args, 'epochs', 0)),
            'steps': int(getattr(args, 'batch_seen', 0)) if hasattr(args, 'batch_seen') else None,
            'throughput_samples_per_sec': None,
            'gpu_name': None,
            'max_memory_train_bytes': None,
            'max_memory_infer_bytes': None,
            'amp_mode': 'bf16' if bool(getattr(args, 'amp', False)) else 'none',
            'total_params': None,
            'trainable_params': None,
        }
        # params
        try:
            total, trainable = count_params(model)
            cc['total_params'] = total
            cc['trainable_params'] = trainable
        except Exception:
            pass
        # env
        env = _collect_env()
        if 'gpu_name' in env:
            cc['gpu_name'] = env['gpu_name']
        # timing estimates if recorded on args
        if hasattr(args, '_train_start_time') and hasattr(args, '_train_end_time'):
            cc['total_seconds'] = float(args._train_end_time - args._train_start_time)

    # Enrich with standardized compute-cost fields (paper-facing)
    try:
        cc.setdefault('memory_unit', 'MiB')
        dl_cfg = getattr(args, '_dataloader', None)
        if dl_cfg is not None:
            cc['dataloader'] = _to_builtin(dl_cfg)
        proto = getattr(args, '_compute_cost_protocol', None)
        if isinstance(proto, dict):
            cc['compute_cost_protocol'] = _to_builtin(proto)
        # training wall time (includes dataloader + val; excludes final test/upload)
        if hasattr(args, '_train_wall_time_s'):
            cc['train_wall_time_s'] = float(getattr(args, '_train_wall_time_s'))
        if hasattr(args, '_train_time_per_epoch_s'):
            cc['train_time_per_epoch_s'] = float(getattr(args, '_train_time_per_epoch_s'))
        if hasattr(args, '_train_epochs_ran'):
            cc['epochs_ran'] = int(getattr(args, '_train_epochs_ran'))
        if hasattr(args, '_best_epoch') and getattr(args, '_best_epoch') is not None:
            cc['best_epoch'] = int(getattr(args, '_best_epoch'))
        if hasattr(args, '_early_stop_epoch') and getattr(args, '_early_stop_epoch') is not None:
            cc['early_stop_epoch'] = int(getattr(args, '_early_stop_epoch'))
        if hasattr(args, '_best_loss') and getattr(args, '_best_loss') is not None:
            cc['best_loss'] = float(getattr(args, '_best_loss'))
        # peak memory (MiB)
        if hasattr(args, '_train_max_memory_allocated_MiB'):
            cc['max_memory_train_allocated_MiB'] = float(getattr(args, '_train_max_memory_allocated_MiB'))
        if hasattr(args, '_train_max_memory_reserved_MiB'):
            cc['max_memory_train_reserved_MiB'] = float(getattr(args, '_train_max_memory_reserved_MiB'))
        if hasattr(args, '_infer_max_memory_allocated_MiB'):
            cc['max_memory_infer_allocated_MiB'] = float(getattr(args, '_infer_max_memory_allocated_MiB'))
        if hasattr(args, '_infer_max_memory_reserved_MiB'):
            cc['max_memory_infer_reserved_MiB'] = float(getattr(args, '_infer_max_memory_reserved_MiB'))
        # inference latency (pure forward, ms)
        if hasattr(args, '_infer_latency_ms'):
            cc['inference_latency_ms'] = float(getattr(args, '_infer_latency_ms'))
        if hasattr(args, '_infer_latency_p50_ms'):
            cc['inference_latency_p50_ms'] = float(getattr(args, '_infer_latency_p50_ms'))
        if hasattr(args, '_infer_latency_p95_ms'):
            cc['inference_latency_p95_ms'] = float(getattr(args, '_infer_latency_p95_ms'))
        if hasattr(args, '_infer_input_shape'):
            cc['infer_input_shape'] = _to_builtin(getattr(args, '_infer_input_shape'))
    except Exception:
        pass

    # Always update end timestamp and upload metrics (supports re-test without training).
    cc['end_ts'] = _now_bj_ts()
    cc['end_ts_bj'] = _now_bj_ts()
    cc['end_ts_utc'] = _now_utc_ts()
    # If start timestamps were not filled previously, patch them in.
    cc.setdefault('start_ts_bj', getattr(args, 'ts_bj', None))
    cc.setdefault('start_ts_utc', getattr(args, 'ts_utc', None))
    if oss_uploads:
        for u in oss_uploads:
            u.setdefault('ts_bj', _now_bj_ts())
            u.setdefault('ts_utc', _now_utc_ts())
        cc.setdefault('oss_uploads', [])
        try:
            cc['oss_uploads'].extend(oss_uploads)
        except Exception:
            cc['oss_uploads'] = oss_uploads
        try:
            ok_items = [u for u in cc.get('oss_uploads', []) if u.get('ok')]
            tot_bytes = sum(int(u.get('bytes') or 0) for u in ok_items)
            tot_sec = sum(float(u.get('seconds') or 0.0) for u in ok_items)
            cc['oss_total_upload_bytes'] = int(tot_bytes)
            cc['oss_total_upload_seconds'] = float(tot_sec)
            cc['oss_total_upload_mib_per_sec'] = (float(tot_bytes) / (1024.0 * 1024.0) / tot_sec) if tot_sec > 0 else None
        except Exception:
            pass
    _safe_json_dump(cc, cc_path)

    # manifest.json (patch/append missing keys for backward compatibility)
    man_path = out_dir / 'manifest.json'
    man = {}
    if man_path.exists():
        try:
            man = json.load(open(man_path, 'r', encoding='utf-8'))
        except Exception:
            man = {}
    repo_root = Path(__file__).resolve().parent.parent
    git_commit = _git_commit(repo_root)
    lag = int(getattr(args, 'lag', 12))
    horizon = int(getattr(args, 'horizon', 12))
    stride = horizon if bool(getattr(args, 'stride_horizon', False)) else 1
    protocol = {
        'lag': lag,
        'horizon': horizon,
        'stride': stride,
        'prefix_boundary': bool(getattr(args, 'prefix_boundary', True)),
        'year_split': bool(getattr(args, 'year_split', False)),
        'interval_min': int(getattr(args, 'interval', 120)),
        'normalizer': str(getattr(args, 'normalizer', '')),
    }
    man.update({
        'dataset': getattr(args, 'dataset', ''),
        'dataset_slug': dataset_slug(getattr(args, 'dataset', '')),
        'model': getattr(args, 'model', ''),
        'seed': int(getattr(args, 'seed', 0)),
        'units': 'TECU' if str(getattr(args, 'dataset', '')).lower() in ['gimtec', 'tec', 'china_hires'] else 'raw',
        'normalized': False,
        'code_entry': 'model/Run.py',
        'git_commit': git_commit or man.get('git_commit'),
        'protocol': protocol,
        'args': {k: getattr(args, k) for k in sorted(vars(args).keys()) if not k.startswith('_')},
        'env': _collect_env(),
        'ts_utc': getattr(args, 'ts_utc', None) or man.get('ts_utc') or _now_utc_ts(),
        'ts_bj': getattr(args, 'ts_bj', None) or man.get('ts_bj') or _now_bj_ts(),
        'run_id': getattr(args, 'wandb_run_id', None) or man.get('run_id'),
        'run_tag': getattr(args, 'run_tag', None) or man.get('run_tag'),
    })
    # Fill optional fields if missing (do not overwrite user edits)
    man.setdefault('years_split', _infer_years_split(args))
    man.setdefault('grid_shape', _infer_grid_shape(args))
    paths = man.get('paths') if isinstance(man.get('paths'), dict) else {}
    paths.update({
        'root': str(out_dir),
        'predictions': str(preds_path) if preds_path.exists() else (str(src_preds) if src_preds is not None and src_preds.exists() else None),
        'predictions_oss': preds_oss_uri or paths.get('predictions_oss'),
        'predictions_ossref': str((out_dir / 'predictions.ossref')) if (out_dir / 'predictions.ossref').exists() else paths.get('predictions_ossref'),
        'ood_predictions': str(ood_path) if ood_path.exists() else (str(src_ood_preds) if src_ood_preds is not None and src_ood_preds.exists() else None),
        'ood_predictions_oss': ood_oss_uri or paths.get('ood_predictions_oss'),
        'ood_predictions_ossref': str((out_dir / 'ood_predictions.ossref')) if (out_dir / 'ood_predictions.ossref').exists() else paths.get('ood_predictions_ossref'),
        'best_model': str(best_ckpt_local) if best_ckpt_local is not None and best_ckpt_local.exists() else paths.get('best_model'),
        'best_model_oss': best_oss_uri or paths.get('best_model_oss'),
        'best_model_ossref': str((out_dir / 'best_model.ossref')) if (out_dir / 'best_model.ossref').exists() else paths.get('best_model_ossref'),
    })
    man['paths'] = paths
    # Truth pointer (local JSON) + OSS payload (preferred)
    man['truth_ref'] = truth_ref_rel or man.get('truth_ref')
    man['truth_ref_abs'] = truth_ref_abs or man.get('truth_ref_abs')
    man['truth_oss_uri'] = truth_oss_uri or man.get('truth_oss_uri')
    man['truth_digest'] = truth_digest or man.get('truth_digest')
    man['oss_prefix'] = f"pi-gwn-tec/{dataset_slug(getattr(args,'dataset',''))}/" \
                        f"{str(getattr(args,'model','')).lower()}/seed_{int(getattr(args,'seed',0))}/" \
                        f"{getattr(args,'run_tag', '')}/"
    _safe_json_dump(man, man_path)

    # Upload small files to W&B
    _wandb_log_artifact(run, metrics_path, atype='results')
    _wandb_log_artifact(run, stepwise_path, atype='results')
    _wandb_log_artifact(run, cc_path, atype='results')
    _wandb_log_artifact(run, man_path, atype='results')

    # Log external references (truth/preds) to W&B if available
    try:
        if wandb is not None and run is not None:
            refs = []
            if truth_oss_uri:
                refs.append(str(truth_oss_uri))
            # If predictions are scheduled for async OSS upload, only write back to W&B after success.
            if preds_oss_uri and (not preds_async_scheduled):
                refs.append(str(preds_oss_uri))
            if ood_oss_uri:
                refs.append(str(ood_oss_uri))
            if best_oss_uri:
                refs.append(str(best_oss_uri))
            if refs:
                art = wandb.Artifact(name=f"oss-refs-{run.id}", type='external')
                for r in refs:
                    art.add_reference(r)
                run.log_artifact(art, aliases=['latest'])
    except Exception:
        pass
