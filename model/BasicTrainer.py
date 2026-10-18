import torch
import os
import time
import copy
import json
import csv
import numpy as np
import math
from time import perf_counter
from datetime import datetime, timezone, timedelta
try:
    from lib.wandb_import import import_wandb
    wandb = import_wandb()
except Exception:
    wandb = None
try:
    import requests, oss2
except Exception:
    requests = None
    oss2 = None
from threading import Thread
from lib import results_io
from lib.logger import get_logger
from lib.metrics import All_Metrics, MAE_torch, RMSE_torch, CORR_torch
try:
    # Optional imports for physics PINN (kept model-agnostic)
    from lib.datasets.gimtec_adj import load_or_build_adj  # unified adjacency for GIMtec/TEC
    from lib.physics.ppinn import PPinnLoss
except Exception:
    load_or_build_adj = None
    PPinnLoss = None

# --- AMP/BF16 helpers -------------------------------------------------------
def _has_bf16_cuda() -> bool:
    """Detect bf16 support on the current CUDA device.
    Prefer torch APIs; fall back to compute capability heuristic (>= 8.0).
    """
    if not torch.cuda.is_available():
        return False
    try:
        # PyTorch 2.x exposes this helper
        return bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    except Exception:
        pass
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False

class Trainer(object):
    def __init__(self, model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        # self.model_stu = model_stu
        self.args = args
        self.loss = loss
        self.loss_kl = loss_kl
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.batch_seen = 0
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        # Best checkpoint naming will be initialized after logger is created
        self.best_run_path = None
        self.best_link_path = None
        self.best_path = None
        self.loss_figure_path = None
        # W&B state
        self._wandb_status_logged = False
        self._wandb_first_log_done = False
        # OSS background uploads state
        self._oss_threads = []
        self._best_direct_uploaded = False
        self._best_bytes = None
        self._best_s3_uri = None
        self._oss_status_logged = False
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        # build informative logfile name: <timestamp>_<dataset>_<model>_<mode>.log
        ts = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        try:
            ds = getattr(args, 'dataset', 'DATASET')
            mode = getattr(args, 'mode', 'train')
            if mode == 'pretrain':
                arch = getattr(args, 'arch_tag', 'gptst')
                tgt = getattr(args, 'target_model', 'generic')
                gtag = getattr(args, 'graph_tag', 'na')
                filename = f"{ts}_{ds}_{arch}_{mode}_{tgt}_{gtag}.log"
                logger_name = arch
            else:
                md = getattr(args, 'model', 'MODEL')
                filename = f"{ts}_{ds}_{md}_{mode}.log"
                logger_name = md
        except Exception:
            filename = f"{ts}_run.log"
            logger_name = getattr(args, 'model', 'run')
        self.log_filename = filename
        self.logger = get_logger(args.log_dir, name=logger_name, debug=args.debug, filename=filename)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # log command line and basic env/device info
        try:
            if hasattr(args, 'cmdline'):
                self.logger.info('Command: {}'.format(args.cmdline))
            import torch
            self.logger.info('Torch: {}  CUDA: {}'.format(torch.__version__, torch.cuda.is_available()))
            if torch.cuda.is_available():
                self.logger.info('GPU: {}'.format(torch.cuda.get_device_name(0)))
                self.logger.info('CUDA_VISIBLE_DEVICES={}'.format(os.environ.get('CUDA_VISIBLE_DEVICES')))
            # dump key args
            for k in sorted(vars(args).keys()):
                try:
                    self.logger.info(f'{k}: {getattr(args,k)}')
                except Exception:
                    pass
        except Exception:
            pass
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

        # Initialize best checkpoint naming now that log_filename is known
        ts = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        base_stem = os.path.splitext(self.log_filename)[0] if getattr(self, 'log_filename', '') else ''
        if not base_stem:
            # Fallback naming
            ds = getattr(args, 'dataset', 'DATASET')
            md = getattr(args, 'model', 'MODEL')
            mode = getattr(args, 'mode', 'train')
            # Pretrain uses arch tag
            if mode == 'pretrain':
                arch = getattr(args, 'arch_tag', 'gptst')
                tgt = getattr(args, 'target_model', 'generic')
                gtag = getattr(args, 'graph_tag', 'na')
                base_stem = f"{ts}_{ds}_{arch}_{mode}_{tgt}_{gtag}"
            else:
                base_stem = f"{ts}_{ds}_{md}_{mode}"
        self.best_run_path = os.path.join(self.args.log_dir, f"{base_stem}.pth")
        self.best_link_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_path = self.best_run_path
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        # ====== Batch D helpers: hybrid residual / LOO holdout ======
        self.hybrid_enabled = False
        self._hybrid_bg_path = str(getattr(self.args, 'hybrid_bg_path', '') or '').strip()
        self._hybrid_bg = {}
        self._hybrid_bg_index = {}
        self._hybrid_bg_meta = None

        self.holdout_enabled = False
        self._holdout_mode = str(getattr(self.args, 'holdout_mode', 'off') or 'off').strip().lower()
        self._holdout_mask_path = str(getattr(self.args, 'holdout_mask_path', '') or '').strip()
        self._holdout_keep_idx = None
        self._holdout_idx = None

        if self._hybrid_bg_path:
            self.hybrid_enabled = True
            # Hybrid residual does not enable PINN in the paper protocol.
            if bool(getattr(self.args, 'use_pinn', False)):
                self.logger.info('Hybrid enabled: forcing use_pinn=False')
            setattr(self.args, 'use_pinn', False)

        if self._holdout_mode not in ('', 'off', 'none', '0', 'false'):
            if not self._holdout_mask_path:
                self.logger.warning('holdout_mode set but -holdout_mask_path is empty; holdout disabled.')
            else:
                self.holdout_enabled = True
                # For LOO experiments, keep PPINN enabled but mask physics loss to keep nodes to avoid leakage.
                # (See PPinnLoss: node_keep_idx/node_holdout_idx.)

        # Holdout mask (cuda-friendly indices)
        if self.holdout_enabled:
            try:
                mpath = self._holdout_mask_path
                if not os.path.isabs(mpath):
                    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    mpath = os.path.join(_repo_root, mpath)
                mask = np.load(mpath).astype(np.bool_, copy=False).reshape(-1)
                num_nodes = int(getattr(self.args, 'num_nodes', mask.size))
                if mask.size != num_nodes:
                    raise ValueError(f"holdout mask size mismatch: {mask.size} != {num_nodes}")
                keep = np.where(~mask)[0].astype(np.int64, copy=False)
                hold = np.where(mask)[0].astype(np.int64, copy=False)
                self._holdout_keep_idx = torch.from_numpy(keep).long().to(self.args.device)
                self._holdout_idx = torch.from_numpy(hold).long().to(self.args.device)
                self.logger.info(
                    f"Holdout enabled: keep={int(keep.size)}, holdout={int(hold.size)} "
                    f"({(float(hold.size) / float(mask.size) * 100.0) if mask.size else 0.0:.1f}%)"
                )
            except Exception as e:
                self.logger.warning(f"Holdout disabled due to mask load error: {e}")
                self.holdout_enabled = False
                self._holdout_keep_idx = None
                self._holdout_idx = None

        # Hybrid background cache (numpy; keyed by global start_idx)
        if self.hybrid_enabled:
            try:
                ds = str(getattr(self.args, 'dataset', '')).lower()
                if ds not in ('china_hires',):
                    raise ValueError(f"hybrid_bg_path currently only supports china_hires, got dataset={ds}")
                bg_path = self._hybrid_bg_path
                if not os.path.isabs(bg_path):
                    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    bg_path = os.path.join(_repo_root, bg_path)
                if not os.path.exists(bg_path):
                    raise FileNotFoundError(bg_path)
                meta = None
                with np.load(bg_path, allow_pickle=True) as z:
                    # meta
                    try:
                        if 'meta_json' in z.files:
                            meta = json.loads(str(z['meta_json'].item()))
                    except Exception:
                        meta = None
                    if meta is None:
                        try:
                            meta_path = os.path.splitext(bg_path)[0] + '.json'
                            if os.path.exists(meta_path):
                                with open(meta_path, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                        except Exception:
                            meta = None
                    self._hybrid_bg_meta = meta if isinstance(meta, dict) else {}
                    proto = (self._hybrid_bg_meta.get('protocol') if isinstance(self._hybrid_bg_meta, dict) else {}) or {}
                    # strict protocol alignment (required for start_idx matching)
                    if int(proto.get('lag', getattr(self.args, 'lag', 0))) != int(getattr(self.args, 'lag', 0)):
                        raise ValueError('hybrid_bg protocol mismatch: lag')
                    if int(proto.get('horizon', getattr(self.args, 'horizon', 0))) != int(getattr(self.args, 'horizon', 0)):
                        raise ValueError('hybrid_bg protocol mismatch: horizon')
                    expected_stride = int(getattr(self.args, 'horizon', 0)) if bool(getattr(self.args, 'stride_horizon', False)) else 1
                    if int(proto.get('stride', expected_stride)) != int(expected_stride):
                        raise ValueError('hybrid_bg protocol mismatch: stride')
                    if bool(proto.get('prefix_boundary', True)) != bool(getattr(self.args, 'prefix_boundary', True)):
                        raise ValueError('hybrid_bg protocol mismatch: prefix_boundary')
                    # normalizer alignment (residual is computed in normalized domain)
                    bg_norm = str(self._hybrid_bg_meta.get('china_normalizer', '') or '').strip().lower()
                    if bg_norm:
                        cur_norm = str(getattr(self.args, 'normalizer', '') or '').strip().lower()
                        if cur_norm and (cur_norm != bg_norm):
                            raise ValueError(f"hybrid_bg normalizer mismatch: bg={bg_norm}, args={cur_norm}")

                    for split in ('train', 'val', 'test', 'ood'):
                        si_key = f'{split}_start_idx'
                        bg_key = f'{split}_bg'
                        if (si_key not in z.files) or (bg_key not in z.files):
                            continue
                        si = np.asarray(z[si_key], dtype=np.int64).reshape(-1)
                        bg = np.asarray(z[bg_key], dtype=np.float16)
                        if bg.ndim != 3:
                            raise ValueError(f"hybrid_bg invalid shape for {bg_key}: {bg.shape}")
                        if bg.shape[0] != si.shape[0]:
                            raise ValueError(f"hybrid_bg windows mismatch for {split}: bg={bg.shape[0]}, start_idx={si.shape[0]}")
                        if int(bg.shape[1]) != int(getattr(self.args, 'horizon', 0)):
                            raise ValueError(f"hybrid_bg horizon mismatch for {split}: {bg.shape}")
                        if int(bg.shape[2]) != int(getattr(self.args, 'num_nodes', bg.shape[2])):
                            raise ValueError(f"hybrid_bg num_nodes mismatch for {split}: {bg.shape[2]} != {getattr(self.args,'num_nodes',None)}")
                        idx_map = {int(s): i for i, s in enumerate(si.tolist())}
                        if len(idx_map) != int(si.shape[0]):
                            raise ValueError(f"hybrid_bg start_idx has duplicates for split={split}")
                        self._hybrid_bg[split] = bg
                        self._hybrid_bg_index[split] = idx_map
                self.logger.info(f"Hybrid enabled: loaded bg cache: {bg_path}")
            except Exception as e:
                self.logger.warning(f"Hybrid disabled due to bg cache load error: {e}")
                self.hybrid_enabled = False
                self._hybrid_bg_path = ''
                self._hybrid_bg = {}
                self._hybrid_bg_index = {}
                self._hybrid_bg_meta = None

        # ====== Physics PINN (optional; minimal intrusion) ======
        self.use_pinn = bool(getattr(self.args, 'use_pinn', False)) and (PPinnLoss is not None)
        self.ppinn = None
        if self.use_pinn:
            try:
                # Build/load adjacency where applicable (e.g., GIMtec/TEC grid8); failure falls back to None
                A = None
                if load_or_build_adj is not None:
                    A_np, _ = load_or_build_adj(getattr(self.args, 'dataset', 'GIMtec'),
                                                getattr(self.args, 'num_nodes', 71*73),
                                                graph_tag=getattr(self.args, 'graph_tag', 'grid8'),
                                                adj_model=getattr(self.args, 'adj_model', None))
                    A = torch.from_numpy(A_np.astype('float32')).to(self.args.device)
            except Exception:
                A = None
            dtm = int(getattr(self.args, 'interval', 120))
            self.ppinn = PPinnLoss(
                scaler_data=self.scaler,
                A=A,
                dt_minutes=dtm,
                tec_ref=float(getattr(self.args, 'tec_ref', 50.0)),
                t_ref_sec=float(getattr(self.args, 't_ref_sec', 7200.0)),
                rot_cap_tecu_per_min=float(getattr(self.args, 'rot_cap', 0.5)),
                roti_cap_scale=float(getattr(self.args, 'roti_cap_scale', 0.7)),
                kappa_nd=float(getattr(self.args, 'kappa_nd', 0.05)),
                roti_w=3,
                use_diffusion=bool(getattr(self.args, 'use_diffusion', True)),
                use_drivers=bool(getattr(self.args, 'use_drivers', False)),
                use_dynamic_rot_cap=bool(getattr(self.args, 'use_dynamic_rot_cap', True)),
                rot_cap_k=float(getattr(self.args, 'rot_cap_k', 0.01)),
            ).to(self.args.device)
            # Ensure PPINN parameters (e.g., log_vars) are optimized
            try:
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    self.optimizer.add_param_group({'params': self.ppinn.parameters(),
                                                    'lr': float(getattr(self.args, 'lr_init', 1e-3))})
                    self.logger.info('PPINN parameters added to optimizer.')
            except Exception as e:
                self.logger.warning(f'Failed to add PPINN params to optimizer: {e}')
        # Prepare robust drivers path under repo root
        try:
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ds = str(getattr(self.args, 'dataset', '')).lower()
            if ds.startswith('china_hires'):
                self._drivers_path = os.path.join(_repo_root, 'data', 'china_hires', 'drivers', 'drivers_2014_2024_2h_cn.npz')
            else:
                # default: GIMtec/TEC drivers timeline
                self._drivers_path = os.path.join(_repo_root, 'data', 'GIMtec', 'drivers', 'drivers_2009_2022_2h.npz')
        except Exception:
            self._drivers_path = "./data/GIMtec/drivers/drivers_2009_2022_2h.npz"

    def _warn_once(self, key: str, msg: str) -> None:
        try:
            if not hasattr(self, '_warn_once_keys'):
                self._warn_once_keys = set()
            if key in self._warn_once_keys:
                return
            self._warn_once_keys.add(key)
            self.logger.warning(msg)
        except Exception:
            pass

    def _build_gwn_drivers_input(self, start_idx, T: int, N: int, args=None):
        """Build [B,T,N,5] drivers tensor when use_drivers_input=True; return None on any failure.

        NOTE: Currently enabled only for {GWN, STPatchTransformer} to avoid changing other predictors unexpectedly.
        """
        if args is None:
            args = self.args
        if str(getattr(args, 'model', '')).lower() not in ('gwn', 'stpatchtransformer'):
            return None
        use_drivers_input = getattr(args, 'use_drivers_input', None)
        if use_drivers_input is None:
            use_drivers_input = bool(getattr(args, 'use_drivers', False))
        if not bool(use_drivers_input):
            return None
        if start_idx is None:
            self._warn_once(
                'gwn_drivers_start_idx',
                'use_drivers_input=True but start_idx is missing; falling back to zero drivers_input.',
            )
            return None
        try:
            from lib.physics.drivers import DriversStore
            # Reuse a single DriversStore instance per Trainer; rebuild if device changed.
            dev = getattr(args, 'device', None) or getattr(self.args, 'device', 'cpu')
            if (not hasattr(self, '_drivers')) or (getattr(self._drivers, 'device', None) != dev):
                self._drivers = DriversStore(self._drivers_path, device=dev)
            if bool(getattr(self._drivers, 'synthetic', False)):
                self._warn_once(
                    'gwn_drivers_synth',
                    f"drivers unavailable ({self._drivers_path}); falling back to zero drivers_input.",
                )
                return None
            d = self._drivers.slice_batched(start_idx, int(T), int(N))
            ae = d['AE'].unsqueeze(-1).expand(-1, -1, int(N))
            dst = d['Dst'].unsqueeze(-1).expand(-1, -1, int(N))
            kp = d['Kp'].unsqueeze(-1).expand(-1, -1, int(N))
            f107 = d['F107'].unsqueeze(-1).expand(-1, -1, int(N))
            cosz = d.get('cosSZA')
            if cosz is None:
                self._warn_once('gwn_drivers_cosz', 'drivers store has no cosSZA; falling back to zero drivers_input.')
                return None
            return torch.stack([ae, dst, kp, f107, cosz], dim=-1)
        except Exception:
            return None

    def _as_scalar_start_idx(self, start_idx):
        """Best-effort convert batched start_idx to a scalar index (use first element)."""
        if start_idx is None:
            return None
        try:
            if isinstance(start_idx, torch.Tensor):
                if start_idx.numel() == 0:
                    return None
                return int(start_idx.reshape(-1)[0].item())
            return int(start_idx)
        except Exception:
            return None

    def _start_idx_list(self, start_idx):
        """Convert DataLoader-collated start_idx to a python list[int]."""
        if start_idx is None:
            return []
        try:
            if isinstance(start_idx, torch.Tensor):
                return [int(x) for x in start_idx.detach().cpu().reshape(-1).tolist()]
            if isinstance(start_idx, (list, tuple)):
                return [int(x) for x in start_idx]
            if isinstance(start_idx, np.ndarray):
                return [int(x) for x in start_idx.reshape(-1).tolist()]
            return [int(start_idx)]
        except Exception:
            return []

    def _hybrid_bg_tensor(self, split_tag: str, start_idx, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Return y_bg in normalized domain for this batch: [B,H,N,1] on device."""
        if not self.hybrid_enabled:
            raise RuntimeError('hybrid_bg_tensor called but hybrid is disabled')
        if start_idx is None:
            raise ValueError('hybrid requires dataset to return start_idx')
        split = str(split_tag).strip().lower()
        if split not in self._hybrid_bg:
            raise KeyError(f'hybrid bg split not found: {split} (available={sorted(self._hybrid_bg.keys())})')
        idx_map = self._hybrid_bg_index.get(split) or {}
        si_list = self._start_idx_list(start_idx)
        if not si_list:
            raise ValueError('hybrid requires non-empty start_idx')
        rows = []
        for s in si_list:
            j = idx_map.get(int(s), None)
            if j is None:
                raise KeyError(f'hybrid bg missing start_idx={int(s)} for split={split}')
            rows.append(int(j))
        bg_np = self._hybrid_bg[split]
        bg_batch = torch.from_numpy(bg_np[np.asarray(rows, dtype=np.int64)])
        bg_batch = bg_batch.to(device=ref_tensor.device, dtype=ref_tensor.dtype, non_blocking=True)
        return bg_batch.unsqueeze(-1)

    def _select_nodes(self, x: torch.Tensor, node_idx=None) -> torch.Tensor:
        """Select subset of nodes along dim=2 for [B,H,N,(D)] tensors."""
        if x is None or node_idx is None:
            return x
        if not isinstance(x, torch.Tensor):
            return x
        if x.dim() == 4:
            return torch.index_select(x, dim=2, index=node_idx)
        if x.dim() == 3:
            return torch.index_select(x, dim=2, index=node_idx)
        if x.dim() == 2:
            return torch.index_select(x, dim=1, index=node_idx)
        return x

    # ------------------------ W&B utilities ------------------------
    def _wandb_prepare(self):
        """Ensure W&B run exists (retry init once) and log status line once.
        Uses local proxy injection around wandb.init to avoid global side effects.
        """
        if wandb is None:
            return None
        run = getattr(wandb, 'run', None)
        if run is None:
            try:
                # reconstruct entity/project/tags like model/Run.py
                from lib import results_io as _rio
                ds_slug = _rio.dataset_slug(getattr(self.args, 'dataset', ''))
                md_slug = str(getattr(self.args, 'model', 'model')).lower()
                use_drivers_input = getattr(self.args, 'use_drivers_input', None)
                if use_drivers_input is None:
                    use_drivers_input = bool(getattr(self.args, 'use_drivers', False))
                tags = [
                    f"seed_{int(getattr(self.args,'seed',0))}",
                    f"amp_{bool(getattr(self.args,'amp',False))}",
                    f"pinn_{bool(getattr(self.args,'use_pinn',False))}",
                    f"drivers_phys_{bool(getattr(self.args,'use_drivers',False))}",
                    f"drivers_in_{bool(use_drivers_input)}",
                    f"adv_{bool(getattr(self.args,'use_adv',False))}",
                    f"nodiff_{not bool(getattr(self.args,'use_diffusion',True))}",
                    f"nochem_{bool(getattr(self.args,'nochem',False))}",
                    f"acc_{int(getattr(self.args,'accumulate_steps',1))}",
                    f"ys_{bool(getattr(self.args,'year_split',False))}",
                ]
                project = os.getenv('WANDB_PROJECT', 'pi-gwn-tec')
                entity = os.getenv('WANDB_ENTITY', None)
                run = wandb.init(project=project,
                                 entity=entity,
                                 config=vars(self.args),
                                 group=f"{ds_slug}-{md_slug}",
                                 tags=tags)
                try:
                    self.args.wandb_run_id = getattr(run, 'id', None)
                except Exception:
                    pass
                try:
                    run.name = _rio.wandb_run_name(self.args)
                except Exception:
                    pass
            except Exception:
                run = None
        # Log status line once
        if not self._wandb_status_logged:
            offline = str(os.getenv('WANDB_MODE', '')).lower() == 'offline'
            status = 'online' if (run is not None and not offline) else 'offline'
            try:
                self.logger.info(f"W&B status: {status}")
            except Exception:
                pass
            self._wandb_status_logged = True
        return run

    def _wandb_log(self, data: dict, step: int = None):
        if wandb is None:
            return
        try:
            run = self._wandb_prepare()
            if run is None:
                return
            wandb.log(data, step=step)
        except Exception:
            pass

    def _wandb_log_artifact(self, artifact, aliases=None):
        if wandb is None or artifact is None:
            return
        try:
            run = self._wandb_prepare()
            if run is None:
                return
            wandb.run.log_artifact(artifact, aliases=aliases or ['latest'])
        except Exception:
            pass

    def _ppinn_logvars(self) -> dict:
        logvars = {}
        try:
            if self.ppinn is None:
                return logvars
            uncert = getattr(self.ppinn, 'uncert', None)
            log_vars = getattr(uncert, 'log_vars', None)
            if log_vars is None:
                return logvars
            for name, param in log_vars.items():
                try:
                    val = float(param.detach().mean().item())
                except Exception:
                    continue
                if val == val and abs(val) != float('inf'):
                    logvars[str(name)] = val
        except Exception:
            pass
        return logvars

    def _append_ppinn_logvars_csv(self, logvars: dict, epoch: int) -> None:
        if not logvars:
            return
        try:
            path = os.path.join(self.args.log_dir, 'logvars.csv')
            header = getattr(self, '_ppinn_logvars_header', None)
            if not header:
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            first = f.readline().strip()
                        if first:
                            header = [h.strip() for h in first.split(',') if h.strip()]
                    except Exception:
                        header = None
                if not header:
                    keys = sorted(logvars.keys())
                    header = ['epoch', 'step'] + [f'log_var_{k}' for k in keys]
                self._ppinn_logvars_header = header

            write_header = not os.path.exists(path)
            row = {'epoch': int(epoch), 'step': int(getattr(self, 'batch_seen', 0))}
            for k, v in logvars.items():
                row[f'log_var_{k}'] = v
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass

    def benchmark_inference(self, model=None, data_loader=None, warmup: int = 20, repeat: int = 100):
        """Benchmark pure-forward inference latency on a prefetched real batch (bs=1)."""
        try:
            if (not torch.cuda.is_available()) or data_loader is None:
                return None
            net = model if model is not None else self.model
            was_train = bool(getattr(net, 'training', False))
            net.eval()

            # Prefetch one real batch and reuse (exclude dataloader overhead from timing).
            batch = next(iter(data_loader))
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, _, _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                data = batch[0]
            else:
                data = batch
            data = data[:1]
            data = data.to(self.args.device, non_blocking=True)
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]

            # AMP dtype selection: bf16 preferred on supported GPUs.
            use_amp_flag = bool(getattr(self.args, 'amp', False)) and torch.cuda.is_available()
            has_bf16 = _has_bf16_cuda()
            amp_dtype = torch.bfloat16 if (use_amp_flag and has_bf16) else (torch.float16 if use_amp_flag else None)
            try:
                from torch.amp import autocast as _autocast
                def _amp_ctx(enabled, dtype):
                    return _autocast('cuda', enabled=enabled, dtype=dtype)
            except Exception:
                from torch.cuda.amp import autocast as _autocast
                def _amp_ctx(enabled, dtype):
                    return _autocast(enabled=enabled, dtype=dtype)

            def _forward_once():
                if getattr(self.args, 'mode', '') == 'pretrain':
                    try:
                        net(data, None, None, int(getattr(self.args, 'epochs', 1)))
                    except TypeError:
                        try:
                            net(data, None, None)
                        except TypeError:
                            net(data)
                else:
                    try:
                        net(data, label=None)
                    except TypeError:
                        try:
                            net(data, None)
                        except TypeError:
                            net(data)

            warmup = int(max(0, warmup))
            repeat = int(max(1, repeat))
            with torch.no_grad():
                for _ in range(warmup):
                    with _amp_ctx(use_amp_flag, amp_dtype):
                        _forward_once()
                torch.cuda.synchronize()
                # Measure peak memory + timed repeats in steady state.
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                starts, ends = [], []
                for _ in range(repeat):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    with _amp_ctx(use_amp_flag, amp_dtype):
                        _forward_once()
                    e.record()
                    starts.append(s); ends.append(e)
                torch.cuda.synchronize()

            import numpy as _np
            times = _np.array([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=_np.float64)
            if times.size == 0:
                return None
            mean_ms = float(times.mean())
            p50_ms = float(_np.percentile(times, 50))
            p95_ms = float(_np.percentile(times, 95))
            max_alloc_mib = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            max_resv_mib = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))

            # Persist to args for results_io
            self.args._infer_latency_ms = mean_ms
            self.args._infer_latency_p50_ms = p50_ms
            self.args._infer_latency_p95_ms = p95_ms
            self.args._infer_max_memory_allocated_MiB = max_alloc_mib
            self.args._infer_max_memory_reserved_MiB = max_resv_mib
            self.args._infer_input_shape = [int(x) for x in list(data.shape)]

            try:
                tf32 = bool(getattr(torch.backends.cuda.matmul, 'allow_tf32', True))
            except Exception:
                tf32 = None
            proto = {
                'bs': 1,
                'warmup': warmup,
                'repeat': repeat,
                'input_source': 'prefetched_real_batch',
                'measure': 'cuda_event',
                'amp': bool(use_amp_flag),
                'amp_dtype': str(amp_dtype) if amp_dtype is not None else 'none',
                'tf32': tf32,
                'mem_unit': 'MiB',
                'reset_peak_memory_stats': 'before_timed_repeats',
            }
            existing = getattr(self.args, '_compute_cost_protocol', None)
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(proto)
                self.args._compute_cost_protocol = merged
            else:
                self.args._compute_cost_protocol = proto

            return {
                'inference_latency_ms': mean_ms,
                'inference_latency_p50_ms': p50_ms,
                'inference_latency_p95_ms': p95_ms,
                'max_memory_infer_allocated_MiB': max_alloc_mib,
                'max_memory_infer_reserved_MiB': max_resv_mib,
            }
        except Exception as e:
            try:
                self.logger.warning(f'inference benchmark skipped: {e}')
            except Exception:
                pass
            return None
        finally:
            try:
                if was_train:
                    net.train()
            except Exception:
                pass

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        count_steps = 0
        val_sse = 0.0
        val_abs = 0.0
        val_cnt = 0
        # val_pred = []
        # val_true = []

        # AMP dtype selection: bf16 preferred on supported GPUs (e.g., H800)
        use_amp_flag = bool(getattr(self.args, 'amp', False)) and torch.cuda.is_available()
        has_bf16 = _has_bf16_cuda()
        amp_dtype = torch.bfloat16 if (use_amp_flag and has_bf16) else (torch.float16 if use_amp_flag else None)
        # Compat autocast context (torch.amp preferred; fallback to torch.cuda.amp)
        try:
            from torch.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast('cuda', enabled=enabled, dtype=dtype)
        except Exception:
            from torch.cuda.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast(enabled=enabled, dtype=dtype)

        dyn_caps_all = []  # collect dynamic rot_cap per-batch (TECU/min) if available
        dyn_valid_rot_all = []
        dyn_valid_roti_all = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # support optional (data, target, start_idx) for driver slicing; default to 2-tuple
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    data, target, start_idx = batch
                else:
                    data, target = batch
                    start_idx = None
                data = data.to(self.args.device, non_blocking=True)
                target = target.to(self.args.device, non_blocking=True)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                if self.args.mode == 'pretrain':
                    label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                else:
                    label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                # Optional drivers input for de_gwn
                drivers_input = self._build_gwn_drivers_input(start_idx, int(data.size(1)), int(data.size(2)))
                with _amp_ctx(use_amp_flag, amp_dtype):
                    if self.args.mode == 'pretrain':
                        output, _, mask, _, _ = self.model(data, label)
                    else:
                        # pass optional drivers_input for de_gwn
                        try:
                            output, _, mask, _, _ = self.model(data, label=None, batch_seen=None, drivers_input=drivers_input)
                        except TypeError:
                            output, _, mask, _, _ = self.model(data, label=None)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label[..., :self.args.output_dim])
                with _amp_ctx(use_amp_flag, amp_dtype):
                    if self.args.mode == 'pretrain':
                        loss, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                    else:
                        # Batch D: hybrid residual + LOO holdout (mask loss only)
                        try:
                            if isinstance(output, torch.Tensor) and output.dim() == 3:
                                output = output.unsqueeze(-1)
                        except Exception:
                            pass
                        label_true = target[..., :self.args.output_dim]
                        label_for_loss = label_true
                        pred_for_metrics = output
                        if self.hybrid_enabled:
                            if int(getattr(self.args, 'output_dim', 1)) != 1:
                                raise ValueError('hybrid residual expects output_dim=1')
                            bg = self._hybrid_bg_tensor('val', start_idx, ref_tensor=output)
                            label_for_loss = label_true - bg
                            pred_for_metrics = output + bg
                        out_for_loss = output
                        if self.holdout_enabled and (self._holdout_keep_idx is not None):
                            out_for_loss = self._select_nodes(out_for_loss, self._holdout_keep_idx)
                            label_for_loss = self._select_nodes(label_for_loss, self._holdout_keep_idx)
                        res = self.loss(out_for_loss, label_for_loss)
                        loss = res[0] if isinstance(res, tuple) else res
                        # streaming RMSE in TECU domain
                        try:
                            pred_m = pred_for_metrics[..., :self.args.output_dim]
                            true_m = label_true
                            if self.holdout_enabled and (self._holdout_keep_idx is not None):
                                pred_m = self._select_nodes(pred_m, self._holdout_keep_idx)
                                true_m = self._select_nodes(true_m, self._holdout_keep_idx)
                            pred_inv = self.scaler.inverse_transform(pred_m)
                            true_inv = self.scaler.inverse_transform(true_m)
                            diff = (pred_inv - true_inv)
                            val_sse += float(torch.nansum(diff * diff).item())
                            val_abs += float(torch.nansum(torch.abs(diff)).item())
                            val_cnt += int(diff.numel())
                        except Exception:
                            pass
                        # Physics PINN (optional) — zero-driver works without start_idx
                        if self.ppinn is not None:
                            try:
                                y_pred01 = output[..., :1]
                                drivers = None
                                # external drivers can be enabled later with dataset returning start_idx
                                if bool(getattr(self.args, 'use_drivers', False)) and (start_idx is not None):
                                    from lib.physics.drivers import DriversStore
                                    if not hasattr(self, '_drivers'):
                                        self._drivers = DriversStore(self._drivers_path, device=self.args.device)
                                    B, T, N = y_pred01.shape[0], y_pred01.shape[1], y_pred01.shape[2]
                                    if bool(getattr(self._drivers, 'synthetic', False)):
                                        raise ValueError('drivers unavailable (synthetic fallback)')
                                    offset = int(getattr(self.args, 'lag', 0))
                                    if bool(getattr(self.args, 'single', False)):
                                        offset += int(getattr(self.args, 'horizon', 0)) - 1
                                    drivers = self._drivers.slice_batched(start_idx + offset, T, N)
                                    drivers['use_adv'] = bool(getattr(self.args, 'use_adv', False))
                                    if bool(getattr(self.args, 'nochem', False)):
                                        drivers['alpha_nd'] = 0.0
                                        drivers['beta_nd'] = 0.0
                                lam = float(getattr(self.args, 'lambda_phys', 1.0))
                                if lam != 0.0:
                                    # Avoid double-counting data term: exclude data from PPINN aggregation
                                    loss_phys, _ = self.ppinn(
                                        y_pred01,
                                        None,
                                        drivers=drivers,
                                        node_keep_idx=self._holdout_keep_idx if self.holdout_enabled else None,
                                        node_holdout_idx=self._holdout_idx if self.holdout_enabled else None,
                                    )
                                    if torch.isfinite(loss_phys).item():
                                        loss = loss + lam * loss_phys
                            except Exception:
                                pass
                # accumulate only finite losses
                if torch.isfinite(loss).item():
                    total_val_loss += loss.item()
                    count_steps += 1
        denom = count_steps if count_steps > 0 else max(1, len(val_dataloader))
        val_loss = total_val_loss / denom
        val_rmse = float(math.sqrt(val_sse / val_cnt)) if val_cnt > 0 else float('nan')
        val_mae = float(val_abs / val_cnt) if val_cnt > 0 else float('nan')
        self._last_val_rmse = val_rmse
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}'.format(
            epoch,
            val_loss,
            0.0 if (val_rmse != val_rmse) else val_rmse,
            0.0 if (val_mae != val_mae) else val_mae,
        ))
        try:
            self._wandb_log({'val/loss': val_loss, 'val/rmse': val_rmse, 'val/mae': val_mae}, step=self.batch_seen)
        except Exception:
            pass

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        count_steps = 0
        total_flow_loss = 0
        total_s_loss = 0
        # initialize training-time trackers
        ema = None
        step_times_ms = []
        samples_seen = 0
        phys_lam = float(getattr(self.args, 'lambda_phys', 1.0))
        phys_n = 0
        phys_loss_sum = 0.0
        phys_terms_sum = {}
        # AMP dtype selection: bf16 preferred on supported GPUs (e.g., H800)
        use_amp_flag = bool(getattr(self.args, 'amp', False)) and torch.cuda.is_available()
        has_bf16 = _has_bf16_cuda()
        amp_dtype = torch.bfloat16 if (use_amp_flag and has_bf16) else (torch.float16 if use_amp_flag else None)
        accum = max(1, int(getattr(self.args, 'accumulate_steps', 1)))
        # AMP helpers: use new torch.amp API if available
        try:
            from torch.amp import GradScaler as _GradScaler
            def _make_scaler(enabled):
                return _GradScaler('cuda', enabled=enabled)
        except Exception:
            from torch.cuda.amp import GradScaler as _GradScaler
            def _make_scaler(enabled):
                return _GradScaler(enabled=enabled)
        try:
            from torch.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast('cuda', enabled=enabled, dtype=dtype)
        except Exception:
            from torch.cuda.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast(enabled=enabled, dtype=dtype)
        # Enable GradScaler only for FP16 (bf16 path runs without scaling)
        scaler = _make_scaler(enabled=(use_amp_flag and not has_bf16))
        grad_clip = float(getattr(self.args, 'grad_clip', 0.0))
        for batch_idx, batch in enumerate(self.train_loader):
            self.batch_seen += 1
            # support optional (data, target, start_idx)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, target, start_idx = batch
            else:
                data, target = batch
                start_idx = None
            data = data.to(self.args.device, non_blocking=True)
            target = target.to(self.args.device, non_blocking=True)
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            try:
                samples_seen += int(data.size(0))
            except Exception:
                pass
            if self.args.mode == 'pretrain':
                label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            else:
                label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
            if batch_idx % accum == 0:
                self.optimizer.zero_grad(set_to_none=True)

            if self.args.mode == 'pretrain':
                with _amp_ctx(use_amp_flag, amp_dtype):
                    output, out_time, mask, probability, eb = self.model(data, label, self.batch_seen, epoch)
                    loss_flow, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                    if epoch > self.args.change_epoch:
                        loss_s = self.loss_kl(probability.log(), eb) * 0.1
                        loss = loss_flow + loss_s
                    else:
                        loss = loss_flow
            else:
                with _amp_ctx(use_amp_flag, amp_dtype):
                    t0 = perf_counter()
                    # Optional drivers input for de_gwn
                    drivers_input = self._build_gwn_drivers_input(start_idx, int(data.size(1)), int(data.size(2)))
                    try:
                        output, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen, drivers_input=drivers_input)
                    except TypeError:
                        t0 = perf_counter()
                        output, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen)
                    # Batch D: hybrid residual + LOO holdout (mask loss only)
                    try:
                        if isinstance(output, torch.Tensor) and output.dim() == 3:
                            output = output.unsqueeze(-1)
                    except Exception:
                        pass
                    label_true = target[..., :self.args.output_dim]
                    label_for_loss = label_true
                    if self.hybrid_enabled:
                        if int(getattr(self.args, 'output_dim', 1)) != 1:
                            raise ValueError('hybrid residual expects output_dim=1')
                        bg = self._hybrid_bg_tensor('train', start_idx, ref_tensor=output)
                        label_for_loss = label_true - bg
                    out_for_loss = output
                    if self.holdout_enabled and (self._holdout_keep_idx is not None):
                        out_for_loss = self._select_nodes(out_for_loss, self._holdout_keep_idx)
                        label_for_loss = self._select_nodes(label_for_loss, self._holdout_keep_idx)
                    res = self.loss(out_for_loss, label_for_loss)
                    # loss functions may return (mean_loss, elementwise) tuples (e.g., mask_mae)
                    loss = res[0] if isinstance(res, tuple) else res
                    # Physics PINN (optional)
                    if self.ppinn is not None:
                        try:
                            y_pred01 = output[..., :1]
                            drivers = None
                            if bool(getattr(self.args, 'use_drivers', False)) and (start_idx is not None):
                                from lib.physics.drivers import DriversStore
                                if not hasattr(self, '_drivers'):
                                    self._drivers = DriversStore(self._drivers_path, device=self.args.device)
                                B, T, N = y_pred01.shape[0], y_pred01.shape[1], y_pred01.shape[2]
                                if bool(getattr(self._drivers, 'synthetic', False)):
                                    raise ValueError('drivers unavailable (synthetic fallback)')
                                offset = int(getattr(self.args, 'lag', 0))
                                if bool(getattr(self.args, 'single', False)):
                                    offset += int(getattr(self.args, 'horizon', 0)) - 1
                                drivers = self._drivers.slice_batched(start_idx + offset, T, N)
                                drivers['use_adv'] = bool(getattr(self.args, 'use_adv', False))
                                if bool(getattr(self.args, 'nochem', False)):
                                    drivers['alpha_nd'] = 0.0
                                    drivers['beta_nd'] = 0.0
                            lam = float(phys_lam)
                            if lam != 0.0:
                                # Avoid double-counting data term in training
                                loss_phys, terms = self.ppinn(
                                    y_pred01,
                                    None,
                                    drivers=drivers,
                                    node_keep_idx=self._holdout_keep_idx if self.holdout_enabled else None,
                                    node_holdout_idx=self._holdout_idx if self.holdout_enabled else None,
                                )
                                if torch.isfinite(loss_phys).item():
                                    loss = loss + lam * loss_phys
                                    try:
                                        phys_n += 1
                                        phys_loss_sum += float(loss_phys.detach().item())
                                        if isinstance(terms, dict):
                                            for k, v in terms.items():
                                                try:
                                                    vv = float(v.detach().item()) if hasattr(v, 'detach') else float(v)
                                                    if vv == vv and abs(vv) != float('inf'):
                                                        kk = str(k)
                                                        phys_terms_sum[kk] = float(phys_terms_sum.get(kk, 0.0)) + vv
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    # step time measurement
                    try:
                        step_times_ms.append((perf_counter() - t0) * 1000.0)
                    except Exception:
                        pass

            loss_to_back = loss / accum
            if use_amp_flag:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            # add max grad clipping
            if grad_clip and grad_clip > 0:
                # unscale before clipping in scaled fp16 path
                if use_amp_flag and not has_bf16:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            elif self.args.grad_norm:
                if use_amp_flag and not has_bf16:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == self.train_per_epoch:
                if use_amp_flag:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
            if torch.isfinite(loss).item():
                total_loss += loss.item()
                count_steps += 1
            # calculate total loss
            if self.args.mode == 'pretrain':
                total_flow_loss += loss_flow.item()
                if epoch > self.args.change_epoch:
                    total_s_loss += loss_s.item()
            #log information
            # maintain EMA
            try:
                if ema is None:
                    ema = loss.item()
                else:
                    ema = 0.9 * ema + 0.1 * loss.item()
                self._ema_loss = ema
            except Exception:
                pass
            if batch_idx % max(1, int(getattr(self.args, 'log_step', 50))) == 0:
                running_avg = (total_loss / max(1, count_steps)) if 'count_steps' in locals() else loss.item()
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} (running_avg: {:.6f})'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item(), running_avg))
                # wandb real-time metrics
                try:
                    logd = {'train/loss': float(loss.item()), 'train/loss_smooth': float(ema)}
                    try:
                        lr = float(self.optimizer.param_groups[0]['lr'])
                        logd['train/lr'] = lr
                    except Exception:
                        pass
                    if bool(getattr(self.args, 'grad_norm', False)):
                        try:
                            total_norm = 0.0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    g = p.grad.detach().data
                                    total_norm += float(torch.norm(g, 2).item()) ** 2
                            logd['train/grad_norm'] = float(math.sqrt(total_norm))
                        except Exception:
                            pass
                    self._wandb_log(logd, step=self.batch_seen)
                except Exception:
                    pass
        denom = count_steps if count_steps > 0 else self.train_per_epoch
        train_epoch_loss = total_loss/denom
        if self.args.mode == 'pretrain':
            train_epoch_flow_loss = total_flow_loss/self.train_per_epoch
            train_epoch_s_loss = total_s_loss / self.train_per_epoch
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} averaged Loss_s: {:.6f}'.format(epoch, train_epoch_flow_loss, train_epoch_s_loss))
        else:
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} (per-batch mean)'.format(epoch, train_epoch_loss))

        # epoch-level average step time
        try:
            if step_times_ms:
                self._wandb_log({'train/avg_step_time_ms_per_epoch': float(sum(step_times_ms)/len(step_times_ms))}, step=self.batch_seen)
        except Exception:
            pass
        # expose epoch throughput to train() for time/samples_per_sec
        try:
            self._last_train_samples = int(samples_seen)
        except Exception:
            pass
        # physics metrics (epoch average, raw terms)
        try:
            if (self.ppinn is not None) and (float(phys_lam) != 0.0) and (int(phys_n) > 0):
                logp = {
                    'physics/loss_phys_raw': float(phys_loss_sum / float(phys_n)),
                    'physics/lambda_phys': float(phys_lam),
                }
                for k, s in phys_terms_sum.items():
                    logp[f'physics/term_{k}'] = float(s / float(phys_n))
                self._wandb_log(logp, step=self.batch_seen)
        except Exception:
            pass
        # PPINN log-variance tracking (epoch-level)
        try:
            logvars = self._ppinn_logvars()
            if logvars:
                logd = {'epoch': int(epoch)}
                logd.update({f'ppinn/log_var_{k}': float(v) for k, v in logvars.items()})
                self._wandb_log(logd, step=self.batch_seen)
                self._append_ppinn_logvars_csv(logvars, epoch=epoch)
        except Exception:
            pass
        # learning rate scheduling (non-plateau handled here; plateau handled in train() after val)
        if self.lr_scheduler is not None and getattr(self.args, 'scheduler', 'none') != 'plateau':
            if self.args.lr_decay:
                self.lr_scheduler.step()
        # train_epoch_flow_loss for params selecting
        if self.args.mode == 'pretrain':
            return train_epoch_flow_loss
        else:
            return train_epoch_loss

    def train(self):
        best_model = None
        best_model_test = None
        monitor_metric = str(getattr(self.args, 'monitor_metric', 'val_loss') or 'val_loss').strip().lower()
        early_stop_metric = str(getattr(self.args, 'early_stop_metric', '') or '').strip().lower()
        if not early_stop_metric:
            early_stop_metric = monitor_metric
        valid_metrics = {'val_loss', 'val_rmse'}
        if monitor_metric not in valid_metrics:
            self.logger.warning(f"Invalid monitor_metric={monitor_metric}, fallback to val_loss")
            monitor_metric = 'val_loss'
        if early_stop_metric not in valid_metrics:
            self.logger.warning(f"Invalid early_stop_metric={early_stop_metric}, fallback to monitor_metric={monitor_metric}")
            early_stop_metric = monitor_metric
        try:
            self.logger.info(f"Monitor metric: {monitor_metric} (early_stop_metric: {early_stop_metric})")
        except Exception:
            pass

        # best checkpoint is selected by monitor_metric (lower is better)
        best_score = float('inf')
        best_val_loss = float('inf')
        best_val_rmse = float('inf')
        # early stop is based on early_stop_metric (lower is better)
        best_early_stop = float('inf')
        not_improved_count = 0
        best_epoch = None
        early_stop_epoch = None
        train_loss_list = []
        val_loss_list = []
        epoch_times_sec = []
        eta_smooth_k = int(getattr(self.args, 'eta_smooth_k', 5) or 5)
        up_epoch = [int(i) for i in list(self.args.up_epoch.split(','))]
        start_time = time.time()
        # Training peak memory (includes dataloader + val; excludes final test/upload)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            # epoch progress
            try:
                total_epochs = int(getattr(self.args, 'epochs', 0) or 0)
                denom = float(max(1, total_epochs))
                self._wandb_log({
                    'epoch': int(epoch),
                    'epoch/total': int(total_epochs),
                    'epoch/progress': float(epoch) / denom,
                }, step=self.batch_seen)
            except Exception:
                pass

            train_start = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            train_sec = time.time() - train_start
            # print(time.time()-epoch_time)
            if epoch in up_epoch:
                best_score = float('inf')
                best_early_stop = float('inf')
            if self.args.mode == 'pretrain':
                if train_epoch_loss < best_score:
                    best_score = train_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
            else:
                if self.val_loader == None:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                val_loss_list.append(val_epoch_loss)
                # step plateau scheduler with validation loss
                if self.lr_scheduler is not None and getattr(self.args, 'scheduler', 'none') == 'plateau':
                    self.lr_scheduler.step(val_epoch_loss)
                min_delta = float(getattr(self.args, 'early_stop_min_delta', 0.0))

                # Resolve validation metrics (best-effort)
                try:
                    val_rmse = float(getattr(self, '_last_val_rmse', float('nan')))
                except Exception:
                    val_rmse = float('nan')
                try:
                    val_loss_f = float(val_epoch_loss)
                except Exception:
                    val_loss_f = float('nan')
                if (val_loss_f != val_loss_f) or (val_loss_f == float('inf')) or (val_loss_f == float('-inf')):
                    val_loss_f = float('inf')
                if (val_rmse != val_rmse) or (val_rmse == float('inf')) or (val_rmse == float('-inf')):
                    val_rmse = float('inf')

                monitor_value = val_rmse if monitor_metric == 'val_rmse' else val_loss_f
                early_stop_value = val_rmse if early_stop_metric == 'val_rmse' else val_loss_f

                # best checkpoint selection (monitor_metric)
                if (best_score - monitor_value) > min_delta:
                    best_score = float(monitor_value)
                    best_val_loss = float(val_loss_f)
                    best_val_rmse = float(val_rmse)
                    best_state = True
                else:
                    best_state = False

                # early stop counter (early_stop_metric)
                if (best_early_stop - early_stop_value) > min_delta:
                    best_early_stop = float(early_stop_value)
                    not_improved_count = 0
                else:
                    not_improved_count += 1

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)

            # epoch-level monitoring: time/ETA/GPU/state
            try:
                epoch_sec = float(time.time() - epoch_start)
                epoch_times_sec.append(epoch_sec)
                k = max(1, int(eta_smooth_k))
                recent = epoch_times_sec[-k:]
                avg_epoch_sec = float(sum(recent) / max(1, len(recent)))
                remaining_epochs = int(max(0, int(getattr(self.args, 'epochs', 0)) - int(epoch)))
                logd = {
                    'epoch': int(epoch),
                    'epoch/total': int(getattr(self.args, 'epochs', 0) or 0),
                    'epoch/progress': float(epoch) / float(max(1, int(getattr(self.args, 'epochs', 0) or 0))),
                    'time/epoch_sec': float(epoch_sec),
                    'time/eta_min': float((avg_epoch_sec * remaining_epochs) / 60.0),
                    'train/patience_counter': int(not_improved_count),
                }
                try:
                    if bool(best_state):
                        logd['train/best_epoch'] = int(epoch)
                    elif best_epoch is not None:
                        logd['train/best_epoch'] = int(best_epoch)
                except Exception:
                    if best_epoch is not None:
                        logd['train/best_epoch'] = int(best_epoch)
                try:
                    if best_score == best_score and best_score != float('inf'):
                        logd['train/best_loss'] = float(best_score)
                except Exception:
                    pass
                # throughput (optional; best-effort)
                try:
                    samples = int(getattr(self, '_last_train_samples', 0) or 0)
                    if float(train_sec) > 0 and samples > 0:
                        logd['time/samples_per_sec'] = float(samples / float(train_sec))
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        logd['gpu/memory_allocated_gb'] = float(torch.cuda.memory_allocated() / 1e9)
                        logd['gpu/memory_reserved_gb'] = float(torch.cuda.memory_reserved() / 1e9)
                        logd['gpu/max_memory_allocated_gb'] = float(torch.cuda.max_memory_allocated() / 1e9)
                        logd['gpu/max_memory_reserved_gb'] = float(torch.cuda.max_memory_reserved() / 1e9)
                    except Exception:
                        pass
                self._wandb_log(logd, step=self.batch_seen)
            except Exception:
                pass

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    early_stop_epoch = int(epoch)
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                best_model_test = copy.deepcopy(self.model)
                best_epoch = int(epoch)
                # Best snapshot: direct-OSS for big files (optional), else local save + async OSS upload, and write snapshot JSON
                try:
                    import os as _os, io as _io, json
                    base_stem = _os.path.splitext(self.log_filename)[0] if hasattr(self, 'log_filename') else 'supervised_best'
                    named_pth = _os.path.join(self.args.log_dir, f"{base_stem}.pth")
                    best_link = _os.path.join(self.args.log_dir, 'best_model.pth')
                    # OSS components
                    ds_slug = results_io.dataset_slug(getattr(self.args, 'dataset', ''))
                    md_slug = str(getattr(self.args, 'model', 'model')).lower()
                    seed = int(getattr(self.args, 'seed', 0))
                    run_tag = getattr(self.args, 'run_tag', '')
                    bucket = os.getenv('REMOTE_ARTIFACT_BUCKET', '')
                    prefix = f"pi-gwn-tec/{ds_slug}/{md_slug}/seed_{seed}/{run_tag}/"
                    s3_uri = f"oss://{bucket}/{prefix}best_model.pth" if bucket else None
                    # Env controls
                    direct_on = str(os.getenv('REMOTE_ARTIFACT_UPLOAD','')).lower() in ('1','true','yes')
                    thr_mb = int(os.getenv('REMOTE_UPLOAD_THRESHOLD_MB','50'))
                    # Default to writing best_model.ossref unless explicitly disabled
                    write_ref = str(os.getenv('WRITE_REMOTE_REF','1')).lower() in ('1','true','yes')
                    trust_env_oss = str(os.getenv('REMOTE_ARTIFACT_TRUST_ENV','')).lower() in ('1','true','yes')
                    # Log OSS / direct-upload status once for easier debugging
                    if not self._oss_status_logged:
                        try:
                            if not bucket:
                                self.logger.info('OSS disabled: REMOTE_ARTIFACT_BUCKET not set; using local checkpoints only.')
                            elif requests is None or oss2 is None:
                                self.logger.info('OSS disabled: requests/oss2 not available; using local checkpoints only.')
                            elif not direct_on:
                                self.logger.info('OSS available but direct upload disabled (REMOTE_ARTIFACT_UPLOAD not set to true); '
                                                 'will use local save + async upload when possible.')
                            else:
                                self.logger.info(f'OSS direct upload enabled: bucket={bucket}, '
                                                 f'threshold={thr_mb}MB, trust_env_oss={trust_env_oss}')
                        except Exception:
                            pass
                        self._oss_status_logged = True
                    # Serialize to memory and decide
                    buf = _io.BytesIO(); torch.save(best_model, buf); size_bytes = buf.tell(); buf.seek(0)
                    did_direct = False
                    if direct_on and bucket and (requests is not None) and (oss2 is not None) and (size_bytes > thr_mb*1024*1024):
                        try:
                            # schedule background direct upload from memory and join at end of training
                            bytes_data = buf.getvalue()
                            self._best_bytes = bytes_data
                            self._best_s3_uri = s3_uri
                            endpoint = os.environ.get('REMOTE_ARTIFACT_ENDPOINT','oss-accelerate.aliyuncs.com')
                            def _direct_upload_bytes(data: bytes, bucket_name: str, key_full: str, endpoint_url: str, trust_env: bool):
                                try:
                                    ak = os.environ['REMOTE_ARTIFACT_ACCESS_KEY_ID']; sk = os.environ['REMOTE_ARTIFACT_ACCESS_KEY_SECRET']
                                    auth = oss2.Auth(ak, sk)
                                    endpoint_full = endpoint_url if str(endpoint_url).startswith('http') else ('https://' + str(endpoint_url))
                                    proxies = None if bool(trust_env) else {}
                                    bkt = oss2.Bucket(auth, endpoint_full, bucket_name, session=oss2.Session(), proxies=proxies)
                                    bkt.put_object(key_full, data)
                                except Exception:
                                    pass
                            t = Thread(target=_direct_upload_bytes, args=(bytes_data, bucket, prefix+'best_model.pth', endpoint, trust_env_oss), daemon=False)
                            t.start(); self._oss_threads.append(t)
                            did_direct = True
                            self._best_direct_uploaded = True
                            self.logger.info(f"Direct-upload scheduled for OSS: {s3_uri}")
                            if write_ref and s3_uri:
                                try:
                                    with open(_os.path.join(self.args.log_dir, 'best_model.ossref'), 'w') as rf:
                                        rf.write(s3_uri)
                                except Exception:
                                    pass
                            # W&B external artifact reference
                            try:
                                if wandb is not None and s3_uri:
                                    art = wandb.Artifact(name=f"best-ossref-{getattr(self.args,'wandb_run_id','')}", type='external')
                                    art.add_reference(s3_uri)
                                    self._wandb_log_artifact(art, aliases=['best'])
                                    self._wandb_log({
                                        'best/monitor_is_rmse': int(monitor_metric == 'val_rmse'),
                                        'best/monitor_value': float(best_score),
                                        'best/val_loss': float(best_val_loss),
                                        'best/val_rmse': float(best_val_rmse),
                                    }, step=self.batch_seen)
                            except Exception:
                                pass
                        except Exception as e:
                            self.logger.warning(f"Direct OSS upload failed, fallback to local save: {e}")
                    if not did_direct:
                        # Local save and async upload
                        torch.save(best_model, named_pth)
                        self.logger.info("Saving current model to " + named_pth)
                        try:
                            if _os.path.islink(best_link) or _os.path.exists(best_link):
                                _os.remove(best_link)
                            _os.symlink(_os.path.basename(named_pth), best_link)
                            self.logger.info("Updated symlink -> best_model.pth")
                        except Exception as e:
                            self.logger.warning(f"Failed to create best_model.pth symlink: {e}")
                        try:
                            if bucket and (requests is not None) and (oss2 is not None):
                                def _upload_best(path: str, key: str):
                                    try:
                                        ak = os.environ['REMOTE_ARTIFACT_ACCESS_KEY_ID']; sk = os.environ['REMOTE_ARTIFACT_ACCESS_KEY_SECRET']
                                        endpoint = os.environ.get('REMOTE_ARTIFACT_ENDPOINT','oss-accelerate.aliyuncs.com')
                                        auth = oss2.Auth(ak, sk)
                                        endpoint_full = endpoint if str(endpoint).startswith('http') else ('https://' + str(endpoint))
                                        proxies = None if bool(trust_env_oss) else {}
                                        bkt = oss2.Bucket(auth, endpoint_full, bucket, session=oss2.Session(), proxies=proxies)
                                        key_full = prefix + key
                                        sz = os.path.getsize(path)
                                        if sz <= 50*1024*1024:
                                            with open(path, 'rb') as f:
                                                bkt.put_object(key_full, f)
                                        else:
                                            import tarfile
                                            tmp = path + '.tar.gz'
                                            with tarfile.open(tmp, 'w:gz') as tar:
                                                tar.add(path, arcname=os.path.basename(path))
                                            oss2.resumable_upload(bkt, key_full+'.tar.gz', tmp, num_threads=6, part_size=10*1024*1024)
                                            try:
                                                os.remove(tmp)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                try:
                                    t = Thread(target=_upload_best, args=(best_link, 'best_model.pth'), daemon=False)
                                    t.start(); self._oss_threads.append(t)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # best_snapshot.json
                    try:
                        snap = {
                            'epoch': int(epoch),
                            'step': int(self.batch_seen),
                            'metrics': {
                                'val_loss': float(best_val_loss),
                                'val_rmse': float(best_val_rmse),
                                'monitor_metric': str(monitor_metric),
                                'monitor_value': float(best_score),
                                'early_stop_metric': str(early_stop_metric),
                                'early_stop_best': float(best_early_stop),
                            },
                            'timestamp': getattr(self.args, 'ts_utc', ''),
                            'run_tag': run_tag,
                            'wandb_run_id': getattr(self.args, 'wandb_run_id', None),
                            'checkpoint_reference_uri': s3_uri if bucket else None,
                            'oss_prefix': prefix,
                        }
                        with open(os.path.join(self.args.log_dir, 'best_snapshot.json'), 'w') as f:
                            json.dump(snap, f, indent=2)
                        try:
                            if wandb is not None:
                                art2 = wandb.Artifact(name=f"best-snapshot-{getattr(self.args,'wandb_run_id','')}", type='results')
                                art2.add_file(os.path.join(self.args.log_dir, 'best_snapshot.json'))
                                self._wandb_log_artifact(art2, aliases=['best'])
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass

        training_time = time.time() - start_time
        # Persist compute-cost fields to args for results_io
        epochs_ran = int(len(train_loss_list))
        try:
            self.args._train_wall_time_s = float(training_time)
            self.args._train_epochs_ran = int(epochs_ran)
            self.args._train_time_per_epoch_s = float(training_time / max(1, epochs_ran))
            self.args._best_epoch = int(best_epoch) if best_epoch is not None else None
            self.args._early_stop_epoch = int(early_stop_epoch) if early_stop_epoch is not None else None
            self.args._best_loss = float(best_score) if best_score == best_score else None
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                self.args._train_max_memory_allocated_MiB = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            except Exception:
                pass
            try:
                self.args._train_max_memory_reserved_MiB = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
            except Exception:
                pass
        # annotate protocol semantics
        try:
            proto = {
                'train_includes_val': True,
                'train_excludes_test': True,
                'train_excludes_upload': True,
                'mem_unit': 'MiB',
            }
            existing = getattr(self.args, '_compute_cost_protocol', None)
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(proto)
                self.args._compute_cost_protocol = merged
            else:
                self.args._compute_cost_protocol = proto
        except Exception:
            pass
        try:
            if self.args.mode == 'pretrain':
                self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_score))
            else:
                self.logger.info(
                    "Total training time: {:.4f}min, best {}: {:.6f} (val_loss@best={:.6f}, val_rmse@best={:.6f})".format(
                        (training_time / 60),
                        monitor_metric,
                        float(best_score),
                        float(best_val_loss),
                        float(best_val_rmse),
                    )
                )
        except Exception:
            self.logger.info("Total training time: {:.4f}min, best score: {:.6f}".format((training_time / 60), best_score))
        # Benchmark pure-forward inference on a prefetched real batch (bs=1), exclude dataloader overhead.
        try:
            model_for_bench = best_model_test if best_model_test is not None else self.model
            bench_loader = self.test_loader
            if isinstance(bench_loader, dict):
                bench_loader = bench_loader.get('test', None) or next((v for v in bench_loader.values() if v is not None), None)
            if bench_loader is not None:
                self.benchmark_inference(model_for_bench, bench_loader, warmup=20, repeat=100)
        except Exception:
            pass
        # Join pending OSS uploads then fallback-sync best if missing
        try:
            for t in getattr(self, '_oss_threads', []):
                try:
                    t.join()
                except Exception:
                    pass
            bucket = os.getenv('REMOTE_ARTIFACT_BUCKET', '')
            if bucket and (requests is not None) and (oss2 is not None):
                ds_slug = results_io.dataset_slug(getattr(self.args, 'dataset', ''))
                md_slug = str(getattr(self.args, 'model', 'model')).lower()
                seed = int(getattr(self.args, 'seed', 0))
                run_tag = getattr(self.args, 'run_tag', '')
                prefix = f"pi-gwn-tec/{ds_slug}/{md_slug}/seed_{seed}/{run_tag}/"
                endpoint = os.environ.get('REMOTE_ARTIFACT_ENDPOINT','oss-accelerate.aliyuncs.com')
                trust_env_oss = str(os.getenv('REMOTE_ARTIFACT_TRUST_ENV','')).lower() in ('1','true','yes')
                try:
                    ak = os.environ['REMOTE_ARTIFACT_ACCESS_KEY_ID']; sk = os.environ['REMOTE_ARTIFACT_ACCESS_KEY_SECRET']
                    auth = oss2.Auth(ak, sk)
                    endpoint_full = endpoint if str(endpoint).startswith('http') else ('https://' + str(endpoint))
                    proxies = None if bool(trust_env_oss) else {}
                    bkt = oss2.Bucket(auth, endpoint_full, bucket, session=oss2.Session(), proxies=proxies)
                    key_full = prefix + 'best_model.pth'
                    ok = False
                    try:
                        ok = bool(bkt.object_exists(key_full))
                    except Exception:
                        ok = False
                    if not ok:
                        try:
                            self.logger.warning('WARN: Background upload for best_model.pth seems to have failed. '
                                                'Retrying synchronously to ensure completion.')
                        except Exception:
                            pass
                        # try memory bytes
                        if self._best_bytes is not None:
                            try:
                                bkt.put_object(key_full, self._best_bytes)
                                ok = True
                                self.logger.info('Fallback uploaded best_model.pth to OSS from memory')
                            except Exception:
                                ok = False
                        # try local file
                        if not ok:
                            try:
                                base_stem = os.path.splitext(self.log_filename)[0] if hasattr(self, 'log_filename') else 'supervised_best'
                                named_pth = os.path.join(self.args.log_dir, f"{base_stem}.pth")
                                best_link = os.path.join(self.args.log_dir, 'best_model.pth')
                                path = best_link if os.path.exists(best_link) else named_pth
                                if os.path.exists(path):
                                    with open(path, 'rb') as f:
                                        bkt.put_object(key_full, f)
                                    ok = True
                                    self.logger.info('Fallback uploaded best_model.pth to OSS from file')
                            except Exception:
                                pass
                        # explicit log for fallback start and failure cases
                        if not ok:
                            try:
                                self.logger.warning('WARN: Fallback synchronous upload for best_model.pth did not succeed. '
                                                    'Please check OSS credentials, endpoint, and network connectivity.')
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

        # Save the best model to file (always) unless direct-upload was requested and used
        try:
            if getattr(self, '_best_direct_uploaded', False):
                pass
            else:
                target_state = best_model if 'best_model' in locals() else self.model.state_dict()
                # pretrain: save named + symlink latest_pretrain.pth; supervised: save best_model.pth
                if getattr(self.args, 'mode', '') == 'pretrain':
                    # save named
                    torch.save(target_state, self.best_path)
                    self.logger.info("Saving pretrain model to " + self.best_path)
                    # create/refresh symlink latest_pretrain.pth
                    try:
                        import os as _os
                        latest_link = _os.path.join(self.args.log_dir, 'latest_pretrain.pth')
                        if _os.path.islink(latest_link) or _os.path.exists(latest_link):
                            try:
                                _os.remove(latest_link)
                            except Exception:
                                pass
                        _os.symlink(_os.path.basename(self.best_path), latest_link)
                        self.logger.info("Updated symlink -> latest_pretrain.pth")
                    except Exception as e:
                        self.logger.warning(f"Failed to create latest_pretrain.pth symlink: {e}")
                else:
                    # supervised: save with log basename and create/refresh symlink best_model.pth
                    try:
                        import os as _os
                        base_stem = _os.path.splitext(self.log_filename)[0] if hasattr(self, 'log_filename') else 'supervised_best'
                        named_pth = _os.path.join(self.args.log_dir, f"{base_stem}.pth")
                        torch.save(target_state, named_pth)
                        self.logger.info("Saving current model to " + named_pth)
                        # symlink best_model.pth -> named file
                        best_link = _os.path.join(self.args.log_dir, 'best_model.pth')
                        try:
                            if _os.path.islink(best_link) or _os.path.exists(best_link):
                                _os.remove(best_link)
                            _os.symlink(_os.path.basename(named_pth), best_link)
                            self.logger.info("Updated symlink -> best_model.pth")
                        except Exception as e:
                            self.logger.warning(f"Failed to create best_model.pth symlink: {e}")
                    except Exception as e:
                        # fallback to legacy path
                        import torch as _torch
                        _torch.save(target_state, self.best_path)
                        self.logger.info("Saving current model to " + self.best_path)
        except Exception as e:
            self.logger.warning(f"Failed to save model: {e}")

        #test
        # self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        if self.args.mode == 'pretrain':
            model_for_test = best_model_test if best_model_test is not None else self.model
            self.test(model_for_test, self.args, self.train_loader, self.scaler, self.logger)
        else:
            # save arrays: save test (+ optional ood) only (float16); disable val arrays by default
            if isinstance(self.test_loader, dict):
                for _tag, _dl in self.test_loader.items():
                    if _dl is None:
                        continue
                    self.test(best_model_test, self.args, _dl, self.scaler, self.logger,
                              save_arrays=True, save_tag=str(_tag), log_filename=self.log_filename)
            else:
                self.test(best_model_test, self.args, self.test_loader, self.scaler, self.logger,
                          save_arrays=True, save_tag='test', log_filename=self.log_filename)
            if self.val_loader is not None:
                self.test(best_model_test, self.args, self.val_loader, self.scaler, self.logger,
                          save_arrays=False, save_tag='val', log_filename=self.log_filename)
            # Optional: year-wise evaluation for GIMtec + CSA_WTConvLSTM to match original repo analysis
            try:
                if getattr(self.args, 'model', None) == 'CSA_WTConvLSTM' and self.args.dataset.lower() in ['gimtec', 'tec']:
                    from lib.eval_gimtec_yearwise import evaluate_per_year
                    evaluate_per_year(best_model_test, self)
                # Always save JSON metrics for GIMtec/TEC at end of training
                if self.args.dataset.lower() in ['gimtec', 'tec'] and not bool(getattr(self.args, 'debug', False)):
                    from lib.eval_gimtec_yearwise import compute_yearwise_metrics
                    metrics = compute_yearwise_metrics(best_model_test, self)
                    # derive json filename from log filename to keep naming consistent
                    base_stem = os.path.splitext(self.log_filename)[0]
                    json_name = f"{base_stem}.json"
                    out_path = os.path.join(self.args.log_dir, json_name)
                    # Inject meta (env/oom/model/data/paths)
                    try:
                        preds_path = os.path.join(self.args.log_dir, f"{base_stem}_test_preds.npy")
                        ckpt_path = getattr(self, 'best_run_path', os.path.join(self.args.log_dir, 'best_model.pth'))
                        metrics['meta'] = {
                            'run_name': base_stem,
                            'paths': {
                                'preds': preds_path,
                                'log': os.path.join(self.args.log_dir, self.log_filename),
                                'ckpt': ckpt_path,
                            },
                            'env': {
                                'amp': 'bf16' if bool(getattr(self.args, 'amp', False)) else 'none',
                                'tf32': bool(getattr(torch.backends.cuda.matmul, 'allow_tf32', True)),
                                'ln_enabled': bool(getattr(self.args, 'use_ln', False)),
                            },
                            'oom_fallback': {
                                'initial_batch_size': int(getattr(self.args, 'batch_size', 0)),
                                'final_batch_size': int(getattr(self.args, 'batch_size', 0)),
                                'events': [],
                            },
                            'model': {
                                'gpt2_name': str(getattr(self.args, 'hf_model_name', 'gpt2-large')),
                                'llm_layers': int(getattr(self.args, 'llm_layers', 0)),
                                'frozen': True,
                            },
                            'data': {
                                'dataset': str(self.args.dataset),
                                'channels': int(getattr(self.args, 'input_base_dim', 1)),
                                'stride': 1,
                                'interval_minutes': int(getattr(self.args, 'interval', 120)),
                                'lag': int(getattr(self.args, 'lag', 12)),
                                'horizon': int(getattr(self.args, 'horizon', 12)),
                            }
                        }
                    except Exception:
                        pass
                    with open(out_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    self.logger.info('Saved JSON metrics to {}'.format(out_path))
            except Exception as e:
                self.logger.warning(f'Year-wise evaluation skipped due to error: {e}')


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        # Save to run_name.pth
        torch.save(state, self.best_run_path)
        # Update best_model.pth symlink (or copy if symlink unsupported)
        try:
            if os.path.islink(self.best_link_path) or os.path.exists(self.best_link_path):
                try:
                    os.remove(self.best_link_path)
                except Exception:
                    pass
            os.symlink(os.path.basename(self.best_run_path), self.best_link_path)
        except Exception:
            # fallback: copy file
            import shutil
            try:
                shutil.copyfile(self.best_run_path, self.best_link_path)
            except Exception:
                pass
        self.logger.info("Saving current best model to " + self.best_run_path + " (link: best_model.pth)")

    def test(self, model, args, data_loader, scaler, logger, path=None, save_arrays=False, save_tag='test', log_filename=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        tag = str(save_tag) if save_tag is not None else 'eval'
        y_pred = []
        y_true = []
        # Physics metrics (posterior checks):
        # - A: nonneg/ROT/ROTI always computed on test.
        # - B: night monotonicity + dynamic caps computed only when drivers are truly available.
        compute_phys = (tag == 'test') and (str(getattr(args, 'mode', '')).lower() != 'pretrain')
        dt_min = float(getattr(args, 'interval', 120))
        rot_cap = float(getattr(args, 'rot_cap', 0.5))
        roti_w = 3
        roti_cap_scale = float(getattr(args, 'roti_cap_scale', 0.7))
        dyn_on = bool(getattr(args, 'use_dynamic_rot_cap', False))
        rot_cap_k = float(getattr(args, 'rot_cap_k', 0.01))

        # Accumulators (numerator/denominator style; skip non-finite values)
        nonneg_num = 0
        nonneg_den = 0
        rot_num = 0
        rot_den = 0
        roti_num = 0
        roti_den = 0

        night_num = 0
        night_den = 0

        dyn_rot_num = 0
        dyn_rot_den = 0
        dyn_roti_num = 0
        dyn_roti_den = 0
        dyn_caps_sum = None  # torch.Tensor[T-1] on CPU
        dyn_caps_cnt = 0

        drivers_store = None
        drivers_ok = False
        try:
            if compute_phys and os.path.exists(getattr(self, '_drivers_path', '')):
                from lib.physics.drivers import DriversStore
                drivers_store = DriversStore(self._drivers_path, device=args.device)
                # Treat synthetic fallback as "not available" to avoid misleading B metrics.
                drivers_ok = not bool(getattr(drivers_store, 'synthetic', False))
        except Exception:
            drivers_store = None
            drivers_ok = False
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    data, target, start_idx = batch
                else:
                    data, target = batch
                    start_idx = None
                data = data.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                data = data[..., :args.input_base_dim + args.input_extra_dim]
                if args.mode == 'pretrain':
                    output, _, mask, _, _ = model(data, None, None, args.epochs)
                    label = data[..., :args.output_dim]
                    y_true.append(label*mask)
                    y_pred.append(output*mask)
                else:
                    drivers_input = self._build_gwn_drivers_input(start_idx, int(data.shape[1]), int(data.shape[2]), args=args)
                    try:
                        output, _, mask, _, _ = model(data, label=None, batch_seen=None, drivers_input=drivers_input)
                    except TypeError:
                        output, _, mask, _, _ = model(data, label=None)
                    label = target[..., :args.output_dim]
                    # Batch D: hybrid residual inference uses y = y_bg + y_res_pred
                    if isinstance(output, torch.Tensor) and output.dim() == 3:
                        output = output.unsqueeze(-1)
                    if isinstance(label, torch.Tensor) and label.dim() == 3:
                        label = label.unsqueeze(-1)
                    if self.hybrid_enabled:
                        bg = self._hybrid_bg_tensor(tag, start_idx, ref_tensor=output)
                        output = output + bg
                    y_true.append(label)
                    y_pred.append(output)
                # ensure 4D outputs for physics metrics
                try:
                    if isinstance(output, torch.Tensor) and output.dim() == 3:
                        output = output.unsqueeze(-1)
                        y_pred[-1] = output
                    if isinstance(label, torch.Tensor) and label.dim() == 3:
                        label = label.unsqueeze(-1)
                        y_true[-1] = label
                except Exception:
                    pass
                # Physics metrics (full-batch accurate) — test only
                try:
                    if compute_phys and isinstance(output, torch.Tensor) and dt_min > 0:
                        T = int(output.shape[1])
                        N = int(output.shape[2])
                        if T < 2:
                            continue

                        # Convert to TECU domain for physics checks
                        yp = scaler.inverse_transform(output)  # [B,T,N,1]

                        # A1) nonneg_rate
                        finite = torch.isfinite(yp)
                        nonneg_den += int(finite.sum().item())
                        nonneg_num += int(((yp >= 0.0) & finite).sum().item())

                        # A2) ROT/ROTI validity (TECU per minute)
                        rot = (yp[:, 1:, ...] - yp[:, :-1, ...]) / dt_min  # [B,T-1,N,1]
                        rot_f = torch.isfinite(rot)
                        rot_den += int(rot_f.sum().item())
                        rot_num += int(((torch.abs(rot) <= rot_cap) & rot_f).sum().item())

                        roti = None
                        roti_f = None
                        if int(rot.shape[1]) >= roti_w:
                            rw = rot.unfold(dimension=1, size=roti_w, step=1)  # [B,T-1-w+1,N,1,w]
                            roti = rw.std(dim=-1, unbiased=False)  # [B,T-1-w+1,N,1]
                            roti_f = torch.isfinite(roti)
                            roti_den += int(roti_f.sum().item())
                            roti_cap = roti_cap_scale * rot_cap
                            roti_num += int(((roti <= roti_cap) & roti_f).sum().item())

                        # B) Drivers-dependent metrics (night monotonicity + dynamic caps)
                        if drivers_store is None or (not drivers_ok) or start_idx is None:
                            continue
                        try:
                            offset = int(getattr(args, 'lag', 0))
                            if bool(getattr(args, 'single', False)):
                                offset += int(getattr(args, 'horizon', 0)) - 1
                            d = drivers_store.slice_batched(start_idx + offset, T, N)
                        except Exception:
                            continue

                        # B1) Night monotonicity: cosSZA<=0 -> night, require ΔTEC<=0
                        if 'cosSZA' in d:
                            try:
                                cosz = d['cosSZA']  # [B,T,N]
                                night = (cosz[:, :-1, :] <= 0.0)
                                delta = (yp[:, 1:, :, 0] - yp[:, :-1, :, 0])
                                ok = (delta <= 0.0)
                                night_den += int(night.sum().item())
                                night_num += int((ok & night).sum().item())
                            except Exception:
                                pass

                        # B2) Dynamic caps from Dst (optional)
                        if dyn_on and ('Dst' in d):
                            try:
                                dst = d['Dst']  # [B,T]
                                dstn = torch.clamp(-dst, min=0.0)
                                cap = rot_cap * (1.0 + rot_cap_k * (dstn / 100.0))  # [B,T]
                                cap_rot = cap[:, 1:]  # [B,T-1]

                                # accumulate per-step caps mean (for dynamic_rot_cap.npy)
                                cap_sum = cap_rot.detach().to(torch.float64).sum(dim=0).cpu()
                                if dyn_caps_sum is None:
                                    dyn_caps_sum = cap_sum
                                else:
                                    dyn_caps_sum += cap_sum
                                dyn_caps_cnt += int(cap_rot.shape[0])

                                cap_rot_b = cap_rot.view(int(cap_rot.shape[0]), int(cap_rot.shape[1]), 1, 1)
                                dyn_rot_den += int(rot_f.sum().item())
                                dyn_rot_num += int(((torch.abs(rot) <= cap_rot_b) & rot_f).sum().item())

                                if roti is not None and roti_f is not None:
                                    roti_T = int(roti.shape[1])
                                    cap_roti = (roti_cap_scale * cap_rot[:, :roti_T]).view(int(cap_rot.shape[0]), roti_T, 1, 1)
                                    dyn_roti_den += int(roti_f.sum().item())
                                    dyn_roti_num += int(((roti <= cap_roti) & roti_f).sum().item())
                            except Exception:
                                pass
                except Exception:
                    pass

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        # For GIMtec/TEC datasets, align reporting with CSA pipeline: drop MAPE from logs
        disable_mape = (args.dataset.lower() in ['gimtec', 'tec'])

        def _corr_with_ratio(pred_t, true_t, mask_value=None):
            # Harmonize shapes like CORR_torch
            x, y = pred_t, true_t
            if x.dim() == 2:  # [B,N]
                x = x.unsqueeze(1).unsqueeze(1); y = y.unsqueeze(1).unsqueeze(1)
            elif x.dim() == 3:  # [B,T,N]
                x = x.transpose(1, 2).unsqueeze(1); y = y.transpose(1, 2).unsqueeze(1)
            elif x.dim() == 4:  # [B,T,N,D]
                x = x.transpose(2, 3); y = y.transpose(2, 3)
            dims = (0, 1, 2)
            xm = x.mean(dim=dims); ym = y.mean(dim=dims)
            xs = x.std(dim=dims); ys = y.std(dim=dims)
            numer = ((x - xm)*(y - ym)).mean(dim=dims)
            denom = (xs*ys)
            corr = numer / denom
            valid = torch.isfinite(corr) & torch.isfinite(denom) & (ys != 0)
            if valid.numel() == 0:
                return float('nan'), 0.0
            valid_vals = corr[valid]
            ratio = float(valid.float().mean().item())
            if valid_vals.numel() == 0:
                return float('nan'), ratio
            return float(torch.nanmean(valid_vals).item()), ratio

        # tag 已在函数入口处确定（用于区分 test/val）
        # Always report only MAE/RMSE/R2/CORR
        def _log_metrics_block(tag_name: str, yp_all: torch.Tensor, yt_all: torch.Tensor) -> None:
            corr_list = []
            for t in range(yt_all.shape[1]):
                mae, _ = MAE_torch(yp_all[:, t, ...], yt_all[:, t, ...], args.mae_thresh)
                rmse = RMSE_torch(yp_all[:, t, ...], yt_all[:, t, ...], args.mae_thresh)
                corr_val, ratio = _corr_with_ratio(yp_all[:, t, ...], yt_all[:, t, ...], args.mae_thresh)
                # R2 in TECU domain via numpy
                try:
                    import numpy as _np
                    yt_np = yt_all[:, t, ...].detach().cpu().numpy()
                    yp_np = yp_all[:, t, ...].detach().cpu().numpy()
                    m = _np.isfinite(yt_np) & _np.isfinite(yp_np)
                    if m.any():
                        mu = float(_np.mean(yt_np[m]))
                        ss_res = float(_np.sum((yp_np[m] - yt_np[m]) ** 2))
                        ss_tot = float(_np.sum((yt_np[m] - mu) ** 2))
                        r2 = float('nan') if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
                    else:
                        r2 = float('nan')
                except Exception:
                    r2 = float('nan')
                corr_list.append(corr_val)
                logger.info("[{}] Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, R2:{:.4f}, CORR:{:.4f} (valid={:.1f}%)".format(
                    tag_name, t + 1, mae, rmse, 0.0 if r2 != r2 else r2,
                    0.0 if corr_val != corr_val else corr_val, ratio * 100))
            mae, _ = MAE_torch(yp_all, yt_all, args.mae_thresh)
            rmse = RMSE_torch(yp_all, yt_all, args.mae_thresh)
            # overall r2 by numpy
            try:
                import numpy as _np
                yt = yt_all.detach().cpu().numpy()
                yp = yp_all.detach().cpu().numpy()
                valid = _np.isfinite(yt) & _np.isfinite(yp)
                if valid.any():
                    ss_res = float(_np.sum((yp[valid] - yt[valid]) ** 2))
                    mu = float(_np.mean(yt[valid]))
                    ss_tot = float(_np.sum((yt[valid] - mu) ** 2))
                    r2_overall = float('nan') if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
                else:
                    r2_overall = float('nan')
            except Exception:
                r2_overall = float('nan')
            corr_vals = [c for c in corr_list if not (c != c)]
            corr_avg = sum(corr_vals) / len(corr_vals) if len(corr_vals) > 0 else float('nan')
            logger.info("[{}] Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, R2:{:.4f}, CORR:{:.4f}".format(
                tag_name, mae, rmse, 0.0 if r2_overall != r2_overall else r2_overall,
                0.0 if corr_avg != corr_avg else corr_avg))
            # Log summary metrics to W&B
            try:
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        f"{tag_name}/MAE": float(mae),
                        f"{tag_name}/RMSE": float(rmse),
                        f"{tag_name}/R2": 0.0 if r2_overall != r2_overall else float(r2_overall),
                        f"{tag_name}/CORR": 0.0 if corr_avg != corr_avg else float(corr_avg),
                    })
            except Exception:
                pass

        _log_metrics_block(tag, y_pred, y_true)
        if self.holdout_enabled and (self._holdout_keep_idx is not None) and (self._holdout_idx is not None):
            try:
                if int(self._holdout_keep_idx.numel()) > 0:
                    _log_metrics_block(f"{tag}_in_domain",
                                       self._select_nodes(y_pred, self._holdout_keep_idx),
                                       self._select_nodes(y_true, self._holdout_keep_idx))
                if int(self._holdout_idx.numel()) > 0:
                    _log_metrics_block(f"{tag}_holdout",
                                       self._select_nodes(y_pred, self._holdout_idx),
                                       self._select_nodes(y_true, self._holdout_idx))
            except Exception:
                pass

        # physics metrics (posterior checks) — test only
        try:
            if compute_phys:
                phys = {
                    'nonneg_rate': (float(nonneg_num) / float(nonneg_den)) if nonneg_den > 0 else None,
                    'rot_validity': (float(rot_num) / float(rot_den)) if rot_den > 0 else None,
                    'roti_validity': (float(roti_num) / float(roti_den)) if roti_den > 0 else None,
                    'night_monotonicity': (float(night_num) / float(night_den)) if night_den > 0 else None,
                    'rot_validity_dynamic': (float(dyn_rot_num) / float(dyn_rot_den)) if dyn_rot_den > 0 else None,
                    'roti_validity_dynamic': (float(dyn_roti_num) / float(dyn_roti_den)) if dyn_roti_den > 0 else None,
                    'dynamic_windows': int(dyn_caps_cnt),
                    # backward compatibility for older downstream scripts
                    'dynamic_sampled_windows': int(dyn_caps_cnt),
                    'protocol': {
                        'interval_min': float(dt_min),
                        'rot_cap_tecu_per_min': float(rot_cap),
                        'roti_window': int(roti_w),
                        'roti_cap_scale': float(roti_cap_scale),
                        'night_definition': 'cosSZA<=0',
                        'night_monotonicity_condition': 'ΔTEC<=0',
                        'dynamic_rot_cap_enabled': bool(dyn_on),
                        'rot_cap_k': float(rot_cap_k),
                        'drivers_available': bool(drivers_ok),
                    },
                }
                if dyn_caps_cnt > 0 and dyn_caps_sum is not None:
                    try:
                        caps_mean = (dyn_caps_sum / float(dyn_caps_cnt)).detach().cpu().numpy().astype('float32')
                        np.save(os.path.join(args.log_dir, 'dynamic_rot_cap.npy'), caps_mean)
                        phys['dynamic_rot_cap_stats'] = {
                            'mean': float(caps_mean.mean()),
                            'min': float(caps_mean.min()),
                            'max': float(caps_mean.max()),
                            'unit': 'TECU_per_min',
                        }
                        phys['dynamic_rot_cap_path'] = 'dynamic_rot_cap.npy'
                    except Exception as e:
                        phys['dynamic_rot_cap_error'] = str(e)
                        logger.warning(f"dynamic_rot_cap.npy save failed: {e}")
                try:
                    from lib import results_io as _rio
                    _rio._safe_json_dump(phys, os.path.join(args.log_dir, 'physics_metrics.json'))
                except Exception:
                    with open(os.path.join(args.log_dir, 'physics_metrics.json'), 'w') as f:
                        json.dump(phys, f, indent=2)
        except Exception as e:
            logger.warning(f"physics_metrics.json skipped: {e}")

        # optionally save arrays for reproduction (float16 to reduce size)
        if save_arrays and log_filename:
            try:
                base_stem = os.path.splitext(log_filename)[0]
                np.save(os.path.join(args.log_dir, f"{base_stem}_{save_tag}_preds.npy"),
                        y_pred.detach().cpu().to(torch.float16).numpy())
                # Do not save ground-truth arrays by default since they are fully deterministic
                # and can be regenerated from the dataset split/windowing.
                logger.info(f"Saved arrays: {base_stem}_{save_tag}_preds.npy (true not saved)")
            except Exception as e:
                logger.warning(f"Failed to save arrays: {e}")
