
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

# The retained recipe adapter shares this public entry point.
if '--paper-config' in sys.argv:
    from lib.paper_recipe import main
    main()
    raise SystemExit(0)

def _load_env_file(path: str) -> None:
    """Best-effort load repo-root .env into process env (do not override existing keys)."""
    try:
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('export '):
                    s = s[len('export '):].strip()
                if '=' not in s:
                    continue
                k, v = s.split('=', 1)
                k = k.strip()
                v = v.strip()
                if not k:
                    continue
                if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                    v = v[1:-1]
                os.environ.setdefault(k, v)
    except Exception:
        pass

# Ensure repo-root .env is loaded regardless of cwd (uv recommended but not required)
_load_env_file(os.path.join(file_dir, '.env'))

import torch
try:
    # 允许 TF32 以提升矩阵乘吞吐（A100/H100/H800 有效）
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
import torch.nn as nn
import configparser
import time
from datetime import datetime, timezone
from datetime import timedelta
try:
    import yaml
except Exception:
    yaml = None

# --- wandb runtime defaults ---
# 固定到 repo 根目录，避免从不同 cwd 运行导致 wandb 目录漂移
os.environ.setdefault('WANDB_DIR', os.path.join(file_dir, 'wandb'))

try:
    from lib.wandb_import import import_wandb
    wandb = import_wandb()
except Exception:
    wandb = None
from pathlib import Path
from model.Pretrain_model.GPTST import GPTST_Model as Network_Pretrain
from model.Model import Enhance_model as Network_Predict
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.Params_pretrain import parse_args
from lib.Params_predictor import get_predictor_params
from lib import results_io

# *************************************************************************#

# Mode = 'ori'            #ori, eval, pretrain
# DATASET = 'PEMS08'      #PEMS08, METR_LA, NYC_BIKE, NYC_TAXI
# model = 'MSDR'     # ASTGCN CCRNN DMVSTNET GWN MSDR MTGNN STWA STFGNN STGCN STGODE STMGCN STSGCN TGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

from lib.model_names import MODEL_ALIASES, canonical_model_name

args = parse_args(device)
args.model = MODEL_ALIASES.get(args.model, args.model)

def _yaml_overlay(args, args_predictor=None):
    """Overlay YAML config onto args/args_predictor. YAML-first policy.
    Priority: explicit -config > conf/<MODEL>/<DATASET>.yaml > no-op.
    """
    try:
        cfg_path = getattr(args, 'config', '')
        base = Path(file_dir) / 'conf' / str(getattr(args, 'model', '')) / f"{getattr(args, 'dataset', '')}.yaml"
        ypath = Path(cfg_path) if cfg_path else base
        if yaml is None or not ypath.exists():
            return args, args_predictor
        with open(ypath, 'r') as f:
            y = yaml.safe_load(f) or {}
        # sections we understand
        cli_tokens = set(sys.argv[1:])
        protected = {
            # identity / routing keys: never let YAML override explicit CLI
            'dataset','model','mode','seed','log_dir','paper_slug',
            'split_profile',
            'china_hires_data_root',
            # ablation knobs that must respect CLI (e.g. lag/horizon sweeps)
            'lag','horizon',
            'batch_size','epochs','val_ratio','test_ratio','amp','year_split',
            # training knobs: respect explicit CLI (critical for sweeps)
            'accumulate_steps','optimizer','scheduler','lr_init','weight_decay',
            # scheduler/early-stop details (avoid silent override by conf/<MODEL>/<DATASET>.yaml)
            'plateau_factor','plateau_patience','plateau_threshold','plateau_threshold_mode','plateau_cooldown','min_lr',
            'cosine_t_max','eta_min_factor',
            'grad_clip',
            'early_stop','early_stop_patience','early_stop_min_delta',
            'use_pinn','use_drivers','use_drivers_input','use_adv','use_diffusion','nochem','lambda_phys',
            # training monitor/early-stop metric (keep CLI override)
            'monitor_metric','early_stop_metric',
        }
        for sec, kv in y.items():
            if not isinstance(kv, dict):
                continue
            for k, v in kv.items():
                # args priorities: keep CLI single-hyphen overrides
                if (k in protected) and (f'-{k}' in cli_tokens):
                    pass
                elif hasattr(args, k):
                    setattr(args, k, v)
                if args_predictor is not None and hasattr(args_predictor, k):
                    setattr(args_predictor, k, v)
        # unify epochs/early stop defaults if provided
        return args, args_predictor
    except Exception:
        return args, args_predictor
if args.mode !='pretrain':
    args_predictor = get_predictor_params(args)
    # CLI-first policy: if user explicitly passed a single-hyphen flag (e.g., -seed/-debug/-epochs),
    # do NOT let predictor args overwrite it. This is crucial for sweeps and reproducibility.
    cli_tokens = set(sys.argv[1:])
    protected_keys = set()
    try:
        for k in vars(args).keys():
            if f'-{k}' in cli_tokens:
                protected_keys.add(k)
    except Exception:
        protected_keys = set()
    attr_list = []
    for arg in vars(args):
        attr_list.append(arg)
    for attr in attr_list:
        if hasattr(args, attr) and hasattr(args_predictor, attr):
            if attr in protected_keys:
                # keep user-specified main arg value
                continue
            setattr(args, attr, getattr(args_predictor, attr))
    # Overlay YAML onto args + predictor (YAML-first)
    args, args_predictor = _yaml_overlay(args, args_predictor)
    # Keep predictor view consistent for CLI-specified keys (avoid split-brain args vs args_predictor).
    try:
        for k in protected_keys:
            if hasattr(args_predictor, k) and hasattr(args, k):
                setattr(args_predictor, k, getattr(args, k))
    except Exception:
        pass
else:
    args_predictor = None
    # Overlay YAML for pretrain too
    args, _ = _yaml_overlay(args, None)

# Canonicalize again because YAML can supply a historical model identifier.
args.model = canonical_model_name(args.model)
if args_predictor is not None and hasattr(args_predictor, "model"):
    args_predictor.model = canonical_model_name(args_predictor.model)

# print effective arguments
for arg in vars(args):
    print(arg, ':', getattr(args, arg))
print('==========')
if args_predictor is not None:
    for arg in vars(args_predictor):
        print(arg, ':', getattr(args_predictor, arg))
init_seed(args.seed, args.seed_mode)
try:
    # Record determinism knobs for reproducibility (warn_only policy).
    det = {
        'seed': int(getattr(args, 'seed', 0)),
        'seed_mode': bool(getattr(args, 'seed_mode', False)),
        'pyhashseed': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }
    try:
        det['cudnn_deterministic'] = bool(torch.backends.cudnn.deterministic)
        det['cudnn_benchmark'] = bool(torch.backends.cudnn.benchmark)
        det['deterministic_algorithms'] = bool(torch.are_deterministic_algorithms_enabled())
        det['deterministic_warn_only'] = True
    except Exception:
        pass
    proto = getattr(args, '_compute_cost_protocol', None)
    if not isinstance(proto, dict):
        proto = {}
    proto = dict(proto)
    proto['determinism'] = det
    args._compute_cost_protocol = proto
except Exception:
    pass
# UTC ts for standardized naming (e.g., 20250101T000000Z)
try:
    # timezone-aware UTC timestamp
    args.ts_utc = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
except Exception:
    args.ts_utc = time.strftime('%Y%m%dT%H%M%SZ')
try:
    # Beijing time for human-facing W&B names (UTC+8)
    args.ts_bj = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S+0800')
except Exception:
    args.ts_bj = None
args._run_start_ts = args.ts_bj or args.ts_utc
args._run_start_ts_utc = args.ts_utc

model_label = 'gptst' if args.mode == 'pretrain' else args.model
print('mode: ', args.mode, '  model: ', model_label, '  dataset: ', args.dataset, '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)


 # config outputs path (unified with run_tag = ts_utc + 8-char run id)

# Initialize W&B early to get run id (skip for -mode test re-run per user preference)
run = None
args.wandb_run_id = None
cli_tokens = set(sys.argv[1:])
if args.mode == 'test':
    # Do not create a new W&B run for re-test.
    os.environ['WANDB_MODE'] = 'disabled'
    # Reuse an existing run directory provided by -log_dir (must contain best_model.pth).
    out_dir = Path(str(getattr(args, 'log_dir', '') or '')).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path(file_dir) / out_dir).resolve()
    args.log_dir = str(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"test mode requires an existing -log_dir, got: {out_dir}")
    if not (out_dir / 'best_model.pth').exists():
        # fallback to the latest *.pth in the directory
        pths = sorted(out_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pths:
            raise FileNotFoundError(f"test mode requires {out_dir}/best_model.pth or any *.pth checkpoint")
    try:
        # Keep run_tag aligned with directory name for OSS keys.
        args.run_tag = out_dir.name
        if '-' in out_dir.name:
            ts_prefix = out_dir.name.split('-', 1)[0]
            # Derive both UTC/BJ timestamps from directory prefix for accurate manifests.
            try:
                from datetime import datetime as _dt
                if ts_prefix.endswith('Z'):
                    args.ts_utc = ts_prefix
                    dt_utc = _dt.strptime(ts_prefix, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
                    args.ts_bj = dt_utc.astimezone(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S+0800')
                elif ts_prefix.endswith('+0800'):
                    args.ts_bj = ts_prefix
                    dt_bj = _dt.strptime(ts_prefix, '%Y%m%dT%H%M%S+0800').replace(tzinfo=timezone(timedelta(hours=8)))
                    args.ts_utc = dt_bj.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                else:
                    # Unknown format: keep prefix as BJ, keep existing ts_utc
                    args.ts_bj = ts_prefix
            except Exception:
                args.ts_bj = ts_prefix
    except Exception:
        pass
else:
    try:
        if wandb is not None:
            ds_slug = results_io.dataset_slug(getattr(args, 'dataset', ''))
            md_slug = str(getattr(args, 'model', 'model')).lower()
            use_drivers_input = getattr(args, 'use_drivers_input', None)
            if use_drivers_input is None:
                use_drivers_input = bool(getattr(args, 'use_drivers', False))
            tags = [
                f"seed_{int(getattr(args,'seed',0))}",
                f"amp_{bool(getattr(args,'amp',False))}",
                f"pinn_{bool(getattr(args,'use_pinn',False))}",
                f"drivers_phys_{bool(getattr(args,'use_drivers',False))}",
                f"drivers_in_{bool(use_drivers_input)}",
                f"adv_{bool(getattr(args,'use_adv',False))}",
                f"nodiff_{not bool(getattr(args,'use_diffusion',True))}",
                f"nochem_{bool(getattr(args,'nochem',False))}",
                f"acc_{int(getattr(args,'accumulate_steps',1))}",
                f"ys_{bool(getattr(args,'year_split',False))}",
            ]
            run = wandb.run or wandb.init(project=os.getenv('WANDB_PROJECT','pi-gwn-tec'),
                                          entity=os.getenv('WANDB_ENTITY') or None,
                                          config=vars(args),
                                          group=f"{ds_slug}-{md_slug}",
                                          tags=tags)
            args.wandb_run_id = getattr(run, 'id', None)
    except Exception:
        run = None
        args.wandb_run_id = None

    rid8 = (args.wandb_run_id or 'local')[:8]
    run_tag = f"{args.ts_bj or args.ts_utc}-{rid8}"
    # Make W&B email title actionable: run.name includes mode/seed/run_tag and optional paper_slug.
    try:
        if wandb is not None and run is not None:
            run.name = results_io.wandb_run_name(args, run_tag=run_tag)
    except Exception:
        pass
    if args.mode == 'pretrain':
        # tag pretrain runs under pseudo-model 'gptst'
        setattr(args, 'model', 'gptst')
    out_dir = results_io.prepare_outdir(args, run_tag=run_tag)
args.load_pretrain_path = args.load_pretrain_path
args.save_pretrain_path = args.save_pretrain_path

# record command line for logging later
args.cmdline = f"python {os.path.basename(__file__)} " + " ".join(sys.argv[1:])
try:
    results_io.save_config_yaml(out_dir, args, args_predictor=args_predictor, overwrite=False)
except Exception:
    pass

#load dataset
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, scaler_holiday = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
try:
    def _dl_info(dl):
        if dl is None:
            return None
        if isinstance(dl, dict):
            return {str(k): _dl_info(v) for k, v in dl.items()}
        info = {
            'batch_size': getattr(dl, 'batch_size', None),
            'num_workers': getattr(dl, 'num_workers', None),
            'pin_memory': bool(getattr(dl, 'pin_memory', False)),
            'prefetch_factor': getattr(dl, 'prefetch_factor', None),
            'persistent_workers': getattr(dl, 'persistent_workers', None),
            'drop_last': getattr(dl, 'drop_last', None),
        }
        try:
            info['dataset_size'] = int(len(getattr(dl, 'dataset', [])))
        except Exception:
            info['dataset_size'] = None
        return info
    args._dataloader = {
        'train': _dl_info(train_loader),
        'val': _dl_info(val_loader),
        'test': _dl_info(test_loader),
    }
except Exception:
    pass
args.scaler_zeros = scaler_data.transform(0)
args.scaler_zeros_day = scaler_day.transform(0)
args.scaler_zeros_week = scaler_week.transform(0)
# args.scaler_zeros_holiday = scaler_holiday.transform(0)
#init model
if args.mode == 'pretrain':
    model = Network_Pretrain(args)
    model = model.to(args.device)
else:
    model = Network_Predict(args, args_predictor)
    model = model.to(args.device)
# optional gradient watching
try:
    if bool(getattr(args, 'watch_grads', False)) and wandb is not None:
        wandb.watch(model, log='gradients', log_freq=max(1, int(getattr(args, 'log_step', 50))))
except Exception:
    pass

# Optional: initialize from a saved checkpoint for supervised training
init_from = getattr(args, 'init_from', '') if hasattr(args, 'init_from') else ''
if init_from:
    try:
        state = torch.load(init_from, map_location=args.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        print(f"Initialized model from: {init_from}")
    except Exception as e:
        print(f"Warning: failed to init from {init_from}: {e}")

if args.xavier:
    for p in model.parameters():
        if p.requires_grad==True:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

print_model_parameters(model, only_num=False)

#init loss function, optimizer

def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        # Optional: TEC cap-aware down-weighting (e.g., china_hires cap=300)
        cap_value = float(getattr(args, 'tec_cap_value', -1.0))
        cap_weight = float(getattr(args, 'tec_cap_weight', 1.0))
        cap_eps = float(getattr(args, 'tec_cap_eps', 1e-3))
        cap_enabled = (cap_value > 0.0) and (0.0 < cap_weight < 1.0)
        # Apply mask_value semantics first (align with MAE_torch)
        if mask_value is not None:
            m = torch.gt(labels, mask_value)
            preds = torch.masked_select(preds, m)
            labels = torch.masked_select(labels, m)
        mae_loss = torch.abs(labels - preds)
        if cap_enabled and mae_loss.numel() > 0:
            cap_t = torch.tensor(cap_value, device=labels.device, dtype=labels.dtype)
            cap_m = torch.isclose(labels, cap_t, rtol=0.0, atol=cap_eps)
            mae_loss = torch.where(cap_m, mae_loss * cap_weight, mae_loss)
        return torch.mean(mae_loss), mae_loss
    return loss

def scaler_huber_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        cap_value = float(getattr(args, 'tec_cap_value', -1.0))
        cap_weight = float(getattr(args, 'tec_cap_weight', 1.0))
        cap_eps = float(getattr(args, 'tec_cap_eps', 1e-3))
        cap_enabled = (cap_value > 0.0) and (0.0 < cap_weight < 1.0)
        # mask_value 语义：只对 true>mask_value 的样本计算
        if mask_value is not None:
            m = torch.gt(labels, mask_value)
            preds = torch.masked_select(preds, m)
            labels = torch.masked_select(labels, m)
        residual = torch.abs(preds - labels)
        delta = 1.0
        condition = torch.le(residual, delta)
        small_res = 0.5 * torch.square(residual)
        large_res = delta * residual - 0.5 * delta * delta
        hub = torch.where(condition, small_res, large_res)
        if cap_enabled and hub.numel() > 0:
            cap_t = torch.tensor(cap_value, device=labels.device, dtype=labels.dtype)
            cap_m = torch.isclose(labels, cap_t, rtol=0.0, atol=cap_eps)
            hub = torch.where(cap_m, hub * cap_weight, hub)
        return torch.mean(hub), None
    return loss

def scaler_mse_loss(scaler):
    """MSE in inverse-transformed domain, with optional TEC cap down-weighting."""
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        cap_value = float(getattr(args, 'tec_cap_value', -1.0))
        cap_weight = float(getattr(args, 'tec_cap_weight', 1.0))
        cap_eps = float(getattr(args, 'tec_cap_eps', 1e-3))
        cap_enabled = (cap_value > 0.0) and (0.0 < cap_weight < 1.0)
        diff2 = torch.square(preds - labels)
        if cap_enabled and diff2.numel() > 0:
            cap_t = torch.tensor(cap_value, device=labels.device, dtype=labels.dtype)
            cap_m = torch.isclose(labels, cap_t, rtol=0.0, atol=cap_eps)
            diff2 = torch.where(cap_m, diff2 * cap_weight, diff2)
        return torch.mean(diff2)
    return loss

def scaler_l1_loss(scaler):
    """L1 in inverse-transformed domain, with optional TEC cap down-weighting."""
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        cap_value = float(getattr(args, 'tec_cap_value', -1.0))
        cap_weight = float(getattr(args, 'tec_cap_weight', 1.0))
        cap_eps = float(getattr(args, 'tec_cap_eps', 1e-3))
        cap_enabled = (cap_value > 0.0) and (0.0 < cap_weight < 1.0)
        l1 = torch.abs(preds - labels)
        if cap_enabled and l1.numel() > 0:
            cap_t = torch.tensor(cap_value, device=labels.device, dtype=labels.dtype)
            cap_m = torch.isclose(labels, cap_t, rtol=0.0, atol=cap_eps)
            l1 = torch.where(cap_m, l1 * cap_weight, l1)
        return torch.mean(l1)
    return loss

if args.mode == 'pretrain':
    # For pretrain, always use mask-aware loss with scaler; fallback to huber if requested.
    if args.loss_func == 'mask_huber':
        loss = scaler_huber_loss(scaler_data, mask_value=None)
        print('============================scaler_huber_loss (pretrain)')
    else:
        loss = scaler_mae_loss(scaler_data, mask_value=None)
        print('============================scaler_mae_loss (pretrain)')
else:
    if args.loss_func == 'mask_mae':
        loss = scaler_mae_loss(scaler_data, mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    elif args.loss_func == 'mask_huber':
        loss = scaler_huber_loss(scaler_data, mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    elif args.loss_func == 'mae':
        # If tec_cap_* is enabled, use inverse-domain L1 for explicit cap handling.
        if float(getattr(args, 'tec_cap_value', -1.0)) > 0.0 and float(getattr(args, 'tec_cap_weight', 1.0)) < 1.0:
            loss = scaler_l1_loss(scaler_data)
        else:
            loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        # If tec_cap_* is enabled, use inverse-domain MSE for explicit cap handling.
        if float(getattr(args, 'tec_cap_value', -1.0)) > 0.0 and float(getattr(args, 'tec_cap_weight', 1.0)) < 1.0:
            loss = scaler_mse_loss(scaler_data)
        else:
            loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError
loss_kl = nn.KLDivLoss(reduction='sum').to(args.device)

# optimizer
opt_name = getattr(args, 'optimizer', 'adam').lower()
weight_decay = getattr(args, 'weight_decay', 0.0)
import inspect
_supports_fused = False
try:
    _supports_fused = 'fused' in set(inspect.signature(torch.optim.Adam).parameters.keys())
except Exception:
    pass

if opt_name == 'adamw':
    try:
        if _supports_fused and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                          weight_decay=weight_decay, amsgrad=False, fused=True)
        else:
            raise RuntimeError('no fused')
    except Exception:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                      weight_decay=weight_decay, amsgrad=False)
elif opt_name == 'sgd':
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr_init, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
else:
    try:
        if _supports_fused and torch.cuda.is_available():
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                         weight_decay=weight_decay, amsgrad=False, fused=True)
        else:
            raise RuntimeError('no fused')
    except Exception:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                     weight_decay=weight_decay, amsgrad=False)
#learning rate decay
lr_scheduler = None
if getattr(args, 'scheduler', 'none') == 'plateau':
    print('Applying ReduceLROnPlateau scheduler.')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
        threshold_mode=args.plateau_threshold_mode,
        cooldown=args.plateau_cooldown,
        min_lr=args.min_lr,
    )
elif getattr(args, 'scheduler', 'none') == 'cosine':
    print('Applying CosineAnnealingLR scheduler.')
    t_max = args.cosine_t_max if getattr(args, 'cosine_t_max', None) else args.epochs
    eta_min = args.lr_init * getattr(args, 'eta_min_factor', 0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=eta_min)
elif args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=lr_decay_steps,
        gamma=args.lr_decay_rate)

# #config log path
# current_dir = os.path.dirname(os.path.realpath(__file__))
# log_dir = os.path.join(current_dir,'SAVE', args.dataset)
# Mkdir(log_dir)
# args.log_dir = log_dir

#start training
args._train_start_time = time.time()
trainer = Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'pretrain':
    trainer.train()
elif args.mode == 'eval':
    trainer.train()
elif args.mode == 'ori':
    trainer.train()
elif args.mode == 'test':
    # Load checkpoint from -log_dir (prefer best_model.pth; otherwise newest *.pth)
    ckpt_path = Path(args.log_dir) / 'best_model.pth'
    if not ckpt_path.exists():
        cands = sorted(Path(args.log_dir).glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in {args.log_dir}")
        ckpt_path = cands[0]
    state = torch.load(str(ckpt_path), map_location=args.device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    print(f"Load saved model from {ckpt_path}")
    # Pure-forward inference benchmark (bs=1, warmup=20, repeat=100)
    try:
        _bench_dl = test_loader
        if isinstance(test_loader, dict):
            _bench_dl = test_loader.get('test', None) or next(iter(test_loader.values()))
        trainer.benchmark_inference(model, _bench_dl, warmup=20, repeat=100)
    except Exception:
        pass
    if isinstance(test_loader, dict):
        for _tag, _dl in test_loader.items():
            if _dl is None:
                continue
            trainer.test(model, trainer.args, _dl, scaler_data, trainer.logger,
                         save_arrays=True, save_tag=str(_tag), log_filename=getattr(trainer,'log_filename',None))
    else:
        trainer.test(model, trainer.args, test_loader, scaler_data, trainer.logger,
                     save_arrays=True, save_tag='test', log_filename=getattr(trainer,'log_filename',None))
    # Year-wise evaluation for GIMtec + CSA_WTConvLSTM to match original repo analysis
    try:
        if args.dataset.lower() in ['gimtec', 'tec'] and not bool(getattr(args, 'debug', False)):
            # Run year-wise log for CSA (optional): keep compatibility
            if getattr(args, 'model', None) == 'CSA_WTConvLSTM':
                from lib.eval_gimtec_yearwise import evaluate_per_year
                evaluate_per_year(model, trainer)
            # Always produce JSON for GIMtec/TEC in test mode
            from lib.eval_gimtec_yearwise import compute_yearwise_metrics
            metrics = compute_yearwise_metrics(model, trainer)
            import json, os
            base_stem = os.path.splitext(trainer.log_filename)[0] if hasattr(trainer, 'log_filename') else 'eval'
            json_name = f"{base_stem}.json"
            out_path = os.path.join(args.log_dir, json_name)
            with open(out_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print('Saved JSON metrics to', out_path)
    except Exception as e:
        print('Year-wise evaluation skipped due to error:', e)
else:
    raise ValueError
# 确认 GIMtec/TEC 的时间分辨率（每帧≈2小时）
try:
    if str(getattr(args, 'dataset', '')).lower() in ['gimtec', 'tec']:
        args.interval = 120
except Exception:
    pass

# finalize compute_cost timestamp
try:
    args._train_end_time = time.time()
except Exception:
    pass

# W&B finalize + small files upload + offline sync
try:
    results_io.post_run_collect_and_upload(run, args, model, out_dir)
    if run is not None:
        run.finish()
except Exception as e:
    print('results_io post-run failed:', e)

# WANDB offline sync at script end
try:
    # Keep only latest 1 W&B run locally; optionally sync+clean old runs.
    in_sweep = bool(os.getenv('WANDB_SWEEP_ID') or os.getenv('WANDB_AGENT_ID'))
    default_clean = '0' if in_sweep else '1'
    do_clean = str(os.getenv('WANDB_CLEAN_AT_END', default_clean)).lower() in ('1', 'true', 'yes')
    if do_clean:
        from lib.wandb_housekeeping import run_housekeeping
        run_housekeeping()
except Exception:
    pass
