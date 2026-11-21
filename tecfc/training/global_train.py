"""Development-only global GWN training and same-prediction loss exposure.

No evaluation role, evaluation-year manifest or implicit dataset discovery is
supported. Importing this module opens no assets. See GLOBAL_TRAIN_CN.txt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import ModuleType

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LOSSES = ("smooth_l1", "quadratic_matched")
SEEDS = (42, 1024, 2025, 3407, 5000)
TRAIN_YEARS = (2009, 2010, 2011, 2012, 2014, 2016, 2018)
VALIDATION_YEARS = (2013, 2017)
ALLOWED_YEARS = frozenset(TRAIN_YEARS + VALIDATION_YEARS)
REFERENCE_HASHES = {
    "gim_data.py": "1e557f539683394e72f88c9e64a1fcb3654a803736e2c73b432430e0aa3d1bf2",
    "global_model.py": "09ae2e5f2ed5a18ddb498aa394fb4a1c108a4ba15c1b8ae59af43c06128f2d32",
    "model/GWN/GWN.py": "3a2ff52ed880498f1faba94be8fadabec9b74d41da4ec450da4c403b29b25b1b",
    "model/GWN/driver_fusion.py": "85c925220fc2f82765ea658172ff102a7bdb49e30ed541308b73156444b6bc45",
    "lib/physics/driver_scaler.py": "8db60ece0ceab95484089c49f93590015b67a3e94c160d1389a9b70df32388bc",
}
ASSET_NAMES = {"prepared/graph_svd_v1.npz", "prepared/train_climatology_v1.npz"}
SUM_FIELDS = ("valid_count", "tail_count", "normalized_sse", "normalized_l1_sum",
              "quadratic_sum", "smooth_l1_sum", "quadratic_minus_smooth_l1_sum",
              "tecu_sse", "tecu_l1_sum")
RANGE_FIELDS = ("output_normalized", "output_tecu", "residual_normalized", "residual_tecu")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save_atomic(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    torch.save(value, temp)
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_digest(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


def loss_and_exposure(prediction, target, valid, scale, loss_name):
    """Use the original SmoothL1 kernel in both arms; diagnostics are detached.

    All normalized diagnostics use float64 arithmetic on the float32 residual
    e = pred.float() - target.float()/scale. TECU diagnostics use the original
    float32 output conversion before float64 subtraction. Output ranges include
    all cells; residual ranges and all sums/counts use original valid labels.
    """
    if loss_name not in LOSSES or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid loss or scale")
    if prediction.shape != target.shape or target.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("prediction, target and boolean valid must have identical shapes")
    if not bool(valid.any()) or not bool(torch.isfinite(prediction).all()):
        raise ValueError("empty valid support or nonfinite prediction")
    p = prediction.float()
    y = target.to(device=p.device).float()
    mask = valid.to(device=p.device)
    e = p[mask] - y[mask] / scale
    if not bool(torch.isfinite(e).all()):
        raise ValueError("nonfinite valid target/residual")
    base = nn.functional.smooth_l1_loss(p[mask], y[mask] / scale, beta=1.0)
    excess = torch.relu(e.abs() - 1.0)
    loss = base if loss_name == "smooth_l1" else base + (0.5 * excess.square()).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite training objective")
    with torch.no_grad():
        ed = e.detach().double()
        absolute = ed.abs()
        quadratic = 0.5 * ed.square()
        difference = 0.5 * torch.relu(absolute - 1.0).square()
        pt = p.detach() * np.float32(scale)
        te = pt[mask].double() - y[mask].double()
        values = torch.stack((ed.square().sum(), absolute.sum(), quadratic.sum(),
                              (quadratic - difference).sum(), difference.sum(),
                              te.square().sum(), te.abs().sum(),
                              p.min().double(), p.max().double(), pt.min().double(), pt.max().double(),
                              ed.min(), ed.max(), te.min(), te.max())).cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("nonfinite exposure diagnostics/output conversion")
        stats = dict(valid_count=ed.numel(), tail_count=int((absolute > 1).sum()),
                     normalized_sse=values[0], normalized_l1_sum=values[1],
                     quadratic_sum=values[2], smooth_l1_sum=values[3],
                     quadratic_minus_smooth_l1_sum=values[4], tecu_sse=values[5], tecu_l1_sum=values[6])
        for field, index in zip(RANGE_FIELDS, (7, 9, 11, 13)):
            stats[field + "_min"], stats[field + "_max"] = values[index:index + 2]
    return loss, stats


def merge_exposure(total, stats):
    if total is None:
        return dict(stats)
    for key in SUM_FIELDS:
        total[key] += stats[key]
    for field in RANGE_FIELDS:
        total[field + "_min"] = min(total[field + "_min"], stats[field + "_min"])
        total[field + "_max"] = max(total[field + "_max"], stats[field + "_max"])
    return total


def allowed_year_path(files, year):
    # This check precedes path lookup, hashing and np.load, including cache hits.
    if type(year) is not int or year not in ALLOWED_YEARS:
        raise ValueError(f"forbidden development year: {year!r}")
    return Path(files[str(year)]["path"])


def validate_config(config):
    if config.get("schema") != "global_gwn_development_v1":
        raise ValueError("unsupported development config schema")
    if type(config.get("seed")) is not int or config["seed"] not in SEEDS or config.get("loss") not in LOSSES:
        raise ValueError("unknown seed or loss")
    if config.get("reference_sha256") != REFERENCE_HASHES:
        raise ValueError("reference source identity differs from the original matched G")
    if set(config.get("asset_sha256", {})) != ASSET_NAMES:
        raise ValueError("only graph and train-scale assets are allowed")
    if set(config.get("data_files", {})) != {str(y) for y in ALLOWED_YEARS}:
        raise ValueError("data_files must contain exactly the nine train/validation source years")
    for year in ALLOWED_YEARS:
        record = config["data_files"][str(year)]
        if set(record) != {"path", "sha256"} or allowed_year_path(config["data_files"], year).name != f"TEC_{year}.npy":
            raise ValueError("year files require explicit original basename, path and sha256")
    for field in ("reference_root", "asset_root", "expected_initial_tensor_sha256", "group_id"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise ValueError(f"missing {field}")
    if set(config.get("runtime", {})) != {"torch", "cuda", "gpu"}:
        raise ValueError("exact torch/CUDA/GPU identities are required")


def validate_runtime(expected):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("historical CUDA bf16 runtime required; CPU is synthetic-test-only")
    observed = dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    if observed != expected:
        raise RuntimeError(f"runtime mismatch: {observed}")
    return observed


def verify_assets(config):
    for name, digest in REFERENCE_HASHES.items():
        if sha(Path(config["reference_root"]) / name) != digest:
            raise ValueError(f"changed reference source: {name}")
    for name, digest in config["asset_sha256"].items():
        if sha(Path(config["asset_root"]) / name) != digest:
            raise ValueError(f"changed prepared asset: {name}")
    for year in sorted(ALLOWED_YEARS):
        if sha(allowed_year_path(config["data_files"], year)) != config["data_files"][str(year)]["sha256"]:
            raise ValueError(f"changed development source year: {year}")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def reference_modules(root, asset_root):
    """Execute unchanged reference sources with temporary private import aliases."""
    root = Path(root)
    names = ("lib", "lib.physics", "lib.physics.driver_scaler", "model", "model.GWN",
             "model.GWN.driver_fusion", "model.GWN.GWN", "_global_dev_gim", "_global_dev_model")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in ("lib", "lib.physics", "model", "model.GWN"):
            package = ModuleType(name); package.__path__ = []
            sys.modules[name] = package
        load_module("lib.physics.driver_scaler", root / "lib/physics/driver_scaler.py")
        load_module("model.GWN.driver_fusion", root / "model/GWN/driver_fusion.py")
        load_module("model.GWN.GWN", root / "model/GWN/GWN.py")
        gim = load_module("_global_dev_gim", root / "gim_data.py")
        model = load_module("_global_dev_model", root / "global_model.py")
        model.HERE = Path(asset_root)  # Only the prepared graph location changes.
        yield gim, model
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class DevelopmentWindows(Dataset):
    def __init__(self, role, config, gim_module):
        if role not in ("train", "validation"):
            raise ValueError(f"forbidden development role: {role}")
        files = config["data_files"]

        class RestrictedGIM(gim_module.GIM):
            def path(self, year):
                return allowed_year_path(files, year)

        self.gim = RestrictedGIM()
        self.days = list(self.gim.windows(role))
        self.restore_grid = gim_module.restore_grid
        for day in self.days:
            for date in (np.datetime64(day), np.datetime64(day) - np.timedelta64(1, "D")):
                allowed_year_path(files, int(str(date)[:4]))
        with np.load(Path(config["asset_root"]) / "prepared/train_climatology_v1.npz", allow_pickle=False) as archive:
            self.scale = float(archive["scale"])
        if self.scale != 147.1 or len(self.days) != {"train": 2553, "validation": 730}[role]:
            raise ValueError("original scale or window counts changed")

    def __len__(self):
        return len(self.days)

    def __getitem__(self, index):
        sample = self.gim.sample(self.days[index])
        return dict(history=torch.from_numpy(np.array(sample["history"] / np.float32(self.scale), dtype=np.float32)),
                    target=torch.from_numpy(np.array(self.restore_grid(sample["target"]), dtype=np.float32)),
                    valid=torch.from_numpy(np.array(self.restore_grid(sample["valid"]), dtype=bool)),
                    origin=int(sample["origin"]), index=index)


def optimizer_for(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.9, patience=5, threshold=1e-5,
        threshold_mode="abs", cooldown=10, min_lr=0.)
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, loss_name, scale, device):
    model.train(); total = None; objective_sum = 0.; order = hashlib.sha256()
    for batch in loader:
        order.update(batch["index"].numpy().astype("<i8").tobytes())
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            prediction = model(batch["history"].to(device), restore=True)
        loss, exposure = loss_and_exposure(prediction, batch["target"], batch["valid"], scale, loss_name)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        objective_sum += float(loss.detach()) * exposure["valid_count"]
        total = merge_exposure(total, exposure)
    if total is None:
        raise ValueError("empty training loader")
    return dict(training_loss=objective_sum / total["valid_count"], training_exposure=total,
                sample_order_sha256=order.hexdigest())


@torch.no_grad()
def evaluate(model, dataset, loss_name, device, prediction_path=None):
    model.eval(); sums = np.zeros(4, np.float64); total = None; dest = None
    for batch in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        with amp(device):
            prediction = model(batch["history"].to(device), restore=True)
        _, exposure = loss_and_exposure(prediction, batch["target"], batch["valid"], dataset.scale, loss_name)
        total = merge_exposure(total, exposure)
        pred = prediction.float().cpu().numpy() * np.float32(dataset.scale)
        truth, valid = batch["target"].numpy(), batch["valid"].numpy()
        error = pred[valid].astype(np.float64) - truth[valid].astype(np.float64)
        sums += [np.square(error).sum(), np.abs(error).sum(), error.sum(), error.size]
        if prediction_path is not None:
            if dest is None:
                dest = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.float32,
                                                shape=(len(dataset),) + tuple(pred.shape[1:]))
            dest[batch["index"].numpy()] = pred
    if not sums[3]:
        raise ValueError("empty validation")
    if dest is not None:
        dest.flush()
    return dict(RMSE=float(np.sqrt(sums[0] / sums[3])), MAE=float(sums[1] / sums[3]),
                bias=float(sums[2] / sums[3]), sse=float(sums[0]), sae=float(sums[1]),
                error_sum=float(sums[2]), n_valid=int(sums[3]), exposure=total)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def training_state(model, optimizer, scheduler, generator, identity, epoch, best, selected, best_state, row,
                   budget_sha256=None):
    return dict(schema="global_gwn_development_checkpoint_v1", identity=identity,
                epoch=epoch, best=best, selected=selected, best_state=best_state,
                model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=rng_state(generator), row=row, budget_sha256=budget_sha256)


def restore_training(state, model, optimizer, scheduler, generator, identity):
    if state.get("schema") != "global_gwn_development_checkpoint_v1" or state.get("identity") != identity:
        raise ValueError("incompatible training checkpoint; diagnostic states cannot resume training")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"], generator)


def batchnorm_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}


@torch.no_grad()
def exposure_pass(model, dataset, loss_name, device, generator, train_mode, record=None):
    model.train(train_mode)
    loader = DataLoader(dataset, batch_size=32, shuffle=train_mode, drop_last=False,
                        generator=generator if train_mode else None, num_workers=0)
    total = None; order = hashlib.sha256()
    for index, batch in enumerate(loader, 1):
        order.update(batch["index"].numpy().astype("<i8").tobytes())
        with amp(device):
            prediction = model(batch["history"].to(device), restore=True)
        _, stats = loss_and_exposure(prediction, batch["target"], batch["valid"], dataset.scale, loss_name)
        total = merge_exposure(total, stats)
        if record is not None:
            record(dict(batch=index, exposure=stats, batchnorm_sha256=tensor_digest(batchnorm_state(model)),
                        torch_rng_sha256=tensor_digest({"rng": torch.get_rng_state()}),
                        cuda_rng_sha256=tensor_digest({str(i): v for i, v in enumerate(torch.cuda.get_rng_state_all())})
                        if torch.cuda.is_available() else None))
    if total is None:
        raise ValueError("empty exposure dataset")
    return dict(exposure=total, sample_order_sha256=order.hexdigest(), batches=index,
                RMSE=math.sqrt(total["tecu_sse"] / total["valid_count"]))


def append_record(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n"); handle.flush(); os.fsync(handle.fileno())


def scientific_config(config):
    """Remove only the three declared path locations; all scientific fields stay."""
    result = json.loads(json.dumps(config, allow_nan=False))
    result.pop("reference_root"); result.pop("asset_root")
    for record in result["data_files"].values():
        record.pop("path")
    return result


def bind_identity(config_path, config_sha256, logical_path, logical_sha256, identity_path=None):
    if sha(config_path) != config_sha256:
        raise ValueError("runtime config hash mismatch")
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    if sha(logical_path) != logical_sha256:
        raise ValueError("frozen logical config hash mismatch")
    logical = json.loads(Path(logical_path).read_text()); validate_config(logical)
    science = scientific_config(config)
    if science != scientific_config(logical):
        raise ValueError("runtime mapping changed fields other than declared asset paths")
    current = Path(__file__).resolve(); entry_sha = sha(current)
    hashes = {"src/training/global_train.py": entry_sha}; source_commit = None
    execution = dict(runtime_config_sha256=config_sha256, logical_config_sha256=logical_sha256,
                     runtime_config_path=str(Path(config_path).resolve()),
                     logical_config_path=str(Path(logical_path).resolve()),
                     source_identity_sha256=None, source_snapshot=None)
    if identity_path is not None:
        source_identity = json.loads(Path(identity_path).read_text())
        snapshot = Path(source_identity["source_snapshot"]).resolve()
        hashes = source_identity["source_hashes"]
        source_commit = source_identity.get("source_commit")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("source_hashes required")
        executable_bound = False
        for relative, digest in hashes.items():
            path = snapshot / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or path.suffix != ".py":
                raise ValueError("run identity may hash source Python files only")
            if sha(path) != digest:
                raise ValueError(f"changed frozen source: {relative}")
            executable_bound |= path.resolve() == current
        if not executable_bound:
            raise ValueError("execute this entrypoint from its bound source snapshot")
        execution.update(source_identity_sha256=sha(identity_path), source_snapshot=str(snapshot),
                         batch_config_sha256=source_identity.get("config_sha256"))
    stable = dict(logical_config_sha256=logical_sha256, scientific_config_sha256=json_digest(science),
                  entrypoint_sha256=entry_sha, source_commit=source_commit, source_hashes=hashes,
                  seed=config["seed"], loss=config["loss"], group_id=config["group_id"])
    return config, stable, execution


def continuation_receipt(path, digest, config, target):
    if target == 200:
        if path is not None:
            raise ValueError("200-stage does not consume an extension receipt")
        return None
    if path is None or digest is None or sha(path) != digest:
        raise ValueError("300 requires an explicitly hash-bound frozen common budget receipt")
    receipt = json.loads(Path(path).read_text())
    try:
        from .loss_budget import inherit_loss_budget
    except ImportError:
        from loss_budget import inherit_loss_budget
    validated = inherit_loss_budget(receipt, 3407)
    if (validated["target_epochs"] != 300 or config["loss"] not in receipt["applicable_arms"]
            or config["seed"] not in receipt["applicable_seeds"]):
        raise ValueError("common budget receipt does not cover this loss/seed")
    # Receipt hash and logical group are stable across remote path remapping.
    return dict(sha256=digest, group_id=config["group_id"], receipt=receipt)


def recover_epoch_log(path, checkpoint):
    """Repair only a final non-newline-terminated JSON fragment from a crash."""
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith("\n"):
                raise ValueError("corrupt epoch log before its final incomplete line")
            write_json(path.with_name(f"log_repair_{checkpoint['epoch']}_{time.time_ns()}.json"),
                       dict(original_log_sha256=sha(path), checkpoint_epoch=checkpoint["epoch"],
                            discarded_truncated_tail=line, recovered_from_checkpoint_row=True))
            continue
        if not isinstance(row, dict) or type(row.get("epoch")) is not int:
            raise ValueError("invalid epoch record")
        rows.append(row)
    rows = [row for row in rows if row["epoch"] < checkpoint["epoch"]] + [checkpoint["row"]]
    if [row["epoch"] for row in rows] != list(range(1, checkpoint["epoch"] + 1)):
        raise ValueError("incomplete or duplicate checkpoint epoch history")
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def verify_completed(completed, out, target):
    expected = {f"last_{target}.pt", f"best_{target}.pt", f"validation_{target}_predictions.npy",
                f"validation_{target}_origins.npy", f"epochs_{target}.jsonl"}
    if set(completed.get("files", {})) != expected or completed.get("target_epochs") != target:
        raise ValueError("incomplete stage completion manifest")
    for name, digest in completed["files"].items():
        path = Path(out) / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"changed/missing completed output: {name}")


def train(model, tr, va, config, identity, out, device, target, budget):
    marker = out / f"training_complete_{target}.json"
    if marker.exists():
        completed = json.loads(marker.read_text())
        if completed["identity"] != identity or completed["budget"] != budget:
            raise ValueError("completed stage identity changed")
        verify_completed(completed, out, target)
        return completed
    seed_all(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=False, generator=generator, num_workers=0)
    optimizer, scheduler = optimizer_for(model)
    last = out / "last.pt"; epoch0 = 0; best = float("inf"); selected = 0; best_state = None
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        restore_training(state, model, optimizer, scheduler, generator, identity)
        epoch0, best, selected, best_state = state["epoch"], state["best"], state["selected"], state["best_state"]
        if epoch0 > 200 and state.get("budget_sha256") != (budget or {}).get("sha256"):
            raise ValueError("continuation budget changed across attempts")
        if epoch0 > target:
            raise ValueError("attempt to shrink a training budget")
        recover_epoch_log(out / "epochs.jsonl", state)
    elif (out / "epochs.jsonl").exists():
        raise ValueError("epoch log exists without its recoverable last.pt")
    started = time.monotonic()
    for epoch in range(epoch0 + 1, target + 1):
        lr = optimizer.param_groups[0]["lr"]
        row = train_epoch(model, loader, optimizer, config["loss"], tr.scale, device)
        validation = evaluate(model, va, config["loss"], device)
        score = validation["RMSE"]; scheduler.step(score)
        if score < best:
            best, selected = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row.update(epoch=epoch, validation=validation, lr=lr, next_lr=optimizer.param_groups[0]["lr"],
                   selected_epoch=selected, elapsed_this_process_seconds=time.monotonic() - started)
        save_atomic(training_state(model, optimizer, scheduler, generator, identity, epoch, best,
                                   selected, best_state, row, (budget or {}).get("sha256")), last)
        append_record(out / "epochs.jsonl", row)
        print(json.dumps(dict(epoch=epoch, validation_rmse=score, selected_epoch=selected)), flush=True)
    if best_state is None:
        raise ValueError("no selected training state")
    best_path = out / f"best_{target}.pt"
    last_snapshot = out / f"last_{target}.pt"
    shutil.copyfile(last, last_snapshot.with_suffix(".pt.partial"))
    os.replace(last_snapshot.with_suffix(".pt.partial"), last_snapshot)
    log_snapshot = out / f"epochs_{target}.jsonl"
    shutil.copyfile(out / "epochs.jsonl", log_snapshot.with_suffix(".jsonl.partial"))
    os.replace(log_snapshot.with_suffix(".jsonl.partial"), log_snapshot)
    save_atomic(dict(schema="global_gwn_development_best_v1", identity=identity,
                     state_dict=best_state, epoch=selected, validation_rmse=best), best_path)
    model.load_state_dict(best_state, strict=True)
    prediction_path = out / f"validation_{target}_predictions.npy"
    validation = evaluate(model, va, config["loss"], device, prediction_path)
    if abs(validation["RMSE"] - best) >= 1e-8:
        raise ValueError("selected validation replay failed")
    origins_path = out / f"validation_{target}_origins.npy"
    np.save(origins_path, np.array([int(np.datetime64(day, "s").astype(np.int64)) for day in va.days], np.int64))
    completed = dict(status="training_and_validation_complete", identity=identity, target_epochs=target,
                     selected_epoch=selected, validation=validation, budget=budget, evaluation_access_count=0,
                     files={path.name: sha(path) for path in (last_snapshot, best_path, prediction_path, origins_path, log_snapshot)})
    write_json(marker, completed)
    return completed


def expose(model, tr, va, config, identity, out, device, state_name, legacy_path=None, legacy_sha=None):
    if state_name not in ("initialization_train", "legacy_best"):
        raise ValueError("unknown exposure state")
    if (out / "exposure_complete.json").exists() or (out / "exposure_batches.jsonl").exists():
        raise ValueError("refuse to repeat or append an exposure run; inspect any partial output")
    if state_name == "legacy_best":
        if legacy_path is None or legacy_sha is None or sha(legacy_path) != legacy_sha:
            raise ValueError("legacy best requires an explicit checkpoint hash")
        checkpoint = torch.load(legacy_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        expected_validation_rmse = float(config["expected_legacy_validation_rmse"])
        checkpoint_validation_rmse = float(checkpoint["validation_rmse"])
        if not math.isfinite(expected_validation_rmse) or expected_validation_rmse < 0:
            raise ValueError("invalid saved legacy validation RMSE")
        if not math.isfinite(checkpoint_validation_rmse) or abs(checkpoint_validation_rmse - expected_validation_rmse) >= 1e-8:
            raise ValueError("legacy checkpoint RMSE does not match frozen reference metadata")
    elif legacy_path is not None or legacy_sha is not None:
        raise ValueError("initialization exposure cannot load legacy weights")
    seed_all(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    before = dict(batchnorm=batchnorm_state(model), rng=rng_state(generator))
    summary = {}
    for role, dataset in (("train", tr),) if state_name == "initialization_train" else (("train", tr), ("validation", va)):
        summary[role] = exposure_pass(model, dataset, config["loss"], device, generator,
                                     state_name == "initialization_train",
                                     lambda row, role=role: append_record(out / "exposure_batches.jsonl", dict(role=role, **row)))
    replay = None
    if state_name == "legacy_best":
        observed = summary["validation"]["RMSE"]
        difference = abs(observed - expected_validation_rmse)
        replay = dict(expected_validation_rmse=expected_validation_rmse, observed_validation_rmse=observed,
                      checkpoint_validation_rmse=checkpoint_validation_rmse,
                      absolute_difference=difference, tolerance=1e-8)
        if difference >= 1e-8:
            raise ValueError(f"legacy best validation RMSE replay mismatch: {replay}")
    diagnostic = dict(schema="global_gwn_exposure_diagnostic_only_v1", identity=identity, state=state_name,
                      before=before, after=dict(batchnorm=batchnorm_state(model), rng=rng_state(generator)),
                      cannot_resume_training=True)
    diagnostic_path = out / "exposure_bn_rng_diagnostic.pt"; save_atomic(diagnostic, diagnostic_path)
    completed = dict(status="exposure_complete_no_optimizer", identity=identity, state=state_name,
                     legacy_checkpoint_sha256=legacy_sha, legacy_validation_replay=replay,
                     summary=summary, optimizer_steps=0,
                     evaluation_access_count=0, initial_model_rebuild_required_for_training=True,
                     files={path.name: sha(path) for path in (diagnostic_path, out / "exposure_batches.jsonl")})
    write_json(out / "exposure_complete.json", completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "exposure"))
    for name in ("config", "config-sha256", "logical-config", "logical-config-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--run-identity")
    parser.add_argument("--target-epochs", type=int, choices=(200, 300), default=200)
    parser.add_argument("--budget-decision"); parser.add_argument("--budget-decision-sha256")
    parser.add_argument("--state", choices=("initialization_train", "legacy_best"))
    parser.add_argument("--legacy-checkpoint"); parser.add_argument("--legacy-sha256")
    args = parser.parse_args()
    if (args.action == "exposure") != (args.state is not None):
        parser.error("exposure requires --state; train must not specify --state")
    if args.action == "train" and (args.legacy_checkpoint or args.legacy_sha256):
        parser.error("training always rebuilds original initialization; no legacy checkpoint")
    if args.action == "exposure" and (args.target_epochs != 200 or args.budget_decision or args.budget_decision_sha256):
        parser.error("exposure does not consume training budgets")
    torch.set_num_threads(4)
    config, identity, execution = bind_identity(args.config, args.config_sha256, args.logical_config,
                                               args.logical_config_sha256, args.run_identity)
    runtime = validate_runtime(config["runtime"])
    budget = continuation_receipt(args.budget_decision, args.budget_decision_sha256, config, args.target_epochs)
    verify_assets(config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    append_record(out / "executions.jsonl", dict(execution=execution, runtime=runtime,
                                                action=args.action, target_epochs=args.target_epochs,
                                                unix_time=time.time(), pid=os.getpid()))
    if args.action == "train" and ((out / "exposure_complete.json").exists() or (out / "exposure_bn_rng_diagnostic.pt").exists()):
        raise ValueError("training and exposure must use separate output directories")
    with reference_modules(config["reference_root"], config["asset_root"]) as (gim, model_module):
        tr, va = DevelopmentWindows("train", config, gim), DevelopmentWindows("validation", config, gim)
        seed_all(config["seed"]); model = model_module.GlobalGWN(torch.device("cuda")).to("cuda")
        initial = tensor_digest(model.state_dict())
        if initial != config["expected_initial_tensor_sha256"]:
            raise ValueError("original initialization tensor digest mismatch")
        record = dict(identity=identity, runtime=runtime, initial_tensor_sha256=initial,
                      train_windows=len(tr), validation_windows=len(va), scale=tr.scale,
                      parameters=sum(value.numel() for value in model.parameters()))
        path = out / "run_config.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("existing output identity changed")
        write_json(path, record)
        if args.action == "train":
            result = train(model, tr, va, config, identity, out, torch.device("cuda"), args.target_epochs, budget)
        else:
            result = expose(model, tr, va, config, identity, out, torch.device("cuda"), args.state,
                            args.legacy_checkpoint, args.legacy_sha256)
    print(json.dumps(dict(status=result["status"])), flush=True)


if __name__ == "__main__":
    main()
