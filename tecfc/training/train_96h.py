"""exp0027 fixed-200 training: train/validation only; exact base G loop (pi_global/train.py)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True
from tecfc.features.h96_inputs import (base, Windows, build_model, PIAuxiliary, planar_bits, tecu_from_planar, scientific_config,  # noqa: E402
                  write, ARMS, SEEDS, LOSS, RHO)


@torch.no_grad()
def evaluate(model, dataset, device, path=None):
    """Exact original G validation order/batch; optional lossless bf16 planar storage."""
    model.eval(); sums = np.zeros(4, np.float64); h = dataset.horizon
    dest = None if path is None else np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(2, len(dataset), h, 71, 72))
    for b in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        with base.amp(device): prediction = model(b['history'].to(device), restore=True)
        pred = prediction.float().cpu().numpy() * np.float32(dataset.scale)
        truth, valid = b['target'].numpy(), b['valid'].numpy()
        if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite prediction')
        error = pred[valid].astype(np.float64) - truth[valid].astype(np.float64)
        sums += [np.square(error).sum(), np.abs(error).sum(), error.sum(), error.size]
        if dest is not None:
            hi, lo = planar_bits(prediction)
            if not np.array_equal(tecu_from_planar(hi, lo), pred): raise ValueError('Planar storage is not lossless')
            index = b['index'].numpy(); dest[0, index] = hi; dest[1, index] = lo
    if dest is not None: dest.flush()
    return dict(RMSE=float(np.sqrt(sums[0] / sums[3])), MAE=float(sums[1] / sums[3]), bias=float(sums[2] / sums[3]),
                sse=float(sums[0]), sae=float(sums[1]), error_sum=float(sums[2]), n_valid=int(sums[3]))


def summarize(rows):
    if not rows: return {'auxiliary_disabled': True}
    result = dict(batches=len(rows), projection_trigger_fraction=float(np.mean([r['projection_triggered'] for r in rows])), terms={})
    for key in ('auxiliary_ratio_before', 'auxiliary_ratio_after', 'data_gradient_norm', 'auxiliary_gradient_norm', 'cosine',
                'fp32_writeback_difference_norm'):
        a = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        result[key] = dict(zip(('min', 'p05', 'median', 'p95', 'max'), map(float, np.quantile(a, [0, .05, .5, .95, 1]))))
        result[key]['mean'] = float(a.mean())
    for key in rows[0]['terms']:
        a = np.array([r['terms'][key] for r in rows]); counts = [r['residual_counts'][key] for r in rows]
        result['terms'][key] = dict(native_mse_mean=float(a.mean()), native_mse_min=float(a.min()), native_mse_max=float(a.max()),
                                    nonzero_batch_fraction=float(np.mean(a != 0)),
                                    nonzero_residual_fraction=sum(r['nonzero'] for r in counts) / sum(r['count'] for r in counts))
    return result


def train(config, identity, out, device, epochs=200, diagnostic=False, epoch1_hook=None):
    arm, seed, horizon = config['arm'], config['seed'], config['horizon']
    if (epochs != 200 and not diagnostic) or (diagnostic and epochs not in (1, 3)): raise ValueError('Unregistered training budget')
    if arm not in ARMS or seed not in SEEDS or config['loss'] != LOSS[arm] or config['rho'] != RHO[arm]: raise ValueError('Unknown arm/seed/loss/rho')
    if not diagnostic and horizon != 48: raise ValueError('Formal runs are 48-frame only')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    run_id = dict(source_commit=identity['source_commit'], config_sha256=base.json_digest(scientific_config(config)),
                  arm=arm, seed=seed, horizon=horizon, diagnostic_only=diagnostic, epochs=epochs)
    write(out / 'run_config.json', run_id)
    with base.reference_modules(config['reference_root'], config['asset_root']) as (_, architecture):
        tr = Windows('train', config, arm, horizon); va = Windows('validation', config, arm, horizon)
        base.seed_all(seed)
        model = build_model(architecture, device, arm, horizon).to(device)
        initial = base.tensor_digest(model.state_dict())
        if config.get('expected_initial_tensor_sha256') and initial != config['expected_initial_tensor_sha256']:
            raise ValueError('Initial tensor mismatch')
        write(out / 'INITIAL_STATE.json', dict(initial_tensor_sha256=initial, seed=seed, horizon=horizon,
              initialization='original fresh tensors, no checkpoint', diagnostic_only=diagnostic))
        optimizer, scheduler = base.optimizer_for(model)
        auxiliary = PIAuxiliary(config['physics'], device, horizon) if config['rho'] else None
        base.seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=False, generator=generator, num_workers=0)
        epoch0, best, selected, best_state = 0, float('inf'), 0, None
        last = out / 'last.pt'
        if last.exists():
            ck = torch.load(last, map_location=device, weights_only=False)
            if ck['identity'] != run_id or diagnostic: raise ValueError('Cannot resume another identity or diagnostic weights')
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
            base.restore_rng(ck['rng'], generator)
            epoch0, best, selected, best_state = ck['epoch'], ck['best'], ck['selected'], ck['best_state']
            base.recover_epoch_log(out / 'epochs.jsonl', ck)
        elif (out / 'epochs.jsonl').exists(): raise ValueError('Log without complete resumable state')
        started = time.monotonic()
        for epoch in range(epoch0 + 1, epochs + 1):
            model.train(); total = 0.; count = 0; order = hashlib.sha256(); diagnostics = []
            lr = optimizer.param_groups[0]['lr']
            for batch_number, b in enumerate(loader, 1):
                order.update(b['index'].numpy().astype('<i8').tobytes())
                optimizer.zero_grad(set_to_none=True)
                with base.amp(device): pred = model(b['history'].to(device), restore=True)
                valid, target = b['valid'].to(device), b['target'].to(device)
                if config['loss'] == 'mae':
                    data = nn.functional.l1_loss(pred.float()[valid], target.float()[valid] / tr.scale)
                else:
                    data = nn.functional.smooth_l1_loss(pred.float()[valid], target.float()[valid] / tr.scale, beta=1.)
                if auxiliary is None:
                    data.backward()
                else:
                    physical, terms, counts = auxiliary(pred, b['origin'].numpy())
                    stats = auxiliary.assign_gradients(data, physical, list(model.parameters()), config['rho'])
                    stats.update(terms=terms, residual_counts=counts); diagnostics.append(stats)
                    if diagnostic: base.append_record(out / 'batches.jsonl', dict(epoch=epoch, batch=batch_number, **stats))
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                n = int(valid.sum()); total += float(data.detach()) * n; count += n
            val = evaluate(model, va, device); score = val['RMSE']; scheduler.step(score)
            if score < best:
                best, selected = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(epoch=epoch, training_loss=total / count, validation=val, lr=lr, next_lr=optimizer.param_groups[0]['lr'],
                       selected_epoch=selected, sample_order_sha256=order.hexdigest(), auxiliary=summarize(diagnostics),
                       elapsed_this_process_seconds=time.monotonic() - started)
            state = dict(identity=run_id, epoch=epoch, best=best, selected=selected, best_state=best_state, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=base.rng_state(generator), row=row)
            base.save_atomic(state, last); base.append_record(out / 'epochs.jsonl', row)
            if diagnostic and epoch == 1 and epoch1_hook is not None:
                rng = base.rng_state(generator); epoch1_hook(model, va, device, out); base.restore_rng(rng, generator)
            print(json.dumps(dict(arm=arm, seed=seed, horizon=horizon, epoch=epoch, RMSE=score, selected=selected)), flush=True)
        model.load_state_dict(best_state)
        base.save_atomic(dict(state_dict=best_state, epoch=selected, validation_rmse=best, identity=run_id), out / f'best_{epochs}.pt')
        validation = evaluate(model, va, device, None if diagnostic else out / 'validation_bf16planar_predictions.npy')
        if abs(validation['RMSE'] - best) >= 1e-8: raise ValueError('Best validation replay differs')
        np.save(out / 'validation_origins.npy', np.array([int(np.datetime64(d, 's').astype(np.int64)) for d in va.days], np.int64))
        result = dict(status='complete', identity=run_id, epochs=epochs, selected_epoch=selected, validation=validation,
                      windows=dict(train=len(tr), validation=len(va), dropped_train=len(tr.dropped), dropped_validation=len(va.dropped)),
                      evaluation_access_count=0, diagnostic_only=diagnostic,
                      files={p.name: base.sha(p) for p in out.iterdir() if p.is_file() and p.name != f'training_complete_{epochs}.json'})
        write(out / f'training_complete_{epochs}.json', result)
        return result
