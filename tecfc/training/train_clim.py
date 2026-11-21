"""Fixed exp0016 G-MAE/C-in; exact original training loop and validation order."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
from tecfc.features.clim_inputs import base, modules, Windows, scientific_config, validate


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(path, value)



@torch.no_grad()
def evaluate(model, dataset, device, path=None):
    # Exact original G validation operation order and batch size.
    model.eval(); sums = np.zeros(4, np.float64)
    dest = None if path is None else np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                                              shape=(len(dataset), 12, 71, 73))
    for b in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        with base.amp(device): pred = model(b['history'].to(device), restore=True)
        pred = pred.float().cpu().numpy() * np.float32(dataset.scale)
        truth, valid = b['target'].numpy(), b['valid'].numpy()
        if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite prediction')
        error = pred[valid].astype(np.float64) - truth[valid].astype(np.float64)
        sums += [np.square(error).sum(), np.abs(error).sum(), error.sum(), error.size]
        if dest is not None: dest[b['index'].numpy()] = pred
    if dest is not None: dest.flush()
    return dict(RMSE=float(np.sqrt(sums[0]/sums[3])), MAE=float(sums[1]/sums[3]), bias=float(sums[2]/sums[3]),
                sse=float(sums[0]), sae=float(sums[1]), error_sum=float(sums[2]), n_valid=int(sums[3]))


def train(config, identity, out, device, epochs=200, diagnostic=False, epoch1_hook=None):
    if (epochs != 200 and not diagnostic) or (diagnostic and epochs not in (1, 3)):
        raise ValueError('Unregistered training budget')
    if config['arm'] not in ('G-control','C-control','G-MAE','C-in') or config['seed'] not in base.SEEDS:
        raise ValueError('Unknown arm/seed')
    if not diagnostic and config['arm'] not in ('G-MAE','C-in'): raise ValueError('Controls are diagnostic only')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    run_id = dict(source_commit=identity['source_commit'], config_sha256=base.json_digest(scientific_config(config)),
                  arm=config['arm'], seed=config['seed'], diagnostic_only=diagnostic, epochs=epochs)
    write(out/'run_config.json', run_id)
    with modules(config, device) as (gim, constructor):
        tr = Windows('train', config, gim)
        va = Windows('validation', config, gim)
        base.seed_all(config['seed'])
        model = constructor().to(device)
        initial = base.tensor_digest(model.state_dict())
        if initial != config['expected_initial_tensor_sha256']: raise ValueError('Initial tensor mismatch')
        write(out/'INITIAL_STATE.json', dict(initial_tensor_sha256=initial, seed=config['seed'],
              initialization='original fresh tensors, no checkpoint', diagnostic_only=diagnostic))
        optimizer, scheduler = base.optimizer_for(model)
        base.seed_all(config['seed'])
        generator = torch.Generator().manual_seed(config['seed'])
        loader = DataLoader(tr, batch_size=32, shuffle=True, drop_last=False, generator=generator, num_workers=0)
        epoch0, best, selected, best_state = 0, float('inf'), 0, None
        last = out/'last.pt'
        if last.exists():
            ck = torch.load(last, map_location=device, weights_only=False)
            if ck['identity'] != run_id: raise ValueError('Cannot resume another identity or diagnostic weights')
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
            base.restore_rng(ck['rng'], generator)
            epoch0, best, selected, best_state = ck['epoch'], ck['best'], ck['selected'], ck['best_state']
            base.recover_epoch_log(out/'epochs.jsonl', ck)
        elif (out/'epochs.jsonl').exists(): raise ValueError('Log without complete resumable state')
        started = time.monotonic()
        for epoch in range(epoch0+1, epochs+1):
            model.train(); total=0.; count=0; order=hashlib.sha256()
            lr = optimizer.param_groups[0]['lr']
            for batch_number, b in enumerate(loader, 1):
                order.update(b['index'].numpy().astype('<i8').tobytes())
                optimizer.zero_grad(set_to_none=True)
                with base.amp(device): pred = model(b['history'].to(device), restore=True)
                valid, target = b['valid'].to(device), b['target'].to(device)
                if config['loss'] == 'mae':
                    data = nn.functional.l1_loss(pred.float()[valid], target.float()[valid]/tr.scale)
                else:
                    data = nn.functional.smooth_l1_loss(pred.float()[valid], target.float()[valid]/tr.scale, beta=1.)
                data.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                n=int(valid.sum()); total+=float(data.detach())*n; count+=n
            val = evaluate(model, va, device); score=val['RMSE']; scheduler.step(score)
            if score < best:
                best, selected = score, epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row=dict(epoch=epoch,training_loss=total/count,validation=val,lr=lr,next_lr=optimizer.param_groups[0]['lr'],
                selected_epoch=selected,sample_order_sha256=order.hexdigest(),
                elapsed_this_process_seconds=time.monotonic()-started)
            state=dict(identity=run_id,epoch=epoch,best=best,selected=selected,best_state=best_state,
                model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=base.rng_state(generator),row=row)
            base.save_atomic(state,last); base.append_record(out/'epochs.jsonl',row)
            if diagnostic and epoch==1:
                rng=base.rng_state(generator)
                if epoch1_hook is not None: epoch1_hook(model,va,device,out)
                base.restore_rng(rng,generator)
            print(json.dumps(dict(arm=config['arm'],seed=config['seed'],epoch=epoch,RMSE=score,selected=selected)),flush=True)
        model.load_state_dict(best_state)
        base.save_atomic(dict(state_dict=best_state,epoch=selected,validation_rmse=best,identity=run_id),out/f'best_{epochs}.pt')
        validation=evaluate(model,va,device,None if diagnostic else out/'validation_predictions.npy')
        if abs(validation['RMSE']-best)>=1e-8: raise ValueError('Best validation replay differs')
        np.save(out/'validation_origins.npy',np.array([int(np.datetime64(d,'s').astype(np.int64)) for d in va.days],np.int64))
        result=dict(status='complete',identity=run_id,epochs=epochs,selected_epoch=selected,validation=validation,
                    evaluation_access_count=0,diagnostic_only=diagnostic,
                    files={p.name:base.sha(p) for p in out.iterdir() if p.is_file() and p.name!=f'training_complete_{epochs}.json'})
        write(out/f'training_complete_{epochs}.json',result)
        return result
