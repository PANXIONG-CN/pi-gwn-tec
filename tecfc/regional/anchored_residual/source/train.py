"""exp0020 r1: original regional training functions behind explicit stage gates.

A1 uses the exact old dataset/model/step/evaluate functions. A2 removes only the
China driver inputs and their first-layer columns. Neither supplies a formal
scientific checkpoint. Further stages must have separately frozen admissions.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def load_legacy(root, region):
    old = root / 'legacy' / region
    src = old / 'shared' if region == 'japan' else old
    sys.path[:0] = [str(src), str(src / 'repo')]
    spec = importlib.util.spec_from_file_location('regional_legacy', src / 'run.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, old


def digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def difference(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(set(a) | set(b)):
            rows.extend(difference(a.get(key), b.get(key), prefix + ('.' if prefix else '') + key))
        return rows
    return [] if a == b else [dict(field=prefix, A1=a, A2=b)]


class ChinaTECOnly(Dataset):
    def __init__(self, original):
        self.__dict__.update(original.__dict__)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        x = np.empty((89, 2800, 1), np.float32)
        x[:, :, 0] = self.archive.history(row['history'], self.native) / np.float32(self.scale)
        y, v = self.archive.target(row)
        y, v = y[np.arange(7, 96, 8)], v[np.arange(7, 96, 8)]
        return torch.from_numpy(x), torch.from_numpy(y[..., None]), torch.from_numpy(v[..., None]), i


def legacy_objects(root, region, stage, seed=42):
    old, path = load_legacy(root, region)
    device = torch.device('cuda'); old.seed_all(seed)
    if region == 'japan':
        ar = old.Archive(path / 'jpld', 'jpld')
        tr = old.Windows(ar, 'train', 'dense_do'); va = old.Windows(ar, 'validation', 'dense_do')
        model = old.model_for(ar, 'dense_do', device, seed)
    else:
        ar = old.Archive(path)
        tr = old.Windows(ar, 'train', 'A_native'); va = old.Windows(ar, 'validation', 'A_native')
        model = old.model_for(path, 'A_native', device, seed)
    configuration = dict(region=region, seed=seed, data_roles=dict(train=list(range(2014, 2021)), validation=[2021]),
        train_windows=len(tr), validation_windows=len(va), scale=ar.scale,
        input_channels=['TEC'] if region == 'japan' else ['TEC','AE','Dst','Kp','F107','cosSZA'],
        first_layer_shape=list(model.start_conv.weight.shape),
        parameters=sum(p.numel() for p in model.parameters()), initial_tensor_sha256=digest(model),
        initialization_projection='none', input_construction='legacy',
        model=dict(history=89, horizon=12, dropout=.3, blocks=4, layers=2, dilations=[1,2,4,8,16,32,16,16]),
        optimizer=dict(name='Adam',lr=.003,weight_decay=0.,gradient_clip=5.),
        schedule=dict(name='ReduceLROnPlateau',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10),
        training_batch=32, validation_batch=4, precision='bf16', loss='masked_TECU_MAE',
        data_order='legacy included rows; torch.Generator seed42; num_workers0; drop_lastFalse')
    a1 = copy.deepcopy(configuration)
    if stage == 'a2':
        assert region == 'china'
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        old_first = model.start_conv
        first = torch.nn.Conv2d(1, 32, (1,1)).to(device)
        with torch.no_grad(): first.weight.copy_(old_first.weight[:, :1]); first.bias.copy_(old_first.bias)
        model.start_conv = first; model.feature_dim = 1
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name][:,:1] if name == 'start_conv.weight' else before[name], rtol=0, atol=0)
        tr, va = ChinaTECOnly(tr), ChinaTECOnly(va)
        configuration.update(input_channels=['TEC'],first_layer_shape=list(first.weight.shape),
            parameters=sum(p.numel() for p in model.parameters()),initial_tensor_sha256=digest(model),
            initialization_projection='start_conv.weight[:,0:1,:,:]; all other tensors exact',input_construction='TEC-only; no driver_values call')
    return old, ar, tr, va, model, a1, configuration


def diagnostic(root, job, region, stage):
    assert stage in ['a1', 'a2']
    runtime = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), bf16=torch.cuda.is_bf16_supported())
    assert runtime['bf16']
    out = root / 'outputs' / job; out.mkdir(parents=True, exist_ok=False)
    legacy, ar, tr, va, model, a1, cfg = legacy_objects(root, region, stage)
    changes = difference(a1, cfg)
    if stage == 'a2':
        assert {r['field'] for r in changes} == {'input_channels','first_layer_shape','parameters','initial_tensor_sha256','initialization_projection','input_construction'}
        assert a1['parameters'] - cfg['parameters'] == 160
        write(out/'CONFIG_DIFF.json',dict(status='passed',A1=a1,A2=cfg,changes=changes,all_other_parameters_exact=True,
            diagnostic_epoch_budget=dict(A1=3,A2=1,note='explicit user-requested check budgets, not a model-configuration difference')))
    write(out/'config.json',dict(**cfg,stage=stage,runtime=runtime,diagnostic_only=True,epochs=3 if stage=='a1' else 1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.9,patience=5,threshold=1e-5,threshold_mode='abs',cooldown=10,min_lr=0.)
    legacy.seed_all(42)
    loader=DataLoader(tr,batch_size=32,shuffle=True,drop_last=False,generator=torch.Generator().manual_seed(42))
    refs=json.loads((root/'reference_epochs.json').read_text())[region]
    results=[]; started=time.monotonic(); device=torch.device('cuda')
    with (out/'epochs.jsonl').open('x') as log:
        for epoch in range(1,4 if stage=='a1' else 2):
            model.train(); total=count=0
            for batch in loader:
                if region=='japan':loss,n,_,_=legacy.step(model,batch,optimizer,ar,device)
                else:loss,n=legacy.step(model,batch,optimizer,ar.scale,device)
                assert np.isfinite(loss); total+=loss*n; count+=n
            val=legacy.evaluate(model,va,device)
            score=val['RMSE'] if region=='japan' else val['common12']['RMSE']
            assert np.isfinite(score); scheduler.step(score)
            row=dict(epoch=epoch,training_MAE=total/count,validation=val,validation_RMSE=score,
                     reference_RMSE=refs[epoch-1] if stage=='a1' else None,
                     absolute_difference=abs(score-refs[epoch-1]) if stage=='a1' else None,
                     elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);results.append(row)
    passed=all(r['absolute_difference']<1e-6 for r in results) if stage=='a1' else True
    write(out/('CHECK_A1.json' if stage=='a1' else 'CHECK_A2.json'),dict(status='passed' if passed else 'failed',
          region=region,stage=stage,epochs=results,absolute_tolerance_exclusive=1e-6 if stage=='a1' else None,
          all_outputs_finite=True,scientific_checkpoint_created=False,diagnostic_only=True,runtime=runtime))
    write(out/'complete.json',dict(status='complete',scientific_check_passed=passed,stage=stage,region=region,
          files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
    return 0 if passed else 2


def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--region',choices=['china','japan'],required=True);p.add_argument('--stage',choices=['a1','a2'],required=True)
    a=p.parse_args();return diagnostic(a.root,a.job,a.region,a.stage)


if __name__=='__main__':sys.exit(main())
