"""Predict with a trained checkpoint of one paper configuration.

    python -m tecfc.training.predict --config configs/gwn_geo.json --seed 42 \
        --checkpoint runs/gwn_geo/42/best_200.pt --split extension_evaluation \
        --gimtec /data/gimtec --assets /data/tecfc_assets --out gwn_geo_42_2023_2025.npy

Writes float32 TECU predictions [windows, 12 or 48, 71, 73] and <out>.origins.npy
(UTC seconds of each forecast origin).  --days restricts the windows.

Release launcher (new code).  For each family it repeats the steps of the
original inference entry point: seed, model constructor, strict weight
loading, the original evaluation dataset, and the original prediction loop
(batch 32, bf16 autocast on CUDA, prediction x 147.1).  The experiments ran
inference on GPU; CPU predictions differ by the bf16 rounding of the GPU runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tecfc.training import families

SPLITS = ('validation', 'original_evaluation', 'extension_evaluation')


class Selected(Dataset):
    """Windows of a dataset restricted to given days; 'index' runs over the selection."""

    def __init__(self, dataset, days):
        position = {str(d): i for i, d in enumerate(dataset.days)}
        missing = [d for d in days if d not in position]
        if missing:
            raise SystemExit(f'days not in this split: {missing}')
        self.dataset, self.rows = dataset, [position[d] for d in days]
        self.days, self.scale = list(days), dataset.scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = dict(self.dataset[self.rows[index]])
        sample['index'] = index
        return sample


@torch.no_grad()
def predict_loop(model, dataset, device, amp, frames):
    """Prediction loop of the original inference entry points."""
    model.eval()
    out = np.empty((len(dataset), frames, 71, 73), np.float32)
    for b in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        with amp(device):
            p = model(b['history'].to(device), restore=True)
        pred = p.float().cpu().numpy() * np.float32(147.1)
        if not np.isfinite(pred).all() or not np.array_equal(pred[..., 0], pred[..., -1]):
            raise ValueError('Prediction integrity')
        out[b['index'].numpy()] = pred
    return out


def select(dataset, days):
    return dataset if not days else Selected(dataset, days)


def inference_config(paper, seed, mapping, years):
    config = families.runtime_config(paper, seed, mapping)
    config['data_files'] = families.gimtec_files(years, mapping)
    families.bind_assets(config, log=lambda _: None)
    return config


def penalty_windows(gim, config, role):
    """Evaluation windows of the penalty and empirical-tendency arms (their inference entry point)."""
    class ReadonlyGIM(gim.GIM):
        def path(self, year):
            if int(year) not in range(2009, 2026): raise ValueError('Forbidden year')
            return Path(config['data_files'][str(int(year))]['path'])

    class Windows(Dataset):
        def __init__(self, role):
            self.gim = ReadonlyGIM(); self.days = list(self.gim.windows(role)); self.scale = 147.1

        def __len__(self): return len(self.days)

        def __getitem__(self, i):
            s = self.gim.sample(self.days[i])
            return dict(history=torch.from_numpy(np.array(s['history'] / np.float32(self.scale), np.float32)),
                        target=torch.from_numpy(np.array(gim.restore_grid(s['target']), np.float32)),
                        valid=torch.from_numpy(np.array(gim.restore_grid(s['valid']), bool)), origin=int(s['origin']), index=i)
    return Windows(role)


def predict(paper, seed, checkpoint, split, mapping, device, days=()):
    """Return (predictions, origin days) for one checkpoint."""
    from tecfc.training import global_train as base
    family = paper['family']
    years = range(2009, 2026)
    if family in ('gwn_reference', 'simvp_reference'):
        return predict_bundle(paper, seed, checkpoint, split, mapping, device, days)
    config = inference_config(paper, seed, mapping, years)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if family == 'penalty':
        with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, architecture):
            model = architecture.GlobalGWN(device).to(device)
            model.load_state_dict(ck['state_dict'])
            ds = select(penalty_windows(gim, config, split), days)
            return predict_loop(model, ds, device, base.amp, 12), ds.days
    if family == 'members':
        from tecfc.features.member_inputs import Windows, build_model
        with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, architecture):
            base.seed_all(config['seed']); model = build_model(architecture, device, config['arm']).to(device)
            model.load_state_dict(ck['state_dict']); model.eval()
            ds = select(Windows(split, config, gim, evaluation=True), days)
            return predict_loop(model, ds, device, base.amp, 12), ds.days
    if family in ('clim', 'combined'):
        module = families.entry(paper)
        base.seed_all(config['seed'])
        with module.modules(config, device) as (gim, constructor):
            model = constructor().to(device); model.load_state_dict(ck['state_dict'])
            ds = select(module.Windows(split, config, gim, inference=True), days)
            return predict_loop(model, ds, device, base.amp, 12), ds.days
    if family == 'ablation':
        from tecfc.objectives.ablation_arms import Windows, build_model
        base.seed_all(config['seed'])
        with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, architecture):
            model = build_model(architecture, device, config['arm']).to(device)
            model.load_state_dict(ck['state_dict'], strict=True)
            ds = select(Windows(split, config, gim, evaluation=True), days)
            return predict_loop(model, ds, device, base.amp, 12), ds.days
    if family == 'h96':
        from tecfc.features.h96_inputs import Windows, build_model
        config['data_files'] = families.gimtec_files(range(2009, 2027) if split == 'evaluation_2026' else years, mapping)
        base.seed_all(config['seed'])
        with base.reference_modules(config['reference_root'], config['asset_root']) as (_, architecture):
            model = build_model(architecture, device, config['arm'], 48).to(device)
            model.load_state_dict(ck['state_dict'], strict=True)
            ds = select(Windows(split, config, config['arm'], 48, inference=True), days)
            return predict_loop(model, ds, device, base.amp, 48), ds.days
    if family in ('penalty_arms', 'conditioning'):
        if split not in SPLITS:
            raise SystemExit(f'{family}: predictions for {split} are not provided by this launcher')
        if family == 'penalty_arms':
            from tecfc.objectives.penalty_arms import Backbone
        else:
            from tecfc.models.conditioning_arms import Backbone
        with Backbone(config['arm'], config, device) as backbone:
            backbone.seed(config['seed'])
            model = backbone.build()
            model.load_state_dict(ck['state_dict'], strict=True)
            ds = select(backbone.windows(split, evaluation=True), days)
            return backbone_predict_loop(model, ds, backbone, base.amp), ds.days
    raise SystemExit('unknown family ' + family)


@torch.no_grad()
def backbone_predict_loop(model, dataset, backbone, amp):
    """Prediction loop of the inference entry point of the penalty and conditioning arms (inputs via backbone.forward)."""
    model.eval()
    out = np.empty((len(dataset), 12, 71, 73), np.float32)
    for b in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        with amp(backbone.device):
            p = backbone.forward(model, b)
        pred = p.float().cpu().numpy() * np.float32(147.1)
        if not np.isfinite(pred).all() or not np.array_equal(pred[..., 0], pred[..., -1]):
            raise ValueError('Prediction integrity')
        out[b['index'].numpy()] = pred
    return out


def predict_bundle(paper, seed, checkpoint, split, mapping, device, days):
    """GWN and SimVP: the evaluation function of their own training directory."""
    import importlib.util
    import sys
    from tecfc.training.run import work_directory
    with tempfile.TemporaryDirectory() as tmp:
        work = work_directory(paper['family'], Path(tmp), mapping, paper['prepared_assets'])
        sys.path.insert(0, str(work))
        try:
            spec = importlib.util.spec_from_file_location('_release_bundle_train', work / 'train_global.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            arm = 'G' if paper['family'] == 'gwn_reference' else 'C'
            model = (module.GlobalGWN(device) if arm == 'G' else module.GlobalAmplitudeConditionedSimVP()).to(device)
            ck = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ck['state_dict'])
            ds = select(module.Windows(split), days)
            path = Path(tmp) / 'predictions.npy'
            module.evaluate(model, ds, arm, device, 32, path)
            return np.array(np.load(path)), ds.days
        finally:
            sys.path.remove(str(work))
            for name in ('gim_data', 'global_model', 'periodic', 'cg_model', 'attentions', 'simvp', 'model',
                         'model.GWN', 'model.GWN.GWN', 'model.GWN.driver_fusion', 'lib', 'lib.physics',
                         'lib.physics.driver_scaler'):
                sys.modules.pop(name, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', required=True, choices=SPLITS + ('evaluation_2026',))
    parser.add_argument('--days', default='', help='comma-separated YYYY-MM-DD forecast days (default: all)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--gimtec'); parser.add_argument('--assets')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    paper = families.load(args.config)
    days = [d for d in args.days.split(',') if d]
    prediction, used = predict(paper, args.seed, args.checkpoint, args.split,
                               families.locations(args.gimtec, args.assets), torch.device(args.device), days)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, prediction)
    np.save(out.with_suffix('.origins.npy'), np.array([int(np.datetime64(d, 's').astype(np.int64)) for d in used], np.int64))
    print(json.dumps(dict(paper=paper['paper_name'], seed=args.seed, split=args.split, windows=len(used),
                          shape=list(prediction.shape), out=str(out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
