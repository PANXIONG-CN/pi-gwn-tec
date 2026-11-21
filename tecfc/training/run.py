"""Train one seed of one paper configuration.

    python -m tecfc.training.run --config configs/gwn_geo.json --seed 42 \
        --gimtec /data/gimtec --assets /data/tecfc_assets --out runs/gwn_geo/42

Release launcher (new code).  It builds the configuration dictionary the
original training function received, runs the checks the original entry point
ran, and calls that function unchanged.  GWN and SimVP keep their original
self-contained training directories (tecfc/models/*_reference); they are copied
into the output directory and run there, as in the experiments.

Training was run on one NVIDIA RTX PRO 6000 Blackwell GPU with bf16 autocast
(PyTorch 2.9.1, CUDA 12.8).  Other GPUs, library versions or CPU execution give
numerically different weights; the recorded runtime is printed for comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from tecfc.training import families

BUNDLES = {'gwn_reference': ('gim_data.py', 'global_model.py', 'train_global.py', 'SCIENTIFIC_FREEZE.json',
                             'DATA_AUDIT.json', 'model/GWN/GWN.py', 'model/GWN/driver_fusion.py',
                             'lib/physics/driver_scaler.py'),
           'simvp_reference': ('gim_data.py', 'global_model.py', 'periodic.py', 'train_global.py',
                               'SCIENTIFIC_FREEZE.json', 'DATA_AUDIT.json', 'shared/attentions.py',
                               'shared/cg_model.py', 'shared/simvp.py')}


def release_identity():
    try:
        commit = subprocess.run(['git', '-C', str(families.REPO), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unversioned'
    return dict(source_commit='pi-gwn-tec:' + commit)


def runtime_note(recorded):
    observed = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')
    print(json.dumps(dict(recorded_training_runtime=recorded, this_runtime=observed,
                          identical=recorded == observed)), flush=True)


def work_directory(bundle, out, mapping, prepared):
    """Copy a reference training directory and link data and prepared assets into it."""
    source = families.TECFC / 'models' / bundle
    out.mkdir(parents=True, exist_ok=True)
    for name in BUNDLES[bundle]:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source / name, target)
    data = out / 'data'; data.mkdir(exist_ok=True)
    for name in ('gim', 'gim_extension'):
        link = data / name
        if not link.exists():
            link.symlink_to(Path(mapping['{GIMTEC}']), target_is_directory=True)
    (out / 'prepared').mkdir(exist_ok=True)
    record = json.loads((families.CONFIGS / 'inputs/assets.json').read_text())['assets']
    for name in prepared:
        source_asset = Path(mapping['{ASSETS}']) / name
        target = out / name
        if not target.exists():
            target.symlink_to(source_asset)
        expected = record[name]
        observed = families.sha(target)
        if observed != expected['sha256'] and families.content_sha256(target) != expected.get('content_sha256'):
            raise SystemExit(f'prepared asset {name} differs from the recorded one')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gimtec'); parser.add_argument('--assets')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--accept-initial-digest', action='store_true',
                        help='continue when this PyTorch build initializes different tensors than the recorded run')
    args = parser.parse_args(argv)
    paper = families.load(args.config)
    mapping = families.locations(args.gimtec, args.assets)
    out = Path(args.out)
    torch.set_num_threads(4)
    if paper['family'] in BUNDLES:
        work = work_directory(paper['family'], out, mapping, paper['prepared_assets'])
        identity = paper['identities'][str(args.seed)]
        print(json.dumps(dict(paper=paper['paper_name'], seed=args.seed, identity=identity, work=str(work))), flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        return subprocess.call([sys.executable, 'train_global.py', 'train', '--identity', identity], cwd=work, env=env)
    config = families.runtime_config(paper, args.seed, mapping)
    families.bind_assets(config)
    families.validate(paper, config)
    runtime_note(config.get('runtime'))
    device = torch.device(args.device)
    if args.accept_initial_digest and config.get('expected_initial_tensor_sha256'):
        print(json.dumps(dict(note='initial tensor digest check relaxed by --accept-initial-digest')), flush=True)
        config = initial_digest_from_this_build(paper, config, device)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'paper_config.json').write_text(json.dumps(dict(paper=paper['paper_name'], seed=args.seed,
                                                           config=args.config), indent=1) + '\n')
    module = families.entry(paper)
    result = module.train(config, release_identity(), out, device)
    print(json.dumps(dict(status=result.get('status'), selected_epoch=result.get('selected_epoch'))), flush=True)
    return 0


def initial_digest_from_this_build(paper, config, device):
    """Replace the recorded initial-tensor digest with the one this PyTorch build produces."""
    from tecfc.training import global_train as base
    family = paper['family']
    if family in ('penalty', 'ablation'):
        from tecfc.objectives import ablation_arms
        with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, architecture):
            base.seed_all(config['seed'])
            model = (architecture.GlobalGWN(device) if family == 'penalty'
                     else ablation_arms.build_model(architecture, device, config['arm'])).to(device)
    elif family in ('clim', 'combined'):
        module = families.entry(paper)
        with module.modules(config, device) as (gim, constructor):
            base.seed_all(config['seed']); model = constructor().to(device)
    elif family == 'penalty_arms':
        from tecfc.objectives import penalty_arms
        with penalty_arms.Backbone(config['arm'], config, device) as backbone:
            backbone.seed(config['seed']); model = backbone.build()
            if penalty_arms.SPEC[config['arm']]['backbone'] == 'simvp':
                digest = backbone.initial_digest(model)
                return dict(config, simvp=dict(config['simvp'], expected_initial_tensor_sha256=digest))
    else:
        return config
    return dict(config, expected_initial_tensor_sha256=base.tensor_digest(model.state_dict()))


if __name__ == '__main__':
    raise SystemExit(main())
