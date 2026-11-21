"""Paper configurations: placeholder resolution, input checks and family dispatch.

Release support code (not part of the original experiments).  A file in configs/
holds the scientific configuration of one paper configuration with logical
locations:

  {TECFC}    this package
  {CONFIGS}  configs/ of this repository
  {RECORDS}  records/ of this repository
  {GIMTEC}   directory with TEC_<year>.npy, drivers_2009_2022_2h.npz and
             drivers_2009_2025_2h.npz (GIMtec data record); --gimtec or TECFC_GIMTEC
  {ASSETS}   directory with gim_adj.npy and prepared/ (scripts/prepare_assets.py;
             prepared/NIGHT_GATE.npz of the gated penalty arms: scripts/prepare_night_gate.py);
             --assets or TECFC_ASSETS

runtime_config(...) returns exactly the dictionary the original training code
received, with these locations filled in.  The checks that the original
entry points ran before training (data and asset hashes, recipe fields) are
called here; the original frozen-snapshot and exact-GPU bindings are not.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np

TECFC = Path(__file__).resolve().parents[1]
REPO = TECFC.parent
CONFIGS = REPO / 'configs'
RECORDS = REPO / 'records'
YEARS = tuple(range(2009, 2027))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def content_sha256(path):
    """Per-array digest of an .npz file: dtype, shape and bytes (independent of zip timestamps)."""
    out = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            array = np.ascontiguousarray(archive[key])
            h = hashlib.sha256()
            h.update(f'{array.dtype.str}|{array.shape}|'.encode())
            h.update(array.tobytes())
            out[key] = h.hexdigest()
    return out


def locations(gimtec=None, assets=None):
    gimtec = gimtec or os.environ.get('TECFC_GIMTEC')
    assets = assets or os.environ.get('TECFC_ASSETS')
    return {'{TECFC}': str(TECFC), '{CONFIGS}': str(CONFIGS), '{RECORDS}': str(RECORDS),
            '{GIMTEC}': None if gimtec is None else str(Path(gimtec).resolve()),
            '{ASSETS}': None if assets is None else str(Path(assets).resolve())}


def substitute(value, mapping):
    if isinstance(value, dict):
        return {k: substitute(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, mapping) for v in value]
    if isinstance(value, str):
        for key, root in mapping.items():
            if key in value:
                if root is None:
                    raise SystemExit(f'{key} is not set: pass --gimtec/--assets or set TECFC_GIMTEC/TECFC_ASSETS')
                value = value.replace(key, root)
    return value


def load(path):
    paper = json.loads(Path(path).read_text())
    for key in ('paper_name', 'family', 'seeds'):
        if key not in paper:
            raise SystemExit(f'{path}: not a paper configuration ({key} missing)')
    return paper


def runtime_config(paper, seed, mapping):
    if int(seed) not in paper['seeds']:
        raise SystemExit(f"seed {seed} is not one of {paper['seeds']}")
    config = dict(paper['runtime_config'], **paper['per_seed'][str(int(seed))])
    return substitute(config, mapping)


def gimtec_files(years, mapping):
    """data_files for evaluation: every year with its recorded SHA-256."""
    record = json.loads((CONFIGS / 'inputs/gimtec_files.json').read_text())['files']
    return {str(y): dict(path=substitute(f'{{GIMTEC}}/TEC_{y}.npy', mapping), sha256=record[f'TEC_{y}.npy'])
            for y in years}


def bind_assets(config, log=print):
    """Accept regenerated .npz assets whose arrays equal the recorded ones.

    The original configurations record file hashes of prepared/*.npz.  np.savez
    stores the creation time in the zip headers, so a regenerated file differs
    in bytes but not in content.  When the bytes differ and the content matches,
    the observed file hash is bound for this run and both hashes are reported.
    """
    record = json.loads((CONFIGS / 'inputs/assets.json').read_text())['assets']
    bound = {}
    for name, expected in list(config.get('asset_sha256', {}).items()):
        path = Path(config['asset_root']) / name
        if not path.is_file():
            raise SystemExit(f'missing asset {path}: run scripts/prepare_assets.py')
        observed = sha(path)
        if observed != expected:
            wanted = record[name].get('content_sha256')
            if wanted is None or content_sha256(path) != wanted:
                raise SystemExit(f'asset {name} differs from the recorded one in content')
            config['asset_sha256'][name] = observed
        bound[name] = dict(recorded_sha256=expected, file_sha256=observed, content_verified=observed != expected)
    log(json.dumps(dict(assets=bound)))
    return bound


def check_physics(config):
    physics = config.get('physics')
    if not physics:
        return
    for key in ('driver', 'adjacency'):
        if sha(physics[key]) != physics[key + '_sha256']:
            raise SystemExit('changed ' + key)
    for name, digest in physics.get('empirical_sha256', {}).items():
        if sha(Path(physics['empirical_root']) / name) != digest:
            raise SystemExit('changed empirical source ' + name)


def validate(paper, config):
    """Scientific checks the original entry points called before training."""
    family = paper['family']
    if family == 'penalty':
        from tecfc.training import global_train as base
        base.validate_config(config); base.verify_assets(config); check_physics(config)
    elif family == 'members':
        from tecfc.training import global_train as base
        from tecfc.features.member_inputs import validate_config, verify_assets
        validate_config(config); verify_assets(config, base)
    elif family == 'clim':
        from tecfc.features.clim_inputs import validate as check
        check(config)
    elif family == 'combined':
        from tecfc.features.combined_inputs import validate as check
        check(config)
    elif family == 'ablation':
        from tecfc.training import global_train as base
        base.validate_config(config); base.verify_assets(config); check_physics(config)
    elif family == 'h96':
        from tecfc.features.h96_inputs import validate as check
        check(config, stage='train')
    elif family == 'penalty_arms':
        from tecfc.training import global_train as base
        base.validate_config(config); base.verify_assets(config); check_physics(config)
        check_penalty_arm_inputs(config)
    elif family == 'conditioning':
        from tecfc.training import global_train as base
        base.validate_config(config); base.verify_assets(config)
        for key in ('driver', 'scaler'):
            if sha(config[key]['path']) != config[key]['sha256']:
                raise SystemExit('changed ' + key)
        for row in config['history_only_files'].values():
            if sha(row['path']) != row['sha256']:
                raise SystemExit('changed history-only source')
    else:
        raise SystemExit('unknown family ' + family)


def check_penalty_arm_inputs(config, log=print):
    """SimVP sources and assets, and the night-gate table of the gated arms.

    The SimVP code is read from simvp.root ({TECFC}/models/simvp_reference) and its
    prepared assets from simvp.asset_root ({ASSETS}).  Regenerated .npz files are
    accepted when their arrays equal the recorded ones (as in bind_assets); the
    observed file hash is then bound for this run.  The night-gate table is built
    by scripts/prepare_night_gate.py and is needed only by the arms that use it.
    """
    from tecfc.objectives.penalty_arms import SPEC
    simvp = config['simvp']
    record = json.loads((CONFIGS / 'inputs/assets.json').read_text())['assets']
    for name, digest in list(simvp['sha256'].items()):
        path = Path(simvp['asset_root'] if name.startswith('prepared/') else simvp['root']) / name
        if not path.is_file():
            raise SystemExit(f'missing {path}: run scripts/prepare_assets.py')
        observed = sha(path)
        if observed != digest:
            wanted = record.get(name, {}).get('content_sha256')
            if wanted is None or content_sha256(path) != wanted:
                raise SystemExit('changed SimVP source or asset ' + name)
            simvp['sha256'][name] = observed
    if SPEC[config['arm']]['gate']:
        gate = config['night_gate']
        path = Path(gate['path'])
        if not path.is_file():
            raise SystemExit(f'missing {path}: run scripts/prepare_night_gate.py')
        observed = sha(path)
        if observed != gate['sha256']:
            wanted = json.loads((CONFIGS / 'inputs/night_gate.json').read_text())['content_sha256']
            if content_sha256(path) != wanted:
                raise SystemExit('night-gate table differs from the recorded one in content')
            log(json.dumps(dict(night_gate=dict(recorded_sha256=gate['sha256'], file_sha256=observed,
                                                content_verified=True))))
            gate['sha256'] = observed


def entry(paper):
    return importlib.import_module(paper['train_entry'])
