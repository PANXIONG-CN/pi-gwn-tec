"""Frozen manuscript physics and EMP adapters, with explicit units and times."""
from contextlib import contextmanager
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'old_global_pi'
TERMS = ('nonneg', 'rot', 'roti', 'diffusion', 'chemistry', 'night')
SOURCE_SHA = 'fbd0a9eea7075fde4c79915fa862a86944ad9b2e7c4f93894743e87bba1ba645'
T0 = int(np.datetime64('2009-01-01', 's').astype(np.int64))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def extract(path, names, namespace):
    """Execute exact selected frozen definitions without importing training entrypoints."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


ppinn = module('_exp0013_ppinn', SOURCE / 'lib/physics/ppinn.py')
normalization = module('_exp0013_normalization', SOURCE / 'lib/normalization.py')
solar = module('_exp0013_solar', HERE / 'vendor/solar_source.py')
projected = extract(SOURCE / 'revisions/gim_tec_only_physics_v1/objective.py',
                    ('merge', 'assign_gradients'), {'torch': torch})
assign_gradients = projected['assign_gradients']


@contextmanager
def original_imports():
    aliases = {'lib.physics.ppinn': ppinn, 'lib.normalization': normalization}
    sm = ModuleType('lib.physics.solar')
    sm.canonical_cos_sza_float64 = solar._cos_sza_chunk
    aliases['lib.physics.solar'] = sm
    previous = {k: sys.modules.get(k) for k in aliases}
    try:
        sys.modules.update(aliases)
        yield
    finally:
        for k, value in previous.items():
            if value is None: sys.modules.pop(k, None)
            else: sys.modules[k] = value


with original_imports():
    driver_module = module('_exp0013_driver_store', SOURCE / 'lib/physics/drivers.py')


class RecordedPPinn(ppinn.PPinnLoss):
    def _aggregate(self, total, terms, term_id, residual):
        with torch.no_grad():
            self.residual_counts[term_id] = dict(nonzero=int(torch.count_nonzero(residual)), count=residual.numel())
        return super()._aggregate(total, terms, term_id, residual)


def build(config, device, original=False):
    """Original function for A; independently mapped frozen constructor for training."""
    adjacency = config['adjacency']
    if original:
        import hashlib
        namespace = dict(np=np, torch=torch, Path=Path, hashlib=hashlib,
                         ADJ_PATH=adjacency, ADJ_SHA=config['adjacency_sha256'], PHYSICS_TERMS=TERMS)
        extract(HERE / 'vendor/manuscript_model.py', ('build_ppinn',), namespace)
        with original_imports():
            result = namespace['build_ppinn'](Path('/'), device)
    else:
        graph = torch.as_tensor(np.load(adjacency, allow_pickle=False), device=device)
        result = RecordedPPinn(normalization.MinMax01Scaler(0, 147.1), graph,
            dataset='gim', normalizer_id='gim_tec01_147p1_v1', model_key='pi_driver',
            use_nonneg=True, use_rot=True, use_roti=True, use_diffusion=True,
            use_chemistry=True, use_night=True, use_dynamic_rot_cap=True, use_advection=False,
            strict_paper_run=True, first_transition_policy='forecast_internal_only',
            dt_minutes=120, tec_ref=50., t_ref_sec=7200., rot_cap_tecu_per_min=.5,
            rot_cap_k=.01, roti_cap_scale=.7, roti_w=3, kappa_nd=.05,
            alpha_nd=.01, beta_nd=.01, ae_factor=.001,
            adjacency_sha256=config['adjacency_sha256'],
            expected_adjacency_sha256=config['adjacency_sha256']).to(device)
        result.residual_counts = {}
    assert tuple(result.uncert.log_vars) == TERMS
    for parameter in result.parameters():
        parameter.requires_grad_(False)
        with torch.no_grad(): parameter.zero_()
    return result


class Drivers:
    def __init__(self, path, device):
        self.store = driver_module.DriversStore(str(path), device=device, strict=True, expected_sha256=SOURCE_SHA)
        self.device = device

    def at(self, origins, arm):
        origins = np.asarray(origins, dtype=np.int64)
        years = origins.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
        if not set(years) <= {2009, 2010, 2011, 2012, 2014, 2016, 2018}:
            raise ValueError('Auxiliary drivers are training-only')
        query = origins[:, None] + np.arange(12, dtype=np.int64) * 7200
        ix = np.searchsorted(self.store.time, query)
        if not np.array_equal(self.store.time[ix], query):
            raise ValueError('Target driver times disagree')
        return self.store.slice_batched(ix[:, 0], 12, 5183,
            fields=('F107', 'AE', 'Dst') if arm == 'PI' else ('F107', 'AE'),
            require_cos_sza=True, include_grid_shape=False)

    def original_at(self, starts):
        namespace = dict(np=np, DriversStore=driver_module.DriversStore, SOURCE_SHA=SOURCE_SHA, T0=T0)
        extract(HERE / 'vendor/manuscript_drivers.py', ('TrainingDrivers',), namespace)
        old = object.__new__(namespace['TrainingDrivers'])
        old.store = self.store
        return old.physics(starts)


class Auxiliary:
    def __init__(self, arm, config, device):
        self.arm = arm
        self.drivers = Drivers(config['driver'], device)
        self.physics = build(config, device) if arm == 'PI' else None
        self.empirical = module('_exp0013_empirical', Path(config['empirical_root']) / 'empirical_tendency.py')

    def __call__(self, prediction, origins):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pred = prediction.float()
            raw = self.drivers.at(origins, self.arm)
            if self.arm == 'PI':
                _, terms = self.physics(pred.reshape(len(pred), 12, 5183, 1), None, drivers=raw)
                loss = .5 * sum(terms.values())
                counts = self.physics.residual_counts.copy()
            else:
                residual = self.empirical.internal_residual(pred * 147.1, raw['AE'], raw['F107'],
                                                            raw['cosSZA'].reshape_as(pred))
                terms = {'empirical': residual.square().mean()}
                loss = .5 * terms['empirical']
                counts = {'empirical': dict(nonzero=int(torch.count_nonzero(residual)), count=residual.numel())}
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite auxiliary loss')
        return loss, {k: float(v.detach()) for k, v in terms.items()}, counts
