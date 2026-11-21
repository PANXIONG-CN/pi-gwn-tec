"""Registered penalty-term, graph-support and SimVP arms over the frozen penalty and backbone components.

Penalty composition (single terms), a data-derived gate for the nighttime term, and a
support x penalty factorial (GWN with both graph supports / fixed graph only / adaptive
adjacency only / no graph convolution, SimVP), all at rho = 0.1.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

sys.dont_write_bytecode = True
from tecfc.training import global_train as base  # noqa: E402
from tecfc.objectives import penalty as pi  # noqa: E402

SEEDS = (42, 1024, 2025, 3407, 5000)
TERMS = pi.TERMS  # ('nonneg', 'rot', 'roti', 'diffusion', 'chemistry', 'night')
TERM_ARMS = {'nonneg': 'term-nonneg', 'rot': 'term-rot', 'roti': 'term-tsi', 'diffusion': 'term-graph',
             'chemistry': 'term-prodloss', 'night': 'term-night'}
NEW_ARMS = ('term-nonneg', 'term-rot', 'term-tsi', 'term-graph', 'term-prodloss', 'term-night',
            'term-night-gated', 'pen10-gated',
            'gwn-fixed', 'gwn-fixed-pen10', 'gwn-adapt', 'gwn-adapt-pen10', 'gwn-nograph', 'gwn-nograph-pen10',
            'simvp-pen10')
REFERENCE_ARMS = ('gwn', 'pen10', 'simvp')  # check-only rebuilds of the published references
SPEC = {
    'gwn': dict(backbone='gwn', support='both', penalty=(), gate=False),
    'pen10': dict(backbone='gwn', support='both', penalty=TERMS, gate=False),
    'simvp': dict(backbone='simvp', support=None, penalty=(), gate=False),
    'term-night-gated': dict(backbone='gwn', support='both', penalty=('night',), gate=True),
    'pen10-gated': dict(backbone='gwn', support='both', penalty=TERMS, gate=True),
    'gwn-fixed': dict(backbone='gwn', support='fixed', penalty=(), gate=False),
    'gwn-fixed-pen10': dict(backbone='gwn', support='fixed', penalty=TERMS, gate=False),
    'gwn-adapt': dict(backbone='gwn', support='adaptive', penalty=(), gate=False),
    'gwn-adapt-pen10': dict(backbone='gwn', support='adaptive', penalty=TERMS, gate=False),
    'gwn-nograph': dict(backbone='gwn', support='none', penalty=(), gate=False),
    'gwn-nograph-pen10': dict(backbone='gwn', support='none', penalty=TERMS, gate=False),
    'simvp-pen10': dict(backbone='simvp', support=None, penalty=TERMS, gate=False),
}
for _term in TERMS:
    SPEC[TERM_ARMS[_term]] = dict(backbone='gwn', support='both', penalty=(_term,), gate=False)
assert set(NEW_ARMS) <= set(SPEC)
RHO = {arm: (.1 if spec['penalty'] else 0.) for arm, spec in SPEC.items()}
SUPPORT_FLAGS = {'both': dict(gcn_bool=True, addaptadj=True, aptonly=False),
                 'fixed': dict(gcn_bool=True, addaptadj=False, aptonly=False),
                 'adaptive': dict(gcn_bool=True, addaptadj=True, aptonly=True),
                 'none': dict(gcn_bool=False, addaptadj=False, aptonly=False)}


def write(path, value):
    base.write_json(Path(path), value)


# ---------------------------------------------------------------------------------------
# Backbones


class SupportGWN(nn.Module):
    """The reference GlobalGWN with only the graph-support flags changed."""

    def __init__(self, architecture, device, support):
        super().__init__()
        flags = SUPPORT_FLAGS[support]
        with np.load(architecture.HERE / 'prepared/graph_svd_v1.npz', allow_pickle=False) as z:
            q = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
            if q.shape != (5112, 5112) or q.nnz != 45576:
                raise ValueError('Unexpected predictor graph')
            u = torch.from_numpy(z['u'].copy()).to(device)
            s = torch.from_numpy(z['s'].copy()).to(device)
            v = torch.from_numpy(z['v'].copy()).to(device)
        args = types.SimpleNamespace(adj_mx=q.toarray(), num_nodes=5112, dropout=.3, blocks=4, layers=2,
                                     adjtype='doubletransition', randomadj=False, kernel_size=2, nhid=32,
                                     residual_channels=32, dilation_channels=32, input_window=12,
                                     output_window=12, driver_fusion='none', **flags)
        original_svd = torch.svd

        def precomputed_svd(a, *unused, **kwargs):
            if tuple(a.shape) != (5112, 5112) or a.device.type != device.type:
                raise ValueError('Unexpected SVD request')
            return u, s, v

        try:
            torch.svd = precomputed_svd
            self.gwn = architecture.GWNET(args, device, dim_in=1, dim_out=1)
        finally:
            torch.svd = original_svd
        self.support = support
        self.support_record = dict(support=support, **flags,
                                   fixed_supports=0 if self.gwn.supports is None else len(self.gwn.supports),
                                   adaptive=bool(flags['gcn_bool'] and flags['addaptadj']),
                                   graph_convolution=bool(flags['gcn_bool']))

    def forward(self, history, restore=True):
        if history.ndim != 4 or tuple(history.shape[1:]) != (12, 71, 72):
            raise ValueError('history must be [B,12,71,72]')
        p = self.gwn(history.reshape(history.shape[0], 12, 5112, 1)).reshape(history.shape[0], 12, 71, 72)
        return torch.cat((p, p[..., :1]), dim=-1) if restore else p


def build_gwn(architecture, device, arm):
    support = SPEC[arm]['support']
    if support == 'both':
        return architecture.GlobalGWN(device)  # identical constructor path to GWN and the six-term penalty
    return SupportGWN(architecture, device, support)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def simvp_modules(root):
    """Load the frozen SimVP sources under their original bare module names."""
    root = Path(root)
    names = ('gim_data', 'periodic', 'attentions', 'simvp', 'cg_model', 'global_model', 'train_global')
    previous = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(root), str(root / 'shared')]
        modules = {name: load(name, root / file) for name, file in (
            ('gim_data', 'gim_data.py'), ('periodic', 'periodic.py'), ('attentions', 'shared/attentions.py'),
            ('simvp', 'shared/simvp.py'), ('cg_model', 'shared/cg_model.py'), ('global_model', 'global_model.py'),
            ('train_global', 'train_global.py'))}
        yield modules
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class SimVPWindows(torch.utils.data.Dataset):
    """The frozen SimVP windows with explicit data roots (unused SimVP conditions are passed unchanged)."""

    def __init__(self, role, config, modules, evaluation=False):
        roles = ('validation', 'original_evaluation', 'extension_evaluation') if evaluation else ('train', 'validation')
        if role not in roles:
            raise ValueError('Forbidden SimVP role: ' + role)
        gim = modules['gim_data']
        roots = config['simvp']['data_roots']
        allowed = set(range(2009, 2026)) if evaluation else set(base.ALLOWED_YEARS)

        class Restricted(gim.GIM):
            def path(self, year):
                if int(year) not in allowed:
                    raise ValueError('Forbidden SimVP label year')
                return super().path(year)

        self.gim = Restricted(roots['old'], roots['extension'])
        self.days = list(self.gim.windows(role))
        expected = dict(train=2553, validation=730, original_evaluation=1826, extension_evaluation=1096)[role]
        if len(self.days) != expected:
            raise ValueError('Changed SimVP windows')
        assets = Path(config['simvp']['asset_root'])
        with np.load(assets / 'prepared/train_climatology_v1.npz', allow_pickle=False) as z:
            self.clim = z['climatology']
            self.scale = float(z['scale'])
        if self.scale != 147.1:
            raise ValueError('Changed scale')
        self.geom = np.load(assets / 'prepared/geometry.npy', mmap_mode='r')
        self.restore = gim.restore_grid

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        d = self.days[i]
        s = self.gim.sample(d, self.clim)
        prev = np.datetime64(d, 'D') - np.timedelta64(1, 'D')
        doy = int((prev - prev.astype('datetime64[Y]').astype('datetime64[D]')).astype(int))
        scale = np.float32(self.scale)
        return dict(history=torch.from_numpy(np.array(s['history'] / scale, dtype=np.float32)),
                    background_history=torch.from_numpy(np.array(s['background_history'] / scale, dtype=np.float32)),
                    background_future=torch.from_numpy(np.array(s['background_future'] / scale, dtype=np.float32)),
                    geometry=torch.from_numpy(np.array(self.geom[doy], dtype=np.float32)),
                    target=torch.from_numpy(np.array(self.restore(s['target']), dtype=np.float32)),
                    valid=torch.from_numpy(np.array(self.restore(s['valid']), dtype=bool)),
                    origin=int(s['origin']), index=i)


class Backbone:
    """Uniform build/windows/forward over the GWN and SimVP references."""

    def __init__(self, arm, config, device):
        self.arm, self.config, self.device = arm, config, device
        self.kind = SPEC[arm]['backbone']
        self._stack = None

    def __enter__(self):
        if self.kind == 'gwn':
            self._stack = base.reference_modules(self.config['reference_root'], self.config['asset_root'])
            self.gim, self.architecture = self._stack.__enter__()
        else:
            self._stack = simvp_modules(self.config['simvp']['root'])
            self.modules = self._stack.__enter__()
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def seed(self, seed):
        if self.kind == 'gwn':
            base.seed_all(seed)
        else:
            self.modules['train_global'].seed_all(seed)

    def build(self):
        if self.kind == 'gwn':
            return build_gwn(self.architecture, self.device, self.arm).to(self.device)
        return self.modules['global_model'].GlobalAmplitudeConditionedSimVP().to(self.device)

    def initial_digest(self, model):
        if self.kind == 'gwn':
            return base.tensor_digest(model.state_dict())
        return self.modules['train_global'].digest(model)

    def windows(self, role, evaluation=False):
        if self.kind == 'gwn':
            if evaluation:
                from tecfc.features import member_inputs as f15
                return f15.Windows(role, dict(self.config, arm='G'), self.gim, evaluation=True)
            return base.DevelopmentWindows(role, self.config, self.gim)
        return SimVPWindows(role, self.config, self.modules, evaluation=evaluation)

    def forward(self, model, batch):
        if self.kind == 'gwn':
            return model(batch['history'].to(self.device), restore=True)
        return model(*(batch[k].to(self.device) for k in ('history', 'background_history', 'background_future',
                                                          'geometry')), arm='C', restore=True)

    def optimizer(self, model):
        if self.kind == 'gwn':
            return base.optimizer_for(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.9, patience=5,
                                                               threshold=1e-5, threshold_mode='abs', cooldown=10,
                                                               min_lr=0.)
        return optimizer, scheduler


# ---------------------------------------------------------------------------------------
# Penalty selection and the data-derived night gate


class NightGate:
    def __init__(self, path, sha256):
        if base.sha(path) != sha256:
            raise ValueError('Changed night gate table')
        with np.load(path, allow_pickle=False) as z:
            self.table = np.asarray(z['gate'], np.float32)
        if self.table.shape != (3, 12, 12, 18):
            raise ValueError('Wrong gate table shape')
        from tecfc.objectives import night_gate
        self.bins = night_gate.transition_bins

    def mask(self, origins, kp, device):
        flat = self.bins(np.asarray(origins, np.int64), np.asarray(kp, np.float64))
        return torch.as_tensor(self.table.reshape(-1)[flat], device=device)  # [B, 11, 5183]


class Auxiliary(pi.Auxiliary):
    """Six frozen terms evaluated as in the six-term penalty; only the registered terms enter P."""

    def __init__(self, arm, config, device):
        super().__init__('PI', config['physics'], device)
        self.arm, self.device = arm, device
        self.selected = SPEC[arm]['penalty']
        if not self.selected:
            raise ValueError('No registered penalty terms')
        self.gate = NightGate(config['night_gate']['path'], config['night_gate']['sha256']) if SPEC[arm]['gate'] else None

    def __call__(self, prediction, origins):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pred = prediction.float().reshape(len(prediction), 12, 5183, 1)
            raw = self.drivers.at(origins, 'PI')
            _, terms = self.physics(pred, None, drivers=raw)
            counts = self.physics.residual_counts.copy()
            extra = {}
            if self.gate is not None:
                origins = np.asarray(origins, np.int64)
                ix = np.searchsorted(self.drivers.store.time, origins[:, None] + np.arange(12, dtype=np.int64) * 7200)
                kp = self.drivers.store.slice_batched(ix[:, 0], 12, 5183, fields=('Kp',), require_cos_sza=False,
                                                      include_grid_shape=False)['Kp'].detach().cpu().numpy()
                y = self.physics._to_nd(self.physics._as_BTN1(pred, field='prediction'))
                validated = self.physics._validate_driver_dependencies(raw, batch=len(pred), horizon=12, nodes=5183,
                                                                       device=y.device, dtype=y.dtype)
                night = (validated['cosSZA'][:, :-1] <= 1e-6).to(y.dtype)
                ungated = night * nn.functional.relu(y[:, 1:, :, 0] - y[:, :-1, :, 0])
                gate = self.gate.mask(origins, kp, y.device).to(y.dtype)
                gated = gate * ungated
                extra = dict(night_ungated_recomputed=float(ungated.square().mean().detach()),
                             night_internal=float(terms['night'].detach()),
                             gate_on_fraction_of_night=float((gate * night).sum() / night.sum().clamp_min(1)))
                terms = dict(terms, night=gated.square().mean())
                counts = dict(counts, night=dict(nonzero=int(torch.count_nonzero(gated)), count=gated.numel()))
            loss = .5 * sum(terms[k] for k in self.selected)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite auxiliary loss')
        recorded = {k: float(v.detach()) for k, v in terms.items()}
        return loss, recorded, counts, extra


def scientific_config(config):
    value = base.scientific_config(config)
    value.pop('reference_epochs', None)
    for key in ('driver', 'scaler', 'input2026', 'origins2026', 'night_gate'):
        if key in value and isinstance(value[key], dict):
            value[key].pop('path', None)
    for row in value.get('history_only_files', {}).values():
        row.pop('path', None)
    for key in ('driver', 'adjacency', 'empirical_root'):
        value['physics'].pop(key, None)
    for row in value.get('check_references', {}).values():
        row.pop('path', None)
    if 'simvp' in value:
        value['simvp'] = {k: v for k, v in value['simvp'].items() if k not in ('root', 'asset_root', 'data_roots')}
    value.pop('candidates', None)
    value.pop('candidate_freeze_sha256', None)
    return value
