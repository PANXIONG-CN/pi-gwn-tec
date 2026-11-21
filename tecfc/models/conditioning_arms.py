"""Registered conditioning arms over the frozen inputs of GWN + cosSZA + indices and the strong recipe.

All arms read exactly the windows of GWN + cosSZA + indices: TEC/147.1 plus the five z-scored
channels (AE, Dst, Kp, F10.7, daylight-clipped cos SZA) from configs/inputs/driver_scaler.json.
Gated fusion, FiLM: the reference GWNET's own residual_gate / film fusion (TEC-only GWN input).
Cross-attention: the earlier node-wise causal temporal cross-attention applied right after start_conv.
Graph-Transformer: the earlier graph-Transformer adaptation (conditioning_vendor/gnntrans.py; design
notes in legacy/model/GNNTrans/ARCHITECTURE.md), on the same inputs.
"""
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / 'conditioning_vendor'))
from tecfc.training import global_train as base  # noqa: E402
from tecfc.features import member_inputs as f15  # noqa: E402

SEEDS = (42, 1024, 2025, 3407, 5000)
NEW_ARMS = ('cossza-idx-gate', 'cossza-idx-film', 'cossza-idx-xattn', 'graph-transformer')
REFERENCE_ARMS = ('cossza-idx',)
SPEC = {'cossza-idx': dict(model='gwn_concat'), 'cossza-idx-gate': dict(model='gwn', fusion='residual_gate'),
        'cossza-idx-film': dict(model='gwn', fusion='film'), 'cossza-idx-xattn': dict(model='gwn', fusion='causal_crossattn'),
        'graph-transformer': dict(model='gnntrans')}
RHO = {arm: 0. for arm in SPEC}


def write(path, value):
    base.write_json(Path(path), value)


def provenance(config):
    """One label binding the z-scored driver sources (2009-2025 archive and the frozen 2026 input)."""
    payload = dict(driver=config['driver']['sha256'], inputs2026=config.get('inputs2026_sha256'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class StartWithCrossAttention(nn.Module):
    """start_conv followed by the earlier cross-attention; drivers are staged per forward."""

    def __init__(self, conv, attention):
        super().__init__()
        self.conv, self.attention, self.drivers = conv, attention, None

    def forward(self, x):
        if self.drivers is None:
            raise ValueError('Driver history was not staged')
        return self.attention(self.conv(x), self.drivers)


class FusionGWN(nn.Module):
    """The reference GlobalGWN construction with driver fusion enabled; TEC is the only GWN input."""

    def __init__(self, architecture, device, fusion, label, scaler_sha):
        super().__init__()
        import scipy.sparse as sp
        with np.load(architecture.HERE / 'prepared/graph_svd_v1.npz', allow_pickle=False) as z:
            q = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
            u = torch.from_numpy(z['u'].copy()).to(device)
            s = torch.from_numpy(z['s'].copy()).to(device)
            v = torch.from_numpy(z['v'].copy()).to(device)
        from model.GWN.driver_fusion import required_film_config
        from lib.physics.driver_scaler import DRIVER_CHANNEL_ORDER, TRAIN_ZSCORE, DriverHistoryBatch
        self.batch_type, self.order, self.scaling = DriverHistoryBatch, DRIVER_CHANNEL_ORDER, TRAIN_ZSCORE
        if tuple(DRIVER_CHANNEL_ORDER) != tuple(f15.DRIVER_ORDER):
            raise ValueError('Driver channel order differs from GWN + cosSZA + indices')
        built = 'residual_gate' if fusion == 'causal_crossattn' else fusion
        args = types.SimpleNamespace(adj_mx=q.toarray(), num_nodes=5112, dropout=.3, blocks=4, layers=2, gcn_bool=True,
                                     addaptadj=True, adjtype='doubletransition', randomadj=False, aptonly=False,
                                     kernel_size=2, nhid=32, residual_channels=32, dilation_channels=32,
                                     input_window=12, output_window=12, driver_fusion=built,
                                     driver_scaling=TRAIN_ZSCORE, predictor_driver_channels=list(DRIVER_CHANNEL_ORDER),
                                     driver_source_sha256=label, driver_scaler_sha256=scaler_sha,
                                     **(required_film_config() if built == 'film' else {}))
        original_svd = torch.svd
        torch.svd = lambda a, *x, **k: (u, s, v)
        try:
            self.gwn = architecture.GWNET(args, device, dim_in=1, dim_out=1)
        finally:
            torch.svd = original_svd
        self.fusion, self.label, self.scaler_sha = fusion, label, scaler_sha
        if fusion == 'causal_crossattn':
            from attention import NodewiseCausalTemporalCrossAttention
            self.gwn.history_gate = None
            self.gwn.start_conv = StartWithCrossAttention(self.gwn.start_conv, NodewiseCausalTemporalCrossAttention())
        if fusion == 'film':
            self.gwn.assert_film_zero_initialization()

    def forward(self, history, origins, restore=True):
        if history.ndim != 5 or tuple(history.shape[1:]) != (12, 71, 72, 6):
            raise ValueError('history must be the [B,12,71,72,6] input of GWN + cosSZA + indices')
        b = history.shape[0]
        source = history[..., :1].reshape(b, 12, 5112, 1)
        values = history[..., 1:].reshape(b, 12, 5112, 5)
        origins = torch.as_tensor(np.asarray(origins, np.int64))
        times = origins[:, None] - torch.arange(12, 0, -1, dtype=torch.int64)[None, :] * 7200
        batch = self.batch_type(values=values, timestamps=times, forecast_origins=origins, channel_order=tuple(self.order),
                                source_driver_sha256=self.label, driver_scaler_sha256=self.scaler_sha, scaling=self.scaling)
        if self.fusion == 'causal_crossattn':
            self.gwn.start_conv.drivers = values
        try:
            p = self.gwn(source, batch).reshape(b, 12, 71, 72)
        finally:
            if self.fusion == 'causal_crossattn':
                self.gwn.start_conv.drivers = None
        return torch.cat((p, p[..., :1]), dim=-1) if restore else p


class GraphTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        import gnntrans
        self.model = gnntrans.GNNTransPyramid()

    def forward(self, history, origins=None, restore=True):
        if history.ndim != 5 or tuple(history.shape[1:]) != (12, 71, 72, 6):
            raise ValueError('history must be the [B,12,71,72,6] input of GWN + cosSZA + indices')
        b = history.shape[0]
        full = torch.cat((history, history[..., :1, :]), dim=-2)            # [B,12,71,73,6], +180 = -180
        tec = full[..., 0].reshape(b, 12, 5183, 1)
        drivers = full[..., 1:].reshape(b, 12, 5183, 5)
        out = self.model(tec, drivers).reshape(b, 12, 71, 73)
        return out if restore else out[..., :72]


class Backbone:
    def __init__(self, arm, config, device):
        self.arm, self.config, self.device = arm, config, device
        self._stack = None

    def __enter__(self):
        self._stack = base.reference_modules(self.config['reference_root'], self.config['asset_root'])
        self.gim, self.architecture = self._stack.__enter__()
        sys.path.insert(0, str(Path(self.config['reference_root'])))
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def seed(self, seed):
        base.seed_all(seed)

    def build(self):
        spec = SPEC[self.arm]
        if spec['model'] == 'gwn_concat':
            return f15.build_model(self.architecture, self.device, 'G-DE').to(self.device)
        if spec['model'] == 'gnntrans':
            return GraphTransformer().to(self.device)
        return FusionGWN(self.architecture, self.device, spec['fusion'], provenance(self.config),
                         self.config['scaler']['sha256']).to(self.device)

    def initial_digest(self, model):
        return base.tensor_digest(model.state_dict())

    def windows(self, role, evaluation=False):
        return f15.Windows(role, dict(self.config, arm='G-DE'), self.gim, evaluation=evaluation)

    def forward(self, model, batch):
        history = batch['history'].to(self.device)
        if SPEC[self.arm]['model'] == 'gwn_concat':
            return model(history, restore=True)
        return model(history, batch['origin'].numpy() if torch.is_tensor(batch['origin']) else batch['origin'],
                     restore=True)

    def optimizer(self, model):
        return base.optimizer_for(model)


def scientific_config(config):
    value = base.scientific_config(config)
    value.pop('reference_epochs', None)
    for key in ('driver', 'scaler'):
        if key in value and isinstance(value[key], dict):
            value[key].pop('path', None)
    for row in value.get('history_only_files', {}).values():
        row.pop('path', None)
    for key in ('driver', 'adjacency', 'empirical_root'):
        value.get('physics', {}).pop(key, None)
    for row in value.get('check_references', {}).values():
        row.pop('path', None)
    for key in ('inputs2026', 'origins2026'):
        if key in value:
            for row in value[key].values():
                row.pop('path', None)
    value.pop('scorer', None)
    value.pop('targets2026_after_freeze', None)
    value.pop('candidates', None)
    value.pop('candidate_freeze_sha256', None)
    return value
