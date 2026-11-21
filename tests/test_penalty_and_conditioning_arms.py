"""Synthetic checks of the penalty-term, graph-support, SimVP-penalty and conditioning arms (no data, no weights).

The GWN builders run on a synthetic periodic eight-neighbour graph with random rank-10 factors in place of
prepared/graph_svd_v1.npz; forward passes use random inputs and one window.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CPU = torch.device('cpu')


def papers(family):
    from tecfc.training import families
    loaded = [families.load(path) for path in sorted((REPO / 'configs').glob('*.json'))]
    return [paper for paper in loaded if paper['family'] == family]


def periodic_graph(folder):
    """Eight-neighbour graph with self loops on 71 x 72 nodes, periodic in longitude (45,576 entries)."""
    import scipy.sparse as sp
    r, c = np.meshgrid(np.arange(71), np.arange(72), indexing='ij')
    rows, cols = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            keep = (r + dr >= 0) & (r + dr < 71)
            rows.append((r * 72 + c)[keep]); cols.append(((r + dr) * 72 + (c + dc) % 72)[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    q = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(5112, 5112))
    assert q.nnz == 45576
    rng = np.random.default_rng(0)
    (folder / 'prepared').mkdir()
    np.savez(folder / 'prepared/graph_svd_v1.npz', data=q.data, indices=q.indices, indptr=q.indptr,
             shape=np.array(q.shape), s=np.ones(10, np.float32),
             u=(rng.normal(size=(5112, 10)) / 72).astype(np.float32), v=(rng.normal(size=(5112, 10)) / 72).astype(np.float32))


@pytest.fixture(scope='module')
def architecture(tmp_path_factory):
    from tecfc.training import global_train as base
    folder = tmp_path_factory.mktemp('assets')
    periodic_graph(folder)
    with base.reference_modules(REPO / 'tecfc/models/gwn_reference', folder) as (_, module):
        yield module


def check_output(out):
    assert tuple(out.shape) == (1, 12, 71, 73) and torch.isfinite(out).all()
    assert torch.equal(out[..., 0], out[..., -1])


def test_night_gate_bins():
    from tecfc.objectives.night_gate import SHAPE, transition_bins
    origin = int(np.datetime64('2010-03-15', 's').astype(np.int64))
    kp = np.array([[1.0, 2.99, 3.0, 4.99, 5.0, 9.0] + [0.0] * 6])
    flat = transition_bins([origin], kp)
    assert flat.shape == (1, 11, 5183) and flat.min() >= 0 and flat.max() < np.prod(SHAPE)
    kp_class, month, local, band = (a.reshape(11, 71, 73) for a in np.unravel_index(flat[0], SHAPE))
    assert (kp_class[:6, 0, 0] == [0, 0, 1, 1, 2, 2]).all() and (month == 2).all()
    # 00 UTC at 180 W is 12 h local solar time (bin 6); each transition starts 2 h later; +180 copies -180.
    assert (local[:, 0, 0] == (6 + np.arange(11)) % 12).all() and (local[..., 0] == local[..., 72]).all()
    assert band[0, 0, 0] == 17 and band[0, 70, 0] == 0  # 87.5 N and 87.5 S


def test_penalty_arm_configurations():
    from tecfc.objectives import penalty_arms as arms
    gate = json.loads((REPO / 'configs/inputs/night_gate.json').read_text())
    assert list(gate['summary']['shape']) == [3, 12, 12, 18] and set(gate['content_sha256']) == {'count', 'increase', 'gate', 'axes'}
    seen = set()
    for paper in papers('penalty_arms'):
        config, arm = paper['runtime_config'], paper['runtime_config']['arm']
        seen.add(arm)
        assert paper['slug'] == arm and paper['train_entry'] == 'tecfc.training.train_penalty_arms'
        assert config['rho'] == arms.RHO[arm] == (0.1 if arms.SPEC[arm]['penalty'] else 0.)
        assert config['night_gate']['sha256'] == gate['sha256']
        recipe = config['recipe']
        assert (recipe['epochs'], recipe['batch'], recipe['lr'], recipe['loss']) == (200, 32, 0.003, 'smooth_l1')
        assert recipe['selection'] == 'first_strict_minimum_pooled_validation_RMSE'
    assert seen == set(arms.NEW_ARMS)
    singles = [arms.SPEC[arm]['penalty'] for arm in arms.TERM_ARMS.values()]
    assert all(len(terms) == 1 for terms in singles) and sorted(t for terms in singles for t in terms) == sorted(arms.TERMS)
    assert arms.SPEC['term-night-gated'] == dict(backbone='gwn', support='both', penalty=('night',), gate=True)
    assert arms.SPEC['pen10-gated'] == dict(backbone='gwn', support='both', penalty=arms.TERMS, gate=True)
    assert arms.SPEC['simvp-pen10'] == dict(backbone='simvp', support=None, penalty=arms.TERMS, gate=False)
    assert {arms.SPEC[a]['support'] for a in ('gwn-fixed', 'gwn-adapt', 'gwn-nograph')} == {'fixed', 'adaptive', 'none'}


def test_conditioning_configurations():
    from tecfc.models import conditioning_arms as arms
    seen = set()
    for paper in papers('conditioning'):
        config = paper['runtime_config']
        seen.add(config['arm'])
        assert paper['slug'] == config['arm'] and config['rho'] == 0. and paper['train_entry'] == 'tecfc.training.train_conditioning'
        assert (config['recipe']['epochs'], config['recipe']['batch'], config['recipe']['lr']) == (200, 32, 0.003)
    assert seen == set(arms.NEW_ARMS)


def test_both_supports_use_the_reference_constructor(architecture):
    from tecfc.objectives import penalty_arms as arms
    model = arms.build_gwn(architecture, CPU, 'term-rot')
    assert type(model) is architecture.GlobalGWN


@pytest.mark.parametrize('arm, support', [('gwn-fixed', 'fixed'), ('gwn-adapt', 'adaptive'), ('gwn-nograph', 'none')])
def test_graph_support_builder(architecture, arm, support):
    from tecfc.objectives import penalty_arms as arms
    torch.manual_seed(0)
    model = arms.build_gwn(architecture, CPU, arm)
    record, gwn, names = model.support_record, model.gwn, set(model.state_dict())
    assert record['support'] == support
    if support == 'fixed':
        assert record['fixed_supports'] == 2 and not record['adaptive'] and gwn.supports_len == 2
        assert 'gwn.nodevec1' not in names and len(gwn.gconv) == 8
    elif support == 'adaptive':
        assert record['fixed_supports'] == 0 and record['adaptive'] and gwn.supports_len == 1
        assert 'gwn.nodevec1' in names and len(gwn.gconv) == 8
    else:
        assert not record['graph_convolution'] and len(gwn.gconv) == 0 and 'gwn.nodevec1' not in names
    model.eval()
    with torch.no_grad():
        check_output(model(torch.rand(1, 12, 71, 72), restore=True))


@pytest.mark.parametrize('fusion', ['residual_gate', 'film', 'causal_crossattn'])
def test_conditioning_fusion(architecture, fusion):
    from tecfc.models import conditioning_arms as arms
    torch.manual_seed(0)
    model = arms.FusionGWN(architecture, CPU, fusion, '0' * 64, '1' * 64).eval()
    if fusion == 'causal_crossattn':
        assert model.gwn.history_gate is None and isinstance(model.gwn.start_conv, arms.StartWithCrossAttention)
    origin = [int(np.datetime64('2013-06-01', 's').astype(np.int64))]
    with torch.no_grad():
        check_output(model(torch.rand(1, 12, 71, 72, 6), origin, restore=True))


def test_graph_transformer_forward():
    from tecfc.models import conditioning_arms as arms
    torch.manual_seed(0)
    model = arms.GraphTransformer().eval()
    assert sum(p.numel() for p in model.parameters()) == 12534385
    history = torch.rand(1, 12, 71, 72, 6)
    history[..., 1:5] = torch.rand(1, 12, 1, 1, 4)  # AE, Dst, Kp and F10.7 are global: equal at every grid point
    with torch.no_grad():
        check_output(model(history, restore=True))


def test_simvp_backbone_loads_the_reference_sources():
    from tecfc.objectives import penalty_arms as arms
    names = ('gim_data', 'periodic', 'attentions', 'simvp', 'cg_model', 'global_model', 'train_global')
    before = {name: sys.modules.get(name) for name in names}
    with arms.simvp_modules(REPO / 'tecfc/models/simvp_reference') as modules:
        modules['train_global'].seed_all(42)
        model = modules['global_model'].GlobalAmplitudeConditionedSimVP().eval()
        x = torch.rand(1, 12, 71, 72)
        with torch.no_grad():
            check_output(model(x, x, x, x, arm='C', restore=True))
    assert {name: sys.modules.get(name) for name in names} == before  # the bare module names are restored


def test_backbone_prediction_loop():
    from tecfc.training import global_train as base
    from tecfc.training.predict import backbone_predict_loop

    class Backbone:
        device = CPU

        def forward(self, model, batch):
            p = model(batch['history'])
            return torch.cat((p, p[..., :1]), dim=-1)

    class Windows(torch.utils.data.Dataset):
        days, scale = ['2023-01-01', '2023-01-02'], 147.1

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return dict(history=torch.full((12, 71, 72), float(index)), index=index)

    out = backbone_predict_loop(torch.nn.Identity(), Windows(), Backbone(), base.amp)
    assert out.shape == (2, 12, 71, 73) and np.array_equal(out[1], np.full((12, 71, 73), np.float32(147.1)))
