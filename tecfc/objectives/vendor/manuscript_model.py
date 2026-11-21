"""Independent GIM residual-gate predictor using the unchanged GWN backbone."""
from types import SimpleNamespace
import hashlib
from pathlib import Path
import numpy as np
import torch
from model.GWN.GWN import GWNET
from lib.physics.driver_scaler import DRIVER_CHANNEL_ORDER
from revisions.gim_pizscore_fixed_recipe_v1.data import SOURCE_SHA, SCALER_SHA

ADJ_PATH = 'data/adj/GIMtec/GIMtec_adj_grid8_20250913_185704.npy'
ADJ_SHA = 'c5e278353e30339f66d78ca08e4c29305a435d1f40db2f9ee3d26ecd6883cd00'

def build_model(root, device, *, test_nodes=None, mode='zscore_concat'):
    if mode != 'zscore_concat':
        raise ValueError('unexpected model mode')
    if test_nodes is None:
        path = Path(root) / ADJ_PATH
        if hashlib.sha256(path.read_bytes()).hexdigest() != ADJ_SHA:
            raise ValueError('adjacency content mismatch')
        adjacency = np.load(path, allow_pickle=False)
        if adjacency.shape != (5183, 5183):
            raise ValueError('adjacency shape mismatch')
        nodes, graph = 5183, True
    else:
        nodes, adjacency, graph = test_nodes, np.eye(test_nodes, dtype=np.float32), True
    args = SimpleNamespace(adj_mx=adjacency, num_nodes=nodes, dropout=.3,
        blocks=4, layers=2, gcn_bool=graph, addaptadj=graph,
        adjtype='doubletransition', randomadj=not graph, aptonly=False,
        kernel_size=2, nhid=32, residual_channels=32, dilation_channels=32,
        input_window=12, output_window=12, driver_fusion=mode,
        driver_scaling='train_zscore', predictor_driver_channels=list(DRIVER_CHANNEL_ORDER),
        driver_source_sha256=SOURCE_SHA, driver_scaler_sha256=SCALER_SHA)
    return GWNET(args, device, dim_in=6 if mode=='zscore_concat' else 1, dim_out=1).to(device)


PHYSICS_TERMS = ('nonneg', 'rot', 'roti', 'diffusion', 'chemistry', 'night')
PHYSICS_CONFIG = 'protocols/cac_r1/candidates/resolved_configs/gim_pi_driver_c07.json'


def build_ppinn(root, device, *, test_nodes=None):
    """No RNG calls; use raw drivers only in the training objective."""
    from lib.physics.ppinn import PPinnLoss
    from lib.normalization import MinMax01Scaler
    if test_nodes is None:
        path = Path(root) / ADJ_PATH
        if hashlib.sha256(path.read_bytes()).hexdigest() != ADJ_SHA:
            raise ValueError('physics adjacency content mismatch')
        adjacency = np.load(path, allow_pickle=False)
        if adjacency.shape != (5183, 5183) or not np.isfinite(adjacency).all():
            raise ValueError('physics adjacency shape/finiteness')
        digest = ADJ_SHA
    else:
        adjacency = np.ones((test_nodes, test_nodes), dtype=np.float32)
        digest = hashlib.sha256(adjacency.tobytes()).hexdigest()
    module = PPinnLoss(MinMax01Scaler(0, 147.1), torch.as_tensor(adjacency, device=device),
        dataset='gim', normalizer_id='gim_tec01_147p1_v1', model_key='pi_driver',
        use_nonneg=True, use_rot=True, use_roti=True, use_diffusion=True,
        use_chemistry=True, use_night=True, use_dynamic_rot_cap=True, use_advection=False,
        strict_paper_run=True, first_transition_policy='forecast_internal_only',
        dt_minutes=120, tec_ref=50., t_ref_sec=7200., rot_cap_tecu_per_min=.5,
        rot_cap_k=.01, roti_cap_scale=.7, roti_w=3, kappa_nd=.05,
        alpha_nd=.01, beta_nd=.01, ae_factor=.001,
        adjacency_sha256=digest, expected_adjacency_sha256=digest).to(device)
    if tuple(module.uncert.log_vars) != PHYSICS_TERMS or sum(p.numel() for p in module.parameters()) != 6:
        raise ValueError('six-parameter contract')
    return module
