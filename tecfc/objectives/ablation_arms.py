"""Registered interventions over the frozen exp0013 and exp0015 components."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import torch

TECFC = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
from tecfc.training import global_train as base
from tecfc.objectives import penalty as pi
from tecfc.features import member_inputs as f15

sys.path.insert(0, str(TECFC / 'objectives/old_global_pi'))
from revisions.gim_pizscore_fixed_recipe_v1.objective import clip_gradients

SEEDS = (42, 1024, 2025, 3407, 5000)
ARMS = ('P-0.1', 'P-1.0', 'D-noSolar', 'D-noGeomag', 'D-noIndex', 'G-nowrap', 'P-adapt')
RHO = {'G': 0., 'PI': .01, 'P-0.1': .1, 'P-1.0': 1., 'P-adapt': 0.,
       'G-DE': 0., 'D-noSolar': 0., 'D-noGeomag': 0., 'D-noIndex': 0., 'G-nowrap': 0.}
CHANNELS = {'D-noSolar': [0, 1, 2, 3, 5], 'D-noGeomag': [0, 4, 5], 'D-noIndex': [0, 5]}
learned = pi.extract(TECFC / 'objectives/old_global_pi/revisions/gim_pizscore_parameter_merge_v1/objective.py',
                     ('assign_gradients',), {'torch': torch})['assign_gradients']


def write(path, value):
    base.write_json(Path(path), value)


class Windows:
    def __new__(cls, role, config, gim, evaluation=False):
        arm = config['arm']
        if arm in CHANNELS or arm == 'G-DE':
            original = dict(config, arm='G-DE')
            dataset = f15.Windows(role, original, gim, evaluation=evaluation)
            return SelectedWindows(dataset, arm)
        if evaluation:
            return f15.Windows(role, dict(config, arm='G'), gim, evaluation=True)
        return base.DevelopmentWindows(role, config, gim)


class SelectedWindows(torch.utils.data.Dataset):
    def __init__(self, dataset, arm):
        self.dataset = dataset
        self.arm = arm
        self.scale = dataset.scale
        self.days = dataset.days
        self.columns = CHANNELS.get(arm, list(range(6)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['history'] = sample['history'][..., self.columns]
        return sample


def build_model(architecture, device, arm):
    if arm in CHANNELS:
        # Existing input-width constructor preserves all graph and RNG operations.
        temporary = 'exp0026_' + arm
        f15.CHANNELS[temporary] = len(CHANNELS[arm])
        try:
            return f15.build_model(architecture, device, temporary)
        finally:
            del f15.CHANNELS[temporary]
    model = f15.build_model(architecture, device, 'G-DE' if arm == 'G-DE' else 'G')
    if arm == 'G-nowrap':
        model.exp0026_graph = remove_seam(model)
    return model


def remove_seam(model):
    supports = model.gwn.supports
    if not isinstance(supports, list) or not supports:
        raise ValueError('Fixed graph support unavailable')
    rows = []
    longitude = torch.arange(5112, device=supports[0].device) % 72
    seam = ((longitude[:, None] == 0) & (longitude[None, :] == 71)) | ((longitude[:, None] == 71) & (longitude[None, :] == 0))
    for index, graph in enumerate(supports):
        if tuple(graph.shape) != (5112, 5112):
            raise ValueError('Unexpected fixed graph shape')
        nonzero = graph != 0
        changed = graph.clone()
        changed[seam] = 0
        rows.append(dict(index=index, original_directed=int(nonzero.sum()),
                         nowrap_directed=int(torch.count_nonzero(changed)),
                         original_undirected=int(torch.triu(nonzero | nonzero.T, diagonal=1).sum()),
                         nowrap_undirected=int(torch.triu((changed != 0) | (changed.T != 0), diagonal=1).sum()),
                         removed=int((nonzero & seam).sum())))
        supports[index] = changed
    if not any(row['removed'] for row in rows):
        raise NoSeamEdges(rows)
    return rows


class NoSeamEdges(ValueError):
    def __init__(self, rows):
        super().__init__('Original fixed graph has no longitude seam edges')
        self.rows = rows


class Auxiliary(pi.Auxiliary):
    def __init__(self, arm, config, device):
        super().__init__('PI', config, device)
        self.learned = arm == 'P-adapt'
        if self.learned:
            for parameter in self.physics.parameters():
                parameter.requires_grad_(True)

    def __call__(self, prediction, origins):
        if not self.learned:
            return super().__call__(prediction, origins)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            pred = prediction.float()
            drivers = self.drivers.at(origins, 'PI')
            loss, terms = self.physics(pred.reshape(len(pred), 12, 5183, 1), None, drivers=drivers)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite learned physics objective')
        return loss, {key: float(value.detach()) for key, value in terms.items()}, self.physics.residual_counts.copy()


def scientific_config(config):
    # All paths are logical runtime mappings. Hashes and scientific fields remain bound.
    value = base.scientific_config(config)
    value.pop('reference_epochs', None)
    for key in ('driver', 'scaler', 'input2026', 'origins2026'):
        if key not in value: continue
        value[key].pop('path', None)
    for row in value['history_only_files'].values():
        row.pop('path', None)
    for key in ('driver', 'adjacency', 'empirical_root'):
        value['physics'].pop(key, None)
    for row in value.get('check_references', {}).values():
        row.pop('path', None)
    value.pop('candidates', None)
    value.pop('candidate_freeze_sha256', None)
    return value

