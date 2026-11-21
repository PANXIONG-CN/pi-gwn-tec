"""exp0018 compositions of the exact frozen exp0015/exp0016 components."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch.utils.data import Dataset

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
from tecfc.training import global_train as base

# Import unchanged historical sources under private names, preserving aliases.
previous = sys.modules.get('geometry')
base.load_module('geometry', HERE/'geometry.py')
f15 = base.load_module('_exp0018_f15', HERE/'member_inputs.py')
if previous is None: sys.modules.pop('geometry', None)
else: sys.modules['geometry'] = previous
c16 = base.load_module('_exp0018_c16', HERE/'clim_inputs.py')

ARMS = ('G-plus-M', 'G-plus-S', 'C-plus')
CONTROLS = ('G-long', 'C-in')
GEOMETRY = dict(G='exact exp0015.features.geometry/history_times and signed_cos_sza',
                C=deepcopy(c16.GEOMETRY))


def g_model(architecture, device, channels):
    original = architecture.GWNET
    def constructor(args, device, dim_in, dim_out):
        return original(args, device, dim_in=channels, dim_out=dim_out)
    try:
        architecture.GWNET = constructor
        model = architecture.GlobalGWN(device)
    finally:
        architecture.GWNET = original
    model.input_channels = channels
    model.forward = types.MethodType(f15.member_forward, model)
    return model


@contextmanager
def modules(config, device):
    with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, architecture):
        if config['arm'] in ('G-long', 'G-plus-M', 'G-plus-S'):
            # G-long uses the original constructor verbatim for gate A.
            yield gim, (lambda: f15.build_model(architecture, device, 'G-long')) if config['arm']=='G-long' else (lambda: g_model(architecture, device, 9))
        else:
            with c16.cg_modules(config['cg_reference_root']) as original:
                if config['arm']=='C-in':
                    yield gim, original
                else:
                    module=sys.modules['_exp0016_cg']
                    shared=sys.modules['cg_model']; original_nn=shared.nn
                    # Change only the first convolution's input width at its
                    # original initialization position, without extra RNG draws.
                    proxy=types.SimpleNamespace(**vars(original_nn))
                    def convolution(in_channels, *args, **kwargs):
                        return original_nn.Conv2d(6 if in_channels==3 else in_channels, *args, **kwargs)
                    proxy.Conv2d=convolution
                    class CGPlus(original):
                        def __init__(self):
                            try:
                                shared.nn=proxy
                                super().__init__()
                            finally: shared.nn=original_nn
                        def forward(self,x,restore=False):
                            if x.ndim!=5 or x.shape[1:]!=(12,6,71,72):raise ValueError('C-plus input shape')
                            b,t,c,h,w=x.shape
                            enc=self.attn(self.spatial_enc(x.reshape(b*t,c,h,w)))
                            _,hidden,hh,ww=enc.shape
                            compressed=self.time_adapter(enc.reshape(b,t,hidden,hh,ww).permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()
                            latent=self.temporal(compressed.reshape(b,self.compressed_frames*hidden,hh,ww))
                            pred=self.time_proj(latent).reshape(b*self.output_frames,hidden,hh,ww)
                            pred=module.restore_height(self.spatial_dec(pred)).reshape(b,self.output_frames,71,72)
                            return module.restore_longitude(pred) if restore else pred
                    yield gim, CGPlus


class Windows(Dataset):
    def __init__(self, role, config, gim, *, inference=False):
        self.arm=config['arm']; self.scale=147.1; self.role=role
        if self.arm=='C-in':
            self.parent=c16.Windows(role, dict(config,arm='C-in'), gim, inference=inference)
        else:
            self.parent=f15.Windows(role, dict(config,arm='G-long'), gim, evaluation=inference)
        self.days=self.parent.days
        self.gim=self.parent.gim
        self.dropped=getattr(self.parent,'dropped',[])
        if self.arm=='C-plus':
            self.cin=c16.Windows(role,dict(config,arm='C-in'),gim,inference=inference)
            self.cin_indices={str(d):i for i,d in enumerate(self.cin.days)}
    def __len__(self):return len(self.days)
    def __getitem__(self,index):
        result=self.parent[index]
        if self.arm in ('G-plus-M','G-plus-S'):
            result['history']=torch.from_numpy(np.concatenate((result['history'].numpy(), f15.geometry(f15.history_times(self.days[index]))),axis=-1))
        elif self.arm=='C-plus':
            cin=self.cin[self.cin_indices[str(self.days[index])]]
            np.testing.assert_array_equal(cin['history'][:,0].numpy(),result['history'][...,0].numpy())
            result['history']=torch.cat((cin['history'],result['history'][...,1:].permute(0,3,1,2)),dim=1)
        return result


def scientific_config(config):
    result=base.scientific_config(config)
    result.pop('cg_reference_root')
    for key in ('history_only_files','reference_epochs'):
        for row in result[key].values():row.pop('path')
    return result


def validate(config, *, inference=False):
    if config['schema']!='exp0018_fixed200_v1' or config['arm'] not in ARMS+CONTROLS:raise ValueError('Unregistered configuration')
    if config['seed'] not in base.SEEDS or config['geometry']!=GEOMETRY:raise ValueError('Changed seed/geometry')
    if config['loss']!=('mae' if config['arm']=='G-plus-M' else 'smooth_l1'):raise ValueError('Changed supervised loss')
    expected=set(range(2009,2026)) if inference else set(base.ALLOWED_YEARS)
    if set(map(int,config['data_files']))!=expected:raise ValueError('Changed role access')
    if set(config['history_only_files'])!={'2015'}:raise ValueError('Changed historical whitelist')
    days=[str(x) for x in np.arange(np.datetime64('2015-12-06'),np.datetime64('2016-01-01'))]
    if config['history_only_files']['2015']['allowed_dates']!=days:raise ValueError('History whitelist changed')
    for key,root in (('reference_sha256','reference_root'),('cg_reference_sha256','cg_reference_root'),('asset_sha256','asset_root')):
        for name,h in config[key].items():
            if base.sha(Path(config[root])/name)!=h:raise ValueError('Changed asset '+name)
    for group in ('data_files','reference_epochs','history_only_files'):
        for row in config[group].values():
            if base.sha(row['path'])!=row['sha256']:raise ValueError('Changed explicit input')
