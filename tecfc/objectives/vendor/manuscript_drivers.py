"""Raw target-window reader, only constructed in PI training."""
import numpy as np
from lib.physics.drivers import DriversStore
from revisions.gim_pizscore_fixed_recipe_v1.data import SOURCE_SHA,T0

class TrainingDrivers:
    def __init__(self,path,device):
        self.store=DriversStore(str(path),device=device,strict=True,expected_sha256=SOURCE_SHA)
    def physics(self, starts):
        starts=np.asarray(starts)
        if starts.dtype.kind not in 'iu' or starts.ndim != 1:
            raise ValueError('starts must be integer history-window starts')
        index=starts[:,None]+12+np.arange(12)[None,:]
        if index.size == 0 or index.min()<0 or index.max()>=len(self.store.time):
            raise ValueError('target driver window outside source')
        expected=T0+index*7200
        if not np.array_equal(self.store.time[index],expected):
            raise ValueError('target driver UTC alignment/cadence mismatch')
        # origin is FIRST target, not the last historical observation.
        return self.store.slice_batched(starts+12,12,5183,
            fields=('F107','AE','Dst'),require_cos_sza=True,include_grid_shape=False)
