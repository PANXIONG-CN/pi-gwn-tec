"""Precompute the frozen NOAA solar geometry at native global historical times."""
import sys,json
from pathlib import Path
import numpy as np
from gim_data import HERE,sha
sys.path.insert(0,str(HERE/'repo'))
from lib.physics.solar import canonical_cos_sza_float64,COS_SZA_ALGORITHM_ID

p=HERE/'prepared/geometry.npy';assert not p.exists()
latitude=np.arange(87.5,-90,-2.5);longitude=np.arange(-180,180,5)
la,lo=np.meshgrid(latitude,longitude,indexing='ij');a=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(366,12,71,72))
start=int(np.datetime64('2020-01-01','s').astype(np.int64))
for day in range(366):
 times=start+day*86400+np.arange(12)*7200
 a[day]=canonical_cos_sza_float64(times,lo.ravel(),la.ravel()).reshape(12,71,72)
a.flush();del a
(HERE/'prepared/GEOMETRY.json').write_text(json.dumps(dict(status='prepared',shape=[366,12,71,72],algorithm=COS_SZA_ALGORITHM_ID,year_index='zero-based day of year, same canonical 365 denominator; 2020 used only for synthetic timestamp indexing',sha256=sha(p),source_sha256=sha(__file__)),indent=2)+'\n')
print('global geometry prepared',flush=True)
