"""Shared native data contract, fixed causal features, and train-only statistics."""
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import hashlib, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'repo'))
from lib.physics.solar import canonical_cos_sza_float64, COS_SZA_ALGORITHM_ID
from lib.physics.driver_scaler import DRIVER_CHANNEL_ORDER, SCALER_SCHEMA_VERSION, SCALER_STATUS

SEEDS=[42,1024,2025,3407,5000]
ARMS=['dense_do','dense_pi','dense_de','dense_pi_drivers','coarse_de']
YEARS=list(range(2014,2022))+[2023,2024,2025]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def days(years):return [str(d) for y in years for d in np.arange(np.datetime64(f'{y}-01-01'),np.datetime64(f'{y+1}-01-01'))]
def required_days():return ['2013-12-31']+days(range(2014,2022))+['2022-12-31']+days([2023,2024,2025])
def unix(d):return int(np.datetime64(d,'s').astype(np.int64))
def previous(d):return str(np.datetime64(d)-np.timedelta64(1,'D'))
def role(d):return 'history_only' if int(d[:4]) in [2013,2022] else 'train' if int(d[:4])<=2020 else 'validation' if int(d[:4])==2021 else 'evaluation'

def fill(a,v,fallback,step_seconds):
    """Six-hour observed-value hold; imputed cells never reset observation age."""
    out=np.asarray(a,np.float32).copy();last=np.full(a.shape[1],-100000,np.int64);value=np.zeros(a.shape[1],np.float32)
    for t in range(len(a)):
        good=v[t];last[good]=t;value[good]=a[t,good]
        recent=~good&((t-last)*step_seconds<=21600)
        out[t,recent]=value[recent];other=~good&~recent;out[t,other]=fallback[t,other]
    assert np.isfinite(out).all();return out

class NativeDays:
    def __init__(self,root):
        self.root=Path(root);rows=[json.loads(s) for s in (self.root/'data/DAYS.jsonl').read_text().splitlines() if s.strip()]
        self.rows={r['date']:r for r in rows}
        assert len(rows)==len(self.rows), 'duplicate day index'
        required=set(required_days())
        assert all(r['status']=='complete' and r['date'] in required for r in rows)
        first=self.read(sorted(self.rows)[0]);self.lat,self.lon=first['lat'],first['lon'];self.n=len(self.lat)*len(self.lon)
        assert np.all(np.diff(self.lat)<0) and np.all(np.diff(self.lon)>0)
    @lru_cache(maxsize=64)
    def read(self,d):
        r=self.rows[d];p=self.root/r['daily_path'];assert sha(p)==r['daily_sha256'],f'changed daily {d}'
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        step=int(r['cadence_seconds']);T=86400//step
        assert step in [300,900] and a['tec'].shape==(T,len(a['lat']),len(a['lon']))
        assert a['valid'].shape==a['tec'].shape and a['valid'].dtype==bool
        assert a['present'].shape==(T,) and a['present'].dtype==bool
        np.testing.assert_array_equal(a['time_utc'],unix(d)+np.arange(T)*step)
        assert not np.any(a['valid']&~(np.isfinite(a['tec'])&a['present'][:,None,None]))
        if hasattr(self,'lat'):
            np.testing.assert_array_equal(self.lat,a['lat']);np.testing.assert_array_equal(self.lon,a['lon'])
        a['cadence']=step;return a
    @lru_cache(maxsize=None)
    def q15(self,d):
        a=self.read(d);stride=900//a['cadence']
        return {k:np.asarray(a[k][::stride]).reshape(96,-1).copy() if k in ['tec','valid'] else a[k][::stride].copy() for k in ['tec','valid','present','time_utc']}

def candidates(archive):
    result=[]
    for d in days(YEARS):
        h=previous(d);why=[]
        for key in [h,d]:
            if key not in archive.rows:why.append('unfinished_or_source_absent:'+key)
        if not why:
            x,y=archive.q15(h),archive.q15(d)
            if not x['present'][:89].all():why.append('history_absent_frame')
            if not y['present'][::8].all():why.append('target_absent_frame')
            if not y['valid'][::8].any():why.append('target_empty')
        result.append(dict(date=d,origin=unix(d),history=h,role=role(d),included=not why,reasons=why))
    return result

def prepare_drivers(source,dest,lat,lon,product,fit_days):
    """Exactly the legacy coarse-timestamp training scaler with regional SZA."""
    assert all(role(d)=='train' for d in fit_days)
    with np.load(source,allow_pickle=False) as z:a={k:z[k] for k in ['time','AE','Dst','Kp','F107']}
    times=a['time'].astype(np.int64)
    assert np.all(np.diff(times)==7200)
    day_strings=times.astype('datetime64[s]').astype('datetime64[D]').astype(str)
    keep=np.isin(day_strings,required_days());a={k:v[keep] for k,v in a.items()};times=a['time'].astype(np.int64)
    train_times=np.array([unix(d)+h*7200 for d in sorted(set(fit_days)) for h in range(12)],np.int64)
    ix=np.searchsorted(times,train_times);assert np.array_equal(times[ix],train_times)
    la,lo=np.meshgrid(lat,lon,indexing='ij');laf,lof=la.ravel(),lo.ravel();N=laf.size
    np.savez_compressed(dest/'drivers.npz',**a,grid_shape=np.array(la.shape,np.int16),lat_deg=laf,lon_deg=lof)
    channels={}
    for k in DRIVER_CHANNEL_ORDER:
        digest=hashlib.sha256();total=total2=0.;count=0;minimum=np.inf;maximum=-np.inf
        for start in range(0,len(ix),128):
            idx=ix[start:start+128]
            values=canonical_cos_sza_float64(times[idx],lof.astype(np.float64),laf.astype(np.float64)) if k=='cosSZA' else a[k][idx].astype(np.float64)
            assert np.isfinite(values).all();digest.update(np.asarray(values,dtype='<f8').tobytes())
            total+=float(values.sum(dtype=np.float64));total2+=float(np.square(values).sum(dtype=np.float64));count+=values.size
            minimum=min(minimum,float(values.min()));maximum=max(maximum,float(values.max()))
        mean=total/count;std=float(np.sqrt(total2/count-mean*mean));assert std>0
        channels[k]=dict(mean=mean,standard_deviation=std,count=count,minimum=minimum,maximum=maximum,input_values_sha256=digest.hexdigest())
    meta=dict(schema_version=SCALER_SCHEMA_VERSION,status=SCALER_STATUS,dataset=product+'_japan',scaler_id=product+'_japan_driver_zscore_train_v1',source_driver_relative_path='drivers.npz',source_driver_sha256=sha(dest/'drivers.npz'),original_source_driver_sha256=sha(source),channel_order=list(DRIVER_CHANNEL_ORDER),fit_roles=['train'],fit_train_years=sorted({int(d[:4]) for d in fit_days}),imputation_policy='none',role_access=dict(included_roles=['train'],validation_values_included=0,evaluation_values_included=0,non_train_values_included=0),algorithm=dict(timestamp_weighting='each_unique_train_timestamp_once',global_channel_weighting='one_value_per_timestamp',cosSZA_weighting='one_value_per_timestamp_per_node',zscore='population_mean_and_standard_deviation',ddof=0,cosSZA_algorithm_id=COS_SZA_ALGORITHM_ID,cosSZA_chunk_size_timestamps=128),timestamp_count=len(train_times),unique_timestamp_count=len(train_times),num_nodes=N,grid_shape=list(la.shape),channels=channels,train_timestamp_sha256=hashlib.sha256(train_times.astype('<i8').tobytes()).hexdigest(),first_train_timestamp_utc=str(train_times[0].astype('datetime64[s]'))+'Z',last_train_timestamp_utc=str(train_times[-1].astype('datetime64[s]'))+'Z')
    write(dest/'driver_scaler.json',meta)

def prepare(root,product,drivers_source,partial=False):
    from lib.datasets.gimtec_adj import build_grid8_adj
    ar=NativeDays(root);dest=Path(root)/('fixture' if partial else 'prepared');dest.mkdir(exist_ok=True)
    if not partial:assert set(ar.rows)==set(required_days()),'Full verified native daily acquisition required before preparation'
    tr=[d for d in sorted(ar.rows) if role(d)=='train'];sums=np.zeros((12,ar.n));counts=np.zeros((12,ar.n),np.int64);maximum=-np.inf
    for d in tr:
        a=ar.q15(d);x,v=a['tec'][::8],a['valid'][::8]
        sums+=np.where(v,x.astype(np.float64),0);counts+=v
        if v.any():maximum=max(maximum,float(x[v].max()))
    assert maximum>0 and np.isfinite(maximum) and counts.sum()>0
    slot_sum,slot_count=sums.sum(1),counts.sum(1);global_mean=sums.sum()/counts.sum()
    slot_mean=np.divide(slot_sum,slot_count,out=np.full(12,global_mean),where=slot_count>0)
    fallback=np.divide(sums,counts,out=np.broadcast_to(slot_mean[:,None],sums.shape).copy(),where=counts>0)
    np.savez(dest/'tec_statistics.npz',scale=np.float64(maximum),fallback=fallback.astype(np.float32),counts=counts,sums=sums,fit_days=np.array(tr),slot_mean=slot_mean)
    prepare_drivers(drivers_source,dest,ar.lat,ar.lon,product,tr)
    np.save(dest/'adjacency.npy',build_grid8_adj(len(ar.lat),len(ar.lon)))
    np.savez(dest/'coordinates.npz',lat=ar.lat,lon=ar.lon,grid_shape=np.array([len(ar.lat),len(ar.lon)]),lead_hours=np.arange(2,25,2))
    rows=candidates(ar);write(dest/'CANDIDATES.json',rows)
    identity=dict(product=product,dataset=product+'_japan',normalizer_id=product+'_japan_trainmax_v1',num_nodes=ar.n,grid_shape=[len(ar.lat),len(ar.lon)],scientific_ready=not partial,fit_train_days=len(tr),files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='IDENTITY.json'},daily_manifest_sha256=sha(Path(root)/'data/DAYS.jsonl'),code_sha256=sha(__file__))
    write(dest/'IDENTITY.json',identity);return identity

class Archive:
    def __init__(self,root,product,fixture=False):
        self.root=Path(root);self.product=product;self.native=NativeDays(root);self.n=self.native.n
        self.prepared=self.root/('fixture' if fixture else 'prepared')
        self.identity=json.loads((self.prepared/'IDENTITY.json').read_text())
        assert self.identity['product']==product
        for k,v in self.identity['files'].items():assert sha(self.prepared/k)==v, 'changed preparation '+k
        with np.load(self.prepared/'tec_statistics.npz') as z:self.scale=float(z['scale']);self.fallback=z['fallback'].astype(np.float32)
        with np.load(self.prepared/'drivers.npz') as z:self.drivers={k:z[k] for k in ['time','AE','Dst','Kp','F107','lat_deg','lon_deg']}
        meta=json.loads((self.prepared/'driver_scaler.json').read_text());self.means=np.array([meta['channels'][k]['mean'] for k in meta['channel_order']],np.float32);self.stds=np.array([meta['channels'][k]['standard_deviation'] for k in meta['channel_order']],np.float32)
    @lru_cache(maxsize=None)
    def history(self,d,coarse=False):
        a=self.native.q15(previous(d));x,v=a['tec'],a['valid']
        return fill(x[::8],v[::8],self.fallback,7200) if coarse else fill(x[:89],v[:89],self.fallback[np.arange(89)//8],900)
    @lru_cache(maxsize=None)
    def history_drivers(self,d,coarse=False):
        times=unix(d)-86400+np.arange(12 if coarse else 89)*(7200 if coarse else 900)
        ix=np.searchsorted(self.drivers['time'],times,side='right')-1
        assert np.all(ix>=0) and np.all(self.drivers['time'][ix]<=times) and np.all(times-self.drivers['time'][ix]<7200) and times[-1]==unix(d)-7200
        a=np.empty((len(times),self.n,5),np.float32)
        for i,k in enumerate(['AE','Dst','Kp','F107']):a[:,:,i]=self.drivers[k][ix,None]
        a[:,:,4]=canonical_cos_sza_float64(times,self.drivers['lon_deg'].astype(np.float64),self.drivers['lat_deg'].astype(np.float64))
        a=(a-self.means)/self.stds;assert np.isfinite(a).all();return a
    def target(self,d):
        a=self.native.q15(d);return a['tec'][::8],a['valid'][::8]
    def persistence(self,d):
        a=self.native.q15(previous(d));return self.history(d)[::8],a['tec'][::8],a['valid'][::8]
