"""One-shot CPU scoring of frozen forecasts; no training, inference, or downloads."""
import json, subprocess, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/'config.json').read_text())
P = {k: Path(v) for k, v in CFG['paths'].items()}
SEEDS = CFG['seeds']; ZERO = ['G', 'C', 'GG', 'F2', 'G5', 'F10']
GROUPS = CFG['rules']['groups']; OFF = np.arange(12)*7200
sys.path.insert(0, str(P['cn_support']))
from prepare import sufficient
from transfer import mapping as china_mapping, remap

def save(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

@lru_cache(None)
def arr(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)

@lru_cache(None)
def lookup(values):
    assert len(set(values)) == len(values)
    return {int(t): i for i, t in enumerate(values)}

def idx(origins, t):
    i = lookup(tuple(origins))[int(t)]
    np.testing.assert_array_equal(origins[i]+OFF, int(t)+OFF)
    assert t % 86400 == 0
    return i

CNINDEX = dict(np.load(P['cn_support']/'support/index.npz'))
CNROWS = [json.loads(x) for x in (P['cn_native']/'data/WINDOWS.jsonl').read_text().splitlines()]
JPROWS = json.loads((P['jp_windows']/'rows.json').read_text())
JPORIG = np.load(P['jp_windows']/'index.npz')['origins']
CNORIG = arr(P['cn_coarse']/'shared/evaluation/origins.npy')
np.testing.assert_array_equal(JPORIG, [r['origin'] for r in JPROWS])
assert arr(P['jp_windows']/'target.npy').shape == arr(P['jp_windows']/'valid.npy').shape == (4011,12,288)
assert arr(P['cn_coarse']/'shared/evaluation/truth.npy').shape == arr(P['cn_coarse']/'shared/evaluation/valid.npy').shape == (1094,12,2800,1)

def origins(region, split):
    rows = CNROWS if region == 'CN' else JPROWS
    role = ('replay' if split == 'evaluation' else 'validation') if region == 'CN' else split
    ts = np.array([r['origin'] for r in rows if r['included'] and r['role'] == role], np.int64)
    assert len(ts) == ({'CN':1077, 'JP':1091}[region] if split == 'evaluation' else {'CN':364, 'JP':365}[region])
    years = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    assert set(years) == ({2023,2024,2025} if split == 'evaluation' else {2021})
    if region == 'CN' and split == 'evaluation': np.testing.assert_array_equal(ts, CNINDEX['origins'])
    return ts

@lru_cache(4)
def native_cn(t):
    d = np.datetime64(int(t), 's').astype('datetime64[D]'); year = int(str(d)[:4])
    assert year == 2021
    day = int((d-np.datetime64(f'{year}-01-01')).astype(int))+1
    z = dict(np.load(P['cn_native']/f'data/china_15min/{year}/jpld{day:03d}0.{year%100:02d}i.npz'))
    np.testing.assert_array_equal(z['time_utc'], int(t)+np.arange(96)*900)
    return z['tec'].reshape(96,2800), z['valid'].reshape(96,2800)

def day(region, split, t):
    if region == 'JP':
        j = idx(JPORIG, t); r = JPROWS[j]
        assert int(np.datetime64(r['history'], 's').astype(int))+86400 == t
        return arr(P['jp_windows']/'target.npy')[j], arr(P['jp_windows']/'valid.npy')[j], arr(P['jp_windows']/'history.npy')[j,::8]
    if split == 'evaluation':
        i = idx(CNINDEX['origins'], t); j = idx(CNORIG, t)
        assert j == CNINDEX['coarse_index'][i] and CNINDEX['history_time'][i,-1] == t-7200
        np.testing.assert_array_equal(CNINDEX['target_time'][i], t+OFF)
        return arr(P['cn_coarse']/'shared/evaluation/truth.npy')[j,...,0], arr(P['cn_coarse']/'shared/evaluation/valid.npy')[j,...,0], arr(P['cn_support']/'support/persistence.npy')[i]
    y,v = native_cn(t); h,hv = native_cn(t-86400); filled = h[:89].copy()
    fallback = np.load(P['cn_native']/'shared/tec_statistics.npz')['fallback'].astype(np.float32)
    last = np.full(2800,-10000); values = np.zeros(2800,np.float32)
    for k in range(89):
        last[hv[k]]=k; values[hv[k]]=h[k,hv[k]]; recent=~hv[k]&((k-last)*900<=21600)
        filled[k,recent]=values[recent]; other=~hv[k]&~recent; filled[k,other]=fallback[k//8,other]
    return y[::8],v[::8],filled[::8]

def gim(t):
    d=np.datetime64(int(t),'s').astype('datetime64[D]'); y=int(str(d)[:4]); assert 2020<=y<=2025
    a=arr(P['gim_old' if y<2023 else 'gim_ext']/f'TEC_{y}.npy'); k=int((d-np.datetime64(f'{y}-01-01')).astype(int))
    assert a.shape==(int((np.datetime64(f'{y+1}-01-01')-np.datetime64(f'{y}-01-01')).astype(int)),12,71,73)
    z=np.asarray(a[k],np.float32)/np.float32(10)
    np.testing.assert_array_equal(z[...,0],z[...,-1])
    return z.reshape(12,5183).astype(np.float64)

def global_forecasts(t, split):
    out={}; split='extension_evaluation' if split=='evaluation' else 'original_evaluation'
    for arm,base in [('G','gwn'),('C','cg')]:
        values=[]
        for seed in SEEDS:
            d=P[base]/'runs'/f'{arm}_{seed}'; ts=arr(d/f'{split}_origins.npy'); a=arr(d/f'{split}_predictions.npy')
            assert a.shape==(len(ts),12,71,73) and a.dtype==np.float32
            values.append(a[idx(ts,t)].reshape(12,5183).astype(np.float64))
        out[arm]=np.stack(values)
    g,c=out['G'],out['C']; out.update(GG=(g+np.roll(g,-1,axis=0))/2,F2=(g+c)/2,G5=g.mean(0,keepdims=True),F10=np.concatenate([g,c]).mean(0,keepdims=True))
    return out

def mapping(region, nearest=False):
    if region=='CN': return china_mapping('nearest' if nearest else 'bilinear')
    z=np.load(P['jp_native']/'prepared/coordinates.npz'); la,lo=np.meshgrid(z['lat'],z['lon'],indexing='ij'); lat,lon=la.ravel(),lo.ravel()
    assert len(lat)==288 and np.all((lat>=30)&(lat<=46)&(lon>=128)&(lon<=146))
    if nearest: return (np.abs(np.arange(87.5,-90,-2.5)[:,None]-lat).argmin(0)*73+np.abs(np.arange(-180,181,5)[:,None]-lon).argmin(0))[None],np.ones((1,288))
    yy=(87.5-lat.astype(float))/2.5; xx=(lon.astype(float)+180)/5; iy=yy.astype(int); ix=xx.astype(int); dy=yy-iy; dx=xx-ix
    assert iy.min()>=0 and iy.max()+1<71 and ix.min()>=0 and ix.max()+1<73
    return np.stack([iy*73+ix,iy*73+ix+1,(iy+1)*73+ix,(iy+1)*73+ix+1]),np.stack([(1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx])

def regional(region, split, t):
    out={}
    if region=='CN':
        i=idx(origins(region,split),t); prefix='replay' if split=='evaluation' else 'validation'
        out['A_native']=np.stack([arr(P['cn_native']/'runs'/f'A_native_{s}'/f'{prefix}_predictions.npy')[i,...,0] for s in SEEDS[:3]]).astype(float)
        out['A_native_mean']=out['A_native'].mean(0,keepdims=True)
        if split=='evaluation':
            j=idx(CNORIG,t)
            for arm in ['do_tec','pi_tec','de_zscore','pi_zscore']: out[arm]=np.stack([arr(P['cn_coarse']/'runs'/arm/f'seed_{s}'/'predictions.npy')[j,...,0] for s in SEEDS]).astype(float)
            out['A_half']=np.stack([arr(P['cn_half']/'predictions'/f'seed_{s}.npy')[i,...,0] for s in SEEDS[:3]]).astype(float)
            out['do_tec_mean']=out['do_tec'].mean(0,keepdims=True)
    else:
        for arm in ['dense_do','dense_pi','dense_de','dense_pi_drivers','coarse_de','B','C_42']:
            values=[]
            for s in ([42] if arm=='C_42' else SEEDS):
                d=P['jp_baseline' if arm in ['B','C_42'] else 'jp_native']/'runs'/(arm if arm=='C_42' else f'{arm}_{s}')
                ts=arr(d/f'{split}_origins.npy'); a=arr(d/f'{split}_predictions.npy'); assert a.shape==(len(ts),12,288,1)
                values.append(a[idx(ts,t),...,0])
            out[arm]=np.stack(values).astype(float)
        for a in ['dense_do','dense_de','B']: out[a+'_mean']=out[a].mean(0,keepdims=True)
    return out

def legacy(t, mp, only_do=False):
    y=str(np.datetime64(int(t),'s'))[:4]; d=P['legacy_cache']/'shared'/y; ts=arr(d/'forecast_origin_utc.npy'); j=idx(ts,t)
    np.testing.assert_array_equal(arr(d/'target_time_utc.npy')[j],t+OFF)
    return {'old_'+a:np.stack([remap(arr(P['legacy_cache']/f'{a}-seed{s}'/y/'prediction_tecu.npy')[j,...,0],mp) for s in SEEDS[:3]]) for a in (['DO-TEC-only'] if only_do else CFG['arms']['china_legacy'])}

def stat(f,y,v):
    assert np.isfinite(y[v]).all()
    return sufficient(f,y,np.broadcast_to(v,f.shape))

def summary(a):
    totals=a.reshape(len(a),-1,4).sum(1); assert np.all(totals[:,0]==totals[0,0]); n=int(totals[0,0])
    values=np.column_stack([np.sqrt(totals[:,1]/n),totals[:,2:]/n]) if n else np.full((len(a),3),np.nan)
    return dict(n=n,n_predictions=len(a),by_seed=[dict(zip(['RMSE','MAE','bias'],[float(x) if n else None for x in row])) for row in values],**{k:float(values[:,j].mean()) if n else None for j,k in enumerate(['RMSE','MAE','bias'])},RMSE_sd=float(values[:,0].std(ddof=1)) if n and len(a)>1 else None)

def anchors():
    checks={}; data={}
    def test(k,a):
        value=summary(a)['RMSE']; ref=CFG['anchors'][k]; ok=abs(value-ref['reference'])<=ref['tolerance']
        checks[k]=dict(value=value,reference=ref['reference'],difference=value-ref['reference'],passed=ok)
        if k=='8': checks[k]['B_42']=summary(a)['by_seed'][0]['RMSE']; ok &= abs(checks[k]['B_42']-ref['B_42'])<=1e-6; checks[k]['passed']=bool(ok)
        save('anchors.json',checks); print('anchor',k,checks[k],flush=True); assert ok, f'Anchor {k} mismatch: {checks[k]}'
    ts=arr(P['gwn']/'runs/G_42/extension_evaluation_origins.npy')
    for a in ['G','F2']: data[a]=np.zeros((5,len(ts),12,4))
    for i,t in enumerate(ts):
        y=gim(t); forecasts=global_forecasts(t,'evaluation')
        for a in ['G','F2']: data[a][:,i]=stat(forecasts[a],y,np.isfinite(y))
        if i%250==0: print('native anchors',i,'/',len(ts),flush=True)
    test('1',data['G']); test('2',data['F2'])
    for region in ['CN','JP']:
        ts=origins(region,'evaluation'); names={'P':1,**({'A_native':3,'old_DO-TEC-only':3} if region=='CN' else {'dense_do':5,'B':5,'P60':1})}
        a={k:np.zeros((n,len(ts),12,4)) for k,n in names.items()}
        for i,t in enumerate(ts):
            y,v,p=day(region,'evaluation',t); f=regional(region,'evaluation',t); f['P']=p[None]
            if region=='CN': f.update(legacy(t,mapping(region),True))
            else: f['P60']=p[None]
            for k in names: a[k][:,i]=stat(f[k],y,v&(y>=60) if k=='P60' else v)
        for k,name in ([('3','P'),('4','A_native'),('5','old_DO-TEC-only')] if region=='CN' else [('6','P'),('7','dense_do'),('8','B'),('9','P60')]): test(k,a[name])
        data.update({region+'_'+k:v for k,v in a.items()})
    np.savez_compressed(HERE/'artifacts/anchor_daily.npz',**data)
    return checks

def intervals(stats, ts, region):
    dates=ts.astype('datetime64[s]').astype('datetime64[D]'); years=dates.astype('datetime64[Y]').astype(int)+1970
    rng=np.random.default_rng(20260924); samples={k:np.zeros((5000,len(v['all']),2)) for k,v in stats.items() if k in ['G','F2','G5','P','A_native' if region=='CN' else 'dense_de']}
    for year in sorted(set(years)):
        start=np.datetime64(f'{year}-01-01'); n=(int((np.datetime64(f'{year+1}-01-01')-start).astype(int))+6)//7; sel=years==year
        ids=((dates[sel]-start).astype(int)//7); draw=rng.integers(n,size=(5000,n))
        for key,out in samples.items():
            daily=stats[key]['all'].sum(2)[...,[1,0]]; b=np.zeros((n,len(daily),2))
            for s in range(len(daily)): np.add.at(b[:,s],ids,daily[s,sel])
            out+=b[draw].sum(1)
    rm={k:np.sqrt(v[...,0]/v[...,1]).mean(1) for k,v in samples.items()}; pairs=[('G','P'),('F2','G'),('G','A_native' if region=='CN' else 'dense_de'),('G5','G')]
    return {a+'-'+b:dict(delta=summary(stats[a]['all'])['RMSE']-summary(stats[b]['all'])['RMSE'],ci95=np.quantile(rm[a]-rm[b],[.025,.975]).tolist()) for a,b in pairs}

def evaluate(region,split):
    ts=origins(region,split); mp=mapping(region); nn=mapping(region,True); stats={}; coverage=[]
    for i,t in enumerate(ts):
        y,v,p=day(region,split,t); g,gp=gim(t),gim(t-86400); valid=v&np.isfinite(g[:,mp[0]]).all(1)&np.isfinite(gp[:,mp[0]]).all(1)
        coverage.append([int(v.sum()),int(valid.sum())]); forecasts=global_forecasts(t,split); f=regional(region,split,t); f['P']=p[None].astype(float)
        mapped=remap(g,mp); f.update(GIM_P=remap(gp,mp)[None],product=mapped[None])
        for a,x in forecasts.items():
            assert np.isfinite(x[...,mp[0]][:,valid[:,None,:].repeat(4,axis=1)]).all()
            f[a]=remap(x,mp); f[a+'_nearest']=remap(x,nn)
        f.update(GIM_P_nearest=remap(gp,nn)[None],product_nearest=remap(g,nn)[None])
        for a in ['G','F2','G5']: f[a+'_half']=np.where((p>=60)&(f[a]<p),(f[a]+p)/2,f[a])
        if region=='CN' and split=='evaluation': f.update(legacy(t,mp))
        masks=[p>=60,y>=60,y>=80,y>=100,(p>=60)&(y>=60),(p<60)&(y>=60),(p>=60)&(y<60)]
        for k,x in f.items():
            if k not in stats: stats[k]=dict(all=np.zeros((len(x),len(ts),12,4)),groups=np.zeros((len(x),len(ts),7,4)),decomp=np.zeros((len(x),len(ts),5)))
            stats[k]['all'][:,i]=stat(x,y,valid)
            for h,mask in enumerate(masks): stats[k]['groups'][:,i,h]=stat(x,y,valid&mask).sum(1)
            if k in ZERO or k in ['G_half','F2_half','G5_half'] or k.startswith('old_') or k in [a+'_nearest' for a in ZERO]:
                a=np.where(valid,x-(remap(g,nn) if k.endswith('_nearest') else mapped),0); b=np.where(valid,(remap(g,nn) if k.endswith('_nearest') else mapped)-y.astype(float),0)
                vals=np.stack([np.full(len(x),valid.sum()),(a*a).sum((1,2)),np.full(len(x),(b*b).sum()),(2*a*b).sum((1,2)),((a+b)**2).sum((1,2))],axis=1)
                np.testing.assert_allclose(vals[:,1:4].sum(1),vals[:,4],rtol=1e-12,atol=1e-6); np.testing.assert_allclose(vals[:,4],stats[k]['all'][:,i,:,1].sum(1),rtol=1e-12,atol=1e-6); stats[k]['decomp'][:,i]=vals
        if i%200==0: print(region,split,i,'/',len(ts),flush=True)
    n=sum(x[1] for x in coverage)
    if split=='evaluation': assert n=={'CN':36099959,'JP':3770496}[region]
    years=ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970; results={}; baseline=summary(stats['P']['all'])['RMSE']
    for k,a in stats.items():
        z=summary(a['all']); z['improvement_percent']=100*(1-z['RMSE']/baseline); z['seed_ids']=SEEDS[:len(a['all'])] if len(a['all'])>1 else ([42] if k=='C_42' else []); z['kind']='seed/group mean' if len(a['all'])>1 else 'single deterministic prediction'
        z['groups']={g:summary(a['groups'][:,:,h]) for h,g in enumerate(GROUPS)}; z['years']={str(y):summary(a['all'][:,years==y]) for y in sorted(set(years))}; z['leads']={str(2*(h+1)):summary(a['all'][:,:,h]) for h in range(12)}
        d=a['decomp'].sum(1); z['decomposition']=dict(zip(['mean_a2','mean_b2','mean_2ab','total_MSE'],(d[:,1:]/d[:,0,None]).mean(0).tolist())) if d[:,0].any() else None; results[k]=z
    np.savez_compressed(HERE/'artifacts'/f'{region}_{split}_daily.npz',origins=ts,fields=np.array(['n','sse','sae','error_sum']),groups=np.array(GROUPS),coverage=np.array(coverage),**{k+'__'+field:a for k,v in stats.items() for field,a in v.items()})
    return dict(days=len(ts),n=n,source_valid=sum(x[0] for x in coverage),methods=results,intervals=intervals(stats,ts,region) if split=='evaluation' else {})

def main():
    assert CFG['source_commit'] and len(CFG['source_commit'])==40
    for name in ['score.py','recheck.py']:
        frozen=subprocess.check_output(['git','show',CFG['source_commit']+':experiments/exp0011/'+name],cwd=HERE.parents[1]); assert frozen==(HERE/name).read_bytes()
    frozen=json.loads(subprocess.check_output(['git','show',CFG['source_commit']+':experiments/exp0011/config.json'],cwd=HERE.parents[1])); frozen['source_commit']=CFG['source_commit']; assert frozen==CFG
    for item in CFG['inputs']: assert Path(item['path']).stat().st_size==item['bytes']
    la,lo=np.meshgrid(np.arange(87.5,-90,-2.5),np.arange(-180,181,5),indexing='ij')
    for region in ['CN','JP']:
        z=np.load(P['cn_support']/'support/grid.npz') if region=='CN' else np.load(P['jp_native']/'prepared/coordinates.npz'); lat,lon=(z['lat'],z['lon']) if region=='CN' else [a.ravel() for a in np.meshgrid(z['lat'],z['lon'],indexing='ij')]
        np.testing.assert_allclose(remap(np.ones(5183),mapping(region)),1,rtol=0,atol=1e-15); np.testing.assert_allclose(remap((3*la+2*lo).ravel(),mapping(region)),3*lat+2*lon,rtol=0,atol=1e-12)
    result=dict(status='anchors_running',source_commit=CFG['source_commit'],rules=CFG['rules'],anchors=anchors(),regions={})
    for region in ['CN','JP']:
        result['regions'][region]={}
        for split in ['evaluation','validation']: result['regions'][region][split]=evaluate(region,split); save('result.json',result)
    result['status']='complete_cpu_development_scoring'; save('result.json',result)
    print('All scoring complete',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        save('STOPPED.json',dict(status='stopped_no_further_scoring',error=repr(exc),source_commit=CFG['source_commit'])); raise
