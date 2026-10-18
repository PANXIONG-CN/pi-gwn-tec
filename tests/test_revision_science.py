"""Scientific invariants for the manuscript's portable recipes; no training."""
from pathlib import Path
import numpy as np
import pytest
import torch
from lib.paper_recipe import Windows, build_physics, supervised_mae
from lib.paper_statistics import masked_statistics,horizon_statistics
from lib.physics.projection import merge
from lib.physics.driver_scaler import validate_history_visibility
from lib.uq.conformal import china_calendar_roles,horizon_qhat,horizon_interval_metrics

# Numerical Source Data are Supplementary Data 1 of the article, not part of the code.
SOURCE_DATA=Path(__file__).resolve().parents[1]/'source_data'
needs_source_data=pytest.mark.skipif(not SOURCE_DATA.is_dir(),reason='extract Supplementary Data 1 to source_data/')


def test_daily_window_times_and_96h_boundaries():
    values=np.arange((365*12+12)*2,dtype=np.float32).reshape(-1,2)
    ds=Windows(values,'2018-12-31',48)
    assert len(ds)==362
    x,y,t=ds[0]
    assert t==np.datetime64('2019-01-01','s').astype(np.int64)
    np.testing.assert_array_equal(x.numpy()[...,0],values[:12]/np.float32(147.1))
    np.testing.assert_array_equal(y.numpy()[...,0],values[12:60]/np.float32(147.1))
    _,_,last=ds[-1] if False else ds[len(ds)-1]
    assert last+47*7200==np.datetime64('2019-12-31T22','s').astype(np.int64)
    # Continuous 2009--2012 and separate later training years.
    assert sum(len(Windows(np.zeros((n*12,1),np.float32),'2009-01-01',48)) for n in [1461,365,366,365])==2541
    assert sum(len(Windows(np.zeros((365*12+12,1),np.float32),'2012-12-31',48)) for _ in range(2))==724


def test_history_visibility_rejects_target_time():
    origins=np.array([np.datetime64('2019-01-01','s').astype(np.int64)])
    history=origins[:,None]-np.arange(12,0,-1)*7200
    validate_history_visibility(history,origins,expected_history_steps=12)
    history[0,-1]=origins[0]
    with pytest.raises(ValueError):validate_history_visibility(history,origins,expected_history_steps=12)


def test_masks_valid_zero_and_aggregation():
    pred=np.array([0.,2.,9.]);truth=np.array([0.,0.,100.]);valid=np.array([True,True,False])
    r=masked_statistics(pred,truth,valid)
    assert r['n_valid']==2 and r['sse']==4 and r['RMSE']==np.sqrt(2)
    p=np.zeros((1,48,1));p[:,36:48]=1;p[:,47]=4
    s=horizon_statistics(p,np.zeros_like(p),np.ones_like(p,dtype=bool))
    assert s['lead_96h']['RMSE']==4
    assert s['day_4']['RMSE']==np.sqrt(27/12)


def test_projection_one_sided_without_renormalization():
    p=torch.nn.Parameter(torch.zeros(2));d=torch.tensor([1.,0.]);a=torch.tensor([-1.,1.])
    r=merge([p],[d],[a],.1)
    np.testing.assert_allclose(p.grad.numpy(),[1.,.1/np.sqrt(2)],rtol=1e-6)
    assert r['projection_triggered'] and r['data_dot_auxiliary_after']==0
    assert r['auxiliary_ratio_after']<r['auxiliary_ratio_before']
    merge([p],[d],[a],0);torch.testing.assert_close(p.grad,d)


def test_native_penalty_and_gradients():
    module=build_physics(np.eye(3,dtype=np.float32),'cpu')
    y=torch.full((1,12,3,1),.3,requires_grad=True)
    driver={'F107':torch.full((1,12),100.),'AE':torch.zeros(1,12),'Dst':torch.zeros(1,12),'cosSZA':torch.ones(1,12,3)}
    loss,terms=module(y,None,drivers=driver)
    assert set(terms)=={'nonneg','rot','roti','diffusion','chemistry','night'}
    p=.5*sum(terms.values())
    torch.testing.assert_close(p,loss.squeeze())
    grad=torch.autograd.grad(p,y)[0]
    assert torch.isfinite(grad).all() and torch.any(grad!=0)
    assert len(module._rot_and_roti(y)[0][0])==11


def test_calendar_and_per_lead_quantiles():
    dates=np.arange(np.datetime64('2015-01-01'),np.datetime64('2016-01-01'))
    roles=china_calendar_roles(dates)
    assert roles['calibration_2015_h1'].sum()==181
    assert roles['development_2015_h2'].sum()==184
    # Independent hand calculation of the order statistic on unequal leads.
    residual=np.stack([np.arange(12),2*np.arange(12)],axis=1).reshape(4,2,3)
    q,meta=horizon_qhat(np.zeros_like(residual),residual,alpha=.25)
    level=min(1,np.ceil(13*.75)/12)
    expected=[sorted(residual[:,h,:].ravel())[int(np.ceil(level*11))] for h in range(2)]
    np.testing.assert_array_equal(q,expected)
    full=horizon_qhat(np.zeros((181,12,2911),np.float32),np.zeros((181,12,2911),np.float32))[1]
    assert full['scores_per_lead']==526891
    metrics=horizon_interval_metrics(np.zeros_like(residual),residual,q,alpha=.25)
    assert metrics[0]['n_days']==4 and metrics[0]['n_element_scores']==12


def test_regional_saturation_boundaries():
    truth=torch.tensor([0.,300.-.001,300.+.001,299.998],dtype=torch.float32)
    pred=truth+1
    assert float(supervised_mae(pred,truth,torch.ones(4,dtype=torch.bool),'china'))==pytest.approx(.65)


def test_removed_driver_is_invariant_but_retained_driver_is_used():
    from lib.paper_recipe import build_predictor
    from lib.physics.driver_scaler import DriverHistoryBatch, DRIVER_CHANNEL_ORDER
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cfg=dict(driver_fusion='film',horizon=12,input_mask=[1,1,1,0,1],
             driver_source_sha256='a'*64,driver_scaler_sha256='b'*64)
    model=build_predictor(cfg,np.eye(3,dtype=np.float32),'cpu',load_only=True).eval()
    with torch.no_grad(): model.history_film.projection.weight.normal_(0,.1)
    x=torch.randn(1,12,3,1)
    t=torch.tensor([1546300800]);times=t[:,None]-torch.arange(12,0,-1)*7200
    drivers=torch.randn(1,12,3,5)
    def batch(v):
        return DriverHistoryBatch(v,times,t,DRIVER_CHANNEL_ORDER,'a'*64,'b'*64,'train_zscore')
    removed=drivers.clone();removed[...,3]+=100
    retained=drivers.clone();retained[...,0]+=10
    with torch.no_grad():
        a=model(x,batch(drivers));b=model(x,batch(removed));c=model(x,batch(retained))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.allclose(a,c)


def test_pooled_block_interval_unequal_counts_and_pairing():
    import pandas as pd
    from lib.paper_statistics import paired_pooled_rmse_interval
    rows=[]
    for seed in [42, 1024]:
        for start,end,n,sa,sb in [('2015-01-01','2015-01-14',14,14,56),
                                  ('2015-01-15','2015-01-15',1,81,1)]:
            for model,sse in [('PI',sa),('DO',sb)]:
                rows.append(dict(variant=model,seed=seed,start=start,end=end,sse=sse,n_valid=n))
    f=pd.DataFrame(rows)
    r=paired_pooled_rmse_interval(f,'PI','DO',repeats=200)
    assert r['mean_pooled_rmse_delta']==pytest.approx(np.sqrt(95/15)-np.sqrt(57/15))
    assert r['mean_pooled_rmse_delta']!=pytest.approx(((1-2)+(9-1))/2)
    assert r==paired_pooled_rmse_interval(f.sample(frac=1,random_state=1),'PI','DO',repeats=200)
    same=f.copy();same.loc[same.variant=='PI','sse']=same.loc[same.variant=='DO','sse'].to_numpy()
    z=paired_pooled_rmse_interval(same,'PI','DO',repeats=200)
    assert z['low']==z['high']==z['mean_pooled_rmse_delta']==0
    with pytest.raises(ValueError):paired_pooled_rmse_interval(f.iloc[:-1],'PI','DO')
    bad=f.copy();bad.loc[0,'n_valid']=15
    with pytest.raises(ValueError):paired_pooled_rmse_interval(bad,'PI','DO')


@needs_source_data
def test_retained_pooled_intervals():
    import json,pandas as pd
    from lib.paper_statistics import paired_pooled_rmse_interval
    source=SOURCE_DATA/'global_matched'
    saved=json.loads((source/'FINAL_ANALYSIS.json').read_text())['comparisons']
    for file,arm,reference,key in [
        ('TEC_PI_DO','PI-TEC-only','DO-TEC-only','TEC-Phys_minus_TEC-Sup'),
        ('DRIVER_PI_DE','arm_a','zscore','TEC+Driver-Phys_minus_TEC+Driver-Sup')]:
        r=paired_pooled_rmse_interval(pd.read_csv(source/f'BLOCK_STATISTICS_{file}.csv'),arm,reference)
        assert r['mean_pooled_rmse_delta']==pytest.approx(saved[key]['mean'],abs=1e-12)
        np.testing.assert_allclose([r['low'],r['high']],saved[key]['pooled_RMSE_14day_CI']['interval'],rtol=0,atol=1e-12)


def test_historical_pi_effective_parser_without_training(monkeypatch):
    # Execute the real overlay function in isolation: importing Run would train.
    import ast,sys,yaml,argparse
    from pathlib import Path
    from lib.Params_pretrain import parse_args
    from model.GWN.args import parse_args as predictor_args
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys,'argv',['Run.py','-dataset','GIMtec','-model','GWN','-mode','ori',
                                  '-config',str(root/'conf/paper/A1_gim_pi_drivers_only.yaml')])
    args=parse_args('cpu')
    pred=predictor_args('GIMtec',argparse.ArgumentParser())
    for name in vars(args):
        if hasattr(pred,name) and '-'+name not in sys.argv:setattr(args,name,getattr(pred,name))
    tree=ast.parse((root/'model/Run.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_yaml_overlay')
    scope=dict(Path=Path,file_dir=str(root),sys=sys,yaml=yaml)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'Run.py','exec'),scope)
    args,_=scope['_yaml_overlay'](args,pred)
    assert {k:getattr(args,k) for k in ['batch_size','accumulate_steps','amp','lambda_phys','use_drivers','use_drivers_input']}==dict(batch_size=96,accumulate_steps=2,amp=True,lambda_phys=.1,use_drivers=True,use_drivers_input=False)


def test_tsi_replicates_computed_standard_deviations():
    from types import SimpleNamespace
    from lib.physics.ppinn import PPinnLoss
    from lib.physics.ppinn_projected import PPinnLoss as ProjectedPhysicsLoss
    # Independent population-SD calculation for valid windows [0,1,4], [1,4,10].
    y = torch.tensor([0., 0., 1., 5., 15.], dtype=torch.float64).reshape(1, 5, 1, 1)
    expected = torch.tensor([np.sqrt(26/9)]*2+[np.sqrt(14)]*2, dtype=torch.float64)
    obj = SimpleNamespace(dt_nd=1., roti_w=3, _delta_t=PPinnLoss._delta_t)
    for method in [PPinnLoss._rot_roti_nd, ProjectedPhysicsLoss._rot_and_roti]:
        rate, tsi = method(obj, y)
        torch.testing.assert_close(rate.flatten(), torch.tensor([0., 1., 4., 10.], dtype=torch.float64))
        torch.testing.assert_close(tsi.flatten(), expected)


@needs_source_data
def test_retained_extension_original_function_and_protocol():
    import csv, json
    from lib.paper_statistics import retained_extension_intervals
    source = SOURCE_DATA/'extension'
    frozen = json.loads((source/'FREEZE.json').read_text())
    protocol = frozen['statistics']
    assert (protocol['replicates'], protocol['rng_seed'], protocol['original_family_size']) == (10000, 20260925, 3)
    assert protocol['no_alpha_reallocation'] and protocol['quantile_method'] == 'linear'
    assert protocol['quantile_probabilities'] == [1/120, 119/120]
    blocks = list(csv.DictReader((source/'BLOCKS.csv').open()))
    for year, days in [(2023, 365), (2024, 366), (2025, 365)]:
        group = [b for b in blocks if int(b['year']) == year]
        assert [int(b['windows']) for b in group] == [14]*25+[days-350]
        assert [int(b['first_window_index']) for b in group] == list(range(0, 364, 14))
        assert [int(b['end_window_index_exclusive']) for b in group] == list(range(14, 351, 14))+[days]
    actual = retained_extension_intervals(source)
    saved = json.loads((source/'RESULTS.json').read_text())['primary']
    def compare(a, b):
        for key, value in a.items():
            if isinstance(value, dict): compare(value, b[key])
            elif isinstance(value, bool): assert value is b[key]
            else: np.testing.assert_allclose(value, b[key], rtol=0, atol=1e-12)
    for name in ['P2', 'P3']: compare(actual[name], saved[name])
