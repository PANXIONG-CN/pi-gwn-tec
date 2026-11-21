"""Procedural F/E checks and preregistered window eligibility, no model errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from tecfc.scoring.confirm_records import ROOT, EXP, sha, read, write, now
from tecfc.data.data2026 import Data, unix


def preflight():
    for letter in ['A','B','C','D','G']:
        assert read(ROOT/f'experiments/exp0025/checks/{letter}.json')['status']=='passed'
    assert read(ROOT/'experiments/exp0025/checks/F_HASHES.json')['status']=='passed'
    # Write before opening the first 2026 value. Even a subsequent abort retains
    # this access event and cannot describe the data as untouched.
    write(EXP/'SCIENTIFIC_ACCESS_STARTED.json',dict(at_utc=now(),first_entry=__file__,
          stage='procedural_preflight',prediction_errors_permitted=False,
          requested_ranges=dict(global_=['2026-01-01','2026-10-31'],regional=['2026-01-01','2026-06-30'])))
    used=ROOT/'USED_2026.md'
    with used.open('x') as f:
        f.write('# 2026 数据已进入一次性确认\n\n2026-11-15：exp0025 已开始读取 2026 科学数组进行预注册的程序性检查。\n'
                '授权范围：全球 origin 2026-01-01—10-31；中国、日本 origin 2026-01-01—06-30。\n'
                '此后这些已读取数据不得再被称为独立确认数据。实际入选范围、是否评分及停止状态以\n'
                '`experiments/exp0025/SCOPE_2026.json`、`result.json`、`HANDOFF.json` 为准。\n'
                '本实验不启动全球 11–12 月或区域 7–12 月延伸期。\n')
    try:
        data=Data();q=data.quality;a=data.raw2026
        assert a.dtype==np.float32 and a.shape[1:]==(12,71,73)
        np.testing.assert_array_equal(q['lat'],np.arange(87.5,-88,-2.5,dtype=np.float32))
        np.testing.assert_array_equal(q['lon'],np.arange(-180,181,5,dtype=np.float32))
        np.testing.assert_array_equal(q['time_utc'],unix('2026-01-01')+np.arange(len(a)*12,dtype=np.int64).reshape(-1,12)*7200)
        assert q['valid'].dtype==bool and q['valid'].shape==a.shape and q['present'].shape==(len(a),12)
        assert np.array_equal(q['valid'],np.isfinite(a))
        plans={};causality=[];contracts=[]
        days=[str(d) for d in np.arange(np.datetime64('2026-01-01'),np.datetime64('2026-11-01'))]
        included=[];excluded=[]
        for day in days:
            ix=int((np.datetime64(day)-np.datetime64('2026-01-01')).astype(int));why=[]
            if ix>=len(a) or not q['present'][ix].all():why.append('one_or_more_target_frames_unpublished')
            for lag in [1,2,3,27]:
                history=str(np.datetime64(day)-np.timedelta64(lag,'D'))
                try:
                    h=data.global_day(history)
                    if not np.isfinite(h).all():why.append(f'nonfinite_history_lag_{lag}')
                except KeyError:why.append(f'unpublished_history_lag_{lag}')
                stamps=unix(day)-lag*86400+np.arange(12)*7200
                assert stamps.max()<=unix(day)-7200
            row=dict(date=day,origin=unix(day),cutoff=unix(day)-7200,target_times=(unix(day)+np.arange(12)*7200).tolist())
            (excluded if why else included).append(dict(row,reasons=why) if why else row)
            causality.append(dict(domain='global',date=day,input_latest=unix(day)-7200,cutoff=unix(day)-7200,lags=[1,2,3,27],passed=True))
        plans['global']=dict(requested_start=days[0],requested_end=days[-1],included=included,excluded=excluded)
        for region in ['china','japan']:
            st=np.load(ROOT/f'experiments/exp0023/artifacts/prepared/stage2/{region}/statistics.npz')
            old=data.regional_day(region,'2025-12-31');inc=[];exc=[]
            for day in [str(d) for d in np.arange(np.datetime64('2026-01-01'),np.datetime64('2026-07-01'))]:
                hist=str(np.datetime64(day)-np.timedelta64(1,'D'));why=[]
                y=data.regional_day(region,day);h=data.regional_day(region,hist)
                for dd,arr in [(day,y),(hist,h)]:
                    np.testing.assert_array_equal(arr['time_utc'],unix(dd)+np.arange(96,dtype=np.int64)*900)
                    for key in ['lat','lon']:np.testing.assert_array_equal(arr[key],st[key]);np.testing.assert_array_equal(arr[key],old[key])
                    assert arr['tec'].dtype==old['tec'].dtype==np.float32 and arr['valid'].dtype==old['valid'].dtype==bool
                    assert arr['tec'].shape==old['tec'].shape and arr['valid'].shape==arr['tec'].shape
                    assert not (arr['valid'] & ~np.isfinite(arr['tec'])).any()
                    expected=arr['present'][:,None,None]&np.isfinite(arr['tec'])&(arr['tecflag']==0)
                    # Source fill-value sentinels are also invalid; no relabeling.
                    assert not (arr['valid'] & ~expected).any()
                    if region=='china':assert not (arr['valid']&(arr['tecflag']==1)).any()
                if not h['present'][:89].all():why.append('history_frame_absent')
                if not y['present'][::8].all():why.append('target_frame_absent')
                stamps=h['time_utc'][:89];assert stamps.max()<=unix(day)-7200
                row=dict(date=day,origin=unix(day),cutoff=unix(day)-7200,history=hist,target_times=y['time_utc'][::8].tolist())
                (exc if why else inc).append(dict(row,reasons=why) if why else row)
                causality.append(dict(domain=region,date=day,input_latest=int(stamps.max()),cutoff=unix(day)-7200,history_frames=89,passed=True))
            plans[region]=dict(requested_start='2026-01-01',requested_end='2026-06-30',included=inc,excluded=exc)
            contracts.append(dict(region=region,grid_matches_2025=True,time_cadence_seconds=900,label_stride=8,
                                  original_valid_mask_preserved=True,negative_values_not_clipped=True,official_final_status='not_declared'))
        drv=data.drivers();np.testing.assert_array_equal(np.diff(drv['time']),np.full(len(drv['time'])-1,7200))
        sinc=[];sexc=[]
        for row in plans['global']['included']:
            ts=row['origin']-86400+np.arange(12)*7200;ix=np.searchsorted(drv['time'],ts)
            why=[]
            if np.any(ix>=len(drv['time'])) or not np.array_equal(drv['time'][ix],ts):why=['driver_timestamp_absent']
            else:
                why=['driver_missing_'+key for key in ['AE','Dst','Kp','F107'] if not np.isfinite(drv[key][ix]).all()]
            (sexc if why else sinc).append(dict(date=row['date'],origin=row['origin'],reasons=why) if why else dict(date=row['date'],origin=row['origin']))
        plans['S1']=dict(included=sinc,excluded=sexc,label='驱动版本与训练不一致的敏感性结果',criteria_eligible=False)
        rng=np.random.default_rng(20260924);spots=[]
        for domain in ['global','china','japan']:
            rows=[r for r in causality if r['domain']==domain]
            for month in sorted({r['date'][:7] for r in rows}):
                selected=[r for r in rows if r['date'][:7]==month]
                mandatory=[r for r in selected if month=='2026-01' and int(r['date'][-2:])<=(27 if domain=='global' else 10)]
                rest=[r for r in selected if r not in mandatory]
                pick=list(rng.choice(len(rest),min(20,len(rest)),replace=False)) if rest else []
                spots.extend(mandatory+[rest[int(i)] for i in pick])
        write(EXP/'checks/F.json',dict(status='passed',at_utc=now(),hash_record_sha256=sha(EXP/'checks/F_HASHES.json'),
              global_shape=list(a.shape),global_partial_year_axis=True,grid_time_mask_contracts_match_2025=True,
              regional_contracts=contracts,driver_time_grid_seconds=7200,no_error_metrics=True))
        write(EXP/'checks/E.json',dict(status='passed',at_utc=now(),all_candidate_window_assertions=causality,spot_checks=spots,
              all_early_January_cross_year_inputs_checked=True,no_post_cutoff_observations=True))
        write(EXP/'SCOPE_2026.draft.json',dict(experiment='exp0025',status='draft_pending_checks_commit',plans=plans,
              half_year_global_tables='separate subsets aligned exactly to each regional included-date list',
              extension_period_not_started=True,criteria_unchanged=True,at_utc=now()))
        print(json.dumps({k:dict(included=len(v['included']),excluded=len(v['excluded'])) for k,v in plans.items()}))
    except Exception as e:
        if not (EXP/'checks/F.json').exists():write(EXP/'checks/F.json',dict(status='failed',phase='procedural_contract',error=type(e).__name__+': '+str(e),at_utc=now()))
        raise


def freeze_scope():
    for letter in 'ABCDEFG':assert read(ROOT/f'experiments/exp0025/checks/{letter}.json')['status']=='passed'
    d=read(EXP/'SCOPE_2026.draft.json');d['status']='frozen_before_prediction_generation';d['checks_sha256']={k:sha(EXP/f'checks/{k}.json') for k in 'ABCDEFG'}
    write(EXP/'SCOPE_2026.json',d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','freeze']);a=p.parse_args()
    preflight() if a.stage=='preflight' else freeze_scope()
