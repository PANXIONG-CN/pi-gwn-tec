"""exp0017 CPU fusion and handoff gate; no import-time reads or writes.

The coordination modes intentionally hash result.json as bytes without parsing
scientific scores. Scientific execution is gated on all three closed handoffs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
HERE = ROOT / 'experiments/exp0017'
UPSTREAM = ("exp0014", "exp0015", "exp0016")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def choose_method(scores, simplicity_order=("C1", "C4", "C2", "C3"), threshold=0.001):
    """Threshold uses the simpler score as denominator; equality is not close."""
    best = min(scores, key=lambda method: (scores[method], simplicity_order.index(method)))
    minimum = float(scores[best])
    for method in simplicity_order:
        if method not in scores:
            continue
        score = float(scores[method])
        if score == minimum or minimum > score * (1. - threshold):
            return method
    raise AssertionError("No minimum selected")


def keep_reference(g5_score, previous_score, threshold=0.001):
    return "exp0014_adopted" if previous_score <= g5_score * (1. - threshold) else "G5"


def forward_select(cross_sum, count, family_order, threshold=0.001):
    """C4 sees fit-year moments only. It never receives held-out data."""
    import numpy as np
    matrix = np.asarray(cross_sum, dtype=np.float64)
    if matrix.ndim == 3:
        matrix = matrix.sum(axis=0)
    if matrix.shape != (len(family_order), len(family_order)):
        raise ValueError("C4 cross_sum has wrong family axes")
    n = float(np.asarray(count, dtype=np.float64).sum())
    if n <= 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("C4 requires finite moments and nonempty fit support")
    selected = [family_order.index("G")]
    trace = []

    def rmse(indices):
        value = float(matrix[np.ix_(indices, indices)].sum()) / (len(indices) ** 2 * n)
        if value < -1e-10:
            raise ValueError("Negative fitted ensemble squared error")
        return float(np.sqrt(max(value, 0.)))

    current = rmse(selected)
    while len(selected) < len(family_order) and current > 0:
        candidates = [(rmse(selected + [index]), index) for index in range(len(family_order)) if index not in selected]
        next_score, index = min(candidates, key=lambda item: (item[0], item[1]))
        improvement = (current - next_score) / current
        accepted = next_score <= current * (1. - threshold)
        trace.append({"candidate": family_order[index], "before_RMSE": current,
                      "after_RMSE": next_score, "relative_improvement": improvement, "accepted": bool(accepted)})
        if not accepted:
            break
        selected.append(index)
        current = next_score
    weights = np.zeros(len(family_order), dtype=np.float64)
    weights[selected] = 1. / len(selected)
    return {"weights": weights.tolist(), "selected_families": [family_order[i] for i in selected],
            "fit_RMSE": current, "trace": trace}


def deterministic_pair(candidate_stats, reference_stats, origins):
    """Time block interval; a multi-seed reference is a mean of seed metrics.

    Statistics use [prediction, day, lead, (n,sse,sae,error_sum)]. The candidate
    has exactly one prediction. Never broadcast it into independent replicates.
    """
    import numpy as np
    a = np.asarray(candidate_stats, dtype=np.float64)
    b = np.asarray(reference_stats, dtype=np.float64)
    if a.ndim != 4 or b.ndim != 4 or a.shape[0] != 1 or a.shape[1:] != b.shape[1:]:
        raise ValueError("Expected one deterministic candidate and aligned reference predictions")
    if not np.array_equal(np.broadcast_to(a[..., 0], b[..., 0].shape), b[..., 0]):
        raise ValueError("Paired supports differ")
    dates = np.asarray(origins).astype("datetime64[s]").astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    rng = np.random.default_rng(20260924)
    daily = [x.sum(axis=2)[..., :2] for x in (a, b)]
    boot = [np.zeros((5000, len(x), 2), np.float64) for x in (a, b)]
    for year in sorted(set(years.tolist())):
        start, end = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year+1}-01-01")
        count = (int((end-start).astype(int)) + 6) // 7
        select = years == year
        ids = ((dates[select] - start).astype(int) // 7).astype(int)
        draws = rng.integers(0, count, (5000, count))
        multiplicity = np.zeros((5000, count), np.int32)
        np.add.at(multiplicity, (np.arange(5000)[:, None], draws), 1)
        for source, sampled in zip(daily, boot):
            block = np.zeros((count, len(source), 2))
            for index in range(len(source)):
                np.add.at(block[:, index], ids, source[index, select])
            sampled += (multiplicity @ block.reshape(count, -1)).reshape(sampled.shape)
    valid = np.all(boot[0][..., 0] > 0, axis=1) & np.all(boot[1][..., 0] > 0, axis=1)
    total = [x.sum(axis=1) for x in daily]
    if any(np.any(x[:, 0] == 0) for x in total):
        return {"delta_RMSE": None, "ci95": None, "usable_replicates": 0, "status": "empty_support"}
    point = [float(np.sqrt(x[:, 1] / x[:, 0]).mean()) for x in total]
    draws_rmse = [np.sqrt(x[valid, :, 1] / x[valid, :, 0]).mean(axis=1) for x in boot]
    difference = draws_rmse[0] - draws_rmse[1]
    return {"delta_RMSE": point[0] - point[1], "candidate_RMSE": point[0], "reference_RMSE": point[1],
            "ci95": np.quantile(difference, [0.025, 0.975]).tolist() if len(difference) else None,
            "usable_replicates": int(valid.sum()), "replicates": 5000, "rng": 20260924, "block_days": 7,
            "reference_prediction_count": len(b),
            "reference_statistic": "mean of individual-model RMSEs" if len(b) > 1 else "deterministic prediction RMSE",
            "scope": "conditional temporal uncertainty for fixed predictions; no seed SD for deterministic fusion"}


def context():
    import importlib.util
    import numpy as np
    snapshot = Path(__file__).resolve().parents[1]
    relocated = {'experiments/exp0014/fusion_map.py': 'ensemble/fusion_map.py', 'src/pi_global/score.py': 'scoring/pooled.py',
                 'experiments/exp0011/score.py': 'regional/mapping_score.py'}
    def module(relative, name):
        spec = importlib.util.spec_from_file_location(name, snapshot / relocated[relative])
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
        return value
    fm = module('experiments/exp0014/fusion_map.py', '_frozen_fusion_map')
    score = module('src/pi_global/score.py', '_frozen_score')
    cfg = read(HERE / 'config.json')
    refcfg = read(HERE / 'reference_config.json')
    return cfg, refcfg, fm, score, module


def inventory():
    import numpy as np
    cfg, refcfg, fm, _, _ = context()
    arms, hashes, sources = {}, {}, {}
    for experiment in UPSTREAM:
        path = ROOT / 'experiments' / experiment / 'FROZEN_PREDICTIONS.json'
        manifest = read(path); sources[experiment] = sha(path)
        for family, records in manifest['arms'].items():
            if family == 'DO':
                continue
            if family in arms:
                assert arms[family] == records
            arms[family] = records
        for path, digest in manifest['files_sha256'].items():
            if path in hashes:
                assert hashes[path] == digest
            hashes[path] = digest
    assert set(arms) == set(cfg['family_order'])
    for path, entry in refcfg['data_files'].items():
        assert int(path) <= 2025
        hashes[entry['path']] = entry['sha256']
    for i, (path, digest) in enumerate(hashes.items()):
        assert sha(path) == digest, path
        if i % 50 == 0:
            print('input hashes', i, '/', len(hashes), flush=True)
    manifest = {'experiment': 'exp0017', 'family_order': cfg['family_order'], 'seeds': cfg['seeds'],
                'arms': {family: arms[family] for family in cfg['family_order']},
                'files_sha256': hashes, 'upstream_manifests_sha256': sources,
                'unit': 'TECU', 'grid': refcfg['grid'], 'input_identity_verified': True}
    audit = []
    for split in fm.SPLITS:
        ts, readers = fm.family_readers(manifest, split)
        np.testing.assert_array_equal(ts, fm.expected_origins(split))
        for family, values in readers.items():
            for seed, reader in zip(cfg['seeds'], values):
                audit.append({'family': family, 'seed': seed, 'split': split, 'shape': list(reader.shape),
                              'dtype': str(reader.dtype), 'origins_equal': True, 'unit': 'TECU'})
                record = next(r for r in arms[family] if r['seed'] == seed)['splits'][split]
                for part in record.get('parts', [record]):
                    pt = fm.load_array(part['origins'], part.get('origins_key'))
                    if 'target_times' in part:
                        np.testing.assert_array_equal(fm.load_array(part['target_times'], part.get('target_times_key')),
                                                      pt[:, None] + np.arange(12)*7200)
    write(HERE / 'FROZEN_PREDICTIONS.json', manifest)
    write(HERE / 'INPUT_AUDIT.json', {'status': 'passed', 'files_hashed': len(hashes), 'records': audit,
          'evaluation_errors_computed': False, 'ready_upstreams': list(UPSTREAM),
          'target_times': 'origin + 0,2,...,22 hours; forecast lead labels +2,...,+24h relative to prior-day 22 UTC'})


def expanded(weights, old_order, order):
    import numpy as np
    result = np.zeros((12, len(order)))
    result[:, [order.index(x) for x in old_order]] = weights
    return result


def reference_specs(cfg):
    import numpy as np
    prior = read(ROOT / 'experiments/exp0014/FROZEN_FUSION.json')
    g = np.zeros((12, len(cfg['family_order']))); g[:, 0] = 1
    f10 = g / 2; f10[:, 1] = .5
    return {'G5': g, 'exp0014_adopted': expanded(prior['full_validation_weights'], prior['family_order'], cfg['family_order']),
            'F10': f10}


def collect(split, specifications, filename, moments=False, export=False):
    import numpy as np
    cfg, refcfg, fm, _, _ = context()
    manifest = read(HERE / 'FROZEN_PREDICTIONS.json'); order = cfg['family_order']
    names = list(specifications)
    selected = list(range(len(order))) if moments else sorted(set([0] + [i for w in specifications.values()
                                            for i in np.flatnonzero(np.asarray(w).any(0)).tolist()]))
    families = [order[i] for i in selected]
    ts, readers = fm.family_readers(manifest, split, families)
    np.testing.assert_array_equal(ts, fm.expected_origins(split))
    stats = np.zeros((len(names), len(ts), 4, 12, 4))
    single = np.zeros((5, len(ts), 4, 12, 4))
    cross = np.zeros((len(ts), 12, len(order), len(order))) if moments else None
    sums = np.zeros((len(ts), 12, len(order))) if moments else None
    counts = np.zeros((len(ts), 12))
    disturbed = fm.disturbance(refcfg, ts); cache = {}
    destination = None; rounding_max = 0.
    if export:
        path = HERE / 'artifacts' / f'{split}_predictions.npy'
        if path.exists():
            raise ValueError('Refuse to overwrite final predictions')
        destination = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(ts),12,71,73))
        np.save(HERE / 'artifacts' / f'{split}_origins.npy', ts)
    for i, t in enumerate(ts):
        p = fm.family_means(readers, i, families).reshape(len(families), 12, 5183)
        truth = fm.truth_day(refcfg, t, cache); valid = np.isfinite(truth)
        assert np.isfinite(p[:, valid]).all()
        for family in families:
            if family != 'DE-ZScore-c05':
                for reader in readers[family]:
                    np.testing.assert_array_equal(reader[i][..., 0], reader[i][..., -1])
        unique = valid.reshape(12,71,73).copy(); unique[..., -1] = False
        masks = [valid, unique.reshape(12,5183), valid & (truth >= 60), valid & disturbed[i]]
        forecasts = np.stack([fm.fuse(p, np.asarray(w)[:, selected]) for w in specifications.values()])
        if destination is not None:
            adopted = forecasts[names.index('adopted')].reshape(12,71,73)
            destination[i] = adopted
            rounding_max = max(rounding_max, float(np.max(np.abs(destination[i].astype(float)-adopted))))
        stats[:, i] = fm.stats_from_errors(np.where(valid, forecasts-truth, 0.), masks)
        g = np.stack([a[i].reshape(12,5183).astype(np.float64) for a in readers['G']])
        single[:, i] = fm.stats_from_errors(np.where(valid, g-truth, 0.), masks)
        counts[i] = valid.sum(1)
        if moments:
            error = np.where(valid, p-truth, 0.).transpose(1,0,2)
            cross[i] = error @ error.transpose(0,2,1)
            sums[i] = error.sum(2)
        if i % 200 == 0:
            print('collect', split, filename, i, '/', len(ts), flush=True)
    if destination is not None:
        destination.flush(); del destination
    payload = {'origins': ts, 'methods': np.array(names), 'stats': stats, 'single_G': single,
               'counts': counts, 'groups': np.array(fm.GROUPS), 'fields': np.array(fm.FIELDS),
               'max_float32_export_rounding_TECU': np.array(rounding_max)}
    if moments:
        payload.update(family_daily_cross=cross, family_error_sum=sums)
    np.savez_compressed(HERE / 'artifacts' / filename, **payload)
    return payload


def load_stats(name):
    import numpy as np
    with np.load(HERE / 'artifacts' / name, allow_pickle=False) as z:
        return dict(z)


def strip_seed_labels(value):
    if isinstance(value, dict):
        return {k: strip_seed_labels(v) for k, v in value.items() if k not in ('by_seed', 'RMSE_seed_sd')}
    return value


def tables(data):
    _, _, fm, score, _ = context()
    return {name: strip_seed_labels(score.tables({g: data['stats'][j:j+1, :, h] for h,g in enumerate(fm.GROUPS)}, data['origins']))
            for j,name in enumerate(data['methods'].tolist())}


def anchors(split):
    cfg, _, fm, score, _ = context()
    data = collect(split, reference_specs(cfg), f'anchors_{split}.npz')
    table = tables(data)
    # Only the requested split's reference fields are exposed to the workflow.
    prior = read(ROOT / 'experiments/exp0014/result.json')['main_table'][split]
    checks = read(HERE/'ANCHORS.json') if (HERE/'ANCHORS.json').exists() else {}
    for name, previous in [('G5','G5'), ('exp0014_adopted','adopted')]:
        fm.check_anchor(split+'/'+name, table[name]['all']['RMSE'], prior[previous]['all']['RMSE'], checks)
    if split != 'validation':
        fm.check_anchor(split+'/G_single_mean', score.summary(data['single_G'][:,:,0])['RMSE'],
                        2.233885478 if split=='original_evaluation' else 4.729342137, checks)
    write(HERE/'ANCHORS.json', checks)
    return data


def validation():
    import numpy as np
    cfg, _, fm, _, _ = context()
    assert not (HERE/'FROZEN_FUSION.json').exists(), 'Fusion already frozen'
    anchors('validation')
    data = collect('validation', reference_specs(cfg), 'validation_moments.npz', moments=True)
    q, n = data['family_daily_cross'], data['counts']
    years = data['origins'].astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    order = cfg['family_order']; methods = {}; full = {}; traces = {}
    for method in cfg['simplicity_order']:
        folds=[]
        for fit_year, held_year in [(2013,2017),(2017,2013)]:
            fit, held = years==fit_year, years==held_year
            if method == 'C4':
                choice=forward_select(q[fit].sum(0),n[fit],order)
                w=np.broadcast_to(choice['weights'],(12,len(order))).copy()
            else:
                choice=None; w=fm.fit_weights(q[fit].sum(0),method)
            folds.append({'fit_year':fit_year,'held_out_year':held_year,'weights':w.tolist(),
                          'fit_RMSE':fm.rmse_from_moments(q[fit].sum(0),n[fit],w),
                          'held_out_RMSE':fm.rmse_from_moments(q[held].sum(0),n[held],w),'forward_selection':choice})
        methods[method]={'folds':folds,'mean_held_out_RMSE':float(np.mean([f['held_out_RMSE'] for f in folds])),
                         'weight_difference_L1_by_lead':np.abs(np.array(folds[0]['weights'])-folds[1]['weights']).sum(1).tolist()}
        if method == 'C4':
            traces[method]=forward_select(q.sum(0),n,order)
            full[method]=np.broadcast_to(traces[method]['weights'],(12,len(order))).copy()
        else:
            full[method]=fm.fit_weights(q.sum(0),method)
    prior=read(ROOT/'experiments/exp0014/FROZEN_FUSION.json'); previous=prior['cross_validation'][prior['adopted_method']]
    for fold in previous['folds']:
        w=expanded(fold['weights'],prior['family_order'],order); held=years==fold['held_out_year']
        assert abs(fm.rmse_from_moments(q[held].sum(0),n[held],w)-fold['held_out_RMSE'])<=1e-6
    g=reference_specs(cfg)['G5']; gfold=[fm.rmse_from_moments(q[years==y].sum(0),n[years==y],g) for y in [2017,2013]]
    gscore=float(np.mean(gfold)); oldscore=previous['mean_held_out_RMSE']
    retained=keep_reference(gscore,oldscore); refscore=gscore if retained=='G5' else oldscore
    chosen=choose_method({m:r['mean_held_out_RMSE'] for m,r in methods.items()},tuple(cfg['simplicity_order']))
    newscore=methods[chosen]['mean_held_out_RMSE']; replaces=newscore<=refscore*(1.-.001)
    if replaces:
        method=chosen; w=full[chosen]; candidate_scope=list(range(len(order))); origin='new_candidate'
    elif retained=='G5':
        method='G5'; w=g; candidate_scope=[0]; origin='retained_G5'
    else:
        method=prior['adopted_method']; w=reference_specs(cfg)['exp0014_adopted']
        candidate_scope=[order.index(f) for f in prior['family_order']]; origin='retained_exp0014'
    drops={}
    for index,family in enumerate(order):
        if not np.any(w[:,index]>0):
            drops[family]={'status':'zero_weight_or_unselected_contribution_zero','weights':w.tolist()};continue
        selected=[j for j in candidate_scope if j!=index and (method in ('C2','C3') or np.any(w[:,j]>0))]
        if not selected:
            drops[family]={'status':'undefined_empty_after_deletion','weights':None};continue
        dw=fm.fit_weights(q.sum(0),method if method in ('C2','C3') else 'C1',selected)
        drops[family]={'status':'same_algorithm_validation_refit' if method in ('C2','C3') else 'remaining_selected_equal_weight','weights':dw.tolist()}
    total=q.sum((0,1)); count=n.sum(); bias=data['family_error_sum'].sum((0,1))/count
    cross=total/count; covariance=cross-bias[:,None]*bias[None,:]; variance=np.diag(covariance)
    structure={'family_order':order,'bias':bias.tolist(),'variance':variance.tolist(),'error_cross_moment':cross.tolist(),
               'centered_covariance':covariance.tolist(),'Pearson':(covariance/np.sqrt(variance[:,None]*variance[None,:])).tolist(),
               'note':'Low correlation alone is not contribution; validation-fitted ablations quantify contribution.'}
    write(HERE/'VALIDATION_ERROR_STRUCTURE.json',structure)
    frozen={'experiment':'exp0017','source_commit':read(HERE/'RUN_IDENTITY.json')['source_commit'],
            'family_order':order,'seeds':cfg['seeds'],'cross_validation':methods,'new_candidate_method':chosen,
            'retained_reference':retained,'retained_reference_score':refscore,'G5_fold_scores':gfold,'G5_cross_year_score':gscore,
            'exp0014_cross_year_score':oldscore,'exp0014_frozen_sha256':sha(ROOT/'experiments/exp0014/FROZEN_FUSION.json'),
            'new_candidate_relative_improvement':(refscore-newscore)/refscore,'new_candidate_replaces_retained':replaces,
            'adoption_origin':origin,'method':method,'adopted_method':method,'adopted_cross_year_score':newscore if replaces else refscore,
            'candidate_scope':[order[i] for i in candidate_scope],'candidate_full_validation_weights':{m:v.tolist() for m,v in full.items()},
            'full_validation_weights':w.tolist(),'weights':w.tolist(),'drop_family':drops,'forward_full_fit':traces,
            'prediction_manifest_sha256':sha(HERE/'FROZEN_PREDICTIONS.json'),
            'validation_statistics_sha256':sha(HERE/'artifacts/validation_moments.npz'),
            'selection_rules':cfg['selection'],'frozen_before_new_evaluation':True,'limitations':cfg['caveats']}
    write(HERE/'FROZEN_FUSION.json',frozen)
    print(json.dumps({'scores':{m:r['mean_held_out_RMSE'] for m,r in methods.items()},'adopted':method,'origin':origin}),flush=True)


def regional(anchor_only):
    import numpy as np
    cfg,refcfg,fm,score,module=context(); ref=module('experiments/exp0011/score.py','_region')
    manifest=read(HERE/'FROZEN_PREDICTIONS.json'); frozen=read(HERE/'FROZEN_FUSION.json')
    specs=reference_specs(cfg); names=list(specs) if anchor_only else ['adopted']
    if anchor_only:
        active=[f for i,f in enumerate(cfg['family_order']) if any(np.any(w[:,i]) for w in specs.values())]
        ts_global,readers=fm.family_readers(manifest,'extension_evaluation',active)
    else:
        ts_global=np.load(HERE/'artifacts/extension_evaluation_origins.npy',allow_pickle=False)
        # Region scores use the same float64 mathematical fusion as global scores.
        active=cfg['family_order']; _,readers=fm.family_readers(manifest,'extension_evaluation',active)
    position={int(t):i for i,t in enumerate(ts_global)}; output={}
    for region in ('CN','JP'):
        ts=ref.origins(region,'evaluation'); mapping=ref.mapping(region); disturbed=fm.disturbance(refcfg,ts)
        stats=np.zeros((len(names),len(ts),4,12,4))
        for i,t in enumerate(ts):
            y,valid,_=ref.day(region,'evaluation',t); label,prior=ref.gim(t),ref.gim(t-86400)
            valid=valid & np.isfinite(label[:,mapping[0]]).all(1) & np.isfinite(prior[:,mapping[0]]).all(1)
            means=fm.family_means(readers,position[int(t)],active).reshape(len(active),12,5183)
            weights=[w[:,[cfg['family_order'].index(f) for f in active]] for w in specs.values()] if anchor_only else [frozen['full_validation_weights']]
            forecasts=np.stack([ref.remap(fm.fuse(means,w),mapping) for w in weights])
            masks=[valid,valid,valid&(y>=60),valid&disturbed[i]]
            stats[:,i]=fm.stats_from_errors(np.where(valid,forecasts-y.astype(float),0.),masks)
        assert int(stats[0,:,0,:,0].sum())=={'CN':36099959,'JP':3770496}[region]
        payload={'origins':ts,'methods':np.array(names),'stats':stats}
        table=tables(payload)
        for value in table.values():value.pop('unique_longitude')
        np.savez_compressed(HERE/'artifacts'/f'regional_{region}_{"anchors" if anchor_only else "formal"}.npz',**payload)
        if anchor_only:
            checks=read(HERE/'ANCHORS.json')
            old11=read(ROOT/'experiments/exp0011/result.json')['regions'][region]['evaluation']['methods']
            old14=read(ROOT/'experiments/exp0014/result.json')['regional_zero_shot'][region]['methods']
            for name in names:
                expected=(old14['adopted'] if name=='exp0014_adopted' else old11[name])
                expected=expected['all']['RMSE'] if 'all' in expected else expected['RMSE']
                fm.check_anchor(region+'/'+name,table[name]['all']['RMSE'],expected,checks)
            write(HERE/'ANCHORS.json',checks)
        output[region]={'days':len(ts),'methods':table,'support':'exp0011 valid AND current/previous GIM finite bilinear nodes'}
    return output


def evaluate():
    import numpy as np
    cfg,_,fm,score,_=context()
    frozen=read(HERE/'FROZEN_FUSION.json')
    assert frozen['prediction_manifest_sha256']==sha(HERE/'FROZEN_PREDICTIONS.json')
    assert not (HERE/'FORMAL_RESULT_BEFORE_ORACLE.json').exists()
    for split in fm.SPLITS[1:]:anchors(split)
    region_reference=regional(True)
    write(HERE/'EVALUATION_ANCHORS_PASSED.json',{'status':'passed_before_new_scores','freeze_sha256':sha(HERE/'FROZEN_FUSION.json')})
    specs=reference_specs(cfg)
    specs.update({m:np.array(w) for m,w in frozen['candidate_full_validation_weights'].items()})
    specs['adopted']=np.array(frozen['full_validation_weights'])
    for family,row in frozen['drop_family'].items():
        if row['weights'] is not None:specs['drop_'+family]=np.array(row['weights'])
    result={'experiment':'exp0017','status':'formal_scored_before_oracle','source_commit':frozen['source_commit'],
            'family_order':cfg['family_order'],'adopted_method':frozen['method'],'adoption_origin':frozen['adoption_origin'],
            'cross_validation':frozen['cross_validation'],'weights':frozen['full_validation_weights'],
            'main_table':{},'single_model_five_seed_mean':{},'paired_time_intervals':{},'drop_family_contribution':{},
            'validation_error_structure':read(HERE/'VALIDATION_ERROR_STRUCTURE.json'),'limitations':cfg['caveats'],
            'prediction_precision':'mathematical fusion and scores float64; delivered forecasts float32; independent delivery metric agreement tolerance 1e-6 TECU'}
    for split in fm.SPLITS:
        data=collect(split,specs,f'{split}_formal.npz',moments=split=='extension_evaluation',export=True)
        table=tables(data);result['main_table'][split]=table
        result['single_model_five_seed_mean'][split]=score.tables({g:data['single_G'][:,:,h] for h,g in enumerate(fm.GROUPS)},data['origins'],cfg['seeds'])
        adopted=data['stats'][data['methods'].tolist().index('adopted'):data['methods'].tolist().index('adopted')+1,:,0]
        pairs={}
        for name in cfg['paired_references']:
            if name=='G_single_five_seed_metric_mean':reference=data['single_G'][:,:,0]
            else:
                j=data['methods'].tolist().index(name);reference=data['stats'][j:j+1,:,0]
            pairs['adopted-'+name]=deterministic_pair(adopted,reference,data['origins'])
        result['paired_time_intervals'][split]=pairs
        result['drop_family_contribution'][split]={f:{'status':row['status'],
            'RMSE':table['drop_'+f]['all']['RMSE'] if row['weights'] is not None else None,
            'delta_RMSE_removed_minus_adopted':(0. if row['status'].startswith('zero_weight') else table['drop_'+f]['all']['RMSE']-table['adopted']['all']['RMSE']) if row['weights'] is not None else None}
            for f,row in frozen['drop_family'].items()}
    result['regional_zero_shot']=regional(False)
    for region in ('CN','JP'):result['regional_zero_shot'][region]['methods'].update(region_reference[region]['methods'])
    result['anchors']=read(HERE/'ANCHORS.json')
    write(HERE/'FORMAL_RESULT_BEFORE_ORACLE.json',result)


def oracle():
    import numpy as np
    cfg,_,fm,_,_=context();result=read(HERE/'FORMAL_RESULT_BEFORE_ORACLE.json')
    assert not (HERE/'result.json').exists()
    q=load_stats('extension_evaluation_formal.npz')['family_daily_cross'].sum(0)
    weights={m:fm.fit_weights(q,m) for m in ('C2','C3')}
    data=collect('extension_evaluation',{'oracle_'+m:w for m,w in weights.items()},'oracle.npz')
    table=tables(data)
    result['posthoc_oracle']={'status':'posthoc_diagnostic_only','weights':{m:w.tolist() for m,w in weights.items()},'table':table,
         'adopted_minus_oracle_RMSE':{m:result['main_table']['extension_evaluation']['adopted']['all']['RMSE']-table['oracle_'+m]['all']['RMSE'] for m in weights},
         'formal_result_sha256':sha(HERE/'FORMAL_RESULT_BEFORE_ORACLE.json'),'no_feedback_into_selection':True}
    result['status']='scored_pending_independent_recheck';write(HERE/'result.json',result)


def main():
    global ROOT,HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check-upstreams", "register-upstreams",'inventory','validation','evaluate','oracle'])
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    if args.repo:
        ROOT=args.repo.resolve();HERE=ROOT/'experiments/exp0017'
    if args.mode in ('inventory','validation','evaluate','oracle'):
        (HERE/'artifacts').mkdir(exist_ok=True)
        globals()[args.mode]();return


if __name__ == "__main__":
    main()
