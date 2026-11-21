"""One-shot CPU exp0019. Explicit stages; no network, GPU or checkpoint loading."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ROOT = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
HERE = ROOT / 'experiments/exp0019'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def committed(path):
    return None  # git-HEAD binding removed in the release

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def audit(event, args):
    if event in ('socket.__new__', 'socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network forbidden in exp0019')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing and not p.is_relative_to(HERE):
            raise RuntimeError('Output outside exp0019: ' + str(p))
        if not writing and (p.name.startswith('TEC_2026') or ('2026' in p.name and p.suffix in ('.nc', '.ionex'))):
            raise RuntimeError('2026 scientific observations forbidden')

CFG = read(HERE / 'config.json')
BASE = module(Path(__file__).resolve().parent / 'final_fusion.py', '_exp0017_reused')
_, REFCFG, FM, SCORE, FROZEN_MODULE = BASE.context()
BASE.HERE = HERE
BASE.context = lambda: (CFG, REFCFG, FM, SCORE, FROZEN_MODULE)
ORDER = CFG['family_order']

def identity():
    record = read(HERE / 'RUN_IDENTITY.json')
    return record

def references():
    old = read(ROOT / 'experiments/exp0017/FROZEN_FUSION.json')
    g = np.zeros((12, len(ORDER))); g[:, ORDER.index('G')] = 1
    return {'exp0017_adopted': BASE.expanded(old['full_validation_weights'], old['family_order'], ORDER), 'G5': g}

def prepare():
    identity()
    h = read(ROOT / 'experiments/exp0018/HANDOFF.json')
    assert h['status'] == 'complete' and h['ready_for_exp0019'] and h['verification_passed'] and h['resources_closed']
    assert h['formal_jobs_completed'] == 15 and h['epochs_per_job'] == 200 and h['active_owned_gpus'] == 0
    assert not h['unperformed_requested_items']
    committed(ROOT / 'experiments/exp0018/HANDOFF.json')
    for rel, digest in h['ready_files'].items():
        assert sha(ROOT / rel) == digest, rel
        committed(ROOT / rel)
    resources = BASE.verify_resources('exp0018', h)
    assert resources['resources_checked'] == 17
    arms, hashes, manifests = {}, {}, {}
    for name in ('exp0017', 'exp0018'):
        p = ROOT / 'experiments' / name / 'FROZEN_PREDICTIONS.json'
        committed(p); m = read(p); manifests[str(p)] = sha(p)
        for family, records in m['arms'].items():
            assert family not in arms
            arms[family] = records
        for path, digest in m['files_sha256'].items():
            assert path not in hashes or hashes[path] == digest
            hashes[path] = digest
    assert set(arms) == set(ORDER)
    for p in ('FROZEN_FUSION.json', 'FUSED_PREDICTIONS.json', 'result.json'):
        path = ROOT / 'experiments/exp0017' / p
        committed(path); hashes[str(path)] = sha(path)
    hashes.update(read(ROOT / 'experiments/exp0017/FUSED_PREDICTIONS.json')['files_sha256'])
    def verify(item):
        path, digest = item
        assert sha(path) == digest, path
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(verify, hashes.items()), 1):
            if i % 75 == 0: print('input hashes', i, '/', len(hashes), flush=True)
    manifest = {'experiment': 'exp0019', 'family_order': ORDER, 'seeds': CFG['seeds'],
                'arms': {f: arms[f] for f in ORDER}, 'files_sha256': hashes,
                'upstream_manifests_sha256': manifests, 'unit': 'TECU', 'grid': REFCFG['grid'], 'input_identity_verified': True}
    rows = []
    for split in FM.SPLITS:
        ts, readers = FM.family_readers(manifest, split)
        np.testing.assert_array_equal(ts, FM.expected_origins(split))
        for family, values in readers.items():
            for seed, reader in zip(CFG['seeds'], values):
                assert reader.shape == (len(ts), 12, 71, 73)
                rows.append({'family': family, 'seed': seed, 'split': split, 'shape': list(reader.shape), 'dtype': str(reader.dtype), 'origins_equal': True, 'unit': 'TECU'})
                rec = next(x for x in arms[family] if x['seed'] == seed)['splits'][split]
                for part in rec.get('parts', [rec]):
                    pt = FM.load_array(part['origins'], part.get('origins_key'))
                    if 'target_times' in part:
                        np.testing.assert_array_equal(FM.load_array(part['target_times'], part.get('target_times_key')), pt[:, None] + np.arange(12)*7200)
    write(HERE / 'FROZEN_PREDICTIONS.json', manifest)
    write(HERE / 'INPUT_AUDIT.json', {'status': 'passed', 'files_hashed': len(hashes), 'records': rows, 'evaluation_errors_computed': False})
    write(HERE / 'UPSTREAM_VERIFIED.json', {'status': 'passed', 'checked_at_utc': now(), 'handoff_sha256': sha(ROOT / 'experiments/exp0018/HANDOFF.json'), 'formal_jobs_completed': 15, 'resource_verification': resources, 'cpu_only_local_verification': True})

def anchor_check(split, table):
    old = read(ROOT / 'experiments/exp0017/result.json')['main_table'][split]
    checks = read(HERE / 'ANCHORS.json') if (HERE / 'ANCHORS.json').exists() else {}
    for key, target in [('exp0017_adopted', 'adopted'), ('G5', 'G5')]:
        FM.check_anchor(split + '/' + key, table[key]['all']['RMSE'], old[target]['all']['RMSE'], checks)
    write(HERE / 'ANCHORS.json', checks)

def c4_crossyear(q, n, years, allowed):
    chosen = [ORDER.index('G')]; trace = []
    def weights(ids):
        w = np.zeros((12, len(ORDER))); w[:, ids] = 1 / len(ids)
        return w
    def score(ids):
        w = weights(ids)
        return float(np.mean([FM.rmse_from_moments(q[years == y].sum(0), n[years == y], w) for y in (2017, 2013)]))
    current = score(chosen)
    while len(chosen) < len(allowed):
        alternatives = [(score(chosen + [j]), j) for j in allowed if j not in chosen]
        best, j = min(alternatives)
        accepted = best <= current * .999
        trace.append({'candidate': ORDER[j], 'before_cross_year_RMSE': current, 'after_cross_year_RMSE': best, 'relative_improvement': (current-best)/current, 'accepted': bool(accepted)})
        if not accepted: break
        chosen.append(j); current = best
    return weights(chosen), {'selected_families': [ORDER[i] for i in chosen], 'cross_year_RMSE': current, 'trace': trace}

def validation():
    ident = identity()
    data = BASE.collect('validation', references(), 'validation_moments.npz', moments=True)
    anchor_check('validation', BASE.tables(data))
    q, n = data['family_daily_cross'], data['counts']
    years = data['origins'].astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    total = q.sum((0, 1)); count = n.sum(); bias = data['family_error_sum'].sum((0, 1))/count
    cross = total/count; covariance = cross-bias[:, None]*bias[None, :]; variance = np.diag(covariance)
    write(HERE / 'VALIDATION_ERROR_STRUCTURE.json', {'family_order': ORDER, 'bias': bias.tolist(), 'variance': variance.tolist(), 'error_cross_moment': cross.tolist(), 'centered_covariance': covariance.tolist(), 'Pearson': (covariance/np.sqrt(variance[:, None]*variance[None, :])).tolist()})
    old = read(ROOT / 'experiments/exp0017/FROZEN_FUSION.json')
    for fold in old['cross_validation'][old['adopted_method']]['folds']:
        w = BASE.expanded(fold['weights'], old['family_order'], ORDER)
        select = years == fold['held_out_year']
        assert abs(FM.rmse_from_moments(q[select].sum(0), n[select], w)-fold['held_out_RMSE']) <= 1e-6
    for system, spec in CFG['systems'].items():
        allowed = [i for i, f in enumerate(ORDER) if f not in spec['exclude']]
        methods, full = {}, {}
        c4w, trace = c4_crossyear(q, n, years, allowed)
        for method in CFG['simplicity_order']:
            folds = []
            for fit_year, held_year in [(2013, 2017), (2017, 2013)]:
                fit, held = years == fit_year, years == held_year
                w = c4w.copy() if method == 'C4' else FM.fit_weights(q[fit].sum(0), method, allowed)
                folds.append({'fit_year': fit_year, 'held_out_year': held_year, 'weights': w.tolist(), 'fit_RMSE': FM.rmse_from_moments(q[fit].sum(0), n[fit], w), 'held_out_RMSE': FM.rmse_from_moments(q[held].sum(0), n[held], w)})
            methods[method] = {'folds': folds, 'mean_held_out_RMSE': float(np.mean([f['held_out_RMSE'] for f in folds]))}
            full[method] = c4w.copy() if method == 'C4' else FM.fit_weights(q.sum(0), method, allowed)
        chosen = BASE.choose_method({m: x['mean_held_out_RMSE'] for m, x in methods.items()})
        w = full[chosen]; drops = {}
        if system == 'S1':
            for index, family in enumerate(ORDER):
                if not np.any(w[:, index] > 0):
                    drops[family] = {'status': 'zero_weight_or_unselected_contribution_zero', 'weights': w.tolist()}; continue
                selected = [j for j in allowed if j != index and (chosen in ('C2', 'C3') or np.any(w[:, j] > 0))]
                dw = FM.fit_weights(q.sum(0), chosen if chosen in ('C2', 'C3') else 'C1', selected) if selected else None
                drops[family] = {'status': 'same_algorithm_validation_refit' if chosen in ('C2', 'C3') else 'remaining_selected_equal_weight', 'weights': dw.tolist() if dw is not None else None}
        score = methods[chosen]['mean_held_out_RMSE']
        frozen = {'experiment': 'exp0019', 'system': system, 'source_commit': ident['source_commit'], 'frozen_at_utc': now(), 'family_order': ORDER, 'eligible_families': [ORDER[i] for i in allowed], 'seeds': CFG['seeds'], 'cross_validation': methods, 'adopted_method': chosen, 'adopted_cross_year_RMSE': score, 'full_validation_weights': w.tolist(), 'candidate_full_validation_weights': {m: a.tolist() for m, a in full.items()}, 'forward_selection': trace, 'drop_family': drops, 'prediction_manifest_sha256': sha(HERE / 'FROZEN_PREDICTIONS.json'), 'validation_statistics_sha256': sha(HERE / 'artifacts/validation_moments.npz'), 'config_sha256': sha(HERE / 'config.json'), 'frozen_before_evaluation': True}
        if system == 'S1':
            frozen.update(replaces_current_best=bool(score <= 2.086171*.999), replacement_reference_RMSE=2.086171, exact_exp0017_cross_year_RMSE=old['adopted_cross_year_score'])
        write(HERE / f'FROZEN_FUSION_{system}.json', frozen)
        print(json.dumps({'system': system, 'adopted': chosen, 'CV': {m: r['mean_held_out_RMSE'] for m, r in methods.items()}, 'active_families': [ORDER[i] for i in range(len(ORDER)) if np.any(w[:, i])]}), flush=True)

def freezes():
    identity()
    out = {}
    for system in CFG['systems']:
        p = HERE / f'FROZEN_FUSION_{system}.json'; committed(p); out[system] = read(p)
        assert out[system]['prediction_manifest_sha256'] == sha(HERE / 'FROZEN_PREDICTIONS.json')
        assert out[system]['config_sha256'] == sha(HERE / 'config.json')
    return out

def regional(specs, suffix, anchors=False):
    ref = FROZEN_MODULE('experiments/exp0011/score.py', '_exp0011_regions')
    manifest = read(HERE / 'FROZEN_PREDICTIONS.json')
    active = [f for i, f in enumerate(ORDER) if any(np.any(w[:, i]) for w in specs.values())]
    ts_global, readers = FM.family_readers(manifest, 'extension_evaluation', active)
    position = {int(t): i for i, t in enumerate(ts_global)}; output = {}; names = list(specs)
    for region in ('CN', 'JP'):
        ts = ref.origins(region, 'evaluation'); mapping = ref.mapping(region); disturbed = FM.disturbance(REFCFG, ts)
        stats = np.zeros((len(names), len(ts), 4, 12, 4))
        for i, t in enumerate(ts):
            y, valid, _ = ref.day(region, 'evaluation', t); label, prior = ref.gim(t), ref.gim(t-86400)
            valid = valid & np.isfinite(label[:, mapping[0]]).all(1) & np.isfinite(prior[:, mapping[0]]).all(1)
            p = FM.family_means(readers, position[int(t)], active).reshape(len(active), 12, 5183)
            forecasts = np.stack([ref.remap(FM.fuse(p, w[:, [ORDER.index(f) for f in active]]), mapping) for w in specs.values()])
            stats[:, i] = FM.stats_from_errors(np.where(valid, forecasts-y.astype(float), 0.), [valid, valid, valid & (y >= 60), valid & disturbed[i]])
        assert int(stats[0, :, 0, :, 0].sum()) == {'CN': 36099959, 'JP': 3770496}[region]
        payload = {'origins': ts, 'methods': np.array(names), 'stats': stats}; table = BASE.tables(payload)
        for value in table.values(): value.pop('unique_longitude')
        np.savez_compressed(HERE / 'artifacts' / f'regional_{region}_{suffix}.npz', **payload)
        if anchors:
            checks = read(HERE / 'ANCHORS.json')
            old = read(ROOT / 'experiments/exp0017/result.json')['regional_zero_shot'][region]['methods']['adopted']
            FM.check_anchor(region + '/exp0017_adopted', table['exp0017_adopted']['all']['RMSE'], old['all']['RMSE'], checks)
            write(HERE / 'ANCHORS.json', checks)
        output[region] = {'days': len(ts), 'methods': table, 'support': 'exact exp0011 validity AND current/previous GIM finite bilinear nodes'}
    return output

def evaluation_anchors():
    freezes()
    for split in FM.SPLITS[1:]:
        data = BASE.collect(split, references(), f'anchors_{split}.npz')
        anchor_check(split, BASE.tables(data))
    r = regional({'exp0017_adopted': references()['exp0017_adopted']}, 'anchors', anchors=True)
    write(HERE / 'REGIONAL_ANCHORS.json', r)
    write(HERE / 'EVALUATION_ANCHORS_PASSED.json', {'status': 'passed_before_new_formal_scores', 'frozen_weights_sha256': {s: sha(HERE / f'FROZEN_FUSION_{s}.json') for s in CFG['systems']}, 'checked_at_utc': now()})

def collect_export(split, specs):
    """exp0017 collection arithmetic, exporting the two prespecified fusions together."""
    manifest = read(HERE / 'FROZEN_PREDICTIONS.json'); names = list(specs)
    ts, readers = FM.family_readers(manifest, split)
    np.testing.assert_array_equal(ts, FM.expected_origins(split))
    stats = np.zeros((len(names), len(ts), 4, 12, 4)); single = np.zeros((5, len(ts), 4, 12, 4))
    moments = split == 'extension_evaluation'
    cross = np.zeros((len(ts), 12, len(ORDER), len(ORDER))) if moments else None
    counts = np.zeros((len(ts), 12)); cache = {}; disturbed = FM.disturbance(REFCFG, ts)
    destinations = {}; rounding = {s: 0. for s in CFG['systems']}
    for system in CFG['systems']:
        folder = HERE / 'artifacts' / system; folder.mkdir(exist_ok=True)
        path = folder / f'{split}_predictions.npy'
        assert not path.exists(), 'Never overwrite a final prediction'
        destinations[system] = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(ts), 12, 71, 73))
        np.save(folder / f'{split}_origins.npy', ts)
    for i, t in enumerate(ts):
        p = FM.family_means(readers, i, ORDER).reshape(len(ORDER), 12, 5183)
        truth = FM.truth_day(REFCFG, t, cache); valid = np.isfinite(truth)
        assert np.isfinite(p[:, valid]).all()
        unique = valid.reshape(12, 71, 73).copy(); unique[..., -1] = False
        masks = [valid, unique.reshape(12, 5183), valid & (truth >= 60), valid & disturbed[i]]
        forecasts = np.stack([FM.fuse(p, w) for w in specs.values()])
        stats[:, i] = FM.stats_from_errors(np.where(valid, forecasts-truth, 0.), masks)
        g = np.stack([a[i].reshape(12, 5183).astype(float) for a in readers['G']])
        single[:, i] = FM.stats_from_errors(np.where(valid, g-truth, 0.), masks)
        counts[i] = valid.sum(1)
        for system, dest in destinations.items():
            prediction = forecasts[names.index(system)].reshape(12, 71, 73); dest[i] = prediction
            rounding[system] = max(rounding[system], float(np.max(np.abs(dest[i].astype(float)-prediction))))
        if moments:
            error = np.where(valid, p-truth, 0.).transpose(1, 0, 2); cross[i] = error @ error.transpose(0, 2, 1)
        if i % 200 == 0: print('formal once', split, i, '/', len(ts), flush=True)
    for dest in destinations.values(): dest.flush()
    payload = {'origins': ts, 'methods': np.array(names), 'stats': stats, 'single_G': single, 'counts': counts, 'groups': np.array(FM.GROUPS), 'fields': np.array(FM.FIELDS)}
    if moments: payload['family_daily_cross'] = cross
    np.savez_compressed(HERE / 'artifacts' / f'{split}_formal.npz', **payload)
    return payload, rounding

def evaluate():
    frozen = freezes(); assert read(HERE / 'EVALUATION_ANCHORS_PASSED.json')['status'] == 'passed_before_new_formal_scores'
    specs = references(); specs.update({s: np.array(f['full_validation_weights']) for s, f in frozen.items()})
    for family, row in frozen['S1']['drop_family'].items():
        if row['weights'] is not None: specs['drop_' + family] = np.array(row['weights'])
    result = {'experiment': 'exp0019', 'status': 'formal_scored_before_oracle', 'source_commit': identity()['source_commit'], 'family_order': ORDER, 'systems': {s: {'adopted_method': f['adopted_method'], 'cross_year_RMSE': f['adopted_cross_year_RMSE'], 'weights': f['full_validation_weights'], 'eligible_families': f['eligible_families']} for s, f in frozen.items()}, 'current_best': 'exp0019_S1' if frozen['S1']['replaces_current_best'] else 'exp0017', 'current_best_replaced': frozen['S1']['replaces_current_best'], 'main_table': {}, 'single_model_five_seed_mean': {}, 'paired_time_intervals': {}, 'drop_family_contribution': {}, 'float32_max_value_rounding_TECU': {}, 'limitations': CFG['caveats'], 'precision': CFG['precision']}
    for split in FM.SPLITS:
        data, rounding = collect_export(split, specs); table = BASE.tables(data)
        table['G_single_seed42'] = BASE.strip_seed_labels(SCORE.tables({g: data['single_G'][:1, :, h] for h, g in enumerate(FM.GROUPS)}, data['origins']))
        result['main_table'][split] = table
        result['single_model_five_seed_mean'][split] = SCORE.tables({g: data['single_G'][:, :, h] for h, g in enumerate(FM.GROUPS)}, data['origins'], CFG['seeds'])
        def stat(name):
            j = data['methods'].tolist().index(name)
            return data['stats'][j:j+1, :, 0]
        result['paired_time_intervals'][split] = {a+'-'+b: BASE.deterministic_pair(stat(a), stat(b), data['origins']) for a, b in [('S1', 'exp0017_adopted'), ('S2', 'S1'), ('S1', 'G5')]}
        result['drop_family_contribution'][split] = {f: {'status': row['status'], 'RMSE': table['drop_'+f]['all']['RMSE'] if row['weights'] is not None else None, 'delta_RMSE_removed_minus_S1': table['drop_'+f]['all']['RMSE']-table['S1']['all']['RMSE'] if row['weights'] is not None else None} for f, row in frozen['S1']['drop_family'].items()}
        result['float32_max_value_rounding_TECU'][split] = rounding
    region = regional({s: specs[s] for s in CFG['systems']}, 'formal')
    reference_region = read(HERE / 'REGIONAL_ANCHORS.json')
    for key in region: region[key]['methods'].update(reference_region[key]['methods'])
    result['regional_zero_shot'] = region; result['anchors'] = read(HERE / 'ANCHORS.json')
    result['validation_error_structure'] = read(HERE / 'VALIDATION_ERROR_STRUCTURE.json')
    write(HERE / 'FORMAL_RESULT_BEFORE_ORACLE.json', result)

def oracle():
    frozen = freezes(); result = read(HERE / 'FORMAL_RESULT_BEFORE_ORACLE.json')
    q = BASE.load_stats('extension_evaluation_formal.npz')['family_daily_cross'].sum(0)
    weights = {}
    for system, f in frozen.items():
        allowed = [ORDER.index(x) for x in f['eligible_families']]
        for method in ('C2', 'C3'): weights[system+'_'+method] = FM.fit_weights(q, method, allowed)
    data = BASE.collect('extension_evaluation', weights, 'posthoc_oracle.npz')
    table = BASE.tables(data)
    result['posthoc_oracle'] = {'status': 'posthoc_diagnostic_only_not_formal_results', 'fit_years': [2023, 2024, 2025], 'weights': {m: w.tolist() for m, w in weights.items()}, 'table': table, 'formal_minus_oracle_RMSE': {m: result['main_table']['extension_evaluation'][m[:2]]['all']['RMSE']-table[m]['all']['RMSE'] for m in weights}, 'formal_result_sha256': sha(HERE / 'FORMAL_RESULT_BEFORE_ORACLE.json'), 'no_feedback_into_selection': True}
    result['status'] = 'scored_pending_independent_recheck'; write(HERE / 'result.json', result)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validation', 'evaluation-anchors', 'evaluate', 'oracle'])
    stage = parser.parse_args().stage
    (HERE / 'artifacts').mkdir(exist_ok=True)
    globals()[stage.replace('-', '_')]()

if __name__ == '__main__':
    main()
