"""Frozen CPU fusion map and reusable readers. Importing this file opens no data.

Public API: load_array, PredictionReader, prediction_set, family_readers,
family_means, fuse, simplex_weights, fit_weights, rmse_from_moments.
All numerical prediction arithmetic is float64; source forecasts remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import numpy as np

SEEDS = [42, 1024, 2025, 3407, 5000]
FAMILIES = ['G', 'C', 'PI', 'EMP', 'DE-ZScore-c05']
SPLITS = ['validation', 'original_evaluation', 'extension_evaluation']
YEARS = {'validation': [2013, 2017], 'original_evaluation': [2015, 2019, 2020, 2021, 2022],
         'extension_evaluation': [2023, 2024, 2025]}
GROUPS = ['all', 'unique_longitude', 'high_TEC', 'disturbed_day']
FIELDS = ['n', 'sse', 'sae', 'error_sum']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def load_array(path, key=None):
    """Read NPY or a stored NPZ member without extraction or a duplicate cache.

    NPZ prediction_key is 'values'; its metadata array_name is 'prediction_tecu'.
    For compressed NPZ members only, numpy's ordinary in-memory reader is used.
    """
    path = Path(path)
    if path.suffix != '.npz':
        if key is not None:
            raise ValueError('A key was supplied for a non-NPZ array')
        return np.load(path, mmap_mode='r', allow_pickle=False)
    if key is None:
        raise ValueError('NPZ arrays require an explicit key')
    with zipfile.ZipFile(path) as archive:
        item = archive.getinfo(key + '.npy')
        if item.compress_type != zipfile.ZIP_STORED:
            with np.load(path, allow_pickle=False) as z:
                return z[key]
        offset = item.header_offset
    with path.open('rb') as handle:
        handle.seek(offset)
        local = handle.read(30)
        if local[:4] != b'PK\x03\x04':
            raise ValueError('Invalid ZIP local header')
        name_length, extra_length = struct.unpack('<HH', local[26:30])
        handle.seek(offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError('Unsupported embedded NPY version')
        data_offset = handle.tell()
    if dtype.hasobject:
        raise ValueError('Object arrays are forbidden')
    return np.memmap(path, mode='r', offset=data_offset, shape=shape, dtype=dtype,
                     order='F' if fortran else 'C')


class PredictionReader:
    """Canonical read-only [N,12,71,73] view over one split or ordered parts."""
    def __init__(self, split_record):
        self.parts = split_record.get('parts', [split_record])
        self.arrays = []
        timestamps = []
        for part in self.parts:
            if part.get('unit', split_record.get('unit', 'TECU')) != 'TECU':
                raise ValueError('Predictions must be in TECU')
            ts = np.asarray(load_array(part['origins'], part.get('origins_key')))
            a = load_array(part['predictions'], part.get('prediction_key'))
            if ts.dtype != np.int64 or ts.ndim != 1 or a.dtype != np.float32:
                raise ValueError('Expected int64 UTC origins and float32 predictions')
            if a.shape not in [(len(ts), 12, 71, 73), (len(ts), 12, 5183, 1)]:
                raise ValueError(f'Unsupported forecast shape {a.shape}')
            timestamps.append(ts)
            self.arrays.append(a.reshape(len(ts), 12, 71, 73))
        self.origins = np.concatenate(timestamps)
        self.ends = np.cumsum([len(x) for x in self.arrays])
        if len(set(self.origins.tolist())) != len(self.origins) or np.any(np.diff(self.origins) <= 0):
            raise ValueError('Repeated or unordered forecast origins')
        self.shape = (len(self.origins), 12, 71, 73)
        self.dtype = np.dtype('float32')

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if not np.isscalar(index):
            ids = np.arange(len(self))[index]
            return np.stack([self[int(i)] for i in ids])
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = int(np.searchsorted(self.ends, index, side='right'))
        start = 0 if part == 0 else int(self.ends[part - 1])
        return self.arrays[part][index - start]


def prediction_set(records, split):
    readers = [PredictionReader(row['splits'][split]) for row in records]
    if [r['seed'] for r in records] != SEEDS:
        raise ValueError('The exact five seeds in their frozen order are required')
    for reader in readers[1:]:
        np.testing.assert_array_equal(reader.origins, readers[0].origins)
    return readers[0].origins, readers


load_prediction_set = prediction_set


def family_readers(manifest, split, families=None):
    families = list(families or manifest['family_order'])
    readers = {}
    origins = None
    for name in families:
        ts, values = prediction_set(manifest['arms'][name], split)
        if origins is not None:
            np.testing.assert_array_equal(origins, ts)
        origins = ts
        readers[name] = values
    return origins, readers


def family_means(readers, index, family_order=None):
    return np.stack([np.mean(np.stack([np.asarray(a[index], np.float64)
                                      for a in readers[name]]), axis=0)
                     for name in (family_order or list(readers))])


def fuse(predictions, weights):
    """Fuse [family,12,...] with [family] or [12,family] weights."""
    p, w = np.asarray(predictions, np.float64), np.asarray(weights, np.float64)
    if p.ndim < 3 or p.shape[1] != 12:
        raise ValueError('Need a family x 12 leads prediction tensor')
    if w.ndim == 1:
        w = np.broadcast_to(w, (12, len(w)))
    if w.shape != (12, len(p)) or np.min(w) < -1e-12 or not np.allclose(w.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError('Weights must be nonnegative and sum to one per lead')
    return np.einsum('lf,fln->ln', w, p.reshape(len(p), 12, -1)).reshape(p.shape[1:])


def simplex_weights(gram):
    """Exact small convex QP by enumerating nonempty active supports.

    No ridge or data-dependent regularization. Equal numerical objectives
    (1e-12 relative scale) prefer fewer active families then frozen family order.
    An equality-constrained KKT system is solved on every support, including
    singleton endpoints. Enumeration is suitable for the frozen 5/10-family task.
    """
    q = np.asarray(gram, np.float64)
    q = (q + q.T) / 2
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.isfinite(q).all():
        raise ValueError('Invalid Gram matrix')
    scale = max(float(np.max(np.abs(q))), 1.)
    qn = q / scale
    n = len(q)
    best = None
    for k in range(1, n + 1):
        for support in itertools.combinations(range(n), k):
            sub = qn[np.ix_(support, support)]
            lhs = np.block([[sub, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            rhs = np.r_[np.zeros(k), 1.]
            try:
                solution = np.linalg.solve(lhs, rhs)[:k]
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(lhs, rhs, rcond=1e-13)[0][:k]
            if np.min(solution) < -1e-10 or abs(solution.sum() - 1) > 1e-8:
                continue
            solution = np.maximum(solution, 0)
            solution /= solution.sum()
            w = np.zeros(n)
            w[list(support)] = solution
            value = float(w @ qn @ w)
            if best is None or value < best[0] - 1e-12:
                best = (value, w)
    if best is None:
        raise ValueError('No feasible simplex solution')
    w = best[1]
    gradient = qn @ w
    level = float(w @ gradient)
    if np.min(gradient - level) < -2e-8 or np.max(np.abs((gradient - level)[w > 1e-8])) > 2e-8:
        raise ValueError('Simplex KKT verification failed')
    return w


def fit_weights(lead_gram, method, allowed=None):
    """Input [12,F,F] pooled uncentered error cross-sums; output [12,F]."""
    q = np.asarray(lead_gram, np.float64)
    n = q.shape[-1]
    allowed = list(range(n)) if allowed is None else list(allowed)
    if not allowed:
        raise ValueError('Empty fusion is undefined')
    out = np.zeros((12, n))
    sub = q[:, allowed][:, :, allowed]
    if method == 'C1':
        out[:, allowed] = 1 / len(allowed)
    elif method == 'C2':
        out[:, allowed] = simplex_weights(sub.sum(0))
    elif method == 'C3':
        out[:, allowed] = np.stack([simplex_weights(x) for x in sub])
    else:
        raise ValueError(method)
    return out


def rmse_from_moments(lead_gram, counts, weights):
    q, w = np.asarray(lead_gram), np.asarray(weights)
    if w.ndim == 1:
        w = np.broadcast_to(w, (12, len(w)))
    return float(np.sqrt(max(float(np.einsum('li,lij,lj->', w, q, w)), 0.) / np.sum(counts)))


def expected_origins(split):
    return np.concatenate([np.arange(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y+1}-01-01'),
                                     dtype='datetime64[D]') for y in YEARS[split]]).astype('datetime64[s]').astype(np.int64)


def truth_day(cfg, origin, cache):
    date = np.datetime64(int(origin), 's').astype('datetime64[D]')
    year = int(str(date)[:4])
    if year > 2025 or str(year) not in cfg['data_files']:
        raise ValueError('Forbidden or undeclared observation year')
    if year not in cache:
        cache[year] = np.load(cfg['data_files'][str(year)]['path'], mmap_mode='r', allow_pickle=False)
    index = int((date - np.datetime64(f'{year}-01-01')).astype(int))
    raw = cache[year][index]
    if raw.dtype != np.float32 or raw.shape != (12, 71, 73):
        raise ValueError('Raw GIM schema mismatch')
    np.testing.assert_array_equal(raw[..., 0], raw[..., -1])
    return (np.asarray(raw, np.float32) / np.float32(10)).reshape(12, 5183).astype(np.float64)


def disturbance(cfg, origins):
    with np.load(cfg['driver_path'], allow_pickle=False) as z:
        times = z['time'].astype(np.int64)
        query = origins[:, None] + np.arange(12) * 7200
        ids = np.searchsorted(times, query)
        np.testing.assert_array_equal(times[ids], query)
        kp, dst = z['Kp'][ids], z['Dst'][ids]
    if not np.isfinite(kp).all() or not np.isfinite(dst).all():
        raise ValueError('Missing disturbance classification')
    return np.any((kp >= 5) | (dst <= -50), axis=1)


def inventory(cfg, out):
    """Hashes, schema, dates, grid/units and provenance only; no skill scores."""
    repo = Path(cfg['repo'])
    prior = read(repo / 'experiments/exp0013/FROZEN_PREDICTIONS.json')
    source = Path(cfg['legacy_source_index'])
    source_rows = read(source)
    legacy = Path(cfg['legacy_study'])
    checkpoints = read(legacy / 'final_freeze/CHECKPOINTS.json')['checkpoints']
    cp = {(r['method'], r['seed']): r for r in checkpoints}
    hashes = {}
    audit = []

    def checked(path, expected=None):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = sha(path)
        if expected is not None and hashes[path] != expected:
            raise ValueError('Input hash mismatch: ' + path)
        return hashes[path]

    for path in cfg['reference_sources']:
        checked(path, cfg['reference_sources'][path])
    checked(source)
    checked(legacy / 'final_freeze/CHECKPOINTS.json')
    arms = {a: json.loads(json.dumps(prior['arms'][a])) for a in ('G', 'PI', 'EMP')}
    arms['C'] = [dict(arm='C', seed=s, splits={split: dict(
        predictions=str(Path(cfg['cg_root']) / f'C_{s}' / f'{split}_predictions.npy'),
        origins=str(Path(cfg['cg_root']) / f'C_{s}' / f'{split}_origins.npy')) for split in SPLITS}) for s in SEEDS]
    for name in ('G', 'C', 'PI', 'EMP'):
        for row in arms[name]:
            for split, part in row['splits'].items():
                part.update(unit='TECU', layout='N,12,71,73; latitude-major C order',
                            grid='lat 87.5..-87.5 step -2.5; lon -180..180 step 5; inclusive endpoints')
                for key in ('predictions', 'origins'):
                    checked(part[key], prior['files_sha256'].get(part[key]))
                reader = PredictionReader(part)
                np.testing.assert_array_equal(reader.origins, expected_origins(split))
                audit.append(dict(family=name, seed=row['seed'], split=split, shape=list(reader.shape),
                                  dtype=str(reader.dtype), unit='TECU', origins_equal=True))
    arms['DE-ZScore-c05'] = []
    reference_only = []

    def extension_record(method, seed):
        weight = cp[method, seed]
        checked(weight['checkpoint'], weight['checkpoint_sha256'])
        parts = []
        for year in YEARS['extension_evaluation']:
            folder = legacy / 'evaluation/native' / f'{method}-seed{seed}' / str(year)
            manifest = read(folder / 'manifest.json')
            checked(folder / 'manifest.json')
            assert manifest['method'] == method and manifest['seed'] == seed and manifest['year'] == year
            assert manifest['unit'] == 'TECU' and manifest['checkpoint_sha256'] == weight['checkpoint_sha256']
            assert manifest['selected_epoch'] == weight['selected_epoch']
            shared = legacy / 'evaluation/native' / manifest['shared_reference']
            for key in ('forecast_origin_utc.npy', 'target_time_utc.npy', 'history_time_utc.npy'):
                checked(shared / key, manifest['shared_file_sha256'][key])
            ts = load_array(shared / 'forecast_origin_utc.npy')
            np.testing.assert_array_equal(load_array(shared / 'target_time_utc.npy'), ts[:, None] + np.arange(12) * 7200)
            np.testing.assert_array_equal(load_array(shared / 'history_time_utc.npy'), ts[:, None] - np.arange(12, 0, -1) * 7200)
            checked(folder / 'prediction_tecu.npy', manifest['file_sha256']['prediction_tecu.npy'])
            parts.append(dict(predictions=str(folder / 'prediction_tecu.npy'), origins=str(shared / 'forecast_origin_utc.npy'),
                              target_times=str(shared / 'target_time_utc.npy'), history_times=str(shared / 'history_time_utc.npy'),
                              unit='TECU', layout='N,12,5183,1 -> reshape(N,12,71,73), latitude-major C order',
                              provenance_manifest=str(folder / 'manifest.json'), year=year))
        record = dict(unit='TECU', parts=parts)
        np.testing.assert_array_equal(PredictionReader(record).origins, expected_origins('extension_evaluation'))
        return record

    for seed in SEEDS:
        row = next(r for r in source_rows if r['seed'] == seed)
        training = row['training']
        identity = cp['DE-ZScore-c05', seed]
        checked(training['durable_checkpoint_path'], training['checkpoint_sha256'])
        assert training['checkpoint_sha256'] == identity['checkpoint_sha256']
        assert training['best_epoch'] == identity['selected_epoch']
        member = dict(arm='DE-ZScore-c05', seed=seed, checkpoint_sha256=training['checkpoint_sha256'],
                      checkpoint_path=training['durable_checkpoint_path'], splits={})
        for split, folder in [('validation', Path(training['durable_checkpoint_path']).parent),
                              ('original_evaluation', Path(row['evaluation_directory']))]:
            manifest = read(folder / 'manifest.json')
            checked(folder / 'manifest.json')
            assert manifest['run_tuple']['seed'] == seed and manifest['run_tuple']['candidate_id'] == 'gim_de_zscore_c05'
            for key in ('predictions', 'mapping_arrays', 'truth_manifest', 'data_runtime', 'runtime_identity', 'prediction_truth_alignment'):
                entry = manifest['artifacts'][key]
                checked(folder / entry['relative_path'], entry['sha256'])
            truthmeta = read(folder / 'truth_manifest.json')
            assert truthmeta['truth_unit'] == 'TECU' and truthmeta['prediction_dtype'] == 'float32'
            if split == 'original_evaluation':
                reuse = read(folder / 'evaluation_reuse.json')
                checked(folder / 'evaluation_reuse.json')
                assert reuse['request']['source_checkpoint_sha256'] == training['checkpoint_sha256']
            with np.load(folder / 'predictions.npz', allow_pickle=False) as z:
                assert str(z['array_name']) == 'prediction_tecu' and str(z['dtype']) == 'float32'
                declared_shape = z['shape'].tolist()
            part = dict(predictions=str(folder / 'predictions.npz'), prediction_key='values', array_name='prediction_tecu',
                        origins=str(folder / 'mapping_arrays.npz'), origins_key='forecast_origin_utc', unit='TECU',
                        target_times=str(folder / 'mapping_arrays.npz'), target_times_key='target_time_utc',
                        history_times=str(folder / 'mapping_arrays.npz'), history_times_key='history_time_utc',
                        layout='N,12,5183,1 -> reshape(N,12,71,73), latitude-major C order',
                        provenance_manifest=str(folder / 'manifest.json'))
            reader = PredictionReader(part)
            np.testing.assert_array_equal(reader.origins, expected_origins(split))
            assert declared_shape == [len(reader), 12, 5183, 1]
            with np.load(folder / 'mapping_arrays.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['target_time_utc'], reader.origins[:, None] + np.arange(12) * 7200)
                np.testing.assert_array_equal(z['history_time_utc'], reader.origins[:, None] - np.arange(12, 0, -1) * 7200)
            data_runtime = read(folder / 'data_runtime.json')
            for entry in data_runtime['source_files']:
                year = str(entry['target_year'])
                if int(year) in YEARS[split]:
                    assert entry['source_file_hash'] == cfg['data_files'][year]['sha256']
            member['splits'][split] = part
            audit.append(dict(family='DE-ZScore-c05', seed=seed, split=split, shape=list(reader.shape),
                              unit='TECU', origins_equal=True, twelve_target_times_equal=True, history_times_equal=True,
                              source_checkpoint_link_verified=True, prediction_key='values', array_name='prediction_tecu'))
        member['splits']['extension_evaluation'] = extension_record('DE-ZScore-c05', seed)
        arms['DE-ZScore-c05'].append(member)
        reference_only.append(dict(arm='PI-RatioProj', seed=seed, checkpoint_sha256=cp['PI-RatioProj', seed]['checkpoint_sha256'],
                                   splits={'extension_evaluation': extension_record('PI-RatioProj', seed)}))
    for entry in cfg['data_files'].values():
        checked(entry['path'], entry['sha256'])
    checked(cfg['driver_path'])
    for path in cfg['regional_inputs']:
        checked(path)
    pi_root = Path(cfg['pi_ratio_training_root'])
    pi_arrays = sorted(str(p) for p in pi_root.rglob('*') if p.is_file() and p.suffix in ('.npy', '.npz'))
    if pi_arrays:
        raise ValueError('PI-RatioProj array inventory changed; inspect eligibility before selection')
    result = dict(schema='exp0014.frozen_predictions.v1', status='input_identity_verified', family_order=FAMILIES,
                  seeds=SEEDS, arms={k: arms[k] for k in FAMILIES}, reference_only={'PI-RatioProj': reference_only},
                  reference_policy='(G5+C5+PI5+EMP5+PI-RatioProj5)/5; five families at 20%; extension only; not a candidate',
                  files_sha256=hashes, created_at_utc=now(), grid=cfg['grid'],
                  canonical_truth='Current GIM raw float32 / float32(10), then float64; old normalized-roundtrip truths not used',
                  old_endpoint_policy='Retain independent old -180/+180 predictions unchanged; score 73 columns and diagnose 72')
    write(out / 'FROZEN_PREDICTIONS.json', result)
    write(out / 'INPUT_AUDIT.json', dict(status='passed', records=audit, files_hashed=len(hashes),
          bytes_hashed=sum(Path(p).stat().st_size for p in hashes), pi_ratio_array_search=pi_arrays,
          grid_evidence=cfg['grid_evidence'], evaluation_errors_computed=False,
          note='DO is the G alias and is not counted twice; only the original four families define the 20-member curves.'))
    print('INPUT_AUDIT passed:', len(hashes), 'files', flush=True)


def refs(cfg):
    repo = Path(cfg['repo'])
    return module(Path(__file__).resolve().parent / 'fusion.py', '_exp14_core'), module(Path(__file__).resolve().parents[1] / 'scoring/pooled.py', '_exp14_score')


def verify_sources(cfg, out):
    identity = read(out / 'RUN_IDENTITY.json')
    for name, expected in identity['source_files_sha256'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError('Frozen source snapshot changed: ' + name)
    for path, expected in cfg['reference_sources'].items():
        if sha(path) != expected:
            raise ValueError('Read-only reference implementation changed')
    return identity


def check_anchor(name, value, expected, checks):
    row = dict(observed=float(value), reference=float(expected), difference=float(value - expected),
               tolerance=1e-6, passed=bool(abs(value - expected) <= 1e-6))
    checks[name] = row
    if not row['passed']:
        raise ValueError('Anchor mismatch: ' + name + ': ' + str(row))


def anchor_global(cfg, out, manifest, split, checks):
    _, score = refs(cfg)
    ts, arrays = family_readers(manifest, split, ['G', 'C', 'PI', 'EMP'])
    names = [f'G_seed{s}' for s in SEEDS] + [f'F2_seed{s}' for s in SEEDS] + ['G5', 'PI5', 'EMP5', 'F10']
    stats = np.zeros((len(names), len(ts), 12, 4))
    cache = {}
    for i, t in enumerate(ts):
        y = truth_day(cfg, t, cache)
        p = {f: np.stack([a[i].reshape(12, 5183).astype(np.float64) for a in arrays[f]]) for f in arrays}
        predictions = np.concatenate([p['G'], (p['G'] + p['C']) / 2,
                                     p['G'].mean(0)[None], p['PI'].mean(0)[None], p['EMP'].mean(0)[None],
                                     ((p['G'].mean(0) + p['C'].mean(0)) / 2)[None]])
        stats[:, i] = score.sufficient(predictions, y, np.isfinite(y))
        if i % 250 == 0:
            print('anchor', split, i, '/', len(ts), flush=True)
    reference = read(Path(cfg['repo']) / 'experiments/exp0013/result.json')['ensemble_table'][split]
    for family in ('G', 'PI', 'EMP'):
        j = names.index(family + '5')
        check_anchor(split + '/' + family + '5', score.summary(stats[j:j+1])['RMSE'], reference[family]['all']['RMSE'], checks)
    if split in ('original_evaluation', 'extension_evaluation'):
        check_anchor(split + '/G_single_five_seed_mean', score.summary(stats[:5])['RMSE'],
                     2.233885478 if split == 'original_evaluation' else 4.729342137, checks)
    if split == 'extension_evaluation':
        check_anchor(split + '/same_seed_GC_five_pair_mean', score.summary(stats[5:10])['RMSE'], 4.491318720, checks)
    np.savez_compressed(out / 'artifacts' / f'anchor_{split}.npz', origins=ts, methods=np.array(names), stats=stats)
    write(out / 'ANCHORS.json', checks)
    print('anchors passed', split, flush=True)


def base_specs():
    specs = {}
    for f, family in enumerate(FAMILIES):
        for j, seed in enumerate(SEEDS):
            w = np.zeros(25); w[f * 5 + j] = 1
            specs[f'{family}_seed{seed}'] = np.broadcast_to(w, (12, 25)).copy()
        w = np.zeros(25); w[f*5:(f+1)*5] = .2
        specs[family + '5'] = np.broadcast_to(w, (12, 25)).copy()
    for label, families in [('G', [0]), ('C', [1]), ('GC', [0, 1]), ('four', [0, 1, 2, 3])]:
        for m in range(1, 6):
            w = np.zeros(25)
            for f in families:
                w[f*5:f*5+m] = 1 / (len(families) * m)
            specs[f'curve_{label}_m{m}'] = np.broadcast_to(w, (12, 25)).copy()
    specs['F10'] = specs['curve_GC_m5'].copy()
    specs['original_four20'] = specs['curve_four_m5'].copy()
    for j, seed in enumerate(SEEDS):
        w = np.zeros(25); w[j] = w[5+j] = .5
        specs[f'F2_seed{seed}'] = np.broadcast_to(w, (12, 25)).copy()
    return specs


def formal_specs(freeze):
    family_specs = {method: np.array(w) for method, w in freeze['candidate_full_validation_weights'].items()}
    family_specs['adopted'] = np.array(freeze['full_validation_weights'])
    for family, row in freeze['drop_family'].items():
        if row['weights'] is not None:
            family_specs['drop_' + family] = np.array(row['weights'])
    return {name: np.repeat(w / 5, 5, axis=1) for name, w in family_specs.items()}


def stats_from_errors(errors, masks):
    """Return [method,group,lead,n/sse/sae/error_sum] with common support."""
    out = np.zeros((len(errors), len(masks), 12, 4))
    squares, absolute = errors * errors, np.abs(errors)
    for g, mask in enumerate(masks):
        out[:, g, :, 0] = mask.sum(1)
        for column, values in [(1, squares), (2, absolute), (3, errors)]:
            out[:, g, :, column] = np.einsum('mln,ln->ml', values, mask)
    return out


def collect(cfg, out, manifest, split, specifications, filename, with_moments=True, ratio=False):
    ts, readers = family_readers(manifest, split)
    np.testing.assert_array_equal(ts, expected_origins(split))
    names = list(specifications)
    weights = np.stack(list(specifications.values()))
    if ratio:
        ratio_ts, ratio_readers = prediction_set(manifest['reference_only']['PI-RatioProj'], split)
        np.testing.assert_array_equal(ts, ratio_ts)
        names += ['PI-RatioProj5_reference_only', 'fixed_five_family20_reference']
    stats = np.zeros((len(names), len(ts), 4, 12, 4))
    years = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
    year_values = sorted(set(years.tolist()))
    seed_cross = np.zeros((len(year_values), 4, 12, 25, 25))
    seed_sums = np.zeros((len(year_values), 4, 12, 25))
    counts = np.zeros((len(year_values), 4, 12))
    family_daily = np.zeros((len(ts), 12, 5, 5))
    seam = {family: dict(max_abs=0., unequal=0, checked=0) for family in FAMILIES}
    disturbed = disturbance(cfg, ts)
    cache = {}
    for i, t in enumerate(ts):
        pred = np.stack([a[i].reshape(12, 5183).astype(np.float64) for f in FAMILIES for a in readers[f]])
        y = truth_day(cfg, t, cache)
        valid = np.isfinite(y)
        if not np.isfinite(pred[:, valid]).all():
            raise ValueError('Nonfinite forecast on required support')
        grid = pred.reshape(5, 5, 12, 71, 73)
        for j, family in enumerate(FAMILIES):
            difference = grid[j, ..., 0] - grid[j, ..., -1]
            seam[family]['max_abs'] = max(seam[family]['max_abs'], float(np.max(np.abs(difference))))
            seam[family]['unequal'] += int(np.count_nonzero(difference))
            seam[family]['checked'] += int(difference.size)
            if j < 4 and np.any(difference):
                raise ValueError('New recipe periodic seam mismatch')
        errors = np.where(valid, pred - y, 0.)
        unique = valid.reshape(12, 71, 73).copy(); unique[..., -1] = False
        masks = [valid, unique.reshape(12, 5183), valid & (y >= 60), valid & disturbed[i]]
        # Batch over leads; never materialize expanded windows on disk.
        mixed = (weights.transpose(1, 0, 2) @ errors.transpose(1, 0, 2)).transpose(1, 0, 2)
        if ratio:
            rp = np.mean([np.asarray(a[i], np.float64).reshape(12, 5183) for a in ratio_readers], axis=0)
            re = np.where(valid, rp - y, 0.)
            fixed = (errors[:20].reshape(4, 5, 12, 5183).mean(1).sum(0) + re) / 5
            mixed = np.concatenate([mixed, re[None], fixed[None]])
        stats[:, i] = stats_from_errors(mixed, masks)
        if with_moments:
            yy = year_values.index(int(years[i]))
            e = errors.transpose(1, 0, 2)
            q = e @ e.transpose(0, 2, 1)
            edge = e.reshape(12, 25, 71, 73)[..., -1]
            qu = q - edge @ edge.transpose(0, 2, 1)
            qh = np.stack([e[h][:, masks[2][h]] @ e[h][:, masks[2][h]].T for h in range(12)])
            for g, cross in enumerate([q, qu, qh, q if disturbed[i] else np.zeros_like(q)]):
                seed_cross[yy, g] += cross
                seed_sums[yy, g] += np.einsum('fln,ln->lf', errors, masks[g])
                counts[yy, g] += masks[g].sum(1)
            family_daily[i] = q.reshape(12, 5, 5, 5, 5).mean((2, 4))
        if i % 100 == 0:
            print('collect', split, filename, i, '/', len(ts), flush=True)
    payload = dict(origins=ts, methods=np.array(names), groups=np.array(GROUPS), fields=np.array(FIELDS),
                   stats=stats, years=np.array(year_values), disturbed=disturbed)
    if with_moments:
        payload.update(seed_cross=seed_cross, seed_error_sum=seed_sums, counts=counts, family_daily_cross=family_daily)
    np.savez_compressed(out / 'artifacts' / filename, **payload)
    write(out / 'artifacts' / (filename + '.seam.json'), seam)
    return payload


def select_and_freeze(cfg, out, data, manifest):
    q = data['family_daily_cross']
    years = data['origins'].astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
    counts = data['stats'][0, :, 0, :, 0]
    methods = {}
    all_weights = {}
    for method_name in ('C1', 'C2', 'C3'):
        folds = []
        for fit_year, held_year in [(2013, 2017), (2017, 2013)]:
            fit, held = years == fit_year, years == held_year
            w = fit_weights(q[fit].sum(0), method_name)
            folds.append(dict(fit_year=fit_year, held_out_year=held_year, weights=w.tolist(),
                              fit_RMSE=rmse_from_moments(q[fit].sum(0), counts[fit], w),
                              held_out_RMSE=rmse_from_moments(q[held].sum(0), counts[held], w)))
        methods[method_name] = dict(folds=folds, mean_held_out_RMSE=float(np.mean([f['held_out_RMSE'] for f in folds])),
                                   weight_difference_L1_by_lead=np.abs(np.array(folds[0]['weights']) - folds[1]['weights']).sum(1).tolist())
        all_weights[method_name] = fit_weights(q.sum(0), method_name).tolist()
    order = ['C1', 'C2', 'C3']
    best = min(order, key=lambda m: (methods[m]['mean_held_out_RMSE'], order.index(m)))
    chosen = best
    comparisons = []
    for simple in order[:order.index(best)]:
        score = methods[simple]['mean_held_out_RMSE']
        improvement = (score - methods[best]['mean_held_out_RMSE']) / score
        comparisons.append(dict(simple=simple, best=best, relative_improvement=improvement, close=improvement < .001))
        if improvement < .001:
            chosen = simple
            break
    w = np.array(all_weights[chosen])
    drop = {}
    for j, family in enumerate(FAMILIES):
        if not np.any(w[:, j] > 0):
            drop[family] = dict(status='zero_weight_contribution_zero', weights=w.tolist())
        else:
            allowed = [k for k in range(5) if k != j]
            drop[family] = dict(status='same_algorithm_refit_on_validation', weights=fit_weights(q.sum(0), chosen, allowed).tolist())
    freeze = dict(schema='exp0014.frozen_fusion.v1', experiment='exp0014', created_at_utc=now(),
                  source_commit=read(out / 'RUN_IDENTITY.json')['source_commit'],
                  prediction_manifest_sha256=sha(out / 'FROZEN_PREDICTIONS.json'),
                  validation_sufficient_statistics_sha256=sha(out / 'artifacts/validation_base.npz'),
                  family_order=FAMILIES, seeds=SEEDS, method=chosen, adopted_method=chosen,
                  cross_validation=methods, raw_best_method=best, simplicity_comparisons=comparisons,
                  candidate_full_validation_weights=all_weights, full_validation_weights=w.tolist(),
                  weights=w.tolist(), drop_family=drop,
                  selection_rule=cfg['selection_rule'], optimizer=cfg['optimizer'],
                  frozen_before_new_evaluation=True,
                  limitations=['Both years already selected base checkpoints; cross-year scores assess fusion stability, not independent generalization.',
                               '2023–2025 has participated in development; evaluation scores are descriptive.'])
    write(out / 'FROZEN_FUSION.json', freeze)
    print('FROZEN', chosen, {m: methods[m]['mean_held_out_RMSE'] for m in order}, flush=True)
    return freeze


def summary_tables(data, score):
    return {name: score.tables({g: data['stats'][j:j+1, :, h] for h, g in enumerate(GROUPS)}, data['origins'])
            for j, name in enumerate(data['methods'].tolist())}


def error_structure(data):
    def one(q, sums, count):
        cross = q / count
        bias = sums / count
        covariance = cross - bias[:, None] * bias[None, :]
        variance = np.diag(covariance)
        correlation = covariance / np.sqrt(variance[:, None] * variance[None, :])
        return dict(bias=bias.tolist(), variance=variance.tolist(), error_cross_moment=cross.tolist(),
                    centered_covariance=covariance.tolist(), Pearson=correlation.tolist())
    q = data['seed_cross'][:, 0].sum((0, 1))
    sums = data['seed_error_sum'][:, 0].sum((0, 1))
    count = data['counts'][:, 0].sum()
    averaging = np.zeros((5, 25))
    for i in range(5):
        averaging[i, i*5:(i+1)*5] = .2
    families = one(averaging @ q @ averaging.T, averaging @ sums, count)
    seeds = one(q, sums, count)
    decomposition = {}
    for i, family in enumerate(FAMILIES):
        block = q[i*5:(i+1)*5, i*5:(i+1)*5] / count
        diagonal = float(np.trace(block) / 5)
        shared = float((block.sum() - np.trace(block)) / 20)
        seed_variance = diagonal - shared
        actual = float(block.mean())
        np.testing.assert_allclose(actual, shared + seed_variance / 5, atol=1e-12, rtol=1e-12)
        decomposition[family] = dict(mean_single_model_MSE=diagonal, shared_error_MSE=shared,
                                    exchangeable_seed_variance=seed_variance, five_seed_ensemble_MSE=actual,
                                    infinite_seed_MSE_extrapolation=shared,
                                    infinite_seed_RMSE_extrapolation=float(np.sqrt(shared)) if shared >= 0 else None,
                                    assumptions='Exchangeable seeds, invariant shared-error component, independent zero-mean seed deviations conditional on sample; estimated from only five seeds, not an observed result.')
    return dict(family_order=FAMILIES, member_order=[f'{f}_{s}' for f in FAMILIES for s in SEEDS],
                families=families, all_members=seeds, decomposition=decomposition,
                within_family={f: {k: np.array(v)[i*5:(i+1)*5, i*5:(i+1)*5].tolist()
                                   for k, v in seeds.items() if np.array(v).ndim == 2} for i, f in enumerate(FAMILIES)},
                note='Low correlation alone is not contribution; variance, bias-inclusive cross moments and validation refits govern interpretation.')


def subset_curves(data, out, split):
    result = {}
    artifact = {}
    q = data['seed_cross'].sum(0)
    count = data['counts'].sum(0)
    for label, families in [('G', [0]), ('C', [1]), ('GC', [0, 1]), ('four', [0, 1, 2, 3])]:
        result[label] = []
        for m in range(1, 6):
            choices = list(itertools.combinations(range(5), m))
            combinations = list(itertools.product(choices, repeat=len(families)))
            w = np.zeros((len(combinations), 25))
            for i, selection in enumerate(combinations):
                for family, subset in zip(families, selection):
                    w[i, np.array(subset) + family*5] = 1 / (m * len(families))
            group_rows = {}
            for g, group in enumerate(GROUPS):
                pooled = q[g].sum(0)
                mse = np.einsum('si,ij,sj->s', w, pooled, w, optimize=True) / count[g].sum() if count[g].sum() else None
                values = np.sqrt(np.maximum(mse, 0.)) if mse is not None else None
                group_rows[group] = dict(mean_RMSE=float(values.mean()), min_RMSE=float(values.min()), max_RMSE=float(values.max())) if values is not None else None
                if group == 'all':
                    artifact[f'{label}_m{m}_RMSE'] = values
            common = np.array([i for i, c in enumerate(combinations) if all(s == c[0] for s in c)])
            common_values = artifact[f'{label}_m{m}_RMSE'][common]
            result[label].append(dict(m=m, member_count=m*len(families), subset_count=len(combinations),
                                     balanced_independent_family_subsets=group_rows,
                                     synchronized_seed_subsets=dict(count=len(common), mean_RMSE=float(common_values.mean()),
                                                                   min_RMSE=float(common_values.min()), max_RMSE=float(common_values.max())),
                                     prefix_method=f'curve_{label}_m{m}'))
            artifact[f'{label}_m{m}_member_indices'] = np.array([[f*5+j for f, s in zip(families, c) for j in s]
                                                                for c in combinations], np.int16)
    np.savez_compressed(out / 'artifacts' / f'{split}_all_subsets.npz', **artifact)
    return result


def regional(cfg, out, manifest, freeze, anchors_only, checks):
    ref = module(Path(__file__).resolve().parents[1] / 'regional/mapping_score.py', '_exp14_region')
    _, score = refs(cfg)
    families = ['G', 'C'] if anchors_only else FAMILIES
    ts_global, readers = family_readers(manifest, 'extension_evaluation', families)
    positions = {int(t): i for i, t in enumerate(ts_global)}
    result = {}
    reference = read(Path(cfg['repo']) / 'experiments/exp0011/result.json')['regions']
    for region in ('CN', 'JP'):
        ts = ref.origins(region, 'evaluation')
        mp = ref.mapping(region)
        names = ['G5', 'F10'] if anchors_only else ['adopted', 'C1']
        stats = np.zeros((len(names), len(ts), 4, 12, 4))
        disturbed = disturbance(cfg, ts)
        for i, t in enumerate(ts):
            y, valid, _ = ref.day(region, 'evaluation', t)
            label, prior = ref.gim(t), ref.gim(t - 86400)
            valid = valid & np.isfinite(label[:, mp[0]]).all(1) & np.isfinite(prior[:, mp[0]]).all(1)
            means = family_means(readers, positions[int(t)], families).reshape(len(families), 12, 5183)
            predictions = [means[0], means.mean(0)] if anchors_only else [fuse(means, freeze['full_validation_weights']),
                                                                         fuse(means, freeze['candidate_full_validation_weights']['C1'])]
            mapped = np.stack([ref.remap(p, mp) for p in predictions])
            masks = [valid, valid, valid & (y >= 60), valid & disturbed[i]]
            stats[:, i] = stats_from_errors(np.where(valid, mapped - y.astype(np.float64), 0.), masks)
            if i % 250 == 0:
                print('regional anchors' if anchors_only else 'regional formal', region, i, '/', len(ts), flush=True)
        assert int(stats[0, :, 0, :, 0].sum()) == {'CN': 36099959, 'JP': 3770496}[region]
        data = dict(origins=ts, methods=np.array(names), stats=stats)
        table = summary_tables(data, score)
        for name in names:
            table[name].pop('unique_longitude')
            table[name]['kind'] = 'single deterministic prediction; no seed SD'
        if anchors_only:
            for name in names:
                check_anchor(region + '/' + name, table[name]['all']['RMSE'], reference[region]['evaluation']['methods'][name]['RMSE'], checks)
            write(out / 'ANCHORS.json', checks)
        np.savez_compressed(out / 'artifacts' / f'regional_{region}_{"anchors" if anchors_only else "formal"}.npz',
                            **data, groups=np.array(GROUPS), fields=np.array(FIELDS))
        result[region] = dict(days=len(ts), methods=table, support='exp0011 native valid AND current/previous GIM bilinear finite support')
    return result


def formal_result(cfg, out, manifest, freeze):
    _, score = refs(cfg)
    result = dict(experiment='exp0014', status='formal_scored_before_oracle', source_commit=read(out / 'RUN_IDENTITY.json')['source_commit'],
                  family_order=FAMILIES, adopted_method=freeze['method'], cross_validation=freeze['cross_validation'],
                  weights=freeze['full_validation_weights'], anchors=read(out / 'ANCHORS.json'),
                  main_table={}, single_model_five_seed_mean={}, same_seed_two_model_five_pair_mean={},
                  curves={}, error_structure={}, paired_time_intervals={}, drop_family_contribution={},
                  source_and_method_notes=cfg['limitations'])
    for split in SPLITS:
        with np.load(out / 'artifacts' / f'{split}_base.npz', allow_pickle=False) as z:
            base = dict(z)
        with np.load(out / 'artifacts' / f'{split}_formal.npz', allow_pickle=False) as z:
            formal = dict(z)
        tables = summary_tables(base, score)
        tables.update(summary_tables(formal, score))
        result['main_table'][split] = tables
        result['single_model_five_seed_mean'][split] = {}
        base_names = base['methods'].tolist()
        for f in FAMILIES:
            ids = [base_names.index(f'{f}_seed{s}') for s in SEEDS]
            result['single_model_five_seed_mean'][split][f] = score.tables(
                {g: base['stats'][ids, :, h] for h, g in enumerate(GROUPS)}, base['origins'], SEEDS)
        ids = [base_names.index(f'F2_seed{s}') for s in SEEDS]
        result['same_seed_two_model_five_pair_mean'][split] = score.tables(
            {g: base['stats'][ids, :, h] for h, g in enumerate(GROUPS)}, base['origins'], SEEDS)
        result['curves'][split] = subset_curves(base, out, split)
        result['error_structure'][split] = error_structure(base)
        form_names = formal['methods'].tolist()
        adopted = formal['stats'][form_names.index('adopted'):form_names.index('adopted')+1, :, 0]
        shared = score.draws(base['origins'], 7)
        adopted_draws = score.time_rmse(adopted, shared)
        comparisons = {}
        for reference in ['G5', 'F10', 'C1']:
            source = base if reference in base_names else formal
            j = source['methods'].tolist().index(reference)
            ref = source['stats'][j:j+1, :, 0]
            samples = adopted_draws - score.time_rmse(ref, shared)
            comparisons['adopted-' + reference] = dict(delta_RMSE=tables['adopted']['all']['RMSE'] - tables[reference]['all']['RMSE'],
                ci95=np.quantile(samples, [.025, .975]).tolist(), block_days=7, replicates=5000, rng=20260924,
                scheme='Year-stratified non-overlapping seven-calendar-day blocks; shared samples for all comparisons',
                interpretation='Conditional temporal variation of fixed deterministic predictions; no training-seed SD')
        result['paired_time_intervals'][split] = comparisons
        result['drop_family_contribution'][split] = {f: dict(status=freeze['drop_family'][f]['status'],
            RMSE=tables['drop_' + f]['all']['RMSE'],
            delta_RMSE_removed_minus_adopted=0. if freeze['drop_family'][f]['status'].startswith('zero_weight') else
                tables['drop_' + f]['all']['RMSE'] - tables['adopted']['all']['RMSE']) for f in FAMILIES}
    result['regional_zero_shot'] = regional(cfg, out, manifest, freeze, False, {})
    write(out / 'FORMAL_RESULT_BEFORE_ORACLE.json', result)
    return result


def oracle(cfg, out, manifest, result):
    """Called last, after formal global/regional scores have been persisted."""
    _, score = refs(cfg)
    with np.load(out / 'artifacts/extension_evaluation_base.npz', allow_pickle=False) as z:
        q = z['family_daily_cross'].sum(0)
    weights = {name: fit_weights(q, name) for name in ['C2', 'C3']}
    specifications = {'oracle_' + name: np.repeat(w / 5, 5, 1) for name, w in weights.items()}
    data = collect(cfg, out, manifest, 'extension_evaluation', specifications, 'extension_oracle.npz', False)
    table = summary_tables(data, score)
    result['posthoc_oracle'] = dict(status='diagnostic_only_not_formal_result', fit_period='2023–2025',
        weights={k: v.tolist() for k, v in weights.items()}, table=table,
        adopted_minus_oracle_RMSE={name: result['main_table']['extension_evaluation']['adopted']['all']['RMSE'] -
                                  table['oracle_' + name]['all']['RMSE'] for name in weights},
        no_feedback_into_selection=True, formal_result_sha256=sha(out / 'FORMAL_RESULT_BEFORE_ORACLE.json'))
    result['status'] = 'scored_pending_independent_recheck'
    write(out / 'result.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['inventory', 'validation', 'evaluate'])
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / 'experiments/exp0014'
    cfg = read(Path(__file__).parent / 'config.json')
    assert cfg['repo'] == str(repo)
    verify_sources(cfg, out)
    if args.phase == 'inventory':
        inventory(cfg, out)
        return
    manifest = read(out / 'FROZEN_PREDICTIONS.json')
    if args.phase == 'validation':
        if (out / 'FROZEN_FUSION.json').exists():
            raise ValueError('Refuse to repeat selection over an existing frozen fusion')
        checks = {}
        anchor_global(cfg, out, manifest, 'validation', checks)
        data = collect(cfg, out, manifest, 'validation', base_specs(), 'validation_base.npz')
        select_and_freeze(cfg, out, data, manifest)
        return
    freeze = read(out / 'FROZEN_FUSION.json')
    assert freeze['prediction_manifest_sha256'] == sha(out / 'FROZEN_PREDICTIONS.json')
    checks = read(out / 'ANCHORS.json')
    for split in SPLITS[1:]:
        anchor_global(cfg, out, manifest, split, checks)
    regional(cfg, out, manifest, freeze, True, checks)
    write(out / 'EVALUATION_ANCHORS_PASSED.json', dict(status='passed_before_new_evaluation', time=now(),
          anchors_sha256=sha(out / 'ANCHORS.json'), frozen_fusion_sha256=sha(out / 'FROZEN_FUSION.json')))
    for split in SPLITS:
        if split != 'validation':
            collect(cfg, out, manifest, split, base_specs(), f'{split}_base.npz', True, split == 'extension_evaluation')
        collect(cfg, out, manifest, split, formal_specs(freeze), f'{split}_formal.npz', False)
    result = formal_result(cfg, out, manifest, freeze)
    oracle(cfg, out, manifest, result)
    print('Formal, regional, subset and final post-hoc oracle scoring complete', flush=True)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
