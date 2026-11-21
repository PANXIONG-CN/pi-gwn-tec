"""Common NumPy scoring entry for frozen reference and experimental predictions."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from tecfc.scoring.ablation_prepare import ROOT, EXP, read, write, sha

spec = importlib.util.spec_from_file_location('exp0026_reference_score', Path(__file__).resolve().parent / 'pooled.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def collect(paths, origins, targets, frozen_hashes, indices=None):
    """Return per-seed and mean-prediction daily lead sufficient statistics."""
    arrays = []
    for path in paths:
        if sha(path) != frozen_hashes[path]:
            raise ValueError('Changed frozen prediction ' + path)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        expected_length = len(origins) if indices is None else len(array)
        if array.shape != (expected_length, 12, 71, 73) or array.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Wrong frozen prediction layout')
        arrays.append(array)
    raws = {}
    stats = np.zeros((len(arrays), len(origins), 12, 4), np.float64)
    ensemble = np.zeros((1, len(origins), 12, 4), np.float64)
    for i, origin in enumerate(origins):
        day = np.datetime64(int(origin), 's').astype('datetime64[D]')
        year = str(day)[:4]
        if year not in raws:
            rec = targets[year]
            if sha(rec['path']) != rec['sha256']:
                raise ValueError('Changed observation source')
            raws[year] = np.load(rec['path'], mmap_mode='r', allow_pickle=False)
        offset = int((day - np.datetime64(year + '-01-01')).astype(int))
        truth = (np.asarray(raws[year][offset], np.float32) / np.float32(10)).reshape(12, 5183).astype(np.float64)
        prediction = np.stack([np.asarray(array[i if indices is None else indices[j][i]], np.float64).reshape(12, 5183) for j, array in enumerate(arrays)])
        valid = np.isfinite(truth)
        stats[:, i] = ref.sufficient(prediction, truth, valid)
        ensemble[:, i] = ref.sufficient(prediction.mean(0, keepdims=True), truth, valid)
    return stats, ensemble


def anchors():
    p13 = read(ROOT / 'experiments/exp0013/FROZEN_PREDICTIONS.json')
    p15 = read(ROOT / 'experiments/exp0015/FROZEN_PREDICTIONS.json')
    cfg = read(ROOT / 'experiments/exp0013/configs/inference.json')
    p25 = read(ROOT / 'experiments/exp0025/FROZEN_PREDICTIONS_2026.json')
    d25 = read(ROOT / 'experiments/exp0025/SCORING_INPUTS.json')['datasets']['global_2026']
    checks = []
    output = EXP / 'artifacts/anchors'
    output.mkdir(parents=True, exist_ok=True)
    for arm, split, expected_single, expected_ensemble in [
        ('G', 'original_evaluation', 2.233885, 2.136732),
        ('G', 'extension_evaluation', 4.729342, 4.492715),
        ('PI', 'extension_evaluation', None, 4.505975),
        ('G-DE', 'extension_evaluation', None, 4.555438),
    ]:
        manifest = p15 if arm == 'G-DE' else p13
        records = manifest['arms'][arm]
        dates = None
        for row in records:
            origin_path = row['splits'][split]['origins']
            if sha(origin_path) != manifest['files_sha256'][origin_path]:
                raise ValueError('Changed frozen origins')
            origins = np.load(origin_path, allow_pickle=False)
            if dates is not None:
                np.testing.assert_array_equal(dates, origins)
            dates = origins
        paths = [row['splits'][split]['predictions'] for row in records]
        stats, ens = collect(paths, dates, cfg['data_files'], manifest['files_sha256'])
        np.savez_compressed(output / (arm + '_' + split + '.npz'), origins=dates, single=stats, ensemble=ens)
        for kind, actual, expected in [('single_seed_mean', ref.summary(stats)['RMSE'], expected_single),
                                       ('ensemble', ref.summary(ens)['RMSE'], expected_ensemble)]:
            if expected is None:
                continue
            delta = abs(actual - expected)
            checks.append(dict(arm=arm, split=split, kind=kind, actual=actual, expected=expected,
                               absolute_difference=delta, passed=delta < 1e-6))
    origin_path = d25['origins']['path']
    if sha(origin_path) != d25['origins']['sha256']:
        raise ValueError('Changed 2026 frozen origins')
    origins = np.load(origin_path, allow_pickle=False)
    for arm, expected in [('G5', 3.341440), ('PI5', 3.356432)]:
        paths = [row['path'] for row in d25['methods'][arm]['arrays']]
        hashes = {path: p25['files'][path]['sha256'] for path in paths}
        stats, ens = collect(paths, origins, d25['target']['years'], hashes)
        actual = ref.summary(stats)['RMSE']
        checks.append(dict(arm=arm, split='2026_descriptive', kind='frozen_ensemble', actual=actual,
                           expected=expected, absolute_difference=abs(actual-expected), passed=abs(actual-expected) < 1e-6))
        np.savez_compressed(output / (arm + '_2026.npz'), origins=origins, ensemble=ens)
    result = dict(status='passed' if all(row['passed'] for row in checks) else 'failed',
                  checks=checks, tolerance_absolute_strict=1e-6, recomputed_from_frozen_predictions=True,
                  new_arm_metrics_computed=False, source_sha256=sha(__file__))
    write(EXP / 'CHECK_C.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['anchors'])
    args = parser.parse_args()
    result = anchors()
    print(json.dumps(dict(status=result['status'], anchors=len(result['checks']))))
    if result['status'] != 'passed':
        raise SystemExit(1)
