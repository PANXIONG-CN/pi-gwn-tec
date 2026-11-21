"""Single post-freeze evaluation using shared, year-stratified time draws."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from tecfc.scoring.ablation_prepare import ROOT, EXP, SEEDS, ARMS, read, write, sha
from tecfc.scoring.ablation_score import ref, collect

SPLITS = ('validation', 'original_evaluation', 'extension_evaluation', '2026')


def statistics(stats):
    result = ref.summary(stats, SEEDS if len(stats) == 5 else None)
    for metric in ('RMSE', 'MAE', 'bias'):
        values = [row[metric] for row in result['by_seed']]
        result[metric + '_seed_sd'] = float(np.std(values, ddof=1)) if len(values) == 5 else None
    result['lead_hours'] = {str(2 * (lead + 1)): ref.summary(stats[:, :, lead], SEEDS if len(stats) == 5 else None)
                            for lead in range(12)}
    for lead in range(12):
        summary = result['lead_hours'][str(2 * (lead + 1))]
        for metric in ('RMSE', 'MAE', 'bias'):
            summary[metric + '_seed_sd'] = float(np.std([r[metric] for r in summary['by_seed']], ddof=1)) if len(stats) == 5 else None
    return result


def shared_draws(origins):
    dates = origins.astype('datetime64[s]').astype('datetime64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    rng = np.random.default_rng(20260924)
    draws = []
    for year in sorted(set(years)):
        selected = years == year
        ids = (dates[selected] - np.datetime64(f'{year}-01-01')).astype(int) // 7
        nblocks = int(ids.max()) + 1
        index = rng.integers(nblocks, size=(5000, nblocks))
        weights = np.zeros((5000, nblocks), np.int32)
        np.add.at(weights, (np.arange(5000)[:, None], index), 1)
        draws.append((selected, ids, weights))
    return draws


def comparison(a, b, ae, be, draws):
    assert a.shape == b.shape and np.array_equal(a[..., 0], b[..., 0])
    av, bv = ref.summary(a), ref.summary(b)
    delta = [x['RMSE'] - y['RMSE'] for x, y in zip(av['by_seed'], bv['by_seed'])]
    ensemble_delta = ref.summary(ae)['RMSE'] - ref.summary(be)['RMSE']
    sampled = ref.time_rmse(ae, draws) - ref.time_rmse(be, draws)
    low, high = np.quantile(sampled, [.025, .975]).tolist()
    label = '无差别'
    if ensemble_delta < 0 and high < 0:
        label = '改善'
    if ensemble_delta > 0 and low > 0:
        label = '变差'
    return dict(seed_order=SEEDS, by_seed_delta=delta, improving_seeds=sum(v < 0 for v in delta),
                seed_mean_delta=float(np.mean(delta)), ensemble_delta=ensemble_delta,
                ensemble_ci95=[low, high], classification=label,
                common_support_metrics=dict(candidate_single=statistics(a), reference_single=statistics(b),
                                            candidate_ensemble=statistics(ae), reference_ensemble=statistics(be)),
                block_days=7, resamples=5000, rng=20260924)


def historical_records():
    p13 = read(ROOT / 'experiments/exp0013/FROZEN_PREDICTIONS.json')
    p15 = read(ROOT / 'experiments/exp0015/FROZEN_PREDICTIONS.json')
    p25 = read(ROOT / 'experiments/exp0025/FROZEN_PREDICTIONS_2026.json')
    records = {arm: p13['arms'][arm] for arm in ('G', 'PI', 'EMP')}
    records['G-DE'] = p15['arms']['G-DE']
    records = json.loads(json.dumps(records))
    for arm, rows in records.items():
        origins = p25['global_']['S1_origins' if arm == 'G-DE' else 'origins']
        for row in rows:
            member = p25['global_']['members'][f"global_{arm}_{row['seed']}"]
            row['splits']['2026'] = dict(predictions=member['path'], origins=origins['path'])
    hashes = dict(p13['files_sha256'], **p15['files_sha256'])
    hashes.update({path: row['sha256'] for path, row in p25['files'].items()})
    return records, hashes


def aligned_collect(records, split, origins, targets, hashes):
    arrays = []
    indexes = []
    for record in records:
        row = record['splits'][split]
        ts = np.load(row['origins'], allow_pickle=False)
        if sha(row['origins']) != hashes[row['origins']]:
            raise ValueError('Changed origins')
        index = np.searchsorted(ts, origins)
        if np.any(index >= len(ts)) or not np.array_equal(ts[index], origins):
            raise ValueError('Missing common-support predictions')
        arrays.append(row['predictions'])
        indexes.append(index)
    return collect(arrays, origins, targets, hashes, indices=indexes)


def evaluate():
    freeze_path = EXP / 'FROZEN_PREDICTIONS.json'
    manifest = read(freeze_path)
    for name, expected in manifest.get('reference_manifests_sha256', {}).items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed frozen reference manifest: ' + name)
    historical, hashes = historical_records()
    records = dict(historical, **manifest['arms'])
    hashes.update(manifest['files_sha256'])
    targets = read(ROOT / 'experiments/exp0013/configs/inference.json')['data_files']
    target26 = read(ROOT / 'experiments/exp0025/SCORING_INPUTS.json')['datasets']['global_2026']['target']['years']
    targets.update(target26)
    tables, ensembles, pairs = {}, {}, {}
    daily = EXP / 'artifacts/daily'
    daily.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        tables[split], ensembles[split], pairs[split] = {}, {}, {}
        cache, draw_cache = {}, {}

        def get(arm, origins):
            key = (arm, tuple(origins))
            if key not in cache:
                cache[key] = aligned_collect(records[arm], split, origins, targets, hashes)
                s, e = cache[key]
                support = 'S1' if split == '2026' and len(origins) == 292 else 'global'
                np.savez_compressed(daily / f'{split}_{arm}_{support}.npz', origins=origins, single=s, ensemble=e)
            return cache[key]

        for arm in ('G', 'PI', 'EMP', 'G-DE', *ARMS):
            if arm not in records or len(records[arm]) != 5:
                continue
            first = records[arm][0]['splits'][split]
            origins = np.load(first['origins'], allow_pickle=False)
            s, e = get(arm, origins)
            tables[split][arm], ensembles[split][arm] = statistics(s), statistics(e)
            tables[split][arm]['windows'] = len(origins)
            ensembles[split][arm]['windows'] = len(origins)
        for arm in ARMS:
            if arm not in tables[split]:
                continue
            for baseline in (('G', 'PI') if arm.startswith('P-') else ('G-DE', 'G') if arm.startswith('D-') else ('G',)):
                ts_a = np.load(records[arm][0]['splits'][split]['origins'], allow_pickle=False)
                ts_b = np.load(records[baseline][0]['splits'][split]['origins'], allow_pickle=False)
                origins = np.intersect1d(ts_a, ts_b)
                if len(origins) != min(len(ts_a), len(ts_b)):
                    raise ValueError('Unexpected comparison support')
                a, ae = get(arm, origins)
                b, be = get(baseline, origins)
                key = tuple(origins)
                if key not in draw_cache:
                    draw_cache[key] = shared_draws(origins)
                paired = comparison(a, b, ae, be, draw_cache[key])
                paired['windows'] = len(origins)
                pairs[split][arm + '-' + baseline] = paired
        write(EXP / 'SCORING_PROGRESS.json', dict(completed_period=split, prediction_freeze_sha256=sha(freeze_path)))
    decision = {}
    for arm in ('P-0.1', 'P-1.0', 'P-adapt'):
        comparisons = [pairs[split].get(arm + '-G') for split in SPLITS[:3]]
        if not all(row is not None for row in comparisons):
            decision[arm] = '未完成'
        else:
            decision[arm] = '该强度有收益' if all(row['classification'] == '改善' for row in comparisons) else '无收益'
    result = dict(experiment='exp0026', status='scored', main_table=tables, ensemble_table=ensembles,
                  paired=pairs, P_decision=decision, checks=read(EXP / 'CHECKS.json'),
                  training_outcomes=read(EXP / 'FROZEN_CHECKPOINTS.json')['outcomes'],
                  checkpoint_freeze_sha256=sha(EXP / 'FROZEN_CHECKPOINTS.json'),
                  prediction_freeze_sha256=sha(freeze_path),
                  limitations=['2023–2025已参与开发。', '2026为未预注册的描述性结果，未用于选择或改变结论。',
                               'D-noIndex与G-DE的2026比较采用共同S1的292窗，完整D-noIndex与G采用304窗。',
                               '时间区间描述固定模型的时间变动，训练种子差异另列。'])
    write(EXP / 'result.json', result)
    return result


if __name__ == '__main__':
    evaluate()
