"""Score global predictions written by tecfc.training.predict against the GIMtec maps.

    python scripts/score_predictions.py --predictions pred/a.npy --gimtec DIR [--reference pred/b.npy] [--out scores.json]

Per forecast day and lead time the sufficient statistics (valid count, squared-error sum, absolute-error sum,
signed-error sum) are computed with the scoring functions of the 2026 confirmation
(tecfc/scoring/confirm_score.py); RMSE, MAE and bias pool them over grid cells, lead times and days (zeros are
valid map values; non-finite labels are excluded). With --reference, the RMSE difference gets the 95% interval of the
seven-day time-block resampling used in the article (blocks from 1 January of each year, years stratified,
5,000 draws, seed 20260924; tecfc/scoring/confirm_final.samples). Writes the daily statistics next to --out.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tecfc.scoring.confirm_score import sufficient, summary  # noqa: E402
from tecfc.scoring.confirm_final import samples, ci  # noqa: E402


def labels(gimtec, origins, frames):
    cache = {}
    for t in origins:
        days = [np.datetime64(int(t), 's').astype('datetime64[D]') + np.timedelta64(k, 'D') for k in range(frames // 12)]
        maps = []
        for day in days:
            year = int(str(day)[:4])
            if year not in cache:
                cache[year] = np.load(Path(gimtec) / f'TEC_{year}.npy', mmap_mode='r', allow_pickle=False)
            maps.append(np.asarray(cache[year][int((day - np.datetime64(f'{year}-01-01')).astype(int))], np.float32) / np.float32(10))
        y = np.concatenate(maps).astype(np.float32)
        yield y, np.isfinite(y)


def daily_statistics(path, gimtec):
    prediction = np.load(path, mmap_mode='r')
    origins = np.load(Path(path).with_suffix('.origins.npy'))
    frames = prediction.shape[1]
    stats = np.zeros((len(origins), frames, 4))
    for i, (y, v) in enumerate(labels(gimtec, origins, frames)):
        stats[i] = sufficient(np.asarray(prediction[i], np.float64), y, v)
    return origins, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True); parser.add_argument('--gimtec', required=True)
    parser.add_argument('--reference'); parser.add_argument('--out', default='scores.json')
    args = parser.parse_args()
    origins, stats = daily_statistics(args.predictions, args.gimtec)
    result = dict(predictions=str(args.predictions), windows=len(origins), **summary(stats),
                  lead_hours={str(2 * (k + 1)): summary(stats[:, k]) for k in range(stats.shape[1])})
    out = Path(args.out)
    np.savez_compressed(out.with_suffix('.daily.npz'), origins=origins, stats=stats,
                        fields=np.array(['n', 'sse', 'sae', 'error_sum']))
    if args.reference:
        ref_origins, ref_stats = daily_statistics(args.reference, args.gimtec)
        if not np.array_equal(origins, ref_origins):
            raise SystemExit('the two prediction sets cover different days')
        with np.errstate(invalid='ignore', divide='ignore'):
            draws, _ = samples(np.stack([stats.sum(1), ref_stats.sum(1)]), origins)
        reference = summary(ref_stats)
        result['reference'] = dict(predictions=str(args.reference), **reference)
        result['RMSE_difference'] = result['RMSE'] - reference['RMSE']
        if np.isfinite(draws).all():
            result['RMSE_difference_ci95'] = ci(draws[:, 0] - draws[:, 1])
        else:  # blocks are counted from 1 January; draws of empty blocks leave no data for a few days only
            result['RMSE_difference_ci95'] = None
            result['note'] = 'too few days for the seven-day block resampling; score whole periods'
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('windows', 'n_valid', 'RMSE', 'MAE', 'bias')}))


if __name__ == '__main__':
    main()
