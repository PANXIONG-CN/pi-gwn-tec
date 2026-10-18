"""CPU evaluation of previous-day persistence and CODE one-day GIM forecasts.

Input arrays are read-only. A day uses the same UTC maps from the previous
day, including the preceding year's final day at a year boundary.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

YEARS = (2015, 2019, 2020, 2021, 2022)
MODELS = ('24-h persistence', 'C1PG')
EXPECTED_N = 113_569_896


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_year(root: Path, name: str, year: int) -> np.ndarray:
    value = np.load(root / f'{name}_{year}.npy', mmap_mode='r', allow_pickle=False)
    expected = (366 if calendar.isleap(year) else 365, 12, 71, 73)
    if value.shape != expected:
        raise ValueError(f'{name}_{year}.npy: {value.shape} != {expected}')
    return value


def sums(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, ...]:
    valid = np.isfinite(prediction) & np.isfinite(truth)
    delta = np.zeros(truth.shape, dtype=np.float64)
    np.subtract(prediction, truth, out=delta, where=valid)
    axes = (0, 2, 3)
    return (np.sum(delta * delta, axis=axes, dtype=np.float64),
            np.sum(np.abs(delta), axis=axes, dtype=np.float64),
            np.sum(valid, axis=axes, dtype=np.int64))


def metric(sse: float, sae: float, n: int) -> dict:
    return dict(n=int(n), SSE=float(sse), SAE=float(sae),
                RMSE=float(np.sqrt(sse / n)), MAE=float(sae / n))


def evaluate(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    totals = {name: [np.zeros(12), np.zeros(12), np.zeros(12, dtype=np.int64)] for name in MODELS}
    annual = []
    for year in YEARS:
        truth_raw = load_year(root, 'TEC', year)
        prefix = load_year(root, 'TEC', year - 1)
        c1pg_raw = load_year(root, 'C1PG', year)
        accum = {name: [np.zeros(12), np.zeros(12), np.zeros(12, dtype=np.int64)] for name in MODELS}
        for start in range(0, len(truth_raw), 16):
            stop = min(start + 16, len(truth_raw))
            truth = np.asarray(truth_raw[start:stop], dtype=np.float64) / 10.0
            persistence_raw = (np.concatenate((prefix[-1:], truth_raw[:stop - 1]), axis=0)
                               if start == 0 else truth_raw[start - 1:stop - 1])
            forecasts = (np.asarray(persistence_raw, dtype=np.float64) / 10.0,
                         np.asarray(c1pg_raw[start:stop], dtype=np.float64) / 10.0)
            for name, forecast in zip(MODELS, forecasts):
                for target, value in zip(accum[name], sums(forecast, truth)):
                    target += value
        for name in MODELS:
            for total, value in zip(totals[name], accum[name]): total += value
            annual.append(dict(model=name, year=year, **metric(*(v.sum() for v in accum[name]))))
    lead = [dict(model=name, lead=i + 1, lead_hours=2 * (i + 1),
                 **metric(*(v[i] for v in totals[name]))) for name in MODELS for i in range(12)]
    summary = []
    for name in MODELS:
        values = metric(*(v.sum() for v in totals[name]))
        summary.append(dict(model=name, **{k: values[k] for k in ('n', 'RMSE', 'MAE')}))
    return summary, lead, annual


def validate(summary: list[dict], lead: list[dict], annual: list[dict]) -> None:
    for row in summary:
        if row['n'] != EXPECTED_N: raise ValueError(f'Unexpected valid count: {row}')
        for partitions in [lead, annual]:
            group = [x for x in partitions if x['model'] == row['model']]
            merged = metric(sum(x['SSE'] for x in group), sum(x['SAE'] for x in group), sum(x['n'] for x in group))
            if merged['n'] != row['n'] or not np.allclose([merged['RMSE'], merged['MAE']], [row['RMSE'], row['MAE']], rtol=0, atol=1e-12):
                raise ValueError(f'Partition closure failed: {row["model"]}')
    p, c = summary
    if abs(p['RMSE'] - 2.808) > .001 or abs(p['MAE'] - 1.631) > .001:
        raise ValueError(f'Persistence acceptance failed: {p}')
    if abs(c['RMSE'] - 2.868) > .001 or not 1.876 <= c['MAE'] <= 1.877:
        raise ValueError(f'C1PG acceptance failed: {c}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gim-root', type=Path, default=Path('data/GIMtec'))
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'source_data/baselines')
    args = parser.parse_args()
    summary, lead, annual = evaluate(args.gim_root)
    validate(summary, lead, annual)
    inputs = sorted({f'TEC_{year}.npy' for year in YEARS + tuple(y - 1 for y in YEARS)} |
                    {f'C1PG_{year}.npy' for year in YEARS} | {'C1PG_2022_imputed_days.json'})
    provenance = dict(
        evaluation_years=list(YEARS), windows=1826, grid=[71, 73], leads=12,
        unit='TECU', source_unit='0.1 TECU', accumulation='float64 SSE and SAE; integer counts; pool before square root',
        valid_mask='isfinite(prediction) & isfinite(truth); zero is valid',
        persistence='Previous day at the same UTC time; January 1 uses December 31 of the preceding year',
        c1pg='CODE one-day predicted GIM; supplied 2022 imputed days included',
        inputs=[dict(file='data/GIMtec/' + name, sha256=sha256(args.gim_root / name)) for name in inputs],
        script=dict(file='tools/paper/trivial_baselines.py', sha256=sha256(Path(__file__))),
        checks=dict(expected_valid_elements=EXPECTED_N, numerical_acceptance=True, lead_and_year_closure=True))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [('summary', summary), ('lead', lead), ('year', annual)]:
        with (args.out_dir / f'trivial_baselines_{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out_dir / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
