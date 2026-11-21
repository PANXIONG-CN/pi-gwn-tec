# pi-gwn-tec

Code for **Physical knowledge improves ionospheric electron content forecasts as inputs rather than as tested loss
penalties** (Xiong et al.). It recomputes the numbers of the article's tables from the daily statistics, and retrains
and scores every configuration of the article.

## Install

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements-verified.txt` pins the versions of the release checks (CPU; Python 3.11.4, PyTorch 2.14.1, NumPy
2.4.6). The experiments were trained on one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU per run with Python
3.12.3, PyTorch 2.9.1+cu128 and CUDA 12.8.

## Data

| Item | Where |
|---|---|
| Global IGS maps 2009-2026, two-hourly indices, regional JPLD arrays (China, Japan) | GIMtec, https://doi.org/10.5281/zenodo.24410367 (`python scripts/download_gimtec_from_zenodo.py --data-root DIR`) |
| Figure and table source data, interval results, preregistration and freeze records | Supplementary Data 1 of the article |
| Daily statistics (`statistics/*.npz` of Supplementary Data 1, about 535 MB) and frozen model checkpoints (indexed by system) | Zenodo record of this code: `Supplementary_Data_1_statistics.zip`, `checkpoints.zip` |
| Per-window prediction arrays (about 170 GB) | from the corresponding author on request (too large for the repository); those of the supplementary penalty-term, graph-support and conditioning controls were not retained and can be regenerated from the checkpoints |
| CODE one-day forecast (C1PG) | NASA CDDIS: `scripts/fetch_c1pg.py` (from 2023), `scripts/build_c1pg_ionex.py` (2015-2022), `scripts/convert_c1pg.py` |

The model assets (fixed graph and its rank-10 SVD initialization, training-year climatology, solar-geometry table of
SimVP) are rebuilt from the GIMtec files and checked against their recorded SHA-256:

```bash
python scripts/prepare_assets.py --gimtec DIR --assets ASSETS        # a few minutes
python scripts/prepare_night_gate.py --gimtec DIR --assets ASSETS    # gated nighttime term only
```

## Recompute the tables (CPU, about one minute)

Unpack the daily statistics into `Supplementary_Data_1/statistics/`, then run

```bash
python scripts/reproduce_tables.py --sd1 /path/to/Supplementary_Data_1
```

The script runs the table and interval functions of `tecfc/scoring` and `tecfc/intervals` on the daily statistics
and compares every reproducible row of `records/EVIDENCE.csv` with the recomputed value (tolerance 1e-6). Rows that
are not functions of the daily statistics (timings, parameter counts, thresholds, values of the earlier
configuration and of the literature) are checked against the files in `records/` and in Supplementary Data 1. The
report is written to `reproduce_report.json`.

## Retrain and score a configuration

`configs/` holds one file per configuration of the article, named after it; `CONFIG_MAP.md` maps every figure and
table to its configurations. Each file holds the configuration of the original runs with the seeds 42, 1024, 2025,
3407 and 5000; data locations are filled in at run time.

```bash
export TECFC_GIMTEC=/data/gimtec TECFC_ASSETS=/data/tecfc_assets
python -m tecfc.training.run --config configs/gwn_geo_hist_mae.json --seed 42 --out runs/gwn_geo_hist_mae/42
python -m tecfc.training.predict --config configs/gwn_geo_hist_mae.json --seed 42 \
    --checkpoint runs/gwn_geo_hist_mae/42/best_200.pt --split extension_evaluation --out pred/gwn_geo_hist_mae_42.npy
python scripts/score_predictions.py --predictions pred/gwn_geo_hist_mae_42.npy --gimtec $TECFC_GIMTEC
```

Global models: batch 32, 200 epochs, Adam (learning rate 0.003), SmoothL1 (or MAE where named) on TEC divided by
147.1 TECU, gradient clipping at 5, ReduceLROnPlateau (factor 0.9, patience 5, cooldown 10); the epoch with the
lowest pooled validation RMSE is kept. Inputs end at 22 UTC of the day before the forecast day; the outputs are the
12 two-hourly maps from +2 to +24 h (48 maps for the 96-hour models).

Other entry points:

- single penalty terms, gated nighttime term, graph-support variants and the penalty on SimVP:
  `tecfc/objectives/penalty_arms.py`, `tecfc/objectives/night_gate.py`; training `tecfc/training/train_penalty_arms.py`;
- index conditioning (gated fusion, FiLM, cross-attention) and the graph-Transformer adaptation:
  `tecfc/models/conditioning_arms.py`; training `tecfc/training/train_conditioning.py`;
- multi-family ensemble: `tecfc/ensemble/` (family means, non-negative weights fitted on the validation years);
- regional systems: `tecfc/regional/` (bilinear mapping of the global forecast, anchored residual model with its
  fallback above 79.5 TECU in China, learning-rate replay);
- prediction intervals: `tecfc/intervals/` (static and trailing 365-day calibration);
- 24-h persistence and CODE one-day forecast: `tecfc/scoring/persistence.py`, `tecfc/scoring/code_forecast.py`.

Seeds, the order of training windows, cuDNN determinism, the ARPACK start vector of the graph SVD and the random
generator of the time-block intervals (seed 20260924) are fixed. Training used bfloat16 autocast on the GPU, so
retrained weights depend on the GPU and library versions; the article reports the seed-to-seed spread.

## Layout

| Path | Content |
|---|---|
| `tecfc/data` | regional windows and anchors, 2026 data access |
| `tecfc/models` | GWN (`gwn_reference`) and SimVP (`simvp_reference`) training directories; conditioning variants and the graph-Transformer adaptation |
| `tecfc/features` | solar geometry, multi-day history, climatology and index inputs |
| `tecfc/objectives` | data loss, the six penalty terms, gradient scaling and projection, empirical-tendency comparator, ablation and single-term variants, nighttime gate |
| `tecfc/training` | training functions; `run.py` and `predict.py` launch them |
| `tecfc/ensemble` | family means, non-negative weights, cross-year check |
| `tecfc/regional` | mapping, anchored residual model, fallback, learning-rate replay, regional scores |
| `tecfc/scoring` | pooled metrics, time-block intervals, strata |
| `tecfc/intervals` | interval calibration |
| `configs/` | one file per configuration; `configs/inputs/` holds the driver scaler and file hashes |
| `records/` | evidence table, name map and the inputs of `scripts/reproduce_tables.py` (`records/README.md`) |
| `scripts/` | data download and conversion, asset preparation, table reproduction, scoring |
| `tests/` | synthetic tests; reproduction tests (need Supplementary Data 1 with its daily statistics) |
| `legacy/` | code of the earlier training configuration, for the earlier-configuration comparison (`legacy/README.md`) |

## Tests

```bash
python -m pytest tests -q
TECFC_SD1=/path/to/Supplementary_Data_1 python -m pytest tests -q   # adds the reproduction tests
```

## License and citation

MIT. The Channel Mixer/OpenSTL components in `legacy/third_party/channel_mixer` keep their Apache-2.0 license.
Please cite the article; citation metadata are in `CITATION.cff`.
