# pi-gwn-tec

Code for **Physics inspired graph learning for ionospheric total electron content forecasting**
(Xiong et al.).

## Install

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Training used NVIDIA RTX PRO 6000 Blackwell GPUs. `requirements-verified.txt` lists the CPU
environment used to verify this release (Python 3.11.4, PyTorch 2.14.1, NumPy 2.4.6).
TEC-MoLLM additionally needs `transformers` and `peft`. Weights & Biases logging is optional;
set `WANDB_MODE=offline` to run without an account.

## Configurations

`CONFIG_MAP.md` maps every table and figure of the article to its configuration file.
Predictor inputs are fixed by each configuration: the auxiliary-loss driver flag never
grants predictor access to drivers. Adaptive-weight and projected-gradient objectives are
separate modules (`lib/physics/ppinn*.py`, `lib/physics/projection.py`). The projection
scales the six-term auxiliary gradient to ratio 0.01 of the data gradient, removes its
conflicting component and does not renormalize.

Driver order is AE, Dst, Kp, F10.7 and daylight-clipped cos(SZA). Saved scalers
(`conf/revised/*_driver_scaler.json`) are fitted on training years only. Category masks
are applied after standardization in training and inference; solar geometry is kept in
every removal. Predictor drivers precede the forecast origin; observed future drivers enter
only the physics-inspired training penalties.

## Data

Data are not stored in this repository.

| Data | Source |
|---|---|
| Global maps and drivers, 2009–2025 | GIMtec, https://doi.org/10.5281/zenodo.23331771 |
| Numerical Source Data for the figures and tables | Supplementary Data 1 of the article |
| Trained weights and prediction arrays | Available from the corresponding author on request |

```bash
GIMTEC=/path/to/gimtec
python scripts/download_gimtec_from_zenodo.py --data-root "$GIMTEC"
python -c "import sys, numpy as np; from lib.datasets.gimtec_adj import build_grid8_adj; np.save(sys.argv[1], build_grid8_adj(71, 73))" "$GIMTEC/gim_adj.npy"
```

The last command rebuilds the fixed eight-neighbour global graph used in the experiments.
Extract Supplementary Data 1 to `source_data/` for the statistics, figure and test commands
below.

`TEC_YEAR.npy` stores GIM values in 0.1 TECU (divided by 10, then by the network scale
147.1). The 2023–2025 maps are IGS COMBINED final IONEX files from NASA CDDIS
(https://cddis.nasa.gov/archive/gnss/products/ionex/), converted with the same grid and
time convention. Two-hour drivers are from NASA OMNI2; the 2009–2022 part of
`drivers_2009_2025_2h.npz` equals `drivers_2009_2022_2h.npz`.

The CODE one-day predicted maps (C1PG) used as a reference forecast are not redistributed.
`scripts/build_c1pg_ionex.py` downloads the daily C1PG IONEX files from NASA CDDIS (an
Earthdata login in `~/.netrc` and the `uncompress` command are needed) and writes
`C1PG_YEAR.npy` in 0.1 TECU on the GIM grid. Missing daily files are filled with the
previous day's map and listed in `C1PG_YEAR_imputed_days.json` (28–31 December 2022).
`tools/paper/trivial_baselines.py` then recomputes the 24-hour persistence and C1PG results
by lead and year; `source_data/baselines/PROVENANCE.json` lists the SHA-256 of every input
array used in the article.

```bash
python scripts/build_c1pg_ionex.py --years 2015-2015 --out_dir "$GIMTEC"
python scripts/build_c1pg_ionex.py --years 2019-2022 --out_dir "$GIMTEC"
python tools/paper/trivial_baselines.py --gim-root "$GIMTEC" --out-dir baselines_out
```

Regional arrays are in TECU (network scale 300) on 41 × 71 cells (15–55° N, 70–140° E).
The regional product uses provider TEC/RMSE grids; no raw-grid rebuild command is provided.
Year roles follow Table 1 of the article. Regional data are not distributed with the code;
see the Data availability statement of the article.

## Evaluate trained models

```bash
python model/Run.py --paper-config conf/revised/gim_pi_tec.yaml \
  --action evaluate --data-root "$GIMTEC" --adjacency "$GIMTEC/gim_adj.npy" \
  --checkpoint /path/to/weights/matched/PI-TEC-only/seed42/best.pt \
  --years 2015 2019 2020 2021 2022 --seed 42 --device cpu --output outputs/pi_tec_42
```

Driver-conditioned configurations also take
`--drivers "$GIMTEC/drivers/drivers_2009_2022_2h.npz" --driver-scaler conf/revised/gim_driver_scaler.json`.
For 2023–2025 use `--years 2023 2024 2025`, `--drivers "$GIMTEC/drivers/drivers_2009_2025_2h.npz"`
and the same weights and scaler; the history for 1 January 2023 is read from `TEC_2022.npy`.
Each evaluation writes `predictions.npy`, `truth.npy` (TECU), `valid.npy`, `origins.npy`
(UTC seconds), `metrics.json` and `manifest.json`. RMSE pools squared errors and valid
counts; zero TEC is valid.

## Train

Use `--action train` with an unused output directory. Seeds are 42, 1024, 2025, 3407 and
5000 (the first three for 96-hour runs). The first strict minimum of validation RMSE is
selected. Separately tuned adaptive PI uses the original entry point:

```bash
python model/Run.py -dataset GIMtec -mode ori -model GWN \
  -config conf/paper/A1_gim_pi_drivers_only.yaml -seed 42
```

TEC-MoLLM runtime options use the `PIGWN_` prefix: `PIGWN_ENABLE_GC=0|1`
controls gradient checkpointing, `PIGWN_SDPA=math|mem|flash` selects attention,
and `PIGWN_NODE_SAFE_CAP` optionally caps the batch-by-node chunk size.

## Calibration, statistics, tables and figures

```bash
python tools/paper/collect_paper_results.py \
  --calibrate-predictions outputs/china_film_42 --out-dir outputs/china_film_42_intervals
python lib/paper_statistics.py --extension-dir source_data/extension
python tools/paper/revised_materials.py --revised-sources source_data --out-dir tables_out
python tools/paper/revised_materials.py --revised-sources source_data --out-dir figures_out --figures-only
```

The figures need Arial, Helvetica or Liberation Sans, which share character widths (on
Debian or Ubuntu, install `fonts-liberation`); other fonts change the panel layout, so the
figure script stops if none of them is installed.

Intervals are calibrated per lead on 1 January–30 June 2015 with the finite-sample "higher"
quantile; coverage is empirical. `lib/paper_statistics.py` implements pooled statistics and
paired 14-day block resampling. In the extracted Supplementary Data 1, `source_data/README.txt` maps
the quantitative figure panels and tables to its Source Data file; `source_data/NAME_MAP.csv` maps
internal identifiers to article names.

## Tests

```bash
PYTHONPATH=. python -m pytest tests -q
```

Two tests of the retained intervals read `source_data/` and are skipped when it is absent.

## License and citation

MIT. Adapted Channel Mixer/OpenSTL inference components keep their Apache-2.0 license in
`third_party/channel_mixer`. The graph–Transformer is an adaptation of Liu & Jiang (2025),
not the authors' code (`model/GNNTrans/ARCHITECTURE.md`). Please cite the article and
`CITATION.cff`.
