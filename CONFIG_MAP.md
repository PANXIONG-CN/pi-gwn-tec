# Figures and tables -> configurations, commands, evidence and data

Numbers: every number of a figure or table is a row of `records/EVIDENCE.csv`; the column "Evidence" gives the id
prefix (`<group>.<system>.<quantity>.<period>`, see `records/README.md`). `python scripts/reproduce_tables.py --sd1
<Supplementary Data 1>` (path 1 of `README.md`) recomputes every row that is a function of the daily statistics and
checks the others against the records they cite; this needs only Supplementary Data 1 with its daily statistics.
Retraining a configuration (path 2) needs the data in the last column.

Data keys: **G** GIMtec record (global maps 2009-2025, indices); **G26** global maps 2026 (new GIMtec version);
**J** regional JPLD arrays (new GIMtec version); **A** prepared assets (`scripts/prepare_assets.py`; the gate table of
the gated nighttime term: `scripts/prepare_night_gate.py`);
**W** trained weights (`checkpoints.zip` of the Zenodo record of this code); **C** CODE C1PG files (NASA CDDIS).

Commands: `run` = `python -m tecfc.training.run --config <config> --seed <s> --out <dir>`;
`predict` = `python -m tecfc.training.predict --config <config> --seed <s> --checkpoint <dir>/best_200.pt --split <split> --out <file>`
(GWN and SimVP: checkpoint `runs/<identity>/best.pt` inside the output directory); `score` =
`python scripts/score_predictions.py --predictions <file> --gimtec <dir> [--reference <file>]`.

## Configurations

| Configuration (article name) | File | Training entry | Data |
|---|---|---|---|
| GWN | `configs/gwn.json` | `tecfc/models/gwn_reference/train_global.py` (run in a copy of the directory) | G, A |
| GWN + penalty (1%) | `configs/gwn_penalty_1pct.json` | `tecfc/training/train_penalty.py` | G, A |
| GWN + empirical-tendency penalty | `configs/gwn_empirical_tendency_penalty.json` | `tecfc/training/train_penalty.py` | G, A |
| GWN + penalty (10%), (100%) | `configs/gwn_penalty_10pct.json`, `configs/gwn_penalty_100pct.json` | `tecfc/training/train_ablation.py` | G, A |
| GWN + cosSZA | `configs/gwn_cossza.json` | `tecfc/training/train_ablation.py` | G, A |
| GWN + cosSZA + geomagnetic indices | `configs/gwn_cossza_geomagnetic_indices.json` | `tecfc/training/train_ablation.py` | G, A |
| GWN + cosSZA + F10.7 | `configs/gwn_cossza_f107.json` | `tecfc/training/train_ablation.py` | G, A |
| GWN + cosSZA + indices | `configs/gwn_cossza_indices.json` | `tecfc/training/train_members.py` | G, A |
| GWN-geo | `configs/gwn_geo.json` | `tecfc/training/train_members.py` | G, A |
| GWN-hist | `configs/gwn_hist.json` | `tecfc/training/train_members.py` | G, A |
| GWN (MAE) | `configs/gwn_mae.json` | `tecfc/training/train_clim.py` | G, A |
| GWN-geo-hist (MAE), (SmoothL1) | `configs/gwn_geo_hist_mae.json`, `configs/gwn_geo_hist_smoothl1.json` | `tecfc/training/train_combined.py` | G, A |
| GWN without longitude wrap | `configs/gwn_without_longitude_wrap.json` | `tecfc/training/train_ablation.py` | G, A |
| SimVP | `configs/simvp.json` | `tecfc/models/simvp_reference/train_global.py` (run in a copy of the directory) | G, A |
| SimVP-clim | `configs/simvp_clim.json` | `tecfc/training/train_clim.py` | G, A |
| SimVP-clim-hist | `configs/simvp_clim_hist.json` | `tecfc/training/train_combined.py` | G, A |
| 96-h GWN, 96-h GWN + penalty (1%), 96-h GWN-geo-hist (MAE) | `configs/gwn_96h.json`, `configs/gwn_penalty_1pct_96h.json`, `configs/gwn_geo_hist_mae_96h.json` | `tecfc/training/train_96h.py` | G, A |
| GWN + adaptive penalty weights, unscaled port (not evidence) | `configs/gwn_adaptive_penalty_unscaled_port.json` | `tecfc/training/train_ablation.py` | G, A |
| GWN + non-negativity, ROT, TSI, graph-temporal, production-loss or nighttime term (10%) | `configs/gwn_nonnegativity_term_10pct.json`, `configs/gwn_rot_term_10pct.json`, `configs/gwn_tsi_term_10pct.json`, `configs/gwn_graph_temporal_term_10pct.json`, `configs/gwn_production_loss_term_10pct.json`, `configs/gwn_nighttime_term_10pct.json` | `tecfc/training/train_penalty_arms.py` | G, A |
| GWN + gated nighttime term (10%); GWN + penalty (10%), gated nighttime term | `configs/gwn_gated_nighttime_term_10pct.json`, `configs/gwn_penalty_10pct_gated_nighttime.json` | `tecfc/training/train_penalty_arms.py`; gate table `scripts/prepare_night_gate.py` (`tecfc/objectives/night_gate.py`) | G, A |
| GWN, fixed graph only; GWN, adaptive graph only; GWN without graph convolution; each + penalty (10%) | `configs/gwn_fixed_graph_only.json`, `configs/gwn_adaptive_graph_only.json`, `configs/gwn_without_graph_convolution.json`, and the same names ending in `_penalty_10pct.json` | `tecfc/training/train_penalty_arms.py` | G, A |
| SimVP + penalty (10%) | `configs/simvp_penalty_10pct.json` | `tecfc/training/train_penalty_arms.py` | G, A |
| GWN + cosSZA + indices, gated fusion; FiLM; cross-attention | `configs/gwn_cossza_indices_gated_fusion.json`, `configs/gwn_cossza_indices_film.json`, `configs/gwn_cossza_indices_cross_attention.json` | `tecfc/training/train_conditioning.py` | G, A |
| graph-Transformer adaptation | `configs/graph_transformer_adaptation.json` | `tecfc/training/train_conditioning.py` | G, A |
| Multi-family ensemble (and the variant with the index family) | weights in `records/multi_family_ensemble_weights.json` of Supplementary Data 1 | fit: `tecfc/ensemble/refusion.py` (`validation`), `tecfc/ensemble/fusion_map.py`; apply: `tecfc/ensemble/apply.py` (`fuse`) | W of the six families |
| Zero-shot mapping | bilinear weights from the regional grid | `tecfc/ensemble/apply.py` (`mapping`, `remap`) | J, ensemble forecast |
| Anchored residual model (China, Japan) | final configuration in `records/regional_final_configuration.json` of Supplementary Data 1 (fallback 79.5 TECU in China, 20 fixed epochs) | `tecfc/regional/anchored_residual/source/entry.py` (stages `train`, `refit`, `predict`); windows `tecfc/data/regional_final_inputs.py`; final mean and fallback `tecfc/regional/final_predictions.py` | J, ensemble forecast |
| Static and trailing interval calibration | `tecfc/intervals/spec.py` | `tecfc/intervals/run.py` | predictions (W) |
| 24-h persistence, CODE one-day forecast | `records/baselines/reference_forecasts_config.json` | `tecfc/scoring/persistence.py`, `tecfc/scoring/code_forecast.py`; download `scripts/fetch_c1pg.py`, `scripts/build_c1pg_ionex.py`, `scripts/convert_c1pg.py` | G, G26, C |

Seeds of every trained configuration: 42, 1024, 2025, 3407, 5000.

## Main figures and tables

| Item | Content | Configurations | Evidence |
|---|---|---|---|
| Figure 1 | schematic and data timeline (PowerPoint) | - | no numerical source data |
| Figure 2a | penalty dose-response, four periods | GWN, GWN + penalty (1%, 10%, 100%) | `p.pen1.vs.gwn-ens.*`, `p.pen10.vs.gwn-ens.*`, `p.pen100.vs.gwn-ens.*`, `k.k4.*` |
| Figure 2b | baseline strength, same-seed differences | GWN + penalty (1%) vs GWN; earlier configuration | `p.pen1.*`, `p.old-*` |
| Figure 2c | gradient conflict during training | GWN + penalty (10%, 100%) | `p.pen10.train.*`, `p.pen100.train.*` |
| Figure 2d | 96-h penalty by forecast day | 96-h GWN, 96-h GWN + penalty (1%) | `h.h96-pen1.vs.h96-gwn.*` |
| Figure 3a | physical inputs against GWN | GWN + cosSZA, GWN-geo, GWN-hist, GWN (MAE), GWN-geo-hist (MAE) | `i.*`, `k.k2.*`, `k.k3.*` |
| Figure 3b | observed indices by group | GWN + cosSZA (+ indices, + geomagnetic, + F10.7) | `i.cossza*` |
| Figure 3c | second backbone | SimVP, SimVP-clim, SimVP-clim-hist | `i.simvp*` |
| Figure 3d | 96-h inputs by forecast day | 96-h GWN, 96-h GWN-geo-hist (MAE) | `h.h96-geohist.vs.h96-gwn.*` |
| Figure 4a | longitude wrap | GWN without longitude wrap | `s.*` |
| Figure 4b-e | regional transfer: leads, years, high-value groups | zero-shot mapping, anchored residual model, persistence | `r.cn.*`, `r.jp.*` |
| Figure 5 | ten preregistered 2026 comparisons with development counterparts | all of K1-K7 | `k.*` |
| Figure 6 | interval coverage and width | static and trailing calibration | `c.*` |
| Table 1 | data roles and years | - | `README.md`, `tecfc/models/gwn_reference/gim_data.py` (global), `tecfc/data/regional_windows.py` (regional) |
| Table 2 | configurations: inputs, loss, graph, training years, data needed at inference | this file, `configs/` | - |
| Table 3 | system comparison and computation cost | multi-family ensemble, GWN five-seed ensemble, persistence, CODE one-day forecast, external Mixer-SimVP checkpoint | `g.*`, `b.*` |

## Supplementary figures and tables

| Item | Content | Evidence |
|---|---|---|
| Figure S1 | RMSE by lead time, four periods | `g.*.h02`-`g.*.h24` |
| Figure S2 | 2026 by month | `g.*.m01`-`g.*.m10` |
| Figure S3 | error correlation of the families and ensemble weights | `o.weight.*`; weights in `records/multi_family_ensemble_weights.json` of Supplementary Data 1 |
| Figure S4 | regional months and product difference | `r.*.devsm.m*`, `r.*.conf26.m*`, `r.*.product.*` |
| Figure S5 | 2014 hold-out | `o.h2014.*` |
| Figure S6 | CODE one-day forecast release times | `b.code.release.*` |
| Figure S7 | earlier configuration: adaptive term weights | no numerical source (earlier article) |
| Earlier-configuration and reference table | old models re-scored, persistence, CODE forecast, Mixer-SimVP, IRI-2020 | `b.*` (`records/baselines/BASELINE_TABLE.csv`; code of the earlier configurations in `legacy/`) |
| Interval tables | coverage and width by system, period, nominal level; quantiles by lead | `c.*` |
| 96-h tables | four days and all 48 frames, four periods | `h.*` |
| Single penalty terms and the gated nighttime term | GWN + one term (non-negativity, ROT, TSI, graph-temporal, production-loss, nighttime) at 10%, + gated nighttime term (10%), + penalty (10%) with the gated nighttime term; four periods; the gate table | `p.term-nonneg*`, `p.term-rot*`, `p.term-tsi*`, `p.term-graph*`, `p.term-prodloss*`, `p.term-night*`, `p.term-night-gated*`, `p.pen10-gated*`; `p.nightgate.*` |
| Graph support x penalty and the penalty on SimVP | GWN with the fixed graph only, the adaptive graph only or no graph convolution, each without and with the penalty (10%); SimVP + penalty (10%); penalty-effect interactions | `p.gwn-fixed*`, `p.gwn-adapt*`, `p.gwn-nograph*`, `p.simvp-pen10*`, `p.fact.*` |
| Conditioning and graph-Transformer adaptation (strong configuration) | GWN + cosSZA + indices with gated fusion, FiLM or cross-attention, and the graph-Transformer adaptation, against GWN + cosSZA + indices, GWN and GWN + cosSZA; 2026 on the 292 windows with complete indices | `b.cossza-idx-gate*`, `b.cossza-idx-film*`, `b.cossza-idx-xattn*`, `b.graph-transformer*`, `b.cossza.rmse.conf26.w292`, `b.gwn-ens.rmse.conf26.w292` |
| Rapid and forecast-time input maps | multi-family ensemble and GWN five-seed ensemble run unchanged with final, rapid and forecast-time input maps; persistence and CODE one-day forecast on the same windows | `b.lat.*` |
| Activity classes and storm events (descriptive) | days per class and storm events by year and period; RMSE, MAE and bias by class (quiet, moderate, storm), globally and over China and Japan; event-level comparisons | `g.days.*`, `g.events.*`, `r.days.*`, `r.events.*`; `g.*` and `r.cn.*`, `r.jp.*` ending in `.quiet`, `.moderate`, `.dist`; `*.evcount.*`, `*.evlower.*`, `*.evdelta*` |
| Computation cost table | two-round CPU timing, GPU training cost and memory; Mixer-SimVP as reported by its authors (`records/baselines/LITERATURE_COST.csv`) | `b.cost.*` (Mixer-SimVP: `b.cost.mixer-simvp.*`) |

## Earlier configuration

The earlier-configuration rows (GWN and penalty under the earlier training configuration, the separately tuned
models, the index-conditioning variants, the pretrained-Transformer and graph-Transformer models, the external
Mixer-SimVP checkpoint) were re-scored from retained predictions and are listed in
`records/baselines/BASELINE_TABLE.csv`. Their code and configuration files are in `legacy/` (see `legacy/README.md`).
The gated, FiLM and cross-attention
conditioning and the graph-Transformer adaptation were also retrained with the strong configuration (rows above).
