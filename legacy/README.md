# Earlier training configuration (unchanged code)

This directory keeps, unchanged, the code and configuration files of the earlier article version that the new
article still cites: the comparison of the training configurations ("baseline strength": GWN and the penalty under the
earlier matched recipe) and the rows of the earlier-configuration table (`records/baselines/BASELINE_TABLE.csv`).
The scores of these models were re-scored from their retained predictions; they are not retrained by this release.

| Earlier-configuration row | Configuration |
|---|---|
| GWN, earlier configuration (matched) | `conf/revised/gim_do_tec.yaml` |
| GWN + physics penalty, projected, earlier configuration | `conf/revised/gim_pi_tec.yaml` |
| GWN, separately tuned | `conf/paper/A1_gim_do_gwn.yaml` |
| GWN + physics penalty, adaptive weights, separately tuned | `conf/paper/A1_gim_pi_drivers_only.yaml` |
| GWN + raw observed-index inputs | `conf/paper/A1_gim_de_gwn.yaml` |
| GWN + standardized observed-index inputs (with projected penalty) | `conf/revised/gim_de_zscore.yaml` (`gim_pi_zscore.yaml`), scaler `conf/revised/gim_driver_scaler.json` |
| GWN + gated, cross-attention, FiLM observed-index conditioning | `conf/revised/gim_de_gated.yaml`, `gim_de_crossattn.yaml`, `gim_de_film.yaml` |
| pretrained Transformer with low-rank adaptation | `conf/paper/A2_gim_tecmollm.yaml`, `conf/TEC_MoLLM/` |
| graph-Transformer adaptation (of Liu and Jiang, 2025; not an exact reproduction) | `conf/revised/gim_gnntrans.yaml`, `model/GNNTrans/ARCHITECTURE.md` |
| external Mixer-SimVP checkpoint | `conf/revised/gim_mixer.yaml`, `lib/external_mixer*.py`, `third_party/channel_mixer/` |

Run from this directory, as in the earlier release, e.g.

```bash
cd legacy
python model/Run.py --paper-config conf/revised/gim_pi_tec.yaml --action evaluate --data-root "$GIMTEC" \
  --adjacency "$ASSETS/gim_adj.npy" --checkpoint /path/to/weights/seed42/best.pt \
  --years 2015 2019 2020 2021 2022 --seed 42 --device cpu --output outputs/pi_tec_42
```
