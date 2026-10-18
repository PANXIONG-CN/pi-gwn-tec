# Configurations used in the article

| Article item | Model | Configuration |
|---|---|---|
| Table 2, Fig. S1–S2 (separately tuned) | DO | `conf/paper/A1_gim_do_gwn.yaml` |
| Table 2, Fig. S1–S2 (separately tuned) | adaptive PI (drivers in loss only; λ 0.1, batch 96, accumulation 2, AMP) | `conf/paper/A1_gim_pi_drivers_only.yaml` |
| Table 2, Fig. S1 (separately tuned) | raw DE | `conf/paper/A1_gim_de_gwn.yaml` |
| Table 2, Table 6 | TEC-MoLLM | `conf/TEC_MoLLM/GIMtec.conf` |
| SI budget table | TEC-MoLLM budget control | `conf/paper/T133_gim_tecmollm_orig_budget.yaml`, `conf/paper/T133_gim_tecmollm_matched_budget_elastic_nc512_gc.yaml` (`PIGWN_ENABLE_GC=0/1`) |
| Fig. 2, Table 2 (matched) | DO TEC / PI TEC projected | `conf/revised/gim_do_tec.yaml`, `conf/revised/gim_pi_tec.yaml` |
| Fig. 2, Table 2 (matched) | DE ZScore / PI ZScore projected | `conf/revised/gim_de_zscore.yaml`, `conf/revised/gim_pi_zscore.yaml` |
| Table 2 | FiLM / gated / cross-attention | `conf/revised/gim_de_film.yaml`, `gim_de_gated.yaml`, `gim_de_crossattn.yaml` |
| Table 2 | Channel Mixer–SimVP (external checkpoint) | `conf/revised/gim_mixer.yaml` |
| Table 2 | Graph–Transformer adaptation | `conf/revised/gim_gnntrans.yaml` |
| Table 4, Fig. 5 | FiLM input-category removal | `conf/revised/{gim,china}_de_film_no_{solar,geomagnetic,activity}.yaml` |
| Table 5, Fig. 6 | 96-hour DO / PI / DE | `conf/revised/gim_do_tec_h96.yaml`, `gim_pi_tec_h96.yaml`, `gim_de_zscore_h96.yaml` |
| Table 3, Fig. 3–4 | regional DO | `conf/paper/T143_china_do_gwn_mrmse_bs96acc2_plateau10_es30.yaml` |
| Table 3, Fig. 3–4 | regional raw DE | `conf/paper/T143_china_de_gwn_mrmse_bs96acc2_plateau10_es30.yaml` |
| Table 3, Fig. 3–4 | regional adaptive PI (λ 0.05, cap 0.25) | `conf/paper/T143_china_pi_gwn_mech_rotcap0p25_mrmse_plateau10_es30.yaml` |
| SI regional tables | regional static PI (λ 0.2, cap 0.5) | `conf/paper/T143_china_pi_gwn_static_mrmse_bs96acc2_plateau10_es30.yaml` |
| Table 3 | regional DE ZScore / FiLM | `conf/revised/china_de_zscore.yaml`, `conf/revised/china_de_film.yaml` |
| SI selection tables | 16 regional selection configurations | listed individually below |
| SI sensitivity tables | coefficient perturbations | `conf/paper/T98_sens_*`, `T132_*`, `T139_*` |

## Regional selection configurations (Supplementary Table S3)

Each of the sixteen configurations was crossed with the five processed regional products (80 candidates;
Supplementary Table S4). The selected candidate is product 5 with configuration 12.

| No. | Identifier | Configuration |
|---|---|---|
| 1 | de_canonical | `conf/paper/C1_china_de_gwn.yaml` |
| 2 | de_combo | `conf/paper/T143_china_de_gwn_mrmse_bs96acc2_plateau10_es30.yaml` |
| 3 | de_mrmse | `conf/paper/T143_china_de_gwn_mrmse.yaml` |
| 4 | do_canonical | `conf/paper/C1_china_do_gwn.yaml` |
| 5 | do_combo | `conf/paper/T143_china_do_gwn_mrmse_bs96acc2_plateau10_es30.yaml` |
| 6 | do_mrmse | `conf/paper/T143_china_do_gwn_mrmse.yaml` |
| 7 | mech_2019_k0p03 | `conf/paper/T142_china_pi_gwn_mech_rotcap0p25_k0p03.yaml` |
| 8 | mech_2019_k0p03_roti0p55 | `conf/paper/T143_china_pi_gwn_mech_rotcap0p25_k0p03_roti0p55.yaml` |
| 9 | mech_2019_rotcap0p30_k0p02_roti0p55 | `conf/paper/T143_china_pi_gwn_mech_rotcap0p30_k0p02_roti0p55.yaml` |
| 10 | mech_agg_k0p02_mrmse_bs96acc2 | `conf/paper/T142_china_pi_gwn_mech_rotcap0p25_k0p02_mrmse_bs96acc2.yaml` |
| 11 | mech_agg_mrmse | `conf/paper/T142_china_pi_gwn_mech_rotcap0p25_mrmse.yaml` |
| 12 | mech_agg_mrmse_plateau10_es30 | `conf/paper/T143_china_pi_gwn_mech_rotcap0p25_mrmse_plateau10_es30.yaml` |
| 13 | mech_anchor_rotcap0p25 | `conf/paper/T139_china_pi_gwn_mech_rotcap0p25.yaml` |
| 14 | pi_static_canonical | `conf/paper/C1_china_pi_gwn_static.yaml` |
| 15 | pi_static_combo | `conf/paper/T143_china_pi_gwn_static_mrmse_bs96acc2_plateau10_es30.yaml` |
| 16 | pi_static_mrmse | `conf/paper/T143_china_pi_gwn_static_mrmse.yaml` |

The ST-Patch Transformer described in the Supplementary Information is configured by
`conf/STPatchTransformer/`, `conf/paper/T98_stpatch_transformer_*` and `T110_*`; it is not used
for results in the revised article. Renamed files are listed in `conf/paper/RENAMED.csv`.

The TEC-MoLLM configuration is named `conf/paper/A2_gim_tecmollm.yaml`; its model parameters are unchanged.
