# Records

The evidence table of the article and the files that `scripts/reproduce_tables.py` reads. `NAME_MAP.csv` translates
the internal experiment numbers (`exp0013`, ...) and system names (`S2`, `G-plus-M`, ...) used in these files and in
the code into the names used in the article. The preregistration texts, commit timeline and freeze records of the
2026 confirmation are in `records/` of Supplementary Data 1.

| Path | Content |
|---|---|
| `EVIDENCE.csv` | one row per number of the article (6,797 rows; the same file as in Supplementary Data 1) with its source record and JSON pointer. Ids read `<group>.<subject>.<quantity>.<period>[.<qualifier>]`: groups `k` (preregistered 2026 criteria), `g` (global systems), `r` (regional systems), `p` (loss penalty), `i` (physical inputs), `s` (longitude wrap), `h` (96-hour forecasts), `c` (interval calibration), `b` (baselines, CODE forecast, cost), `o` (other checks); periods `val`, `orig`, `dev`, `devsm`, `conf26`, `h2014` |
| `NAME_MAP.csv` | internal name, article name and evidence id slug |
| `scoring/` | scoring descriptors of the 2026 tables and recorded values that are not functions of the daily statistics (training-time diagnostics, thresholds, counts) |
| `baselines/` | baseline table, CODE one-day forecast scores and release times, persistence by period, computation cost, the cost reported for Mixer-SimVP (Liu et al. 2024, Table 2) and a configuration template for `tecfc/scoring/code_forecast.py` |
| `earlier_configuration/` | summaries of the earlier training configuration |
| `training_logs/` | per-epoch training logs referenced by the configurations |
