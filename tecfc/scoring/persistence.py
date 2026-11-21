"""24-h persistence (previous UT day, same epochs) on all four periods plus per-year splits, same scoring entry."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tecfc.scoring import reference_common as e29  # noqa: E402


def main():
    out = dict(experiment='exp0029', stage='persistence_periods', at_utc=datetime.now(timezone.utc).isoformat(), periods={})
    for name in ['validation', 'old_evaluation', 'extension_2023_2025', 'descriptive_2026']:
        dates = e29.period_dates(name)
        desc = e29.make_desc(name, dates, dict(persistence=e29.persistence_method()))
        ts, names, st = e29.stats_for(desc, progress=500)
        s = st[0]
        row = dict(windows=int(len(ts)), **{k: v for k, v in e29.rmse_of(s).items()})
        years = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
        row['by_year'] = {str(y): dict(windows=int((years == y).sum()), RMSE=e29.rmse_of(s[years == y])['RMSE'], MAE=e29.rmse_of(s[years == y])['MAE']) for y in sorted(set(years))}
        if name == 'extension_2023_2025':
            months = ts.astype('datetime64[s]').astype('datetime64[M]').astype(int) % 12 + 1
            sel = months <= 10
            row['january_to_october_subset'] = dict(windows=int(sel.sum()), RMSE=e29.rmse_of(s[sel])['RMSE'])
        out['periods'][name] = row
        print(name, row['windows'], row['RMSE'], flush=True)
    frozen = json.loads((e29.ROOT / 'experiments/exp0025/result.json').read_text())['datasets']
    f26 = frozen['global_2026']['full']['methods']['persistence']['all']['RMSE']
    fdev = frozen['global_development']['full']['methods']['persistence']['all']['RMSE']
    out['cross_check_vs_exp0025_frozen'] = dict(
        descriptive_2026=dict(frozen=f26, here=out['periods']['descriptive_2026']['RMSE'], equal=bool(f26 == out['periods']['descriptive_2026']['RMSE'])),
        extension_Jan_Oct=dict(frozen=fdev, here=out['periods']['extension_2023_2025']['january_to_october_subset']['RMSE'],
                               equal=bool(fdev == out['periods']['extension_2023_2025']['january_to_october_subset']['RMSE'])))
    path = e29.EXP / 'PERSISTENCE_PERIODS.json'
    assert not path.exists()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out['cross_check_vs_exp0025_frozen']))


if __name__ == '__main__':
    main()
