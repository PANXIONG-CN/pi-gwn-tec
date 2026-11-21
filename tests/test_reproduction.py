"""Key numbers of the article recomputed from the daily statistics (no weights).

Runs when TECFC_SD1 points to Supplementary Data 1 with its daily statistics unpacked into statistics/ (available
on request), or TECFC_EXPERIMENT_REPO to the experiment repository; otherwise skipped.
"""
import csv
import os
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
SD1, EXP = os.environ.get('TECFC_SD1'), os.environ.get('TECFC_EXPERIMENT_REPO')
pytestmark = [pytest.mark.skipif(not (SD1 or EXP), reason='set TECFC_SD1 (or TECFC_EXPERIMENT_REPO)'),
              pytest.mark.skipif(bool(SD1) and not EXP and not any(Path(SD1, 'statistics').glob('*.npz')),
                                 reason='TECFC_SD1 has no daily statistics (statistics/*.npz, available on request)')]

KEY = ['k.k1.delta.conf26', 'k.k2.delta.conf26', 'k.k3.delta.conf26', 'k.k4.delta.conf26',
       'k.k5.cn.delta.conf26', 'k.k5.jp.delta.conf26', 'k.k6.cn.delta.conf26', 'k.k6.jp.delta.conf26',
       'k.k7.cn.delta.conf26', 'k.k7.jp.delta.conf26',
       'g.mfe.rmse.conf26', 'g.mfe.rmse.dev', 'g.gwn-ens.rmse.conf26', 'r.cn.anchres.rmse.conf26', 'r.jp.anchres.rmse.conf26']


@pytest.fixture(scope='module')
def documents():
    import reproduce_tables as R
    stats = R.Statistics(EXP, None) if EXP else R.Statistics(None, SD1)
    return {source: builder(stats) for source, builder in R.BUILDERS.items()
            if source in ('experiments/exp0025/result.json', 'experiments/exp0019/result.json')}


def test_key_numbers(documents):
    import reproduce_tables as R
    rows = [r for r in csv.DictReader(open(REPO / 'records/EVIDENCE.csv')) if r['id'] in KEY]
    assert {r['id'] for r in rows} == set(KEY)
    report = R.compare(rows, documents)
    assert all(e['status'] == 'match' for e in report), [e for e in report if e['status'] != 'match']


def test_every_reproducible_row(documents):
    import reproduce_tables as R
    assert R.main(['--experiment-repo', EXP] if EXP else ['--sd1', SD1]) == 0
