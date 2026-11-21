"""Build the training-year table of the gated nighttime penalty term and check it against the record.

    python scripts/prepare_night_gate.py --gimtec /data/gimtec --assets /data/tecfc_assets

Needed only by the configurations with the gated nighttime term
(configs/gwn_gated_nighttime_term_10pct.json, configs/gwn_penalty_10pct_gated_nighttime.json).
Run scripts/prepare_assets.py first: the training windows are enumerated by the
original GWN data code, which reads prepared/train_climatology_v1.npz.

Writes into --assets:
  prepared/NIGHT_GATE.npz    count, increase and gate [Kp class, month, local-time bin, latitude band]
  prepared/NIGHT_GATE.json   definition and summary

The table is computed by tecfc/objectives/night_gate.py derive() from the seven
training years and the penalty driver store only; no validation or evaluation
year is opened.  The arrays are compared with configs/inputs/night_gate.json
by content, so that a NumPy version writing different zip headers is accepted
(with NumPy 2.4.6 the rebuilt file is byte-identical to the recorded one).
Nothing is written when the arrays differ.  CPU, under a minute.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tecfc.training import families  # noqa: E402

CONFIG = REPO / 'configs/gwn_gated_nighttime_term_10pct.json'
TRAIN_YEARS = (2009, 2010, 2011, 2012, 2014, 2016, 2018)


def compare(path, record):
    observed = families.content_sha256(path)
    return {name: observed.get(name) == digest for name, digest in record['content_sha256'].items()} | \
        {name: False for name in observed if name not in record['content_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gimtec', required=True); parser.add_argument('--assets', required=True)
    parser.add_argument('--force', action='store_true', help='rebuild even if prepared/NIGHT_GATE.npz exists')
    args = parser.parse_args(argv)
    record = json.loads((REPO / 'configs/inputs/night_gate.json').read_text())
    target = Path(args.assets) / record['file']
    if target.exists() and not args.force:
        same = compare(target, record)
        print(json.dumps(dict(file=str(target), existing=True, identical_arrays=same), indent=1))
        return 0 if all(same.values()) else 1
    paper = families.load(CONFIG)
    config = families.runtime_config(paper, 42, families.locations(args.gimtec, args.assets))
    for year in TRAIN_YEARS:
        row = config['data_files'][str(year)]
        if families.sha(row['path']) != row['sha256']:
            raise SystemExit(f'TEC_{year}.npy differs from the recorded file')
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / 'config.json'
        config_path.write_text(json.dumps(config))
        from tecfc.objectives import night_gate
        summary = night_gate.derive(config_path, Path(tmp) / 'out')
        made = Path(tmp) / 'out' / 'NIGHT_GATE.npz'
        same = compare(made, record)
        expected = record['summary']
        counts = {key: (summary[key], expected[key]) for key in ('training_days', 'eligible_elements',
                                                                  'increase_elements', 'gate_on_bins', 'bins_with_data')}
        print(json.dumps(dict(identical_arrays=same, counts_observed_recorded=counts), indent=1))
        if not all(same.values()):
            raise SystemExit('night-gate table differs from the recorded one; nothing written')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(made, target)
        shutil.copyfile(Path(tmp) / 'out' / 'NIGHT_GATE.json', target.with_suffix('.json'))
    print(json.dumps(dict(written=str(target), sha256=families.sha(target), recorded_file_sha256=record['sha256'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
