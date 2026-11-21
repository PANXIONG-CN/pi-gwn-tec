"""Build the prepared model assets from the GIMtec files and check them against the recorded ones.

    python scripts/prepare_assets.py --gimtec /data/gimtec --assets /data/tecfc_assets

Writes into --assets:
  gim_adj.npy                          eight-neighbour 71 x 73 GIM graph (legacy/lib/datasets/gimtec_adj.py)
  prepared/graph_svd_v1.npz            periodic 72-longitude graph and its rank-10 SVD (prepare_graph.py)
  prepared/train_climatology_v1.npz    training-year climatology and the 147.1 TECU scale (gim_data.py climatology)
  prepared/geometry.npy                cos(SZA) at the native GIM times, SimVP (prepare_geometry.py)

The three preparation programs are the original ones; each runs in a copy of
its training directory, as in the experiments.  Every result is compared with
configs/inputs/assets.json: byte-identical files for .npy, identical arrays for
.npz (zip headers carry the creation time).  Reading the seven training years
takes a few minutes.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tecfc.training import families  # noqa: E402


def build_adjacency(target):
    sys.path.insert(0, str(REPO / 'legacy'))
    spec = importlib.util.spec_from_file_location('_legacy_gimtec_adj', REPO / 'legacy/lib/datasets/gimtec_adj.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import numpy as np
    np.save(target, module.build_grid8_adj(71, 73))


def run_in_copy(bundle, files, command, gimtec):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in files:
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / 'tecfc/models' / bundle / name, work / name)
        (work / 'data').mkdir()
        for name in ('gim', 'gim_extension'):
            (work / 'data' / name).symlink_to(Path(gimtec).resolve(), target_is_directory=True)
        (work / 'prepared').mkdir()
        subprocess.run([sys.executable, '-B'] + command, cwd=work, check=True)
        return {p.name: p.read_bytes() for p in (work / 'prepared').iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gimtec', required=True); parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    assets = Path(args.assets); (assets / 'prepared').mkdir(parents=True, exist_ok=True)
    record = json.loads((REPO / 'configs/inputs/assets.json').read_text())['assets']
    adjacency = assets / 'gim_adj.npy'
    if not adjacency.exists():
        build_adjacency(adjacency)
    made = {}
    # gim_data.py's own command line opens its default data locations; call its
    # build_climate() with the GIMtec directory instead.
    made.update(run_in_copy('gwn_reference', ('gim_data.py',),
                            ['-c', 'import sys, gim_data as g; g.build_climate(g.GIM(sys.argv[1], sys.argv[1]))',
                             str(Path(args.gimtec).resolve())], args.gimtec))
    made.update(run_in_copy('gwn_reference', ('prepare_graph.py', 'gim_data.py'),
                            ['prepare_graph.py', str(adjacency.resolve())], args.gimtec))
    made.update(run_in_copy('simvp_reference', ('prepare_geometry.py', 'gim_data.py', 'repo/lib/physics/solar.py'),
                            ['prepare_geometry.py'], args.gimtec))
    for name in ('train_climatology_v1.npz', 'graph_svd_v1.npz', 'geometry.npy'):
        target = assets / 'prepared' / name
        if not target.exists():
            target.write_bytes(made[name])
    report = {}
    for name, expected in record.items():
        path = assets / name
        same_bytes = families.sha(path) == expected['sha256']
        same_content = same_bytes or ('content_sha256' in expected and families.content_sha256(path) == expected['content_sha256'])
        report[name] = dict(identical_bytes=same_bytes, identical_content=same_content)
    print(json.dumps(report, indent=1))
    if not all(r['identical_content'] for r in report.values()):
        raise SystemExit('prepared assets differ from the recorded ones')


if __name__ == '__main__':
    main()
