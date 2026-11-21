"""The earlier-configuration code in legacy/ still imports without the removed modules."""
import subprocess
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / 'legacy'


def test_legacy_modules_import():
    code = ('import sys; sys.argv=["x"]; '
            'import model.Model, model.GWN.GWN, model.GNNTrans.gnntrans, model.TEC_MoLLM.tec_mollm, '
            'lib.paper_recipe, lib.paper_statistics, lib.external_mixer, lib.physics.ppinn_projected, lib.datasets.gimtec_adj')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=LEGACY, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
