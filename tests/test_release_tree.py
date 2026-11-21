"""Release-tree checks: every tecfc import resolves to a source file of this tree, and no Python source is git-ignored
(a git-ignored source would be missing from a release archive built from the repository).
"""
import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'.git', '__pycache__'}


def _sources():
    for p in sorted(REPO.rglob('*.py')):
        rel = p.relative_to(REPO)
        if SKIP_PARTS & set(rel.parts) or any(part.startswith('.venv') for part in rel.parts):
            continue
        yield p, rel


def _tecfc_imports():
    found = set()
    for p, rel in _sources():
        tree = ast.parse(p.read_text(encoding='utf-8'), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                    and node.module.split('.')[0] == 'tecfc':
                found.add((node.module, str(rel)))
            elif isinstance(node, ast.Import):
                found.update((a.name, str(rel)) for a in node.names if a.name.split('.')[0] == 'tecfc')
    return sorted(found)


def _resolves(module):
    path = REPO.joinpath(*module.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def test_every_tecfc_import_resolves():
    imports = _tecfc_imports()
    assert imports, 'no tecfc imports found; the scan is broken'
    missing = [(m, f) for m, f in imports if not _resolves(m)]
    assert not missing, f'imports without a source file in this tree: {missing}'


@pytest.mark.skipif(not (REPO / '.git').exists(), reason='not a git checkout')
def test_no_python_source_is_git_ignored():
    out = subprocess.run(['git', '-C', str(REPO), 'ls-files', '--others', '--ignored', '--exclude-standard', '--', '*.py'],
                         capture_output=True, text=True, check=True).stdout.split()
    ignored = [f for f in out if not any(part.startswith('.venv') for part in Path(f).parts)]
    assert not ignored, f'git-ignored Python sources (they would be missing from a release): {ignored}'
