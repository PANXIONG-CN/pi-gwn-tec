"""Bind registered jobs to exact donor configuration and frozen source hashes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
EXP = ROOT / 'experiments/exp0026'
SEEDS = [42, 1024, 2025, 3407, 5000]
ARMS = ['P-0.1', 'P-1.0', 'D-noSolar', 'D-noGeomag', 'D-noIndex', 'G-nowrap', 'P-adapt']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
