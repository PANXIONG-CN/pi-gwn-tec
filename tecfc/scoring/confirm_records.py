"""Metadata-only registration and strict A/B audit for exp0025.

No numpy/torch import, observation parsing, fitting, inference or scoring.
Expected hashes are inherited from committed records, never from current weights.
"""
import argparse
import hashlib
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
EXP = ROOT
SEEDS = [42, 1024, 2025, 3407, 5000]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
