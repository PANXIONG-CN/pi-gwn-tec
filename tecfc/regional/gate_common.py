"""Paths and immutable-record helpers for the single exp0021 experiment."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE
REPO = Path(os.environ.get('TECFC_OUTPUT', '.')).resolve()
PROJECT = REPO
EXP = REPO
OLD = REPO
PREP = EXP / 'artifacts/prepared'
META = EXP / 'artifacts/ops'
SEEDS = [42, 1024, 2025, 3407, 5000]
REGIONS = ['china', 'japan']
ARMS = ['R1', 'R1-B']
CANDIDATES = [f'{arm}/{gate}' for arm in ARMS for gate in ['G0', 'G1', 'G2']]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''): h.update(data)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def spec(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

def now(): return datetime.now(timezone.utc).isoformat()

def job_name(region, arm, seed): return f'{region}_{arm.lower().replace("-", "")}_{seed}'

def output(job): return EXP / 'artifacts/remote' / job / 'outputs' / job
