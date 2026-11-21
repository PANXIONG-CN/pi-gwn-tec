"""Explicit paths, immutable records, and finite exp0023 identities."""
import hashlib
import importlib.util
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
ARMS = ['R2', 'R2-B']
CANDIDATES = [f'{arm}/{gate}' for arm in ARMS for gate in ['G0', 'G1']]
TRAIN1 = [2014, 2015, 2019, 2020, 2021]
TRAIN2 = TRAIN1 + [2022]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''): h.update(data)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def spec(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m

def job_name(region, arm, seed, stage='train'):
    return f'{region}_{stage}_{arm.lower().replace("-", "")}_{seed}'

def output(job): return EXP / 'artifacts/remote' / job / 'outputs' / job
