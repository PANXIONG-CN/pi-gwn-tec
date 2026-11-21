"""Download CODE one-day predicted GIM (C1PG; long name COD0OPSP0D) from NASA CDDIS, 2023-01-01 to 2026-10-31.

Authentication uses the existing ~/.netrc through requests (never read, printed or logged here).
Only CDDIS paths under /archive/gnss/products/ionex/ are requested. 404 is recorded as a missing day, not retried.
Downloaded files are untrusted data: they are only written to the dated output directory and parsed elsewhere.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

BASE = 'https://cddis.nasa.gov/archive/gnss/products/ionex'
THREAD = threading.local()
LOCK = threading.Lock()


class AuthBlocked(RuntimeError):
    pass


def session():
    if not hasattr(THREAD, 's'):
        THREAD.s = requests.Session()
        THREAD.s.headers.update({'User-Agent': 'tecGPT-exp0029-c1pg-fetch/1.0', 'Connection': 'close'})
    return THREAD.s


def now():
    return datetime.now(timezone.utc).isoformat()


def name_for(y, d):
    return f'COD0OPSP0D_{y}{d:03d}0000_01D_01H_GIM.INX.gz'


def listing(y, d, out):
    path = out / 'LISTINGS' / f'{y}_{d:03d}.txt'
    if path.exists():
        return
    try:
        r = session().get(f'{BASE}/{y}/{d:03d}/', timeout=(15, 60))
        if urlparse(r.url).hostname != 'cddis.nasa.gov':
            raise AuthBlocked('listing redirected to ' + str(urlparse(r.url).hostname))
        names = sorted(set(n.strip() for n in re.findall(r'href="([^"?/][^"]*)"', r.text)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(url=f'{BASE}/{y}/{d:03d}/', http_status=r.status_code, retrieved_utc=now(), names=names), indent=1) + '\n')
    except AuthBlocked:
        raise
    except Exception as e:  # listing is only evidence for missing days
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(url=f'{BASE}/{y}/{d:03d}/', error=type(e).__name__, retrieved_utc=now()), indent=1) + '\n')


def fetch_one(day, out):
    y, d = int(str(day)[:4]), int((day - np.datetime64(str(day)[:4] + '-01-01')).astype(int)) + 1
    fname = name_for(y, d)
    url = f'{BASE}/{y}/{d:03d}/{fname}'
    dest = out / 'raw' / str(y) / f'{d:03d}' / fname
    rec = dict(date=str(day), year=y, doy=d, url=url, file=str(dest.relative_to(out)))
    if dest.exists() and dest.stat().st_size > 0:
        try:
            gzip.decompress(dest.read_bytes())
            rec.update(status='present', http_status=200, bytes=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), reused=True)
            return rec
        except Exception:
            dest.unlink()
    errors = []
    for attempt in range(1, 6):
        try:
            r = session().get(url, timeout=(15, 120))
            host = urlparse(r.url).hostname
            if host == 'urs.earthdata.nasa.gov' or r.status_code in (401, 403):
                raise AuthBlocked('CDDIS/Earthdata authentication failed')
            if r.status_code == 404:
                rec.update(status='missing_404', http_status=404, attempts=attempt, errors=errors)
                listing(y, d, out)
                return rec
            r.raise_for_status()
            if host != 'cddis.nasa.gov' or 'html' in r.headers.get('Content-Type', '').lower():
                raise AuthBlocked('Non-product/HTML response rejected')
            body = r.content
            if r.headers.get('Content-Length') and int(r.headers['Content-Length']) != len(body):
                raise IOError('short body')
            gzip.decompress(body)  # integrity of the gzip stream
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + '.part')
            tmp.write_bytes(body)
            tmp.replace(dest)
            rec.update(status='downloaded', http_status=r.status_code, last_modified=r.headers.get('Last-Modified'),
                       content_length=r.headers.get('Content-Length'), retrieved_utc=now(), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest(), attempts=attempt, errors=errors)
            return rec
        except AuthBlocked:
            raise
        except Exception as e:
            errors.append(dict(attempt=attempt, type=type(e).__name__))
            time.sleep(min(2 ** attempt, 20))
    rec.update(status='failed_after_5_attempts', attempts=5, errors=errors)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--first', default='2023-01-01')
    p.add_argument('--last', default='2026-10-31')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    out = a.out
    days = np.arange(np.datetime64(a.first), np.datetime64(a.last) + 1, dtype='datetime64[D]')
    started = now()
    manifest = out / 'MANIFEST.jsonl'
    done = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            if r.get('status') in ('downloaded', 'present', 'missing_404'):
                done[r['date']] = r
    todo = [d for d in days if str(d) not in done]
    print(json.dumps(dict(total=len(days), already=len(done), todo=len(todo), started=started)), flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for rec in ex.map(lambda d: fetch_one(d, out), todo):
            with LOCK:
                with manifest.open('a') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done[rec['date']] = rec
            n += 1
            if n % 100 == 0:
                print(json.dumps(dict(progress=n, of=len(todo), last=rec['date'], status=rec['status'])), flush=True)
    status = {}
    for r in done.values():
        status[r['status']] = status.get(r['status'], 0) + 1
    summary = dict(source=BASE + '/{YYYY}/{DOY}/', product='CODE 1-day predicted GIM (C1PG; long name COD0OPSP0D)',
                   first=a.first, last=a.last, days_requested=len(days), status_counts=status, started_utc=started, finished_utc=now(),
                   missing_days=sorted(r['date'] for r in done.values() if r['status'] == 'missing_404'),
                   failed_days=sorted(r['date'] for r in done.values() if r['status'].startswith('failed')),
                   python=sys.version.split()[0], requests=requests.__version__,
                   credentials='existing ~/.netrc via requests; never read or logged')
    (out / 'DOWNLOAD_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
