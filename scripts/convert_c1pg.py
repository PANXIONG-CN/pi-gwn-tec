"""IONEX reader, per-file audit and yearly conversion for CODE C1PG (COD0OPSP0D) files.

Primary reading method follows the old converter (tools/build_c1pg_ionex.py): line scan, token split, first-day maps,
UT hours 0,2,...,22, value * 10**(EXPONENT+1) stored as 0.1 TECU float32; no spatial or temporal interpolation.
Audit and conversion never fill a missing or rejected day.
"""
import argparse
import csv
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GRID = (71, 73)
LAT = np.arange(87.5, -87.6, -2.5)
EVEN = list(range(0, 24, 2))
MONTHS = {m: i + 1 for i, m in enumerate(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}


def label(line):
    return line[60:80].strip()


def parse_header(lines):
    h = dict(comments=[], descriptions=0)
    for ln in lines:
        lab = label(ln)
        if lab == 'END OF HEADER':
            break
        if lab == 'PGM / RUN BY / DATE':
            h['pgm'] = ln[:20].strip(); h['run_by'] = ln[20:40].strip(); h['stamp'] = ln[40:60].strip()
        elif lab == 'COMMENT':
            h['comments'].append(ln[:60].strip())
        elif lab == 'DESCRIPTION':
            h['descriptions'] += 1
        elif lab == 'EPOCH OF FIRST MAP':
            h['first_epoch'] = tuple(int(x) for x in ln[:60].split())
        elif lab == 'EPOCH OF LAST MAP':
            h['last_epoch'] = tuple(int(x) for x in ln[:60].split())
        elif lab == 'INTERVAL':
            h['interval'] = int(ln[:60].split()[0])
        elif lab == '# OF MAPS IN FILE':
            h['n_maps'] = int(ln[:60].split()[0])
        elif lab == 'EXPONENT':
            h['exponent'] = int(ln[:60].split()[0])
        elif lab == 'LAT1 / LAT2 / DLAT':
            h['lat'] = [float(x) for x in ln[:60].split()]
        elif lab == 'LON1 / LON2 / DLON':
            h['lon'] = [float(x) for x in ln[:60].split()]
        elif lab == 'HGT1 / HGT2 / DHGT':
            h['hgt'] = [float(x) for x in ln[:60].split()]
        elif lab == 'BASE RADIUS':
            h['base_radius'] = float(ln[:60].split()[0])
        elif lab == 'MAP DIMENSION':
            h['map_dimension'] = int(ln[:60].split()[0])
        elif lab == 'IONEX VERSION / TYPE':
            h['version'] = ln[:60].split()[0]
    return h


def parse_maps(lines):
    """All TEC maps: list of (epoch tuple, int array (71,73) in file units); row count and lat labels are verified."""
    i = 0
    while 'END OF HEADER' not in lines[i]:
        i += 1
    i += 1
    maps = []
    while i < len(lines):
        if label(lines[i]) == 'START OF TEC MAP':
            epoch = tuple(int(x) for x in lines[i + 1][:60].split())
            i += 2
            grid = np.zeros(GRID, np.int64)
            row = 0
            lats = []
            while label(lines[i]) != 'END OF TEC MAP':
                if label(lines[i]) == 'LAT/LON1/LON2/DLON/H':
                    nums = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', lines[i][:32].replace('-', ' -'))]
                    lats.append(nums)
                    i += 1
                    vals = []
                    while len(vals) < GRID[1]:
                        vals.extend(int(x) for x in lines[i].split())
                        i += 1
                    if len(vals) != GRID[1]:
                        raise ValueError('latitude row has %d values' % len(vals))
                    grid[row] = vals
                    row += 1
                else:
                    i += 1
            if row != GRID[0]:
                raise ValueError('latitude rows %d' % row)
            maps.append((epoch, grid, lats))
        i += 1
    return maps


def read_file(path):
    text = gzip.decompress(Path(path).read_bytes()).decode('ascii', errors='strict')
    lines = text.splitlines()
    header = parse_header(lines)
    maps = parse_maps(lines)
    return header, maps


def audit(path, date, header, maps):
    """Return (record, even-hour array (12,71,73) float32 in 0.1 TECU or None)."""
    y, doy = int(str(date)[:4]), int((date - np.datetime64(str(date)[:4] + '-01-01')).astype(int)) + 1
    reasons = []
    com = ' '.join(header.get('comments', []))
    m = re.search(r"CODE'S\s+(\d+)-DAY PREDICTED IONOSPHERE MAPS FOR DAY\s+(\d+),\s*(\d{4})", com)
    if not m:
        reasons.append('header_not_predicted_maps')
    else:
        if int(m.group(1)) != 1:
            reasons.append('header_lead_%s_day' % m.group(1))
        if int(m.group(2)) != doy or int(m.group(3)) != y:
            reasons.append('header_day_mismatch')
    d0 = np.datetime64(date)
    nxt = d0 + 1
    first = (y, int(str(d0)[5:7]), int(str(d0)[8:10]), 0, 0, 0)
    if header.get('first_epoch') != first:
        reasons.append('first_epoch_not_day_start')
    if header.get('interval') != 3600:
        reasons.append('interval_not_3600')
    if header.get('n_maps') != 25 or len(maps) != 25:
        reasons.append('map_count_not_25')
    else:
        want = [(y, int(str(d0)[5:7]), int(str(d0)[8:10]), h, 0, 0) for h in range(24)] + [(int(str(nxt)[:4]), int(str(nxt)[5:7]), int(str(nxt)[8:10]), 0, 0, 0)]
        if [e for e, _, _ in maps] != want:
            reasons.append('epoch_axis_not_hourly_consecutive')
    if header.get('lat') != [87.5, -87.5, -2.5]:
        reasons.append('lat_axis')
    if header.get('lon') != [-180.0, 180.0, 5.0]:
        reasons.append('lon_axis')
    if header.get('hgt') != [450.0, 450.0, 0.0]:
        reasons.append('height')
    if header.get('exponent') != -1:
        reasons.append('exponent_not_minus1')
    if header.get('map_dimension') != 2 or header.get('base_radius') != 6371.0:
        reasons.append('map_dimension_or_radius')
    for e, g, lats in maps:
        if [round(x[0], 1) for x in lats] != [round(float(v), 1) for v in LAT] or any(x[1:4] != [-180.0, 180.0, 5.0] for x in lats):
            reasons.append('row_latitude_or_longitude_labels'); break
    allv = np.stack([g for _, g, _ in maps]) if maps else np.zeros((0,) + GRID, np.int64)
    n9999 = int((allv == 9999).sum())
    tec = allv * 0.1
    n_neg = int((tec < 0).sum()); n_hi = int((tec > 300).sum())
    if n9999:
        reasons.append('sentinel_9999')
    if n_neg or n_hi:
        reasons.append('out_of_range_0_300_TECU')
    stamp = header.get('stamp', '')
    created = None
    try:
        ms = re.match(r'(\d{2})-([A-Z]{3})-(\d{2})\s+(\d{2}):(\d{2})', stamp)
        created = datetime(2000 + int(ms.group(3)), MONTHS[ms.group(2)], int(ms.group(1)), int(ms.group(4)), int(ms.group(5)))
    except Exception:
        reasons.append('creation_stamp_unreadable')
    rec = dict(date=str(date), doy=doy, maps=len(maps), tec_min_TECU=float(tec.min()) if tec.size else None, tec_max_TECU=float(tec.max()) if tec.size else None,
               n_sentinel_9999=n9999, n_negative=n_neg, n_over_300=n_hi, pgm=header.get('pgm'), stamp=stamp,
               status='ok' if not reasons else 'rejected', reasons=reasons)
    if created is not None:
        day_start = datetime(y, int(str(d0)[5:7]), int(str(d0)[8:10]))
        cutoff = datetime(int(str(d0 - 1)[:4]), int(str(d0 - 1)[5:7]), int(str(d0 - 1)[8:10]), 22)
        rec['created_hours_after_day_start'] = (created - day_start).total_seconds() / 3600
        rec['created_hours_after_cutoff_prev_day_22'] = (created - cutoff).total_seconds() / 3600
    arr = None
    if not reasons:
        ix = [k for k, (e, _, _) in enumerate(maps) if e[3] in EVEN and e[2] == int(str(d0)[8:10])]
        assert len(ix) == 12
        arr = np.stack([allv[k] for k in ix]).astype(np.float32)  # EXPONENT -1: file unit is already 0.1 TECU
    return rec, arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='download root (read only)')
    p.add_argument('--converted', type=Path, required=True, help='new empty directory for yearly arrays')
    p.add_argument('--audit-out', type=Path, required=True)
    p.add_argument('--first', default='2023-01-01')
    p.add_argument('--last', default='2026-10-31')
    a = p.parse_args()
    manifest = {}
    for line in (a.root / 'MANIFEST.jsonl').read_text().splitlines():
        r = json.loads(line); manifest[r['date']] = r
    days = np.arange(np.datetime64(a.first), np.datetime64(a.last) + 1, dtype='datetime64[D]')
    a.converted.mkdir(parents=True, exist_ok=False)
    arrays = {}
    daily = []
    for d in days:
        key = str(d)
        r = manifest.get(key)
        y = int(key[:4]); doy = int((d - np.datetime64(f'{y}-01-01')).astype(int)) + 1
        if y not in arrays:
            ndays = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
            arrays[y] = np.full((ndays,) + (12,) + GRID, np.nan, np.float32)
        if r is None or r['status'] not in ('downloaded', 'present'):
            daily.append(dict(date=key, doy=doy, status='missing', reasons=['download_status_' + (r['status'] if r else 'absent')], sha256=None))
            continue
        path = a.root / r['file']
        try:
            header, maps = read_file(path)
            rec, arr = audit(path, d, header, maps)
        except Exception as e:  # unreadable file counts as rejected, never filled
            rec, arr = dict(date=key, doy=doy, status='rejected', reasons=['unreadable:' + type(e).__name__]), None
        rec['sha256'] = r['sha256']; rec['bytes'] = r['bytes']; rec['last_modified'] = r.get('last_modified')
        if arr is not None:
            arrays[y][doy - 1] = arr
        daily.append(rec)
    for y, arr in arrays.items():
        np.save(a.converted / f'C1PG_{y}.npy', arr)
    valid = [r['date'] for r in daily if r['status'] == 'ok']
    (a.converted / 'C1PG_VALID_DAYS.json').write_text(json.dumps(valid) + '\n')
    cols = ['date', 'doy', 'status', 'reasons', 'stamp', 'created_hours_after_day_start', 'created_hours_after_cutoff_prev_day_22',
            'tec_min_TECU', 'tec_max_TECU', 'n_sentinel_9999', 'maps', 'bytes', 'sha256', 'last_modified']
    with open(a.audit_out / 'C1PG_AUDIT_DAILY.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in daily:
            w.writerow([';'.join(r[c]) if c == 'reasons' else r.get(c) for c in cols])
    ok = [r for r in daily if r['status'] == 'ok']
    created = np.array([r['created_hours_after_day_start'] for r in ok if 'created_hours_after_day_start' in r])
    after_cut = np.array([r['created_hours_after_cutoff_prev_day_22'] for r in ok if 'created_hours_after_cutoff_prev_day_22' in r])
    q = lambda x: dict(min=float(x.min()), p05=float(np.quantile(x, .05)), median=float(np.median(x)), p95=float(np.quantile(x, .95)), max=float(x.max())) if len(x) else None  # noqa: E731
    by_period = {}
    for name, lo, hi in [('2023', '2023-01-01', '2023-12-31'), ('2024', '2024-01-01', '2024-12-31'), ('2025', '2025-01-01', '2025-12-31'), ('2026_Jan_Oct', '2026-01-01', '2026-10-31')]:
        sel = [r for r in daily if lo <= r['date'] <= hi]
        by_period[name] = dict(days_in_period=len(sel), valid_days=sum(r['status'] == 'ok' for r in sel),
                               missing=[r['date'] for r in sel if r['status'] == 'missing'], rejected=[dict(date=r['date'], reasons=r['reasons']) for r in sel if r['status'] == 'rejected'])
    summary = dict(experiment='exp0029', stage='c1pg_audit', at_utc=datetime.now(timezone.utc).isoformat(), days_requested=len(days),
                   valid_days=len(valid), missing_days=[r['date'] for r in daily if r['status'] == 'missing'],
                   rejected_days=[dict(date=r['date'], reasons=r['reasons']) for r in daily if r['status'] == 'rejected'], by_period=by_period,
                   value_range_TECU=dict(min=min(r['tec_min_TECU'] for r in ok), max=max(r['tec_max_TECU'] for r in ok)),
                   creation_stamp_hours_after_day_00=q(created), creation_stamp_hours_after_cutoff_prev_day_22=q(after_cut),
                   stamp_time_zone='not stated in the file header', filling='none', converted_dir=str(a.converted),
                   converted_files={f'C1PG_{y}.npy': dict(shape=list(arr.shape), finite_days=int(np.isfinite(arr[:, 0, 0, 0]).sum())) for y, arr in arrays.items()})
    (a.audit_out / 'C1PG_AUDIT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('by_period', 'converted_files')}, ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main()
