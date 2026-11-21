"""
Download and convert CODE 1-day prediction (C1PG) IONEX products to yearly numpy arrays.

Target output (aligned with data/GIMtec/TEC_YYYY.npy):
  - data/GIMtec/C1PG_<YEAR>.npy
  - shape: (days_in_year, 12, 71, 73)
  - unit: 0.1 TECU (same storage unit as TEC_YYYY.npy)

Notes:
  - Input IONEX files are downloaded from CDDIS (requires Earthdata login).
  - Credentials should be provided via ~/.netrc (NOT committed to repo).
  - We sample 2-hour cadence maps at UT hours {0,2,...,22} from the hourly IONEX file.
"""

from __future__ import annotations

import argparse
import netrc
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests


CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"
GRID_SHAPE = (71, 73)
HOURS_2H = list(range(0, 24, 2))  # 0..22


def _is_leap_year(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def _days_in_year(y: int) -> int:
    return 366 if _is_leap_year(y) else 365


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def _default_cache_dir() -> Path:
    # Under data/ (already gitignored) so large raw downloads won't be tracked.
    return _repo_root() / "data" / "GIMtec" / "c1pg_ionex"


def _default_out_dir() -> Path:
    return _repo_root() / "data" / "GIMtec"


def _c1pg_filename(year: int, doy: int) -> str:
    yy = int(year) % 100
    return f"c1pg{int(doy):03d}0.{yy:02d}i.Z"


def _c1pg_url(year: int, doy: int) -> str:
    return f"{CDDIS_BASE}/{int(year)}/{int(doy):03d}/{_c1pg_filename(year, doy)}"


def _netrc_auth() -> Tuple[str, str]:
    """Return (login, password) from ~/.netrc without printing secrets."""
    n = netrc.netrc()
    for host in ("urs.earthdata.nasa.gov", "cddis.nasa.gov"):
        a = n.authenticators(host)
        if a and a[0] and a[2]:
            return a[0], a[2]
    raise RuntimeError("Earthdata credentials not found in ~/.netrc for urs.earthdata.nasa.gov")


def _download(session: requests.Session, url: str, dst: Path, timeout_sec: int = 120, retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, int(retries) + 1):
        try:
            with session.get(url, stream=True, timeout=timeout_sec, allow_redirects=True) as r:
                try:
                    r.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    try:
                        code = int(getattr(getattr(e, "response", None), "status_code", -1))
                    except Exception:
                        code = -1
                    if code == 404:
                        raise FileNotFoundError(f"404 not found: {url}") from e
                    raise
                ctype = str(r.headers.get("Content-Type", "")).lower()
                if "text/html" in ctype:
                    raise RuntimeError(f"unexpected content-type text/html (auth failed?) for {url}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            try:
                if tmp.stat().st_size <= 0:
                    raise RuntimeError("download produced empty file")
            except Exception as e:
                raise RuntimeError(f"download produced invalid file: {tmp}") from e
            tmp.replace(dst)
            return
        except FileNotFoundError:
            # Do not retry 404s; caller decides how to handle missing days.
            raise
        except Exception as e:
            if attempt >= int(retries):
                raise RuntimeError(f"download failed: {url} -> {dst}: {e}") from e
            time.sleep(1.5 * attempt)


def _uncompress_text(path_z: Path) -> str:
    if not path_z.exists():
        raise FileNotFoundError(str(path_z))
    # Use system uncompress (ncompress). Avoid adding python deps for .Z handling.
    try:
        p = subprocess.run(
            ["uncompress", "-c", str(path_z)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError("uncompress not found (please install ncompress)") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"uncompress failed for {path_z}: {msg}") from e
    return p.stdout.decode("utf-8", errors="ignore")


@dataclass
class IonexMap:
    # (year, month, day) of map epoch (UTC)
    day: Tuple[int, int, int]
    # utc hour in [0,24]
    hour: int
    # grid in 0.1 TECU, float32, shape [71,73]
    tec_01: np.ndarray


_RE_NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def _parse_exponent(lines: List[str]) -> int:
    for ln in lines:
        if "EXPONENT" in ln:
            nums = _RE_NUM.findall(ln[:20])
            if not nums:
                break
            return int(float(nums[0]))
        if "END OF HEADER" in ln:
            break
    # Default per IONEX common convention
    return -1


def _parse_tec_maps_2h(text: str) -> List[IonexMap]:
    """Parse TEC maps and keep only 2-hour cadence maps (0..22 UT)."""
    lines = text.splitlines()
    exp = _parse_exponent(lines)
    scale_to_01 = float(10.0 ** (exp + 1))  # store in 0.1 TECU

    # Find header end
    i = 0
    while i < len(lines) and ("END OF HEADER" not in lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("IONEX: END OF HEADER not found")
    i += 1

    maps: List[IonexMap] = []
    while i < len(lines):
        ln = lines[i]
        if "START OF TEC MAP" in ln:
            # Next line: EPOCH OF CURRENT MAP
            if i + 1 >= len(lines):
                break
            epoch = lines[i + 1]
            toks = epoch.split()
            if len(toks) < 4:
                raise ValueError(f"IONEX: bad epoch line: {epoch!r}")
            try:
                y = int(toks[0]); m = int(toks[1]); d = int(toks[2]); hour = int(toks[3])
            except Exception as e:
                raise ValueError(f"IONEX: bad epoch line: {epoch!r}") from e
            keep = (hour in HOURS_2H)
            i += 2

            if not keep:
                # Fast-skip this map
                while i < len(lines) and ("END OF TEC MAP" not in lines[i]):
                    i += 1
                i += 1
                continue

            grid = np.zeros(GRID_SHAPE, dtype=np.float32)
            lat_row = 0
            while i < len(lines) and ("END OF TEC MAP" not in lines[i]):
                if "LAT/LON1/LON2/DLON/H" in lines[i]:
                    nums = _RE_NUM.findall(lines[i][:40])
                    if len(nums) < 5:
                        raise ValueError(f"IONEX: bad LAT/LON header: {lines[i]!r}")
                    i += 1
                    vals: List[float] = []
                    while i < len(lines) and len(vals) < GRID_SHAPE[1]:
                        if ("LAT/LON1/LON2/DLON/H" in lines[i]) or ("END OF TEC MAP" in lines[i]):
                            break
                        vals.extend([float(x) for x in lines[i].split()])
                        i += 1
                    if len(vals) < GRID_SHAPE[1]:
                        raise ValueError("IONEX: insufficient lon values in a latitude row")
                    row = (np.asarray(vals[: GRID_SHAPE[1]], dtype=np.float32) * scale_to_01).astype(np.float32)
                    if lat_row >= GRID_SHAPE[0]:
                        raise ValueError("IONEX: too many latitude rows")
                    grid[lat_row, :] = row
                    lat_row += 1
                else:
                    i += 1

            if lat_row != GRID_SHAPE[0]:
                raise ValueError(f"IONEX: latitude rows mismatch: got {lat_row}, expect {GRID_SHAPE[0]}")
            maps.append(IonexMap(day=(y, m, d), hour=hour, tec_01=grid))
            # consume END OF TEC MAP
            while i < len(lines) and ("END OF TEC MAP" not in lines[i]):
                i += 1
            i += 1
            continue

        i += 1

    if not maps:
        raise ValueError("IONEX: no 2h TEC maps parsed")

    # IONEX daily files often include the next-day 00:00 map (hour=0 again).
    # Keep only maps belonging to the first day encountered in the file.
    base_day = maps[0].day
    maps = [m for m in maps if m.day == base_day]
    maps.sort(key=lambda m: m.hour)

    hours = [m.hour for m in maps]
    if hours != HOURS_2H:
        raise ValueError(f"IONEX: expected 2h hours {HOURS_2H} for day={base_day}, got {hours}")
    return maps


def build_c1pg_year(year: int, cache_dir: Path, out_dir: Path, overwrite: bool = False) -> Path:
    year = int(year)
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    out_path = out_dir / f"C1PG_{year}.npy"
    if out_path.exists() and (not overwrite):
        return out_path

    login, password = _netrc_auth()
    session = requests.Session()
    session.auth = (login, password)
    session.headers.update({"User-Agent": "pi-gwn-tec-c1pg-fetch/1.0"})

    ndays = _days_in_year(year)
    tmp_path = out_dir / f"C1PG_{year}.npy.tmp"
    prog_path = out_dir / f"C1PG_{year}.progress.json"
    sidecar_path = out_dir / f"C1PG_{year}_imputed_days.json"

    # resume / init memmap
    start_doy = 1
    if overwrite:
        try:
            out_path.unlink()
        except Exception:
            pass
        try:
            tmp_path.unlink()
        except Exception:
            pass
        try:
            prog_path.unlink()
        except Exception:
            pass
    if tmp_path.exists():
        arr = np.lib.format.open_memmap(str(tmp_path), mode="r+", dtype=np.float32, shape=(ndays, 12, GRID_SHAPE[0], GRID_SHAPE[1]))
        if prog_path.exists():
            try:
                import json
                prog = json.load(open(prog_path, "r", encoding="utf-8"))
                start_doy = int(prog.get("next_doy", 1))
            except Exception:
                start_doy = 1
    else:
        arr = np.lib.format.open_memmap(str(tmp_path), mode="w+", dtype=np.float32, shape=(ndays, 12, GRID_SHAPE[0], GRID_SHAPE[1]))
        start_doy = 1

    def _save_progress(next_doy: int) -> None:
        import json
        prog_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"year": year, "next_doy": int(next_doy)}, open(prog_path, "w", encoding="utf-8"), indent=2)

    imputed_days: List[int] = []
    # best-effort restore imputed day list from previous partial runs
    try:
        import json
        if sidecar_path.exists():
            old = json.load(open(sidecar_path, "r", encoding="utf-8"))
            xs = old.get("imputed_doys") if isinstance(old, dict) else None
            if isinstance(xs, list):
                for x in xs:
                    try:
                        imputed_days.append(int(x))
                    except Exception:
                        pass
    except Exception:
        pass
    for doy in range(int(start_doy), ndays + 1):
        url = _c1pg_url(year, doy)
        dst = cache_dir / str(year) / f"{doy:03d}" / _c1pg_filename(year, doy)
        # download + parse with retry (handles truncated/empty cache files)
        last_err: Optional[Exception] = None
        maps = None
        imputed = False
        for attempt in range(1, 4):
            try:
                if dst.exists() and dst.stat().st_size <= 0:
                    dst.unlink()
            except Exception:
                pass
            try:
                _download(session, url, dst)
                text = _uncompress_text(dst)
                maps = _parse_tec_maps_2h(text)
                last_err = None
                break
            except FileNotFoundError as e:
                # Some recent days may be missing on CDDIS (observed for 2022 DOY>=362).
                # For paper protocol coverage, we carry-forward the previous day's map as an explicit imputation.
                if doy <= 1:
                    last_err = e
                    break
                try:
                    arr[doy - 1] = np.asarray(arr[doy - 2], dtype=np.float32)
                    imputed_days.append(int(doy))
                    _save_progress(doy + 1)
                    last_err = None
                    imputed = True
                except Exception as ee:
                    last_err = ee
                break
            except Exception as e:
                last_err = e
                try:
                    if dst.exists():
                        dst.unlink()
                except Exception:
                    pass
                time.sleep(1.0 * attempt)
        if last_err is not None:
            raise last_err
        if imputed:
            continue
        if maps is None:
            raise RuntimeError(f"internal error: maps is None for year={year} doy={doy}")
        day_maps = np.stack([m.tec_01 for m in maps], axis=0).astype(np.float32)  # [12,71,73]
        arr[doy - 1] = day_maps
        _save_progress(doy + 1)

        if (doy % 20) == 0 or doy == ndays:
            try:
                arr.flush()
            except Exception:
                pass
            print(f"[{year}] {doy:03d}/{ndays} done")

    try:
        arr.flush()
    except Exception:
        pass
    os.replace(tmp_path, out_path)
    try:
        prog_path.unlink()
    except Exception:
        pass
    # Write a sidecar for transparency if we had to impute missing days.
    if imputed_days:
        try:
            import json
            sidecar = sidecar_path
            dedup = sorted({int(x) for x in imputed_days})
            json.dump(
                {
                    "year": int(year),
                    "source": "CDDIS/Earthdata",
                    "reason": "missing daily file on CDDIS (404); carry-forward previous day map",
                    "imputed_doys": dedup,
                },
                open(sidecar, "w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            pass
    print("[ok] saved:", str(out_path))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020-2022", help="inclusive range, e.g. 2020-2022")
    ap.add_argument("--cache_dir", default=str(_default_cache_dir()))
    ap.add_argument("--out_dir", default=str(_default_out_dir()))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    y0, y1 = [int(x) for x in str(args.years).split("-", 1)]
    for y in range(y0, y1 + 1):
        build_c1pg_year(y, Path(args.cache_dir), Path(args.out_dir), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
