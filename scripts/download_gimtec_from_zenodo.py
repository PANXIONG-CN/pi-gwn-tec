from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, NamedTuple

import requests


DEFAULT_RECORD_ID = "23331771"  # GIMtec (2009-2025)
DEFAULT_DATA_ROOT = Path("data/GIMtec")
ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"


class DownloadPlan(NamedTuple):
    key: str
    url: str
    destination: Path
    checksum: str
    size: int


def _destination_for_key(key: str, data_root: Path) -> Path:
    # The 2009-2022 record has drivers_2009_2022_2h.npz; the 2009-2025 record adds drivers_2009_2025_2h.npz.
    if key.startswith("drivers_") and key.endswith(".npz"):
        return data_root / "drivers" / key
    return data_root / key


def plan_downloads(metadata: dict, data_root: Path) -> list[DownloadPlan]:
    files = metadata.get("files") or []
    planned: list[DownloadPlan] = []
    for item in files:
        key = str(item.get("key") or "").strip()
        url = str((item.get("links") or {}).get("self") or "").strip()
        if not key or not url:
            continue
        planned.append(
            DownloadPlan(
                key=key,
                url=url,
                destination=_destination_for_key(key, Path(data_root)),
                checksum=str(item.get("checksum") or "").strip(),
                size=int(item.get("size") or 0),
            )
        )
    return planned


def fetch_record_metadata(record_id: str, timeout: float = 30.0) -> dict:
    url = ZENODO_RECORD_API.format(record_id=str(record_id))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _split_checksum(checksum: str) -> tuple[str, str]:
    if ":" not in checksum:
        return "", ""
    algo, value = checksum.split(":", 1)
    algo = algo.strip().lower()
    value = value.strip().lower()
    if algo not in {"md5", "sha256"} or not value:
        return "", ""
    return algo, value


def _digest_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checksum_matches(path: Path, checksum: str) -> bool:
    algo, expected = _split_checksum(checksum)
    if not algo:
        return True
    if not path.exists():
        return False
    return _digest_file(path, algo) == expected


def _download_once(plan: DownloadPlan, tmp_path: Path, timeout: float, chunk_size: int) -> None:
    with requests.get(plan.url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_file(
    plan: DownloadPlan,
    *,
    force: bool = False,
    retries: int = 3,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> str:
    destination = Path(plan.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force and _checksum_matches(destination, plan.checksum):
        return "skipped"

    tmp_path = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            _download_once(plan, tmp_path, timeout=timeout, chunk_size=chunk_size)
            if plan.checksum and not _checksum_matches(tmp_path, plan.checksum):
                raise RuntimeError(f"checksum mismatch for {plan.key}")
            os.replace(tmp_path, destination)
            return "downloaded"
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 10.0))

    raise RuntimeError(f"failed to download {plan.key}: {last_error}")


def download_all(planned: Iterable[DownloadPlan], *, force: bool = False) -> dict:
    summary = {"downloaded": 0, "skipped": 0, "files": []}
    for plan in planned:
        status = download_file(plan, force=force)
        summary[status] = int(summary.get(status, 0)) + 1
        summary["files"].append({"key": plan.key, "status": status, "path": str(plan.destination)})
        print(f"[{status}] {plan.key} -> {plan.destination}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download GIMtec files from the Zenodo record.")
    parser.add_argument("--data-root", "--data_root", default=str(DEFAULT_DATA_ROOT), help="target GIMtec data root")
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID, help="Zenodo record id")
    parser.add_argument("--force", action="store_true", help="redownload files even if local checksum is valid")
    args = parser.parse_args(argv)

    try:
        data_root = Path(args.data_root)
        metadata = fetch_record_metadata(str(args.record_id))
        planned = plan_downloads(metadata, data_root)
        if not planned:
            raise RuntimeError(f"Zenodo record has no downloadable files: {args.record_id}")
        summary = download_all(planned, force=bool(args.force))
        (data_root / "zenodo_record_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
