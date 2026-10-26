#!/usr/bin/env python3
"""Download only PhysioNet EEGBCI left/right motor-imagery runs (04, 08, 12).

Run from the directory containing eeg_pipeline.py:
  python download_eeg_data.py --start 6 --end 30 --workers 5

Each EDF is saved to data/S###/S###R##.edf; existing valid files are skipped.
Uses only Python's standard library. Data source:
https://physionet.org/content/eegmmidb/1.0.0/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/eegmmidb/1.0.0"
RUNS = (4, 8, 12)
MIN_REASONABLE_EDF_BYTES = 500_000


def is_valid_edf(path: Path) -> bool:
    """Quick sanity check, not a replacement for loading with MNE."""
    if not path.is_file() or path.stat().st_size < MIN_REASONABLE_EDF_BYTES:
        return False
    with path.open("rb") as f:
        return f.read(8) == b"0       "


def download_one(subject: int, run: int, data_dir: Path, retries: int = 4):
    filename = f"S{subject:03d}R{run:02d}.edf"
    subdir = data_dir / f"S{subject:03d}"
    dest = subdir / filename
    # Recognize the two layouts supported by eeg_pipeline.py.
    flat_dest = data_dir / filename
    if is_valid_edf(dest) or is_valid_edf(flat_dest):
        return filename, "already present"
    subdir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/S{subject:03d}/{filename}"
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EEGBCI-study-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=75) as response, tmp.open("wb") as out:
                expected = response.headers.get("Content-Length")
                written = 0
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
            if expected is not None and written != int(expected):
                raise OSError(f"Incomplete response ({written} of {expected} bytes)")
            if not is_valid_edf(tmp):
                raise ValueError(f"Downloaded file failed EDF sanity check ({written} bytes)")
            os.replace(tmp, dest)
            return filename, f"downloaded ({written / 1_000_000:.1f} MB)"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                return filename, f"FAILED: {exc}"
            time.sleep(min(2 * attempt, 8))
    raise AssertionError("Unreachable")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=6, help="First participant ID (default: 6)")
    p.add_argument("--end", type=int, default=30, help="Last participant ID, inclusive (default: 30)")
    p.add_argument("--workers", type=int, default=5, help="Parallel downloads (default: 5; try 3 if rate-limited)")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
                   help="Where to save EDFs (default: data/ next to this script)")
    args = p.parse_args()
    if not (1 <= args.start <= args.end <= 109):
        p.error("Choose subject IDs between 1 and 109, with --start <= --end")
    if not (1 <= args.workers <= 8):
        p.error("--workers must be between 1 and 8")

    jobs = [(s, run) for s in range(args.start, args.end + 1) for run in RUNS]
    print(f"Fetching {len(jobs)} motor-imagery EDFs for participants {args.start}-{args.end}", flush=True)
    print(f"Target: {args.data_dir.resolve()} | Concurrent downloads: {args.workers}", flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(download_one, s, r, args.data_dir): (s, r) for s, r in jobs}
        for n, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            s, r = future_map[future]
            try:
                name, status = future.result()
            except Exception as exc:
                name, status = f"S{s:03d}R{r:02d}.edf", f"FAILED: {exc}"
            print(f"[{n}/{len(jobs)}] {name}: {status}", flush=True)
            if status.startswith("FAILED"):
                failures.append(name)
    if failures:
        print(f"\n{len(failures)} file(s) failed: {', '.join(failures)}\nRun the same command to retry; complete downloads are skipped.", file=sys.stderr)
        return 1
    print("\nAll requested EEG files are available. Run eeg_pipeline.py with --local-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
