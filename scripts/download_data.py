"""Download the primary dataset: UCI Online Retail II.

No credentials are required. The UCI repository serves the file over plain
HTTPS, which is the main reason it was chosen as the primary dataset for
PBL-1 (see docs/dataset_acquisition.md for the full rationale).

    .venv\\Scripts\\python scripts\\download_data.py

This script NEVER fabricates data. If the download fails it says so and
exits non-zero.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
EXPECTED_MEMBER = "online_retail_II.xlsx"
EXPECTED_ROWS = 1_067_371


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    pct = 100.0 * done / total
    sys.stdout.write(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)")
    sys.stdout.flush()


def main() -> int:
    cfg = load_config()
    xlsx = resolve(cfg["paths"]["raw_online_retail"])
    target_dir = xlsx.parent
    ensure_dir(target_dir)

    if xlsx.exists():
        size = xlsx.stat().st_size / 1e6
        print(f"Already present: {xlsx.relative_to(ROOT)}  ({size:.1f} MB)")
        print("Delete the file and re-run to force a fresh download.")
        return 0

    zip_path = target_dir / "online_retail_II.zip"
    print("Downloading UCI Online Retail II")
    print(f"  source: {URL}")
    print(f"  target: {target_dir.relative_to(ROOT)}")
    try:
        urllib.request.urlretrieve(URL, zip_path, reporthook=_progress)
        print()
    except Exception as exc:  # noqa: BLE001
        print(f"\nDownload FAILED: {exc}", file=sys.stderr)
        print(
            "\nManual fallback:\n"
            f"  1. Open {URL}\n"
            f"  2. Save the archive into {target_dir}\n"
            "  3. Extract online_retail_II.xlsx alongside it\n"
            "  4. Re-run this script to verify",
            file=sys.stderr,
        )
        return 1

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"  sha256: {digest}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if EXPECTED_MEMBER not in names:
            print(f"Archive does not contain {EXPECTED_MEMBER}. Found: {names}",
                  file=sys.stderr)
            return 1
        z.extract(EXPECTED_MEMBER, target_dir)

    print(f"  extracted -> {xlsx.relative_to(ROOT)} "
          f"({xlsx.stat().st_size / 1e6:.1f} MB)")
    print(
        f"\nExpected contents: {EXPECTED_ROWS:,} invoice lines across two "
        "sheets (Year 2009-2010, Year 2010-2011).\n"
        "Verify by running: python scripts/prepare_data.py"
    )
    print("\nNote: data/raw/ is git-ignored. Do not commit the dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
