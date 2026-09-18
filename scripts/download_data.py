#!/usr/bin/env python3
"""One-time download of PGDA / PDS rasters used by the prototype.

Never fabricates files. If a URL fails, prints a clear error and continues.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import RAW  # noqa: E402

FILES = [
    # Site04 Shackleton rim — 5 m LDEM / slope / count (Barker et al. 2021)
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04/Site04_final_adj_5mpp_surf.tif",
        "Site04_final_adj_5mpp_surf.tif",
        "40980806",
    ),
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04/Site04_final_adj_5mpp_slp.tif",
        "Site04_final_adj_5mpp_slp.tif",
        "40980772",
    ),
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04/Site04_final_adj_5mpp_ldec.tif",
        "Site04_final_adj_5mpp_ldec.tif",
        "40980740",
    ),
    # Site07 Peak near Shackleton — centre only for the NASA registry record
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site07/Site07_final_adj_5mpp_surf.tif",
        "Site07_final_adj_5mpp_surf.tif",
        "40980808",
    ),
    # Mazarico 18.6-year mean solar illumination, 85S–90S, 60 m
    (
        "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608.TIF",
        "AVGVISIB_85S_060M_201608.TIF",
        "33994464",
    ),
    # Same product family: mean Earth visibility (DTE proxy source)
    (
        "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608_EARTH.TIF",
        "AVGVISIB_85S_060M_201608_EARTH.TIF",
        "24862958",
    ),
    # 60 m PSR mask 85S–90S (illumination product). Small; used for distance.
    (
        "https://pgda.gsfc.nasa.gov/data/MoonIllumination/LPSR_85S_060M_201608.TIF",
        "LPSR_85S_060M_201608.TIF",
        "656183",
    ),
]


def _check_size(name: str, size: int, hint: str | None) -> None:
    """Sizes are recorded from the products used to build this prototype. A mismatch is
    reported, never corrected: PGDA may have revised the product."""
    if not hint:
        return
    if size != int(hint):
        print(f"  NOTE {name}: {size} bytes, expected {hint}. The product may have been revised — re-read DATA_LIMITS.md.")


def _download(url: str, dest: Path, hint: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (exists) {dest.name} ({dest.stat().st_size} bytes)")
        _check_size(dest.name, dest.stat().st_size, hint)
        return
    print(f"GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    size = dest.stat().st_size
    print(f"  -> {dest.name} {size} bytes")
    _check_size(dest.name, size, hint)


def _head(url: str) -> tuple[int, str]:
    """Reachability check without downloading — run this the day before a demo."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("Content-Length", "?")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download (or just check) the PGDA/PDS rasters.")
    ap.add_argument("--check", action="store_true", help="HEAD every URL and report, download nothing")
    args = ap.parse_args()

    if args.check:
        bad = []
        for url, name, hint in FILES:
            try:
                status, length = _head(url)
                flag = "" if hint in (None, length) else f"  (expected {hint})"
                print(f"{status} {length:>10} bytes  {name}{flag}")
                if status != 200:
                    bad.append(name)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL              {name}: {exc}")
                bad.append(name)
        print("\n" + ("All URLs reachable." if not bad else f"{len(bad)} URL(s) unreachable: {', '.join(bad)}"))
        return 0 if not bad else 1

    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for url, name, hint in FILES:
        dest = RAW / name
        try:
            _download(url, dest, hint)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            failed.append((name, url, str(exc)))
            print(f"FAILED {name}: {exc}")
    if failed:
        print("\nSome files failed. Do not invent replacements.")
        for name, url, err in failed:
            print(f"  - {name}\n    {url}\n    {err}")
        return 1
    print("\nAll listed files present under data/raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
