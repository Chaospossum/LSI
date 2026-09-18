#!/usr/bin/env python3
"""One-time download of PGDA / PDS rasters used by the prototype.

Never fabricates files. If a URL fails, prints a clear error and continues.
"""
from __future__ import annotations

import sys
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
    # Optional vector PSRs > 1 km² (Barker 2023 south-pole view)
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LPSR_80S_20MPP_ADJ_1km2.SHP",
        "LPSR_80S_20MPP_ADJ_1km2.SHP",
        None,
    ),
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LPSR_80S_20MPP_ADJ_1km2.SHX",
        "LPSR_80S_20MPP_ADJ_1km2.SHX",
        None,
    ),
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LPSR_80S_20MPP_ADJ_1km2.DBF",
        "LPSR_80S_20MPP_ADJ_1km2.DBF",
        None,
    ),
    (
        "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LPSR_80S_20MPP_ADJ_1km2.PRJ",
        "LPSR_80S_20MPP_ADJ_1km2.PRJ",
        None,
    ),
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (exists) {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  -> {dest.name} {dest.stat().st_size} bytes")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for url, name, _hint in FILES:
        dest = RAW / name
        try:
            _download(url, dest)
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
