"""Load Site04 rasters and align coarser layers onto the 5 m grid.

Every layer carries source URL, resolution, projection, and
measured | interpolated | proxy. Nothing is invented: missing files stay missing.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src import paths
from src.crs import LUNAR_SP_STEREO, stereo_from_raster, to_lonlat


@dataclass
class LayerMeta:
    name: str
    path: str
    source_url: str
    resolution_m: float
    projection: str
    kind: str  # measured | interpolated | proxy
    units: str
    notes: str


def _apply_scale(src, data: np.ndarray) -> np.ndarray:
    """PDS GeoTIFFs store int16 DN; physical value = DN * scale + offset."""
    scale = src.scales[0] if src.scales else 1.0
    offset = src.offsets[0] if src.offsets else 0.0
    if scale in (None, 0):
        scale = 1.0
    if offset is None:
        offset = 0.0
    if scale == 1.0 and offset == 0.0:
        return data
    return (data * float(scale) + float(offset)).astype(np.float32)


def _open_array(path: Path):
    src = rasterio.open(path)
    data = src.read(1).astype(np.float32)
    nd = src.nodata
    if nd is not None and np.isfinite(nd):
        data = np.where(data == nd, np.nan, data)
    else:
        data = np.where(~np.isfinite(data), np.nan, data)
    data = _apply_scale(src, data)
    return src, data


def _reproject_to(src_path: Path, dst_profile, resampling=Resampling.bilinear) -> np.ndarray:
    with rasterio.open(src_path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            arr = np.where(arr == nd, np.nan, arr)
        arr = _apply_scale(src, arr)
        out = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return out


def hillshade(z: np.ndarray, pixel_m: float, azimuth=315.0, altitude=45.0) -> np.ndarray:
    """DEM-derived hillshade. No external tiles."""
    dy, dx = np.gradient(z, pixel_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1).astype(np.float32)


def chamfer_distance(binary: np.ndarray, cell_m: float, cap_m: float) -> np.ndarray:
    """Approximate Euclidean distance to True cells, capped. Pure numpy (no scipy)."""
    inf = cap_m + cell_m
    d = np.where(binary, 0.0, inf).astype(np.float32)
    n = int(np.ceil(cap_m / cell_m)) + 2
    diag = cell_m * np.sqrt(2.0)
    for _ in range(n):
        up = np.pad(d[1:, :], ((0, 1), (0, 0)), constant_values=inf)
        down = np.pad(d[:-1, :], ((1, 0), (0, 0)), constant_values=inf)
        left = np.pad(d[:, 1:], ((0, 0), (0, 1)), constant_values=inf)
        right = np.pad(d[:, :-1], ((0, 0), (1, 0)), constant_values=inf)
        ul = np.pad(d[1:, 1:], ((0, 1), (0, 1)), constant_values=inf)
        ur = np.pad(d[1:, :-1], ((0, 1), (1, 0)), constant_values=inf)
        dl = np.pad(d[:-1, 1:], ((1, 0), (0, 1)), constant_values=inf)
        dr = np.pad(d[:-1, :-1], ((1, 0), (1, 0)), constant_values=inf)
        d = np.minimum.reduce(
            [d, up + cell_m, down + cell_m, left + cell_m, right + cell_m, ul + diag, ur + diag, dl + diag, dr + diag]
        )
    return np.clip(d, 0, cap_m)


def raster_center_lonlat(path: Path) -> tuple[float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
        cx = 0.5 * (b.left + b.right)
        cy = 0.5 * (b.bottom + b.top)
        lon, lat = to_lonlat(stereo_from_raster(src)).transform(cx, cy)
    return float(lon), float(lat)


def load_site(cap_psr_m: float = 2000.0) -> dict:
    """Load Site04 + aligned illumination / PSR / Earth-visibility layers."""
    for p in (paths.SITE04_DEM, paths.SITE04_SLOPE, paths.SITE04_COUNT):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run: python scripts/download_data.py")

    dem_src, dem = _open_array(paths.SITE04_DEM)
    slope_src, slope = _open_array(paths.SITE04_SLOPE)
    count_src, count = _open_array(paths.SITE04_COUNT)

    profile = dem_src.profile.copy()
    crs = stereo_from_raster(dem_src)
    pixel_m = float(abs(dem_src.transform.a))
    proj_txt = crs.to_wkt() if crs else LUNAR_SP_STEREO

    layers = [
        LayerMeta(
            name="dem",
            path=str(paths.SITE04_DEM),
            source_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04/Site04_final_adj_5mpp_surf.tif",
            resolution_m=pixel_m,
            projection=proj_txt,
            kind="interpolated",
            units="m",
            notes="LOLA 5 m LDEM; empty pixels filled by interpolation (Barker et al. 2021).",
        ),
        LayerMeta(
            name="slope",
            path=str(paths.SITE04_SLOPE),
            source_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04/Site04_final_adj_5mpp_slp.tif",
            resolution_m=pixel_m,
            projection=proj_txt,
            kind="interpolated",
            units="deg",
            notes="Slope derived from the interpolated LDEM, not independently measured at 5 m.",
        ),
        LayerMeta(
            name="count",
            path=str(paths.SITE04_COUNT),
            source_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04/Site04_final_adj_5mpp_ldec.tif",
            resolution_m=pixel_m,
            projection=proj_txt,
            kind="measured",
            units="spots/pixel",
            notes="LOLA return count. Pixels with count < 1 are interpolated in the LDEM.",
        ),
    ]

    illum = earth = psr = None
    if paths.ILLUM.exists():
        illum = _reproject_to(paths.ILLUM, {**profile, "height": dem.shape[0], "width": dem.shape[1], "crs": dem_src.crs, "transform": dem_src.transform})
        # PDS maps are typically 0–1 fraction or 0–100. Normalise later in score.py.
        layers.append(
            LayerMeta(
                name="illumination",
                path=str(paths.ILLUM),
                source_url="https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608.TIF",
                resolution_m=60.0,
                projection=proj_txt,
                kind="interpolated",
                units="fraction (resampled)",
                notes="Mazarico/PDS AVGVISIB 85S 60 m (DN×4e-5 = 0–1 fraction), bilinear onto 5 m. Not a 5 m measurement.",
            )
        )
    if paths.EARTH_VIS.exists():
        earth = _reproject_to(paths.EARTH_VIS, {**profile, "height": dem.shape[0], "width": dem.shape[1], "crs": dem_src.crs, "transform": dem_src.transform})
        layers.append(
            LayerMeta(
                name="earth_visibility",
                path=str(paths.EARTH_VIS),
                source_url="https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608_EARTH.TIF",
                resolution_m=60.0,
                projection=proj_txt,
                kind="interpolated",
                units="fraction (resampled)",
                notes="PDS AVGVISIB Earth 85S 60 m (DN×4e-5), bilinear onto 5 m. Optimistic vs DSN (any Earth disk).",
            )
        )
    if paths.PSR_RASTER.exists():
        psr_raw = _reproject_to(
            paths.PSR_RASTER,
            {**profile, "height": dem.shape[0], "width": dem.shape[1], "crs": dem_src.crs, "transform": dem_src.transform},
            resampling=Resampling.nearest,
        )
        # After PDS scale/offset, LPSR is ~0 (lit) or ~1 (permanently shadowed).
        psr = np.where(np.isnan(psr_raw), 0, (psr_raw > 0.5).astype(np.float32))
        layers.append(
            LayerMeta(
                name="psr",
                path=str(paths.PSR_RASTER),
                source_url="https://pgda.gsfc.nasa.gov/data/MoonIllumination/LPSR_85S_060M_201608.TIF",
                resolution_m=60.0,
                projection=proj_txt,
                kind="interpolated",
                units="binary mask",
                notes="PDS LPSR 85S 60 m, nearest-neighbour onto 5 m. Distance computed from this mask.",
            )
        )

    # Distance to PSR on a 4× coarsened grid (25 m), then upsample. Cap 2 km.
    psr_dist = None
    if psr is not None:
        step = 4
        coarse = psr[::step, ::step] > 0.5
        cell = pixel_m * step
        d_coarse = chamfer_distance(coarse, cell, cap_psr_m)
        psr_dist = np.repeat(np.repeat(d_coarse, step, axis=0), step, axis=1)
        psr_dist = psr_dist[: dem.shape[0], : dem.shape[1]]

    hs = hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)), pixel_m)

    interpolated = np.isnan(count) | (count < 1.0)
    valid = np.isfinite(dem) & np.isfinite(slope)
    pct_interp = 100.0 * np.mean(interpolated[valid]) if valid.any() else float("nan")

    lon0, lat0 = raster_center_lonlat(paths.SITE04_DEM)
    site07 = None
    if paths.SITE07_DEM.exists():
        site07 = raster_center_lonlat(paths.SITE07_DEM)

    b = dem_src.bounds
    summary = {
        "site": "Site04 Shackleton rim",
        "crs": proj_txt,
        "pixel_m": pixel_m,
        "width": int(dem.shape[1]),
        "height": int(dem.shape[0]),
        "bounds_stereo_m": {"left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top},
        "center_lonlat": {"lon": lon0, "lat": lat0},
        "site07_center_lonlat": None if site07 is None else {"lon": site07[0], "lat": site07[1]},
        "dem_minmax_m": [float(np.nanmin(dem)), float(np.nanmax(dem))],
        "slope_minmax_deg": [float(np.nanmin(slope)), float(np.nanmax(slope))],
        "count_minmax": [float(np.nanmin(count)), float(np.nanmax(count))],
        "pct_interpolated": float(pct_interp),
        "illum_present": illum is not None,
        "earth_present": earth is not None,
        "psr_present": psr is not None,
    }

    paths.DATA.mkdir(parents=True, exist_ok=True)
    paths.LAYER_JSON.write_text(json.dumps({"layers": [asdict(x) for x in layers], "summary": summary}, indent=2))

    bundle = {
        "dem": dem,
        "slope": slope,
        "count": count,
        "illum": illum,
        "earth": earth,
        "psr": psr,
        "psr_dist": psr_dist,
        "hillshade": hs,
        "interpolated": interpolated,
        "transform": dem_src.transform,
        "crs": crs,
        "profile": profile,
        "pixel_m": pixel_m,
        "layers": layers,
        "summary": summary,
        "dem_src": dem_src,
    }
    slope_src.close()
    count_src.close()
    return bundle


def print_summary(bundle: dict) -> None:
    s = bundle["summary"]
    print("=== LUNAR SITE INTEL — ingest ===")
    for k, v in s.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    print_summary(load_site())
