"""Planned-operation registry. Coordinates are never silently upgraded from placeholder."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, mapping, shape

from src import paths
from src.crs import LONLAT_CRS, STEREO_CRS, to_stereo

KIND_BUFFER_M = {"lander": 2000.0, "rover": 500.0, "infrastructure": 1000.0}


def load_registry(path: Path | None = None) -> gpd.GeoDataFrame:
    path = path or paths.REGISTRY
    gdf = gpd.read_file(path)
    # GeoJSON often tags CRS84 (Earth). Coordinates are lunar lon/lat on a 1737.4 km sphere.
    gdf = gdf.set_crs(LONLAT_CRS, allow_override=True)
    return gdf


def default_buffer_m(kind: str, override: float | None = None, user_buffers: dict | None = None) -> float:
    """Sidebar sliders win when provided; else GeoJSON buffer_m; else kind default."""
    if user_buffers and kind in user_buffers:
        return float(user_buffers[kind])
    if override is not None:
        try:
            v = float(override)
            if v == v:  # not NaN
                return v
        except (TypeError, ValueError):
            pass
    return float(KIND_BUFFER_M.get(kind, 2000.0))


def to_polar(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project to south polar stereographic before any buffer. Never buffer in lat/lon."""
    out = gdf.copy()
    out = out.set_crs(LONLAT_CRS, allow_override=True)
    return out.to_crs(STEREO_CRS)


def buffer_features(gdf: gpd.GeoDataFrame, user_buffers: dict | None = None) -> gpd.GeoDataFrame:
    polar = to_polar(gdf)
    bufs = []
    for _, row in polar.iterrows():
        r = default_buffer_m(str(row.get("kind", "lander")), row.get("buffer_m"), user_buffers)
        # geopandas may store NaN buffer_m
        if r != r:  # NaN
            r = default_buffer_m(str(row.get("kind", "lander")), None, user_buffers)
        bufs.append(row.geometry.buffer(float(r)))
    polar = polar.copy()
    polar["buffer_geom"] = bufs
    polar["buffer_used_m"] = [
        default_buffer_m(str(row.get("kind", "lander")), row.get("buffer_m"), user_buffers)
        for _, row in polar.iterrows()
    ]
    return polar


def seed_whatif(lat: float, lon: float, t_start: str, t_end: str, name: str = "what-if lander") -> dict:
    return {
        "type": "Feature",
        "geometry": mapping(Point(lon, lat)),
        "properties": {
            "actor": "user",
            "mission": name,
            "kind": "lander",
            "lat": lat,
            "lon": lon,
            "buffer_m": 2000,
            "t_start": t_start,
            "t_end": t_end,
            "rf_band": None,
            "coord_source": "placeholder",
            "source_url": "",
        },
    }


def append_whatif(gdf: gpd.GeoDataFrame, feat: dict) -> gpd.GeoDataFrame:
    extra = gpd.GeoDataFrame(
        [{**feat["properties"], "geometry": shape(feat["geometry"])}],
        crs=LONLAT_CRS,
    )
    return gpd.pd.concat([gdf, extra], ignore_index=True)


def write_registry(gdf: gpd.GeoDataFrame, path: Path | None = None) -> None:
    path = path or paths.REGISTRY
    out = gdf.to_crs(LONLAT_CRS) if gdf.crs else gdf
    path.write_text(json.dumps(json.loads(out.to_json()), indent=2))
