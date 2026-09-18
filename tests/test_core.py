"""Synthetic scoring, buffers, and projection round-trip. No NASA files required."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from src.conflict import find_conflicts, time_overlap
from src.crs import to_lonlat, to_stereo
from src.registry import KIND_BUFFER_M, buffer_features
from src.score import DEFAULT_WEIGHTS, criterion_maps, score as score_fn, top_sites


def _toy_bundle(n=50):
    y, x = np.mgrid[0:n, 0:n]
    dem = 1000 + 0.2 * x + 2 * np.sin(y / 8.0)
    slope = np.clip(np.abs(x - n / 2) * 0.4, 0, 30)
    count = np.where((x + y) % 5 == 0, 3.0, 0.0)
    illum = 0.3 + 0.7 * (y / n)
    earth = np.full((n, n), 0.6)
    psr = np.zeros((n, n))
    psr[:5, :5] = 1
    dist = np.sqrt((x) ** 2 + (y) ** 2) * 5.0
    dist = np.clip(dist, 0, 2000)
    from rasterio.transform import from_origin
    from src.crs import STEREO_CRS

    return {
        "dem": dem.astype(np.float32),
        "slope": slope.astype(np.float32),
        "count": count.astype(np.float32),
        "illum": illum.astype(np.float32),
        "earth": earth.astype(np.float32),
        "psr": psr,
        "psr_dist": dist.astype(np.float32),
        "hillshade": np.full((n, n), 0.5, np.float32),
        "interpolated": count < 1,
        "transform": from_origin(-8000, 8000, 5, 5),
        "crs": STEREO_CRS,
        "profile": {"height": n, "width": n, "count": 1, "dtype": "float32", "transform": from_origin(-8000, 8000, 5, 5), "crs": STEREO_CRS},
        "pixel_m": 5.0,
        "layers": [],
        "summary": {},
    }


def test_score_shape_and_mask():
    b = _toy_bundle()
    sc = score_fn(b, DEFAULT_WEIGHTS, slope_max=15)
    assert sc["score"].shape == (50, 50)
    assert (sc["score"][b["slope"] > 15] == 0).all()
    assert sc["score"].max() > 0


def test_contributions_sum_logic():
    b = _toy_bundle()
    sc = score_fn(b)
    parts = sum(sc["contrib"][k] for k in sc["contrib"])
    # score = parts * (1-penalty) * mask  — on unmasked high-count pixels penalty is small
    c = criterion_maps(b)
    unmasked = c["hard_mask"]
    assert np.nanmax(sc["score"][unmasked]) <= np.nanmax(parts[unmasked]) + 1e-5


def test_top5_separated():
    b = _toy_bundle()
    sc = score_fn(b)
    tops = top_sites(sc, b, n=5, min_sep_m=50)
    assert 1 <= len(tops) <= 5
    for i, a in enumerate(tops):
        for c in tops[i + 1 :]:
            d2 = (a["row"] - c["row"]) ** 2 + (a["col"] - c["col"]) ** 2
            assert d2 >= (50 / 5) ** 2 - 1e-6


def test_time_overlap():
    assert time_overlap("2027-01-01", "2027-12-31", "2027-06-01", "2027-08-01")
    assert not time_overlap("2027-01-01", "2027-03-01", "2028-01-01", "2028-02-01")


def test_buffer_and_conflict_toy():
    import geopandas as gpd
    from src.crs import LONLAT_CRS

    gdf = gpd.GeoDataFrame(
        {
            "actor": ["A", "B"],
            "mission": ["one", "two"],
            "kind": ["lander", "lander"],
            "buffer_m": [2000, 2000],
            "t_start": ["2027-01-01", "2027-06-01"],
            "t_end": ["2027-12-31", "2027-12-31"],
            "coord_source": ["placeholder", "placeholder"],
            "geometry": [Point(0, -89.7), Point(2, -89.7)],
        },
        crs=LONLAT_CRS,
    )
    polar = buffer_features(gdf)
    assert polar.crs.to_string().startswith("+proj=stere") or "Stereographic" in polar.crs.to_wkt()
    # buffers are in metres, much larger than 2e3 deg would have been
    assert polar.iloc[0]["buffer_geom"].area > 1e6
    cf = find_conflicts(gdf)
    assert len(cf) >= 1
    assert cf.iloc[0]["severity"] > 0


def test_projection_roundtrip():
    lon, lat = 123.4, -88.8
    x, y = to_stereo().transform(lon, lat)
    lon2, lat2 = to_lonlat().transform(x, y)
    assert abs(lon - lon2) < 1e-6
    assert abs(lat - lat2) < 1e-6
    # south polar stereo: near-pole radius is small
    assert abs(x) < 80_000 and abs(y) < 80_000


def test_slider_overrides_geojson_buffer():
    import geopandas as gpd
    from src.crs import LONLAT_CRS

    gdf = gpd.GeoDataFrame(
        {
            "actor": ["A"],
            "mission": ["one"],
            "kind": ["lander"],
            "buffer_m": [2000],
            "t_start": ["2027-01-01"],
            "t_end": ["2027-12-31"],
            "coord_source": ["placeholder"],
            "geometry": [Point(0, -89.7)],
        },
        crs=LONLAT_CRS,
    )
    polar = buffer_features(gdf, user_buffers={"lander": 500})
    assert polar.iloc[0]["buffer_used_m"] == 500


def test_folium_uses_simple_crs():
    from src.viz import make_folium_map, score_rgba
    import geopandas as gpd
    from src.crs import LONLAT_CRS

    b = _toy_bundle()
    sc = score_fn(b)
    rgba = score_rgba(sc["score"], b["hillshade"], sc["criteria"]["hard_mask"])
    gdf = gpd.GeoDataFrame(
        {
            "actor": ["A", "B"],
            "mission": ["one", "two"],
            "kind": ["lander", "lander"],
            "buffer_m": [2000, 2000],
            "t_start": ["2027-01-01", "2027-06-01"],
            "t_end": ["2027-12-31", "2027-12-31"],
            "coord_source": ["placeholder", "placeholder"],
            "geometry": [Point(0, -89.7), Point(2, -89.7)],
        },
        crs=LONLAT_CRS,
    )
    polar = buffer_features(gdf)
    m = make_folium_map(b, sc, polar, find_conflicts(gdf), rgba)
    html = m.get_root().render()
    assert "L.CRS.Simple" in html
    assert "EPSG3857" not in html
    assert "data:image" in html
