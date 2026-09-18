"""Streamlit UI. Run: streamlit run app.py  (after python scripts/download_data.py)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import Point, shape
from streamlit_folium import st_folium

from src import paths
from src.conflict import find_conflicts
from src.crs import to_lonlat
from src.ingest import load_site, raster_center_lonlat
from src.registry import buffer_features, load_registry, seed_whatif
from src.score import DEFAULT_WEIGHTS, describe_pixel, score, sensitivity, top_sites
from src.viz import figure_conflict, figure_score_map, make_folium_map, save_overlay_png, score_rgba

st.set_page_config(page_title="LUNAR SITE INTELLIGENCE", layout="wide")


@st.cache_resource(show_spinner="Loading Site04 rasters…")
def _bundle():
    b = load_site()
    if paths.SITE07_DEM.exists():
        b["_site07"] = raster_center_lonlat(paths.SITE07_DEM)
    else:
        b["_site07"] = None
    return b


def _registry(bundle):
    gdf = load_registry()
    s07 = bundle.get("_site07")
    if s07 is not None:
        m = gdf["mission"].astype(str).str.contains("Site07", na=False)
        if m.any():
            lon, lat = s07
            gdf.loc[m, "lon"] = lon
            gdf.loc[m, "lat"] = lat
            gdf.loc[m, "geometry"] = [Point(lon, lat)] * int(m.sum())
    return gdf


def pixel_from_click(bundle, leaflet_lat, leaflet_lng):
    """Leaflet Simple CRS: lng=column, lat=h-row (origin upper)."""
    h, w = bundle["dem"].shape
    r = int(round(h - leaflet_lat))
    c = int(round(leaflet_lng))
    if 0 <= r < h and 0 <= c < w:
        return r, c
    return None


def main():
    st.title("LUNAR SITE INTELLIGENCE")
    st.caption("Shackleton rim (PGDA Site04) screening tool — not survey-grade. Placeholders are labelled.")

    try:
        bundle = _bundle()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    tab_map, tab_limits = st.tabs(["Map", "Data & limits"])

    with st.sidebar:
        st.header("Weights (renormalised to 1)")
        w_slope = st.slider("slope (lower better)", 0.0, 1.0, DEFAULT_WEIGHTS["slope"], 0.05)
        w_illum = st.slider("illumination (higher better)", 0.0, 1.0, DEFAULT_WEIGHTS["illum"], 0.05)
        w_psr = st.slider("near PSR (closer better)", 0.0, 1.0, DEFAULT_WEIGHTS["psr"], 0.05)
        w_comms = st.slider("Earth visibility (higher better)", 0.0, 1.0, DEFAULT_WEIGHTS["comms"], 0.05)
        slope_max = st.slider("hard slope exclude (°)", 8.0, 25.0, 15.0, 0.5)
        st.header("Buffers (m)")
        b_lander = st.slider("lander", 200, 5000, 2000, 100)
        b_rover = st.slider("rover", 50, 2000, 500, 50)
        b_infra = st.slider("infrastructure", 100, 4000, 1000, 100)
        st.header("Layers")
        show_score = st.checkbox("score + hillshade overlay", True)
        show_buffers = st.checkbox("registry buffers", True)
        show_conflicts = st.checkbox("conflict highlights", True)
        st.header("What-if lander")
        st.caption("Click the map. Dates default to 2027 so they overlap the seeded landers.")
        wi_start = st.date_input("t_start", value=dt.date(2027, 6, 1))
        wi_end = st.date_input("t_end", value=dt.date(2027, 12, 31))

    weights = {"slope": w_slope, "illum": w_illum, "psr": w_psr, "comms": w_comms}
    scored = score(bundle, weights, slope_max=slope_max)
    sites = top_sites(scored, bundle, n=5)

    gdf = _registry(bundle)
    user_buf = {"lander": b_lander, "rover": b_rover, "infrastructure": b_infra}
    polar = buffer_features(gdf, user_buffers=user_buf)
    conflicts = find_conflicts(gdf, user_buffers=user_buf)

    overlay = paths.DERIVED / "score_overlay.png"
    rgba = score_rgba(
        scored["score"] if show_score else scored["score"] * 0,
        bundle["hillshade"],
        scored["criteria"]["hard_mask"] if show_score else scored["criteria"]["hard_mask"] & False,
    )
    save_overlay_png(rgba, overlay)

    with tab_map:
        left, right = st.columns([1.4, 1])
        with left:
            st.image(
                rgba[::4, ::4],
                caption="Site04 score + hillshade (north up). Leaflet below: click a pixel for the score breakdown.",
            )
            fmap = make_folium_map(
                bundle,
                scored,
                polar,
                conflicts,
                rgba,
                show_buffers=show_buffers,
                show_conflicts=show_conflicts,
            )
            out = st_folium(fmap, height=640, returned_objects=["last_clicked"], use_container_width=True)
            clicked = (out or {}).get("last_clicked")
            if clicked:
                rpix = pixel_from_click(bundle, float(clicked["lat"]), float(clicked["lng"]))
                if rpix:
                    r, c = rpix
                    x_m, y_m = bundle["transform"] * (c + 0.5, r + 0.5)
                    lon, lat = to_lonlat(bundle["crs"]).transform(x_m, y_m)
                    st.subheader("Pixel breakdown")
                    st.write(describe_pixel(scored, r, c))
                    st.json(
                        {
                            "row": r,
                            "col": c,
                            "x_m": float(x_m),
                            "y_m": float(y_m),
                            "lat": lat,
                            "lon": lon,
                            "slope_deg": float(bundle["slope"][r, c]),
                            "count": float(bundle["count"][r, c]),
                            "interpolated": bool(bundle["interpolated"][r, c]),
                        }
                    )
                    feat = seed_whatif(lat, lon, str(wi_start), str(wi_end))
                    extra = gpd.GeoDataFrame(
                        [{**feat["properties"], "geometry": shape(feat["geometry"])}],
                        crs=gdf.crs,
                    )
                    extra = gpd.GeoDataFrame(pd.concat([gdf, extra], ignore_index=True), crs=gdf.crs)
                    wi_cf = find_conflicts(extra, user_buffers=user_buf)
                    hit = wi_cf[(wi_cf["a"] == "what-if lander") | (wi_cf["b"] == "what-if lander")]
                    st.markdown("**What-if conflicts for a lander at this click**")
                    if hit.empty:
                        st.success("No buffer+time overlap with the registry.")
                    else:
                        st.dataframe(hit.drop(columns=["geometry"]), hide_index=True)
                else:
                    st.warning("Click is off the Site04 5 m grid.")

        with right:
            st.subheader("Top 5 sites")
            if sites:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "rank": s["rank"],
                                "lat": round(s["lat"], 5),
                                "lon": round(s["lon"], 5),
                                "score": round(s["score"], 3),
                                "why": describe_pixel(scored, s["row"], s["col"]),
                            }
                            for s in sites
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            st.subheader("Conflicts")
            if conflicts.empty:
                st.info("No spatial+temporal buffer overlaps.")
            else:
                show = conflicts.drop(columns=["geometry"]).copy()
                show["severity"] = show["severity"].map(lambda x: f"{x:.2f}")
                st.dataframe(show, hide_index=True, use_container_width=True)
                st.caption("Placeholder coordinates are not real landing sites.")

            if st.button("Export score_map.png"):
                figure_score_map(bundle, scored, paths.FIGURES / "score_map.png")
                st.success(str(paths.FIGURES / "score_map.png"))
            if st.button("Export conflict_demo.png"):
                figure_conflict(bundle, scored, polar, conflicts, paths.FIGURES / "conflict_demo.png")
                st.success(str(paths.FIGURES / "conflict_demo.png"))

            with st.expander("Weight sensitivity ±20%"):
                sens = sensitivity(bundle, scored["weights"])
                st.dataframe(pd.DataFrame(sens["rows"]), hide_index=True)
                st.write("All perturbations keep the same top-5:", "yes" if sens["all_stable"] else "no")

    with tab_limits:
        limits = ROOT / "DATA_LIMITS.md"
        st.markdown(limits.read_text() if limits.exists() else "_DATA_LIMITS.md missing_")
        st.subheader("Layer metadata")
        st.json(
            {
                "layers": [x.__dict__ if hasattr(x, "__dict__") else x for x in bundle["layers"]],
                "summary": bundle["summary"],
            }
        )


if __name__ == "__main__":
    main()
