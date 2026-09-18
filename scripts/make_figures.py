#!/usr/bin/env python3
"""Write the two report PNGs from the current score + registry."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import paths
from src.conflict import find_conflicts
from src.ingest import load_site, raster_center_lonlat
from src.registry import buffer_features, load_registry
from src.score import score
from src.uncertainty import candidate_sites, footprint_maps, monte_carlo, separation_verdict
from src.viz import figure_conflict, figure_ranking_intervals, figure_score_map


def main() -> None:
    bundle = load_site()
    sc = score(bundle)
    gdf = load_registry()
    if paths.SITE07_DEM.exists():
        lon, lat = raster_center_lonlat(paths.SITE07_DEM)
        m = gdf["mission"].astype(str).str.contains("Site07", na=False)
        if m.any():
            from shapely.geometry import Point

            gdf.loc[m, "lon"] = lon
            gdf.loc[m, "lat"] = lat
            gdf.loc[m, "geometry"] = [Point(lon, lat)] * int(m.sum())
    polar = buffer_features(gdf)
    cf = find_conflicts(gdf)
    figure_score_map(bundle, sc, paths.FIGURES / "score_map.png")
    figure_conflict(bundle, sc, polar, cf, paths.FIGURES / "conflict_demo.png")

    fp = footprint_maps(bundle)
    mc = monte_carlo(bundle, fp, candidate_sites(fp, bundle, n=25), n_draws=1000)
    figure_ranking_intervals(mc, paths.FIGURES / "ranking_intervals.png")

    print("wrote", paths.FIGURES / "score_map.png")
    print("wrote", paths.FIGURES / "conflict_demo.png")
    print("wrote", paths.FIGURES / "ranking_intervals.png")
    print("conflicts:", 0 if cf.empty else len(cf))
    print(separation_verdict(mc))


if __name__ == "__main__":
    main()
