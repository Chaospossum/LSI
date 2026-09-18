"""Spatial + temporal deconfliction on polar-stereographic buffers."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from shapely.geometry import mapping

from src.registry import buffer_features, load_registry


def _parse(ts) -> datetime | None:
    if ts is None or (isinstance(ts, float) and ts != ts):
        return None
    s = str(ts).strip().replace("Z", "")
    if not s or s.lower() in ("none", "nan"):
        return None
    for fmt, n in (("%Y-%m-%dT%H:%M:%S", 19), ("%Y-%m-%d", 10), ("%Y-%m", 7)):
        try:
            return datetime.strptime(s[:n], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def time_overlap(a_start, a_end, b_start, b_end) -> bool:
    a0, a1 = _parse(a_start), _parse(a_end)
    b0, b1 = _parse(b_start), _parse(b_end)
    if None in (a0, a1, b0, b1):
        return False
    return a0 <= b1 and b0 <= a1


def find_conflicts(gdf=None, user_buffers: dict | None = None) -> pd.DataFrame:
    if gdf is None:
        gdf = load_registry()
    polar = buffer_features(gdf, user_buffers=user_buffers)
    rows = []
    n = len(polar)
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = polar.iloc[i], polar.iloc[j]
            if not time_overlap(ai.get("t_start"), ai.get("t_end"), aj.get("t_start"), aj.get("t_end")):
                continue
            gi, gj = ai["buffer_geom"], aj["buffer_geom"]
            if gi is None or gj is None or gi.is_empty or gj.is_empty:
                continue
            if not gi.intersects(gj):
                continue
            inter = gi.intersection(gj)
            area = float(inter.area)
            smaller = min(float(gi.area), float(gj.area))
            sev = area / smaller if smaller > 0 else 0.0
            rows.append(
                {
                    "a": str(ai.get("mission")),
                    "b": str(aj.get("mission")),
                    "a_actor": str(ai.get("actor")),
                    "b_actor": str(aj.get("actor")),
                    "overlap_m2": area,
                    "severity": sev,
                    "a_source": str(ai.get("coord_source")),
                    "b_source": str(aj.get("coord_source")),
                    "geometry": inter,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["a", "b", "a_actor", "b_actor", "overlap_m2", "severity", "a_source", "b_source", "geometry"])
    return pd.DataFrame(rows).sort_values("severity", ascending=False).reset_index(drop=True)


def conflicts_geojson_like(df: pd.DataFrame) -> list[dict]:
    feats = []
    for _, r in df.iterrows():
        feats.append(
            {
                "type": "Feature",
                "properties": {k: r[k] for k in df.columns if k != "geometry"},
                "geometry": mapping(r["geometry"]),
            }
        )
    return feats
