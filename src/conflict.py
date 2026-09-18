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


# Coordinate uncertainty by provenance. ASSUMED values — a published coordinate is not
# a surveyed pad, and an "approximate" one is a press-release region. Override them
# rather than trusting these.
POSITION_SIGMA_M = {"published": 150.0, "approximate": 5000.0, "placeholder": None}


def conflict_probability(
    gdf=None,
    user_buffers: dict | None = None,
    n_draws: int = 2000,
    seed: int = 0,
    sigma_by_source: dict | None = None,
) -> pd.DataFrame:
    """P(buffer overlap) given coordinate uncertainty, instead of a binary yes/no.

    A binary conflict flag computed from coordinates that are themselves uncertain by
    kilometres is a statement about the GeoJSON, not about the Moon. Buffers are discs,
    so two operations conflict exactly when their centre separation falls below the sum
    of their radii; each centre is perturbed by an isotropic Gaussian whose σ comes from
    `coord_source`, and the probability is the fraction of draws that overlap.

    Placeholder geometry has no meaningful σ: those rows return NaN and stay labelled,
    because inventing an uncertainty for a made-up point would launder it into a result.
    """
    import numpy as np

    from src.registry import buffer_features, load_registry

    sigmas = dict(POSITION_SIGMA_M if sigma_by_source is None else sigma_by_source)
    if gdf is None:
        gdf = load_registry()
    polar = buffer_features(gdf, user_buffers=user_buffers)
    rng = np.random.default_rng(seed)

    rows = []
    n = len(polar)
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = polar.iloc[i], polar.iloc[j]
            if not time_overlap(ai.get("t_start"), ai.get("t_end"), aj.get("t_start"), aj.get("t_end")):
                continue
            ri = float(ai.get("buffer_used_m") or 0.0)
            rj = float(aj.get("buffer_used_m") or 0.0)
            d0 = float(ai.geometry.distance(aj.geometry))
            si = sigmas.get(str(ai.get("coord_source")))
            sj = sigmas.get(str(aj.get("coord_source")))
            if si is None or sj is None:
                p = float("nan")
                basis = "placeholder geometry — no positional uncertainty defined"
            else:
                sd = float(np.hypot(si, sj))
                dxy = rng.normal(0.0, sd, size=(n_draws, 2))
                d = np.hypot(d0 + dxy[:, 0], dxy[:, 1])
                p = float((d < (ri + rj)).mean())
                basis = f"σ_sep = {sd:,.0f} m from coord_source"
            rows.append(
                {
                    "a": str(ai.get("mission")),
                    "b": str(aj.get("mission")),
                    "separation_m": round(d0, 1),
                    "buffer_sum_m": round(ri + rj, 1),
                    "P_conflict": p if p != p else round(p, 3),
                    "nominal_conflict": bool(d0 < ri + rj),
                    "basis": basis,
                }
            )
    cols = ["a", "b", "separation_m", "buffer_sum_m", "P_conflict", "nominal_conflict", "basis"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("separation_m").reset_index(drop=True)
