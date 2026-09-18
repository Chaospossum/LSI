"""Map overlays and report figures.

Leaflet Web Mercator cannot show ~90°S. The interactive map uses Leaflet Simple CRS
in Site04 raster pixels (north up). Report PNGs use the same row/col view.
"""
from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np


def _rowcol(transform, x, y, h):
    """Stereo metres → Leaflet Simple coords. origin=upper ⇒ leaflet_y = h - row."""
    col, row = (~transform) @ (x, y)
    return [h - float(row), float(col)]


def pixel_from_leaflet(lat, lng, h, w):
    """Inverse of _rowcol: Leaflet Simple (lat, lng) → raster (row, col).

    _rowcol emits continuous coordinates, so a click inside pixel `row` arrives as
    lat ∈ (h-row-1, h-row]. floor is the exact inverse; round() would land one pixel
    off for half the grid and push the last row/column outside the raster.
    """
    r = int(np.floor(h - float(lat)))
    c = int(np.floor(float(lng)))
    if 0 <= r < h and 0 <= c < w:
        return r, c
    return None


def make_folium_map(
    bundle,
    scored,
    registry_polar,
    conflicts_df,
    overlay_rgba: np.ndarray | None,
    click_xy=None,
    show_buffers=True,
    show_conflicts=True,
):
    """Leaflet Simple CRS in raster pixels (north up). Not Web Mercator."""
    h, w = scored["score"].shape
    t = bundle["transform"]
    bounds = [[0, 0], [h, w]]
    m = folium.Map(
        location=[h / 2, w / 2],
        tiles=None,
        crs="Simple",
        zoom_start=-2,
        min_zoom=-4,
        max_zoom=3,
        control_scale=False,
    )
    m.get_root().html.add_child(
        folium.Element(
            "<div style='position:fixed;bottom:8px;left:8px;z-index:9999;background:#111;color:#ccc;"
            "font:11px sans-serif;padding:4px 8px;'>Simple CRS: Site04 pixels (x=column, y=h-row). Not Web Mercator.</div>"
        )
    )
    if overlay_rgba is not None:
        img = overlay_rgba[::4, ::4]
        folium.raster_layers.ImageOverlay(
            name="suitability + hillshade",
            image=img,
            bounds=bounds,
            origin="upper",
            opacity=1.0,
            interactive=True,
            zindex=1,
            mercator_project=False,
        ).add_to(m)
    m.fit_bounds(bounds)

    colors = {"lander": "#ffcc00", "rover": "#66ccff", "infrastructure": "#ff66aa"}
    draw = registry_polar if show_buffers else registry_polar.iloc[0:0]
    for _, row in draw.iterrows():
        loc = _rowcol(t, float(row.geometry.x), float(row.geometry.y), h)
        src = row.get("coord_source", "")
        color = colors.get(str(row.get("kind")), "#ffffff")
        folium.CircleMarker(
            location=loc,
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{row.get('mission')}</b><br>{row.get('actor')} · {row.get('kind')}<br>"
                f"{src}<br>{row.get('t_start')} → {row.get('t_end')}",
                max_width=280,
            ),
        ).add_to(m)
        buf = row["buffer_geom"]
        if buf is not None and not buf.is_empty:
            xs, ys = buf.exterior.xy
            folium.Polygon(
                locations=[_rowcol(t, x, y, h) for x, y in zip(xs, ys)],
                color=color,
                weight=2,
                fill=False,
                dash_array="4" if src == "placeholder" else None,
            ).add_to(m)

    if show_conflicts and conflicts_df is not None and len(conflicts_df):
        for _, row in conflicts_df.iterrows():
            g = row["geometry"]
            if g is None or g.is_empty:
                continue
            polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            for poly in polys:
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                xs, ys = poly.exterior.xy
                folium.Polygon(
                    locations=[_rowcol(t, x, y, h) for x, y in zip(xs, ys)],
                    color="#ff2222",
                    weight=3,
                    fill=True,
                    fill_opacity=0.35,
                    popup=f"CONFLICT {row['a']} ∩ {row['b']}  sev={row['severity']:.2f}",
                ).add_to(m)

    if click_xy is not None:
        folium.Marker(location=[click_xy[1], click_xy[0]], tooltip="inspect").add_to(m)
    folium.LayerControl().add_to(m)
    return m


def score_rgba(score: np.ndarray, hill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Hillshade always visible (offline basemap); viridis only on scored pixels.

    Built through a 256-entry uint8 LUT in integer arithmetic: calling the colormap
    on the full grid returns float64 RGBA (~0.6 GB on a Site04-sized raster) and the
    laptop this has to demo on does not have that to spare.
    """
    s = np.nan_to_num(score, nan=0.0)
    hs = np.clip(np.nan_to_num(hill, nan=0.4), 0, 1)
    gray = (hs * 255.0).astype(np.uint8)
    vmax = np.percentile(s[mask], 98) if mask.any() else 1.0
    vmax = max(float(vmax), 1e-6)
    idx = (np.clip(s / vmax, 0, 1) * 255.0).astype(np.uint8)
    lut = (np.asarray(plt.get_cmap("viridis")(np.linspace(0, 1, 256)))[:, :3] * 255.0).astype(np.uint8)
    color = lut[idx]  # H×W×3 uint8
    # 0.40 hillshade + 0.60 viridis, exactly, without leaving integers
    blend = ((2 * gray[..., None].astype(np.uint16) + 3 * color.astype(np.uint16)) // 5).astype(np.uint8)
    use = (mask & (s > 0))[..., None]
    rgba = np.empty((*hs.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.where(use, blend, gray[..., None])
    rgba[..., 3] = 255
    return rgba



def figure_score_map(bundle, scored, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = scored["score"]
    hs = bundle["hillshade"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(hs, cmap="gray", origin="upper")
    im = ax.imshow(np.ma.masked_where(~scored["criteria"]["hard_mask"] | (s <= 0), s), cmap="viridis", origin="upper", alpha=0.75)
    fig.colorbar(im, ax=ax, fraction=0.046, label="suitability (0–1)")
    ax.set_title("Site04 Shackleton rim — landing-site screening score\n5 m LOLA LDEM; illum/PSR/Earth from 60 m (resampled)")
    ax.set_xlabel("column (5 m)")
    ax.set_ylabel("row (5 m)")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def figure_conflict(bundle, scored, registry_polar, conflicts_df, dest: Path) -> None:
    """Site04 window only. Off-grid registry points are omitted (they still appear in the app table)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = bundle["transform"]
    h, w = bundle["dem"].shape
    from shapely.geometry import box

    rb = box(t.c, t.f + t.e * h, t.c + t.a * w, t.f)  # stereo metres, unordered
    rb = box(min(rb.bounds[0], rb.bounds[2]), min(rb.bounds[1], rb.bounds[3]),
             max(rb.bounds[0], rb.bounds[2]), max(rb.bounds[1], rb.bounds[3]))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(bundle["hillshade"], cmap="gray", origin="upper")
    ax.imshow(
        np.ma.masked_where(~scored["criteria"]["hard_mask"], scored["score"]),
        cmap="viridis",
        origin="upper",
        alpha=0.45,
    )

    def xy_to_colrow(x, y):
        col, row = (~t) @ (x, y)
        return col, row

    for _, row in registry_polar.iterrows():
        if not row.geometry.intersects(rb.buffer(row.get("buffer_used_m") or 2000)):
            continue
        col, r = xy_to_colrow(row.geometry.x, row.geometry.y)
        ax.plot(col, r, "o", color="yellow", markersize=6)
        ax.text(col + 8, r, str(row.get("mission")), color="white", fontsize=7)
        buf = row["buffer_geom"]
        if buf is not None and not buf.is_empty:
            xs, ys = buf.exterior.xy
            cr = [xy_to_colrow(x, y) for x, y in zip(xs, ys)]
            ax.plot([c for c, _ in cr], [rr for _, rr in cr], color="yellow", lw=0.8, ls="--")

    if conflicts_df is not None:
        for _, row in conflicts_df.iterrows():
            g = row["geometry"]
            if g.is_empty or not g.intersects(rb):
                continue
            g = g.intersection(rb)
            polys = list(g.geoms) if getattr(g, "geom_type", "") == "MultiPolygon" else [g]
            for poly in polys:
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                xs, ys = poly.exterior.xy
                cr = [xy_to_colrow(x, y) for x, y in zip(xs, ys)]
                ax.fill([c for c, _ in cr], [rr for _, rr in cr], color="red", alpha=0.4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(
        "Flagged buffer+time conflicts on Site04 (red)\n"
        "Yellow dashed = buffers in polar stereographic. Off-grid missions omitted here."
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


# One hue for one series, muted ink for text, recessive grid. No rainbow, no legend
# for a single series — the title names it.
_INK = "#1f2328"
_INK_MUTED = "#6b7280"
_HUE = "#2f6fb5"
_GRID = "#d8dce1"


def figure_ranking_intervals(mc: dict, dest: Path, top_n: int = 10) -> None:
    """Candidate ranking as an interval plot — the honest form for this data.

    A bar chart of the index would imply the differences are real. They mostly are not:
    what the Monte Carlo produces is an interval per site, so the chart draws the
    interval and lets the overlap speak. Sites statistically tied with rank 1 are
    banded and labelled in text, never by colour alone.
    """
    from src.uncertainty import pairwise_dominance, summary_table

    dest.parent.mkdir(parents=True, exist_ok=True)
    table = summary_table(mc).head(top_n)
    if table.empty:
        return

    s = mc["samples"]
    order = np.argsort(-s.mean(axis=1))[:top_n]
    best = int(order[0])
    tied = [k for k, i in enumerate(order) if pairwise_dominance(mc, best, int(i)) < 0.95]

    y = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(table) + 2.4))

    if tied:
        hi = max(table["ci95"].iloc[k] for k in tied)
        ax.axhspan(y[max(tied)] - 0.5, y[min(tied)] + 0.5, color=_GRID, alpha=0.35, lw=0)
        ax.text(
            hi, y[min(tied)] + 0.62,
            f"not separable from rank 1 at 95% ({len(tied)} sites)",
            fontsize=8, color=_INK_MUTED, ha="right", va="bottom",
        )

    for k, yy in enumerate(y):
        row = table.iloc[k]
        ax.plot([row["ci05"], row["ci95"]], [yy, yy], color=_HUE, lw=2, solid_capstyle="round", alpha=0.75)
        ax.plot([row["index_mean"]], [yy], "o", color=_HUE, markersize=8, zorder=3)
        ax.text(
            row["ci95"] + 0.004, yy, f"P(top 5) = {row['P_top5']:.2f}",
            fontsize=8, color=_INK_MUTED, va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"#{int(r['rank'])}  {r['lat']:.3f}°, {r['lon']:.3f}°" for _, r in table.iterrows()],
        fontsize=8, color=_INK,
    )
    ax.set_xlabel("suitability index (footprint mean) — 90% credible interval", fontsize=9, color=_INK)
    ax.set_title(
        "Site04 candidate ranking under input and weight uncertainty\n"
        f"{mc['n_draws']} Monte-Carlo draws · {mc['pad_m']:.0f} m pad · "
        "preference index, not a probability of mission success",
        fontsize=10, color=_INK, loc="left",
    )
    ax.grid(axis="x", color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(axis="x", colors=_INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
