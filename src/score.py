"""Landing-site suitability score.

score = hard_mask * sum(w_i * criterion_i) * (1 - confidence_penalty)

Criteria (each 0–1):
  slope     — lower better; hard exclude > slope_max (default 15°)
  illum     — higher better
  psr       — closer to a PSR better, capped at cap_m (default 2 km)
  comms     — higher Earth-visibility better

Count map is a penalty, not a 5th weight.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio

from src import paths
from src.crs import to_lonlat
from src.ingest import load_site

DEFAULT_WEIGHTS = {"slope": 0.35, "illum": 0.30, "psr": 0.20, "comms": 0.15}
CONF_PENALTY_MAX = 0.15  # subtracted fraction when count == 0


def _finite01(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _maybe_unit_interval(a: np.ndarray) -> np.ndarray:
    """Expect a 0–1 fraction after ingest applies the PDS scale; percent also accepted.

    Raw DN (max ≫ 100) is NOT rescaled by the in-window maximum: that would invent a
    fraction and make the best pixel in this window look perfectly lit. The criterion
    is returned as NaN instead, which zeroes it and leaves the layer flagged.
    """
    a = _finite01(a)
    mx = np.nanmax(a)
    if not np.isfinite(mx):
        return a
    if 1.5 < mx <= 100.5:
        return np.clip(a / 100.0, 0, 1)
    if mx > 100.5:
        return np.full_like(a, np.nan)
    return np.clip(a, 0, 1)


def criterion_maps(bundle: dict, slope_max: float = 15.0, cap_m: float = 2000.0, exclude_psr: bool = True) -> dict:
    slope = bundle["slope"]
    c_slope = np.clip(1.0 - slope / slope_max, 0, 1).astype(np.float32)

    if bundle["illum"] is None:
        c_illum = np.full_like(slope, np.nan, dtype=np.float32)
    else:
        c_illum = _maybe_unit_interval(bundle["illum"]).astype(np.float32)

    if bundle["psr_dist"] is None:
        c_psr = np.full_like(slope, np.nan, dtype=np.float32)
    else:
        c_psr = np.clip(1.0 - bundle["psr_dist"] / cap_m, 0, 1).astype(np.float32)

    if bundle["earth"] is None:
        c_comms = np.full_like(slope, np.nan, dtype=np.float32)
    else:
        c_comms = _maybe_unit_interval(bundle["earth"]).astype(np.float32)

    # A pixel with at least one LOLA return is measured; only pixels the LDEM had to
    # interpolate (count < 1) take the full penalty. Normalising by the maximum count
    # instead would penalise a measured pixel almost as hard as an empty one.
    count = np.nan_to_num(bundle["count"], nan=0.0)
    conf = np.clip(count, 0.0, 1.0)
    penalty = CONF_PENALTY_MAX * (1.0 - conf)

    hard = (slope <= slope_max) & np.isfinite(slope) & np.isfinite(bundle["dem"])
    if exclude_psr and bundle.get("psr") is not None:
        # Solar-lander screening: do not rank pads *inside* a PSR. Proximity stays a criterion.
        hard = hard & (np.nan_to_num(bundle["psr"], nan=0.0) < 0.5)
    warn = (slope > 8.0) & hard

    return {
        "slope": c_slope,
        "illum": c_illum,
        "psr": c_psr,
        "comms": c_comms,
        "penalty": penalty.astype(np.float32),
        "hard_mask": hard,
        "slope_warn": warn,
    }


def normalize_weights(weights: dict) -> dict:
    keys = ("slope", "illum", "psr", "comms")
    w = {k: float(weights.get(k, DEFAULT_WEIGHTS[k])) for k in keys}
    s = sum(w.values())
    if s <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / s for k, v in w.items()}


def score(
    bundle: dict,
    weights: dict | None = None,
    slope_max: float = 15.0,
    cap_m: float = 2000.0,
    exclude_psr: bool = True,
) -> dict:
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    c = criterion_maps(bundle, slope_max=slope_max, cap_m=cap_m, exclude_psr=exclude_psr)
    contrib = {}
    acc = np.zeros(bundle["dem"].shape, dtype=np.float32)
    for k in ("slope", "illum", "psr", "comms"):
        layer = np.nan_to_num(c[k], nan=0.0)
        contrib[k] = (w[k] * layer).astype(np.float32)
        acc += contrib[k]
    acc = acc * (1.0 - np.nan_to_num(c["penalty"], nan=0.0))
    acc = np.where(c["hard_mask"], acc, 0.0).astype(np.float32)
    return {"score": acc, "contrib": contrib, "criteria": c, "weights": w}


def _xy(transform, row, col) -> tuple[float, float]:
    x, y = rasterio.transform.xy(transform, int(row), int(col), offset="center")
    return float(x), float(y)


def top_sites(scored: dict, bundle: dict, n: int = 5, min_sep_m: float = 500.0) -> list[dict]:
    """Greedy local maxima so 'top 5' are not 5 adjacent pixels on a ridge."""
    s = scored["score"]
    mask = scored["criteria"]["hard_mask"] & (s > 0)
    if not mask.any():
        return []
    order = np.argsort(s[mask].ravel())[::-1]
    rows, cols = np.nonzero(mask)
    pix = bundle["pixel_m"]
    min_sep_px = max(1, int(min_sep_m / pix))
    picked = []
    chosen = []
    for idx in order:
        r, c = int(rows[idx]), int(cols[idx])
        if any((r - rr) ** 2 + (c - cc) ** 2 < min_sep_px**2 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        x, y = _xy(bundle["transform"], r, c)
        lon, lat = to_lonlat(bundle["crs"]).transform(x, y)
        total = float(s[r, c])
        parts = {k: float(scored["contrib"][k][r, c]) for k in scored["contrib"]}
        pen = float(scored["criteria"]["penalty"][r, c])
        picked.append(
            {
                "rank": len(picked) + 1,
                "row": r,
                "col": c,
                "x": x,
                "y": y,
                "lon": lon,
                "lat": lat,
                "score": total,
                "parts": parts,
                "confidence_penalty": pen,
                "slope_deg": float(bundle["slope"][r, c]),
            }
        )
        if len(picked) >= n:
            break
    return picked


def describe_pixel(scored: dict, r: int, c: int) -> str:
    """Human-readable breakdown. The printed arithmetic closes exactly.

    score = (weighted criteria) − (confidence penalty), where the penalty is a
    fraction of the weighted sum, not a flat subtraction.
    """
    parts = scored["contrib"]
    keys = ("slope", "illum", "psr", "comms")
    raw = float(sum(float(parts[k][r, c]) for k in keys))
    pen_frac = float(scored["criteria"]["penalty"][r, c])
    total = float(scored["score"][r, c])
    bits = " + ".join(f"{k} {float(parts[k][r, c]):.2f}" for k in keys)
    if not bool(scored["criteria"]["hard_mask"][r, c]):
        return f"score 0.00 — excluded by the hard slope mask (criteria alone would give {raw:.2f})"
    if pen_frac <= 0.0005:
        return f"score {total:.2f} = {bits} — no confidence penalty (LOLA-measured pixel)"
    lost = max(raw - total, 0.0)
    return (
        f"score {total:.2f} = {bits} = {raw:.2f} − {lost:.2f} confidence penalty"
        f" ({pen_frac * 100:.0f}% of {raw:.2f} — no LOLA return at this pixel)"
    )


def sensitivity(bundle: dict, weights: dict, n: int = 5, delta: float = 0.20) -> dict:
    """Rerun with ±20% on each weight (then renormalise). Report top-n stability."""
    base = score(bundle, weights)
    base_top = {(p["row"], p["col"]) for p in top_sites(base, bundle, n=n)}
    rows = []
    for key in ("slope", "illum", "psr", "comms"):
        for sign, label in ((1.0, "+"), (-1.0, "-")):
            w = dict(weights)
            w[key] = max(1e-6, w[key] * (1.0 + sign * delta))
            w = normalize_weights(w)
            sc = score(bundle, w)
            top = {(p["row"], p["col"]) for p in top_sites(sc, bundle, n=n)}
            stay = len(base_top & top)
            rows.append({"perturb": f"{key}{label}{int(delta*100)}%", "top_n_overlap": stay, "stable": stay == n})
    n_stable = sum(1 for r in rows if r["stable"])
    return {"rows": rows, "all_stable": n_stable == len(rows), "base_top": list(base_top)}


def write_score_tif(scored: dict, bundle: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    prof = bundle["profile"].copy()
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(dest, "w", **prof) as dst:
        dst.write(scored["score"], 1)


def write_top_csv(sites: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "lat", "lon", "score", "slope_contrib", "illum_contrib", "psr_contrib", "comms_contrib", "conf_penalty", "slope_deg"])
        for p in sites:
            w.writerow(
                [
                    p["rank"],
                    f"{p['lat']:.6f}",
                    f"{p['lon']:.6f}",
                    f"{p['score']:.4f}",
                    f"{p['parts']['slope']:.4f}",
                    f"{p['parts']['illum']:.4f}",
                    f"{p['parts']['psr']:.4f}",
                    f"{p['parts']['comms']:.4f}",
                    f"{p['confidence_penalty']:.4f}",
                    f"{p['slope_deg']:.3f}",
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Score Site04 and write GeoTIFF + top-5 CSV")
    p.add_argument("--out-tif", default=str(paths.DERIVED / "score.tif"))
    p.add_argument("--out-csv", default=str(paths.DERIVED / "top5.csv"))
    args = p.parse_args()
    bundle = load_site()
    sc = score(bundle)
    sites = top_sites(sc, bundle, n=5)
    write_score_tif(sc, bundle, Path(args.out_tif))
    write_top_csv(sites, Path(args.out_csv))
    print(f"wrote {args.out_tif}")
    print(f"wrote {args.out_csv}")
    for s in sites:
        print(f"  #{s['rank']} lat={s['lat']:.4f} lon={s['lon']:.4f}  {describe_pixel(sc, s['row'], s['col'])}")
    sens = sensitivity(bundle, sc["weights"])
    print("sensitivity ±20% weights; top-5 fully stable in", sum(1 for r in sens["rows"] if r["stable"]), "/", len(sens["rows"]), "perturbations")


if __name__ == "__main__":
    main()
