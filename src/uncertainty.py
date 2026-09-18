"""Uncertainty propagation and rank statistics for the screening score.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
The suitability score is a **multi-criteria preference index**, not a probability of
mission success. Nothing here converts it into one. What these statistics answer is
narrower and answerable:

    given the published error of the input rasters and a stated uncertainty on the
    weights, how stable is the *ranking* of candidate sites?

So the outputs are P(site is in the top 5), P(site A beats site B), and a credible
interval on each site's index — all **conditional on the error model below**. If the
error model is wrong the intervals are wrong, so every assumption is a named constant
with a citation or an explicit "assumed", and `python -m src.uncertainty` prints them.

THREE STATISTICAL PROBLEMS IN THE PIXEL SCORE, AND WHAT IS DONE ABOUT THEM
-------------------------------------------------------------------------
1. **A lander needs a pad, not a pixel.** A single 5 m pixel is the noisiest possible
   estimator, and its uncertainty is the full per-pixel error. Candidates are therefore
   scored over a **footprint** (default 50 m radius, i.e. a 101×101 px box) and the
   hard constraint becomes *what fraction of the pad exceeds the slope limit*, which is
   a decision-relevant quantity rather than a pass/fail on one pixel.

2. **Winner's curse.** Taking the maximum over ~10⁷ pixels selects for favourable noise
   as much as for favourable terrain: the top pixel's score is biased high, and the
   bias grows with the number of pixels searched. Averaging over the footprint shrinks
   the per-site noise by roughly √N_eff, and the reported score for each candidate is
   its **Monte-Carlo mean**, not the value that won the search.

3. **One-at-a-time sensitivity is not sensitivity analysis.** Moving one weight ±20%
   with the others fixed explores a measure-zero subset of the weight simplex and
   systematically understates interaction. The weights are instead drawn from a
   **Dirichlet** centred on the nominal weights, so the whole simplex neighbourhood is
   sampled jointly.

ERROR MODEL (all assumptions, all adjustable)
---------------------------------------------
Slope error is split into a systematic part (shared by every pixel in a realisation —
a bias in the LDEM does not average away) and a random part that is spatially
correlated at ~50 m, so footprint averaging reduces it by √N_eff rather than √N.
Interpolated pixels (LOLA count < 1) take an inflated random term. Illumination and
Earth-visibility carry a representativeness error from the 60 m → 5 m resampling,
estimated per site from the local variability of the coarse field, plus a relative
scale uncertainty. Distance-to-PSR carries the 60 m mask resolution.

None of these σ are measured by this prototype. They are read from Barker et al.
(2021) where published and assumed otherwise — see DATA_LIMITS.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.crs import to_lonlat
from src.score import DEFAULT_WEIGHTS, criterion_maps, normalize_weights


@dataclass(frozen=True)
class ErrorModel:
    """Every number here is an assumption. Change it and the intervals change."""

    slope_rms_measured_deg: float = 1.5
    """Slope RMS where LOLA has returns. Barker et al. 2021 report ~1.5-2.5°; the
    optimistic end is used for measured pixels. VERIFY against the paper's table."""

    slope_rms_interpolated_deg: float = 2.5
    """Same range, pessimistic end, for pixels the LDEM interpolated (count < 1)."""

    slope_bias_sigma_deg: float = 0.5
    """Systematic slope error shared across a realisation. Assumed. This is the term
    that does NOT average away over a footprint, so it dominates large pads."""

    slope_correlation_length_m: float = 50.0
    """Spatial correlation length of the random slope error. Assumed. Sets N_eff, i.e.
    how much footprint averaging actually buys."""

    correlated_variance_fraction: float = 0.7
    """Share of the random slope variance that is correlated within a footprint.
    Assumed. Higher ⇒ averaging helps less."""

    visib_relative_sigma: float = 0.05
    """Relative (5%) uncertainty on the illumination / Earth-visibility fractions,
    shared within a site. Assumed; covers the 18.6-yr mean vs an actual mission epoch."""

    visib_representativeness_k: float = 0.5
    """Within-60 m-cell variability is estimated as k × (local range of the coarse
    field over the footprint). Assumed."""

    psr_distance_sigma_m: float = 60.0
    """Distance-to-PSR uncertainty: the mask is 60 m and the transform runs on a 25 m
    grid. Taken as the mask resolution."""

    weight_relative_sigma: float = 0.20
    """Dirichlet spread on the weights, matched to the ±20% the OAT report used, so the
    two are comparable."""

    def slope_sigma(self, interpolated: np.ndarray) -> np.ndarray:
        return np.where(
            interpolated, self.slope_rms_interpolated_deg, self.slope_rms_measured_deg
        ).astype(np.float32)


# ----------------------------------------------------------------------------- windows


def box_mean(a: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean over a (2r+1)² window, and the count of finite pixels in it.

    Summed-area table on a zero-padded copy, read back by plain slicing: O(N) instead
    of O(N·r²), and no full-size fancy-indexing temporaries (which cost about a
    gigabyte on a Site04-sized grid). Edge windows are partial, and dividing by the
    finite-pixel count handles that and NaN in one step.

    float64 throughout: a summed-area table over 10⁷ cells accumulates to ~10⁷, and
    recovering a ~10²-sized window from it in float32 loses most of the answer to
    cancellation.
    """
    finite = np.isfinite(a)
    all_finite = bool(finite.all())
    v = (a if all_finite else np.nan_to_num(a, nan=0.0)).astype(np.float64)
    pad = ((r + 1, r), (r + 1, r))  # the extra leading zero row/col is the SAT's origin

    def sat(x):
        x = np.pad(x, pad)
        np.cumsum(x, axis=0, out=x)
        np.cumsum(x, axis=1, out=x)
        return x

    def win(s):
        k = 2 * r + 1
        return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]

    tot = win(sat(v))
    if all_finite:
        # every pixel counts; the per-window totals depend only on shape and r, and most
        # layers here are fully finite, so this halves the work on the common path
        cnt = _window_counts(a.shape, r)
    else:
        cnt = win(sat(finite.astype(np.float64))).astype(np.float32)
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), np.nan)
    return out.astype(np.float32), cnt.astype(np.float32)


@lru_cache(maxsize=8)
def _window_counts(shape: tuple[int, int], r: int) -> np.ndarray:
    """Finite-pixel count per window when nothing is NaN: separable, so build it from
    the 1-D edge profile instead of a second summed-area table."""
    h, w = shape
    rows = np.minimum(np.arange(h) + r, h - 1) - np.maximum(np.arange(h) - r, 0) + 1
    cols = np.minimum(np.arange(w) + r, w - 1) - np.maximum(np.arange(w) - r, 0) + 1
    return np.outer(rows, cols).astype(np.float32)


def footprint_maps(
    bundle: dict,
    weights: dict | None = None,
    radius_m: float = 50.0,
    slope_max: float = 15.0,
    cap_m: float = 2000.0,
    max_exceedance: float = 0.05,
) -> dict:
    """Score a *pad*, not a pixel.

    Returns the footprint-averaged index plus the two quantities a flight-safety
    reviewer would actually ask for: what fraction of the pad breaks the slope limit,
    and what fraction of it the LDEM had to interpolate.
    """
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    c = criterion_maps(bundle, slope_max=slope_max, cap_m=cap_m)
    r = max(1, int(round(radius_m / float(bundle["pixel_m"]))))

    acc = np.zeros(bundle["dem"].shape, dtype=np.float32)
    for k in ("slope", "illum", "psr", "comms"):
        mean_k, _ = box_mean(np.where(np.isfinite(c[k]), c[k], np.nan), r)
        acc += (w[k] * np.nan_to_num(mean_k, nan=0.0)).astype(np.float32)

    exceed, _ = box_mean((bundle["slope"] > slope_max).astype(np.float32), r)
    interp, _ = box_mean(bundle["interpolated"].astype(np.float32), r)
    penalty_mean, _ = box_mean(c["penalty"], r)

    index = acc * (1.0 - np.nan_to_num(penalty_mean, nan=0.0))
    usable = np.isfinite(bundle["dem"]) & (np.nan_to_num(exceed, nan=1.0) <= max_exceedance)
    index = np.where(usable, index, 0.0).astype(np.float32)

    return {
        "index": index,
        "exceed_frac": np.nan_to_num(exceed, nan=1.0).astype(np.float32),
        "interp_frac": np.nan_to_num(interp, nan=1.0).astype(np.float32),
        "usable": usable,
        "radius_px": r,
        "radius_m": r * float(bundle["pixel_m"]),
        "weights": w,
        "criteria": c,
        "slope_max": slope_max,
        "cap_m": cap_m,
        "max_exceedance": max_exceedance,
    }


def candidate_sites(fp: dict, bundle: dict, n: int = 25, min_sep_m: float = 500.0) -> list[dict]:
    """Greedy separated maxima of the footprint index — the pool the MC then ranks.

    The pool is deliberately larger than the reported top 5: selecting it from the
    noisy map is itself a source of selection bias, and a generous pool keeps sites
    that a single noise draw would have pushed out of the top 5.
    """
    import rasterio.transform as rtransform

    idx = fp["index"]
    mask = fp["usable"] & (idx > 0)
    if not mask.any():
        return []
    rows, cols = np.nonzero(mask)
    order = np.argsort(idx[mask].ravel())[::-1]
    min_sep_px = max(1, int(min_sep_m / float(bundle["pixel_m"])))
    chosen: list[tuple[int, int]] = []
    out: list[dict] = []
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        if any((r - rr) ** 2 + (c - cc) ** 2 < min_sep_px**2 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        x, y = rtransform.xy(bundle["transform"], r, c, offset="center")
        lon, lat = to_lonlat(bundle["crs"]).transform(float(x), float(y))
        out.append(
            {
                "rank_point": len(out) + 1,
                "row": r,
                "col": c,
                "lon": float(lon),
                "lat": float(lat),
                "index": float(idx[r, c]),
                "exceed_frac": float(fp["exceed_frac"][r, c]),
                "interp_frac": float(fp["interp_frac"][r, c]),
            }
        )
        if len(out) >= n:
            break
    return out


def _patch(a: np.ndarray, r: int, c: int, rad: int) -> np.ndarray:
    r0, r1 = max(0, r - rad), min(a.shape[0], r + rad + 1)
    c0, c1 = max(0, c - rad), min(a.shape[1], c + rad + 1)
    return a[r0:r1, c0:c1]


def dirichlet_alpha(weights: dict, rel_sigma: float) -> np.ndarray:
    """Dirichlet concentration whose marginal CV matches `rel_sigma` on the mean weight.

    Var(w_i) = w_i(1-w_i)/(κ+1), so for a representative weight w̄:
        κ = (1 - w̄)/(w̄ · rel_sigma²) - 1
    """
    w = np.array([weights[k] for k in ("slope", "illum", "psr", "comms")], dtype=np.float64)
    wbar = float(w.mean())
    kappa = max((1.0 - wbar) / (wbar * rel_sigma**2) - 1.0, 1.0)
    return w * kappa


def monte_carlo(
    bundle: dict,
    fp: dict,
    sites: list[dict],
    n_draws: int = 1000,
    model: ErrorModel | None = None,
    seed: int = 0,
    top_k: int = 5,
) -> dict:
    """Propagate input error and weight uncertainty through the footprint index.

    Per draw: one systematic slope bias for the whole map, one correlated slope error
    per site, independent per-pixel residuals, a relative visibility offset per site, a
    PSR-distance offset, and one Dirichlet weight vector. The index is then recomputed
    from the perturbed inputs over each site's footprint.
    """
    if not sites:
        return {
            "samples": np.zeros((0, n_draws), np.float32),
            "sites": [],
            "model": model or ErrorModel(),
            "n_draws": n_draws,
            "pad_m": 0.0,
        }
    m = model or ErrorModel()
    rng = np.random.default_rng(seed)
    rad = fp["radius_px"]
    px = float(bundle["pixel_m"])
    slope_max, cap_m = fp["slope_max"], fp["cap_m"]

    # how much of the random slope error survives footprint averaging
    n_px = (2 * rad + 1) ** 2
    corr_px = max(1.0, (m.slope_correlation_length_m / px) ** 2)
    n_eff = max(1.0, n_px / corr_px)

    patches = []
    for s in sites:
        r, c = s["row"], s["col"]
        illum = fp["criteria"]["illum"]
        comms = fp["criteria"]["comms"]
        patches.append(
            {
                "slope": np.asarray(_patch(bundle["slope"], r, c, rad), dtype=np.float32),
                "illum": np.asarray(_patch(illum, r, c, rad), dtype=np.float32),
                "comms": np.asarray(_patch(comms, r, c, rad), dtype=np.float32),
                "psr_dist": (
                    None if bundle["psr_dist"] is None
                    else np.asarray(_patch(bundle["psr_dist"], r, c, rad), dtype=np.float32)
                ),
                "sigma_slope": m.slope_sigma(_patch(bundle["interpolated"], r, c, rad)),
                "penalty": np.asarray(_patch(fp["criteria"]["penalty"], r, c, rad), dtype=np.float32),
            }
        )

    alpha = dirichlet_alpha(fp["weights"], m.weight_relative_sigma)
    wdraws = rng.dirichlet(alpha, size=n_draws)  # (draws, 4) — joint, not one-at-a-time

    n_s = len(sites)
    scores = np.zeros((n_s, n_draws), dtype=np.float32)
    exceed = np.zeros((n_s, n_draws), dtype=np.float32)

    bias = rng.normal(0.0, m.slope_bias_sigma_deg, size=n_draws).astype(np.float32)
    for i, p in enumerate(patches):
        sig = p["sigma_slope"]
        sig_corr = sig * np.sqrt(m.correlated_variance_fraction)
        sig_ind = sig * np.sqrt(1.0 - m.correlated_variance_fraction)
        # correlated part: one draw shared by the whole pad; independent part: per pixel,
        # but only √n_eff of the pad's pixels are statistically independent
        e_corr = rng.normal(0.0, 1.0, size=(n_draws, 1)).astype(np.float32) * sig_corr.mean()
        e_ind = rng.normal(0.0, 1.0, size=(n_draws, 1)).astype(np.float32) * (
            sig_ind.mean() / np.sqrt(n_eff)
        )
        # per-pixel realisation for the exceedance fraction (the hard constraint)
        slope_flat = p["slope"].ravel()[None, :]
        pad_noise = rng.normal(0.0, 1.0, size=(n_draws, slope_flat.shape[1])).astype(np.float32)
        pad_noise *= sig_ind.ravel()[None, :]
        slope_draws = slope_flat + bias[:, None] + e_corr + pad_noise
        exceed[i] = (slope_draws > slope_max).mean(axis=1)

        c_slope = np.clip(1.0 - (p["slope"].mean() + bias + e_corr[:, 0] + e_ind[:, 0]) / slope_max, 0, 1)

        v_off = rng.normal(0.0, m.visib_relative_sigma, size=n_draws).astype(np.float32)
        rep_illum = m.visib_representativeness_k * float(np.nanmax(p["illum"]) - np.nanmin(p["illum"]) if p["illum"].size else 0.0)
        rep_comms = m.visib_representativeness_k * float(np.nanmax(p["comms"]) - np.nanmin(p["comms"]) if p["comms"].size else 0.0)
        base_illum = float(np.nan_to_num(np.nanmean(p["illum"]), nan=0.0))
        base_comms = float(np.nan_to_num(np.nanmean(p["comms"]), nan=0.0))
        c_illum = np.clip(base_illum * (1.0 + v_off) + rng.normal(0.0, max(rep_illum, 1e-6), n_draws), 0, 1)
        c_comms = np.clip(base_comms * (1.0 + v_off) + rng.normal(0.0, max(rep_comms, 1e-6), n_draws), 0, 1)

        if p["psr_dist"] is None:
            c_psr = np.zeros(n_draws, dtype=np.float32)
        else:
            d = float(np.nan_to_num(np.nanmean(p["psr_dist"]), nan=cap_m))
            d_draws = np.clip(d + rng.normal(0.0, m.psr_distance_sigma_m, n_draws), 0, cap_m)
            c_psr = np.clip(1.0 - d_draws / cap_m, 0, 1)

        pen = float(np.nan_to_num(np.nanmean(p["penalty"]), nan=0.0))
        crit = np.stack([c_slope, c_illum, c_psr, c_comms], axis=1)  # (draws, 4)
        scores[i] = (crit * wdraws).sum(axis=1).astype(np.float32) * (1.0 - pen)

    # a draw where the pad breaks the slope budget is not a candidate in that draw
    feasible = exceed <= fp["max_exceedance"]
    eff = np.where(feasible, scores, -np.inf)
    ranks = np.argsort(np.argsort(-eff, axis=0), axis=0)  # 0 = best within each draw
    p_top = (ranks < top_k).mean(axis=1)
    p_best = (ranks == 0).mean(axis=1)
    p_feasible = feasible.mean(axis=1)

    return {
        "samples": scores,
        "exceedance": exceed,
        "p_top_k": p_top,
        "p_best": p_best,
        "p_feasible": p_feasible,
        "sites": sites,
        "weights_drawn": wdraws,
        "n_eff": n_eff,
        "pad_m": (2 * rad + 1) * px,
        "model": m,
        "top_k": top_k,
        "n_draws": n_draws,
    }


def summary_table(mc: dict) -> pd.DataFrame:
    """One row per candidate: the index with a 90% interval, not a bare number."""
    if not mc["sites"]:
        return pd.DataFrame(
            columns=["rank", "lat", "lon", "index_mean", "ci05", "ci95", "P_top5", "P_best",
                     "P_pad_within_slope_budget", "exceed_frac_nominal", "interp_frac"]
        )
    s = mc["samples"]
    order = np.argsort(-s.mean(axis=1))
    rows = []
    for new_rank, i in enumerate(order, start=1):
        site = mc["sites"][i]
        draws = s[i]
        rows.append(
            {
                "rank": new_rank,
                "lat": round(site["lat"], 5),
                "lon": round(site["lon"], 5),
                "index_mean": round(float(draws.mean()), 4),
                "ci05": round(float(np.percentile(draws, 5)), 4),
                "ci95": round(float(np.percentile(draws, 95)), 4),
                "P_top5": round(float(mc["p_top_k"][i]), 3),
                "P_best": round(float(mc["p_best"][i]), 3),
                "P_pad_within_slope_budget": round(float(mc["p_feasible"][i]), 3),
                "exceed_frac_nominal": round(site["exceed_frac"], 4),
                "interp_frac": round(site["interp_frac"], 3),
            }
        )
    return pd.DataFrame(rows)


def pairwise_dominance(mc: dict, i: int, j: int) -> float:
    """P(site i scores above site j) over the SAME draws — paired, so the shared
    systematic error cancels where it should and the comparison is not double-counted."""
    s = mc["samples"]
    return float((s[i] > s[j]).mean())


def separation_verdict(mc: dict) -> str:
    """Plain-language answer to 'is the winner actually the winner?'"""
    if len(mc["sites"]) < 2:
        return "Fewer than two candidates — no ranking to defend."
    s = mc["samples"]
    order = np.argsort(-s.mean(axis=1))
    a, b = int(order[0]), int(order[1])
    p = pairwise_dominance(mc, a, b)
    if p >= 0.95:
        return f"Rank 1 beats rank 2 in {p * 100:.0f}% of draws — the ordering is robust to the stated errors."
    if p >= 0.75:
        return (
            f"Rank 1 beats rank 2 in only {p * 100:.0f}% of draws — a leaning, not a result. "
            "Do not present these two as separable."
        )
    return (
        f"Rank 1 beats rank 2 in {p * 100:.0f}% of draws — statistically indistinguishable. "
        "Report them as a tied shortlist."
    )


def spearman_matrix(bundle: dict, fp: dict, n_sample: int = 200_000, seed: int = 0) -> pd.DataFrame:
    """Rank correlation between criteria.

    Criteria that correlate strongly are not independent evidence: weighting both is
    double-counting the same terrain fact, and it quietly inflates the winner's margin.
    This is the diagnostic that says whether the four-criterion split is honest.
    """
    keys = ("slope", "illum", "psr", "comms")
    c = fp["criteria"]
    valid = fp["usable"] & np.isfinite(bundle["dem"])
    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        return pd.DataFrame(np.full((4, 4), np.nan), index=keys, columns=keys)
    rng = np.random.default_rng(seed)
    if idx.size > n_sample:
        idx = rng.choice(idx, size=n_sample, replace=False)
    cols = []
    for k in keys:
        v = np.nan_to_num(c[k], nan=0.0).ravel()[idx].astype(np.float64)
        cols.append(np.argsort(np.argsort(v)).astype(np.float64))
    m = np.corrcoef(np.vstack(cols))
    return pd.DataFrame(np.round(m, 3), index=keys, columns=keys)


def assumptions_table(model: ErrorModel) -> pd.DataFrame:
    src = {
        "slope_rms_measured_deg": "Barker et al. 2021 (verify table)",
        "slope_rms_interpolated_deg": "Barker et al. 2021 (verify table)",
        "slope_bias_sigma_deg": "assumed",
        "slope_correlation_length_m": "assumed",
        "correlated_variance_fraction": "assumed",
        "visib_relative_sigma": "assumed",
        "visib_representativeness_k": "assumed",
        "psr_distance_sigma_m": "product resolution (60 m LPSR)",
        "weight_relative_sigma": "chosen to match the ±20% OAT report",
    }
    return pd.DataFrame(
        [{"parameter": k, "value": v, "source": src[k]} for k, v in asdict(model).items()]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Rank statistics for the Site04 screening score")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--candidates", type=int, default=25)
    p.add_argument("--radius-m", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", default=str(paths.DERIVED / "ranking_stats.csv"))
    args = p.parse_args()

    from src.ingest import load_site

    bundle = load_site()
    fp = footprint_maps(bundle, radius_m=args.radius_m)
    sites = candidate_sites(fp, bundle, n=args.candidates)
    mc = monte_carlo(bundle, fp, sites, n_draws=args.draws, seed=args.seed)
    table = summary_table(mc)

    print(f"=== footprint {fp['radius_m']:.0f} m ({2 * fp['radius_px'] + 1}² px), "
          f"{args.draws} draws, N_eff per pad ≈ {mc['n_eff']:.0f} ===")
    print(table.head(10).to_string(index=False))
    print("\n" + separation_verdict(mc))
    print("\nCriterion rank correlation (high |ρ| ⇒ the criteria are not independent evidence):")
    print(spearman_matrix(bundle, fp).to_string())
    print("\nError model — these drive every interval above:")
    print(assumptions_table(mc["model"]).to_string(index=False))
    print("\nThe index is a preference score, not a probability of mission success.")

    dest = Path(args.out_csv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
