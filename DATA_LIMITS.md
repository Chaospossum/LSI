# Data limits — LUNAR SITE INTELLIGENCE

This is a **landing-site screening tool**, not a survey-grade site certificate. Do not treat pixel scores as a go/no-go for flight.

## LOLA 5 m DEM (Site04 Shackleton rim)

- Source: PGDA high-resolution south-pole site products (Barker et al. 2021, PSS).
- Grid: 5 m/pixel, south polar stereographic, MOON_ME / DE421.
- **~90% of 5 m polar LOLA pixels are interpolated between tracks.** LOLA’s cross-track and inter-spot spacing does not fill a 5 m grid. The LDEM looks continuous because empty pixels are filled.
- Use the **LDEC count map**: pixels with count &lt; 1 have **no LOLA spot**. Those elevations are interpolated. This prototype applies a **confidence penalty** (not a 5th weight) from that count map: a pixel with at least one LOLA return keeps its full score, a pixel with none loses **15%** of its weighted score. Nothing in between — the count map answers *measured or not*, not *how well measured*.
- Median RMS height error on these products is on the order of **0.3–0.5 m**; slope RMS **~1.5–2.5°** (Barker et al. 2021). Interpolation error grows with gap size and slope.
- The 5 m slope GeoTIFF is derived from the interpolated LDEM. It is **not** an independent 5 m measurement.

## Illumination and Earth visibility (60 m → 5 m)

- A public **5 m** illumination-fraction raster for this site was **not** used as a native product.
- The app uses Mazarico / LOLA **18.6-year mean** solar illumination and Earth-visibility maps at **60 m/pixel**, resampled onto the 5 m grid.
- Kind: **interpolated** (coarser modeled maps, bilinear onto 5 m). These are **not** 5 m measurements.
- Earth-visibility maps treat the Earth as a disc (any unobscured limb counts). That is **optimistic** versus a DSN radio link.
- DN → fraction uses the documented PDS factor (**×4e-5**). If the GeoTIFF carries no scale tag the factor is applied explicitly and `layers.json` records that it came from the PDS label rather than the file.
- If those files are missing, do not invent values: that criterion is zeroed and the layer is marked absent in metadata. A map that still looks like raw DN after scaling is **not** stretched to 0–1 by its in-window maximum — that would invent a fraction — the criterion is dropped instead.

## Permanently shadowed regions

- Distance-to-PSR uses the **60 m** LPSR raster aligned to Site04. Vector PSR products (Barker 2023, &gt; 1 km²) are **not** used.
- Kind: **interpolated** onto 5 m. Small PSRs and 5 m shadow edges are **not** resolved.
- The distance transform runs on a **4× coarsened (25 m) grid** and is upsampled, so distance-to-PSR is quantised to ~25 m and capped at 2 km.

## Registry coordinates

- `coord_source` is one of `published` | `approximate` | `placeholder`.
- **Placeholder points are not real landing sites.** Blue Moon MK1 Endurance, IM-4, and MAGPIE are demo geometry only.
- Chang’e-7 (~88.8°S, 123.4°E) is **approximate** and may sit **outside** the Site04 5 m window. Conflicts still run in polar stereographic; pixel inspect only works on Site04.
- NASA “Peak near Shackleton” uses the **published Site07 GeoTIFF centre**, not a NASA-announced pad.

## Map display

- Leaflet **Web Mercator cannot show 90°S** (clips near 85°). The app map uses Leaflet **Simple CRS** with **south polar stereographic metres**. Click coordinates are (x, y) metres, converted to lon/lat only for the inspect panel and what-if lander.
- This is still not a lunar web-tile globe. Report figures (`figures/*.png`) are the undistorted raster view.

## Statistical validity — what the numbers support, and what they do not

The suitability score is a **multi-criteria preference index**. It is not a probability
of mission success, not a certification, and it is not calibrated against any landing
outcome, because no such training set exists at 5 m for this site. Treating it as a
success probability would be the single most expensive mistake a reader of this tool
could make.

What *is* defensible is a statement about **ranking stability**: given the published
error of the inputs and a stated uncertainty on the weights, how likely is a candidate
to be in the top 5? `python -m src.uncertainty` answers that and prints every
assumption behind it.

### Three problems with a pixel score, and what the tool does about them

- **A lander needs a pad, not a pixel.** A single 5 m pixel is the noisiest possible
  estimator. Candidates are scored over a square pad (default 105 m across) and the
  hard constraint becomes *what fraction of the pad exceeds the slope limit* (default
  tolerance 5%), which is what a flight-safety reviewer actually asks.
- **Winner's curse.** Selecting the maximum over ~10⁷ pixels selects for favourable
  noise as well as favourable terrain, so the winning pixel's score is biased high.
  Footprint averaging shrinks the noise and the reported figure is the Monte-Carlo
  **mean**, not the value that won the search.
- **One-at-a-time weight sensitivity is not sensitivity analysis.** The ±20% OAT table
  explores a measure-zero slice of the weight simplex. Weights are also drawn jointly
  from a **Dirichlet** centred on the nominal values.

### The error model is an assumption, and it is printed

| Term | Value | Source |
|---|---|---|
| Slope RMS, measured pixels | 1.5° | Barker et al. 2021 — **verify the table before quoting** |
| Slope RMS, interpolated pixels | 2.5° | Barker et al. 2021 — **verify** |
| Systematic slope bias | σ = 0.5° | **assumed** — does not average away over a pad |
| Slope error correlation length | 50 m | **assumed** — sets how much averaging buys |
| Correlated share of slope variance | 0.7 | **assumed** |
| Illumination / Earth-visibility scale | 5% relative | **assumed** |
| 60 m → 5 m representativeness | 0.5 × local range | **assumed** |
| Distance-to-PSR | σ = 60 m | LPSR product resolution |
| Weight spread | 20% relative (Dirichlet) | matched to the OAT report |

Errors are treated as Gaussian and only partly independent. Real DEM error is neither.
If the σ above are wrong, every interval this tool prints is wrong with them.

### Criterion independence

`spearman_matrix` reports the rank correlation between the four criteria. Two criteria
that correlate strongly are **the same terrain fact counted twice**, which inflates the
winner's apparent margin. Read that matrix before defending the weights.

### Conflicts

`conflict_probability` replaces the binary overlap flag with P(overlap) under coordinate
uncertainty (assumed: published σ = 150 m, approximate σ = 5 km). Placeholder geometry
returns **NaN**, never a number: inventing an uncertainty for a made-up point would
launder it into a result.

## What this tool is for

Regional comparison of slope, illumination, PSR proximity, Earth visibility, and operation buffers — with explicit metadata and a confidence penalty from the LOLA count map.
