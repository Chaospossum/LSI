# Data limits — LUNAR SITE INTELLIGENCE

This is a **landing-site screening tool**, not a survey-grade site certificate. Do not treat pixel scores as a go/no-go for flight.

## LOLA 5 m DEM (Site04 Shackleton rim)

- Source: PGDA high-resolution south-pole site products (Barker et al. 2021, PSS).
- Grid: 5 m/pixel, south polar stereographic, MOON_ME / DE421.
- **~90% of 5 m polar LOLA pixels are interpolated between tracks.** LOLA’s cross-track and inter-spot spacing does not fill a 5 m grid. The LDEM looks continuous because empty pixels are filled.
- Use the **LDEC count map**: pixels with count &lt; 1 have **no LOLA spot**. Those elevations are interpolated. This prototype applies a **confidence penalty** (not a 5th weight) from that count map.
- Median RMS height error on these products is on the order of **0.3–0.5 m**; slope RMS **~1.5–2.5°** (Barker et al. 2021). Interpolation error grows with gap size and slope.
- The 5 m slope GeoTIFF is derived from the interpolated LDEM. It is **not** an independent 5 m measurement.

## Illumination and Earth visibility (60 m → 5 m)

- A public **5 m** illumination-fraction raster for this site was **not** used as a native product.
- The app uses Mazarico / LOLA **18.6-year mean** solar illumination and Earth-visibility maps at **60 m/pixel**, resampled onto the 5 m grid.
- Kind: **interpolated** (coarser modeled maps, bilinear onto 5 m). These are **not** 5 m measurements.
- Earth-visibility maps treat the Earth as a disc (any unobscured limb counts). That is **optimistic** versus a DSN radio link.
- If those files are missing, do not invent values: that criterion is zeroed and the layer is marked absent in metadata.

## Permanently shadowed regions

- Distance-to-PSR uses a **60 m** LPSR raster (and, if present, Barker 2023 PSRs &gt; 1 km²) aligned to Site04.
- Kind: **interpolated** onto 5 m. Small PSRs and 5 m shadow edges are **not** resolved.

## Registry coordinates

- `coord_source` is one of `published` | `approximate` | `placeholder`.
- **Placeholder points are not real landing sites.** Blue Moon MK1 Endurance, IM-4, and MAGPIE are demo geometry only.
- Chang’e-7 (~88.8°S, 123.4°E) is **approximate** and may sit **outside** the Site04 5 m window. Conflicts still run in polar stereographic; pixel inspect only works on Site04.
- NASA “Peak near Shackleton” uses the **published Site07 GeoTIFF centre**, not a NASA-announced pad.

## Map display

- Leaflet **Web Mercator cannot show 90°S** (clips near 85°). The app map uses Leaflet **Simple CRS** with **south polar stereographic metres**. Click coordinates are (x, y) metres, converted to lon/lat only for the inspect panel and what-if lander.
- This is still not a lunar web-tile globe. Report figures (`figures/*.png`) are the undistorted raster view.

## What this tool is for

Regional comparison of slope, illumination, PSR proximity, Earth visibility, and operation buffers — with explicit metadata and a confidence penalty from the LOLA count map.
