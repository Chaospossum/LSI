# LUNAR SITE INTELLIGENCE

Laptop prototype for a student space challenge: screen the **Shackleton rim (PGDA Site04)** with a transparent suitability score, keep a small multi-actor surface-ops registry, and flag buffer+time conflicts. Honest about data limits.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # once; needs network. Then fully offline.
python -m pytest tests/ -q
streamlit run app.py
```

Optional (report PNGs and statistics without the UI):

```bash
python -m src.score                # data/derived/score.tif + top5.csv
python -m src.uncertainty          # ranking with 90% intervals + P(top 5), and the assumptions
python scripts/make_figures.py     # figures/score_map.png, conflict_demo.png, ranking_intervals.png
```

After the download, `streamlit run app.py` is the demo command. No tiles, no APIs.

## Ranking statistics

A pixel score alone cannot survive a hard question. The **Ranking statistics** tab (and
`python -m src.uncertainty`) scores a *pad* rather than a pixel, propagates the published
input error and a Dirichlet spread on the weights through ~1000 Monte-Carlo draws, and
reports each candidate's index with a 90% interval, P(top 5), and the probability its pad
stays inside the slope budget. It says out loud when the leaders are **not separable** —
usually they are not. Conflicts get the same treatment: P(overlap) under coordinate
uncertainty instead of a binary flag.

The index is a **preference score, not a probability of mission success**, and every σ in
the error model is printed with its source (or labelled *assumed*). See `DATA_LIMITS.md`.

## What this does NOT do

- No ML crater / boulder detection.
- No sub-5 m hazard maps (rocks, ice, trafficability).
- No orbital / cislunar deconfliction.
- No legal or ITU weight — a **voluntary notification** sketch only.
- No survey-grade site certification (see `DATA_LIMITS.md`).
- No calibrated probability of landing success — the index is a preference score, and nothing here is fitted to landing outcomes.

## Layout

`app.py` · `src/{ingest,score,uncertainty,registry,conflict,viz}.py` · `registry/sites.geojson` · `scripts/download_data.py` · `data/` (gitignored)

Weights (slope, illumination, near-PSR, Earth visibility) sum to 1. Hard mask: slope &gt; 15° (slider). LOLA count is a **penalty**, not a criterion: a pixel with no LOLA return loses 15% of its score. Placeholders in the registry stay labelled placeholder. Weight-sensitivity (±20%) re-scores the grid 8 times, so it runs on a button, not on every slider move.

Before a demo, `python scripts/download_data.py --check` HEADs every source URL and downloads nothing.
