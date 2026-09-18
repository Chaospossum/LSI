"""Published sources this prototype actually uses. No invented papers."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Citation:
    key: str
    authors: str
    year: int
    title: str
    venue: str
    doi: str
    url: str
    used_for: str

    def apa(self) -> str:
        doi = f" https://doi.org/{self.doi}" if self.doi else ""
        return (
            f"{self.authors} ({self.year}). {self.title}. {self.venue}.{doi}"
        ).strip()


CITATIONS: dict[str, Citation] = {
    "barker2021": Citation(
        key="barker2021",
        authors="Barker, M. K., Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., & Head, J. W.",
        year=2021,
        title=(
            "Improved LOLA elevation maps for south pole landing sites: "
            "Error estimates and their impact on illumination conditions"
        ),
        venue="Planetary and Space Science, 203, 105119",
        doi="10.1016/j.pss.2020.105119",
        url="https://doi.org/10.1016/j.pss.2020.105119",
        used_for="Site04 5 m LDEM, slope, LDEC count, slperr, toterr (PGDA product 78)",
    ),
    "mazarico2011": Citation(
        key="mazarico2011",
        authors="Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., & Torrence, M. H.",
        year=2011,
        title="Illumination conditions of the lunar polar regions using LOLA topography",
        venue="Icarus, 211(2), 1066–1081",
        doi="10.1016/j.icarus.2010.10.030",
        url="https://doi.org/10.1016/j.icarus.2010.10.030",
        used_for="60 m AVGVISIB solar/Earth maps and LPSR mask (PGDA MoonIllumination)",
    ),
    "nasa2022": Citation(
        key="nasa2022",
        authors="NASA",
        year=2022,
        title="NASA identifies candidate regions for landing next Americans on Moon",
        venue="News release (Artemis III candidate regions, including Peak Near Shackleton)",
        doi="",
        url="https://www.nasa.gov/news-release/nasa-identifies-candidate-regions-for-landing-next-americans-on-moon/",
        used_for="Artemis III region name Peak Near Shackleton — not a pad coordinate",
    ),
    "kumari2022": Citation(
        key="kumari2022",
        authors="Kumari, N., et al.",
        year=2022,
        title="Surface conditions and resource accessibility at potential Artemis landing sites 007 and 011",
        venue="The Planetary Science Journal, 3, 224",
        doi="10.3847/PSJ/ac88c2",
        url="https://doi.org/10.3847/PSJ/ac88c2",
        used_for="Published context for Site 007 inside Peak Near Shackleton (not a pad)",
    ),
    "gracy2024": Citation(
        key="gracy2024",
        authors="Gracy, S., & Lee, P.",
        year=2024,
        title=(
            "Candidate landing sites for Artemis 3 in two NASA candidate landing "
            "regions nearest the lunar south pole"
        ),
        venue="55th Lunar and Planetary Science Conference, abstract 1695",
        doi="",
        url="https://www.hou.usra.edu/meetings/lpsc2024/pdf/1695.pdf",
        used_for="Literature pad 89.01701°S, 126.27302°E inside Peak Near Shackleton",
    ),
}

LAYER_CITE = {
    "dem": "barker2021",
    "slope": "barker2021",
    "count": "barker2021",
    "slope_err": "barker2021",
    "height_err": "barker2021",
    "illumination": "mazarico2011",
    "earth_visibility": "mazarico2011",
    "psr": "mazarico2011",
}


def get(key: str) -> Citation:
    return CITATIONS[key]


def citations_table() -> pd.DataFrame:
    rows = [
        {
            "key": c.key,
            "year": c.year,
            "citation": c.apa(),
            "doi": c.doi or "—",
            "url": c.url,
            "used_for": c.used_for,
        }
        for c in CITATIONS.values()
    ]
    return pd.DataFrame(rows)
