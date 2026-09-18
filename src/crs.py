"""Moon south-polar stereographic CRS helpers.

PGDA Site04/07 GeoTIFFs are south polar stereographic, metres, MOON_ME / DE421.
We do not assume a WKT string until ingest reads the file; this is the PROJ
fallback matching the product description (sphere R = 1737.4 km, true scale at pole).
"""
from __future__ import annotations

from pyproj import CRS, Transformer

MOON_R_M = 1737400.0

# lon_0=0 is the PGDA/LOLA polar convention used by the 5 m products.
LUNAR_SP_STEREO = (
    f"+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    f"+R={MOON_R_M} +units=m +no_defs"
)
LUNAR_LONLAT = f"+proj=lonlat +R={MOON_R_M} +no_defs"

STEREO_CRS = CRS.from_proj4(LUNAR_SP_STEREO)
LONLAT_CRS = CRS.from_proj4(LUNAR_LONLAT)


def stereo_from_raster(src) -> CRS:
    """Prefer the GeoTIFF CRS; fall back to the documented PROJ string."""
    if src.crs:
        return CRS.from_user_input(src.crs)
    return STEREO_CRS


def to_stereo(crs: CRS | None = None) -> Transformer:
    return Transformer.from_crs(LONLAT_CRS, crs or STEREO_CRS, always_xy=True)


def to_lonlat(crs: CRS | None = None) -> Transformer:
    return Transformer.from_crs(crs or STEREO_CRS, LONLAT_CRS, always_xy=True)
