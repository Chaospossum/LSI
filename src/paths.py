from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"
REGISTRY = ROOT / "registry" / "sites.geojson"
FIGURES = ROOT / "figures"
LAYER_JSON = DATA / "layers.json"

# Canonical filenames after download_data.py
SITE04_DEM = RAW / "Site04_final_adj_5mpp_surf.tif"
SITE04_SLOPE = RAW / "Site04_final_adj_5mpp_slp.tif"
SITE04_COUNT = RAW / "Site04_final_adj_5mpp_ldec.tif"
SITE07_DEM = RAW / "Site07_final_adj_5mpp_surf.tif"
ILLUM = RAW / "AVGVISIB_85S_060M_201608.TIF"
EARTH_VIS = RAW / "AVGVISIB_85S_060M_201608_EARTH.TIF"
PSR_RASTER = RAW / "LPSR_85S_060M_201608.TIF"
PSR_SHP = RAW / "LPSR_80S_20MPP_ADJ_1km2.SHP"
