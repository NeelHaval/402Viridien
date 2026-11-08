"""Build data/reference/ from public sources. Idempotent: each step is independent and skips existing files."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.validation.config import COARSE_CLASSES, coarse_class  # noqa: E402

OUT_DIR = ROOT / "data" / "reference"
MACROSTRAT_URL = "https://macrostrat.org/api/v2/units"
LAT_RANGE, LON_RANGE, STEP_DEG = (24.0, 31.0), (-98.0, -80.0), 0.5
WORKERS, TIMEOUT_S, RETRIES = 6, 20, 3
GULF_BOX = {"lat": (17.0, 32.0), "lon": (-98.5, -80.0)}
TARGET_SPACING_KM = 1.0
LITH_TYPE_CLASS = {"evaporite": "evaporite", "carbonate": "carbonate", "igneous": "volcanic", "volcanic": "volcanic"}
GEBCO_PAGE = "https://www.gebco.net/data-products/gridded-bathymetry-data"
GEBCO_APP = "https://download.gebco.net"
BOEM_MAPPING_PAGE = "https://www.boem.gov/oil-gas-energy/mapping-and-data"
BOEM_DATA_CENTER = "https://www.data.boem.gov/"
BOEM_BOREHOLE_PAGE = "https://www.data.boem.gov/Well/Borehole/Default.aspx"
WELL_ALIASES = {
    "api": ["api", "api_well_number", "api_number", "uwi"],
    "lat": ["lat", "latitude", "surf_latitude", "surface_latitude", "surf_lat"],
    "lon": ["lon", "long", "longitude", "surf_longitude", "surface_longitude", "surf_lon"],
    "water_depth_ft": ["water_depth_ft", "water_depth", "wd", "water_depth_feet"],
    "total_depth_ft": ["total_depth_ft", "bh_total_md", "total_md", "total_depth", "td", "total_measured_depth"],
}
BLOCK_ALIASES = {
    "area_code": ["area_code", "protraction_code", "prot_abbr", "area", "prot_acronym"],
    "area_name": ["area_name", "protraction_name", "prot_name", "area_description"],
    "block": ["block", "block_number", "blk_num", "block_num", "blocknumber"],
    "lat": ["lat", "latitude", "center_lat", "centroid_lat", "y"],
    "lon": ["lon", "long", "longitude", "center_lon", "centroid_lon", "x"],
}
AREA_NAMES = {
    "AC": "Alaminos Canyon", "AT": "Atwater Valley", "BA": "Brazos", "DC": "Desoto Canyon", "EB": "Ewing Bank",
    "EC": "East Cameron", "EI": "Eugene Island", "GB": "Garden Banks", "GC": "Green Canyon", "GI": "Grand Isle",
    "HI": "High Island", "KC": "Keathley Canyon", "LL": "Lloyd Ridge", "MC": "Mississippi Canyon", "MP": "Main Pass",
    "MO": "Mobile", "PL": "Pensacola", "SM": "South Marsh Island", "SP": "South Pass", "SS": "Ship Shoal",
    "ST": "South Timbalier", "VK": "Viosca Knoll", "VR": "Vermilion", "WC": "West Cameron", "WD": "West Delta",
    "WR": "Walker Ridge", "EW": "Ewing Bank", "NO": "New Orleans", "DD": "Destin Dome", "LB": "Lund",
}


def classify_units(units: list[dict]) -> tuple[dict[str, float], int]:
    """Thickness- and proportion-weighted coarse-class shares for the units at one point."""
    shares = {c: 0.0 for c in COARSE_CLASSES}
    for unit in units:
        thick = (float(unit.get("min_thick") or 0) + float(unit.get("max_thick") or 0)) / 2 or 1.0
        for lith in unit.get("lith", []):
            cls = coarse_class(lith.get("name", ""))
            if cls == "other":
                cls = LITH_TYPE_CLASS.get(str(lith.get("type", "")).lower(), "other")
            shares[cls] += thick * float(lith.get("prop") or 0.0)
    total = sum(shares.values())
    return ({c: v / total for c, v in shares.items()} if total > 0 else shares), len(units)


def fetch_point(lat: float, lon: float) -> tuple[float, float, dict[str, float], int] | None:
    """Query Macrostrat for one point with retries; None when the request keeps failing."""
    for attempt in range(RETRIES):
        try:
            resp = requests.get(MACROSTRAT_URL, params={"lat": lat, "lng": lon, "response": "long", "format": "json"}, timeout=TIMEOUT_S)
            resp.raise_for_status()
            shares, n = classify_units(resp.json().get("success", {}).get("data", []))
            return lat, lon, shares, n
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return None


def step_macrostrat(path: Path, force: bool) -> str:
    """Download the Macrostrat grid; write a partial file if interrupted."""
    if path.exists() and not force:
        return "skipped (exists)"
    points = [(float(a), float(b)) for a in np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-9, STEP_DEG)
              for b in np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-9, STEP_DEG)]
    rows, failed = [], 0
    pool = ThreadPoolExecutor(WORKERS)
    try:
        for res in pool.map(lambda p: fetch_point(*p), points):
            if res is None:
                failed += 1
            elif res[3] > 0:
                rows.append({"lat": res[0], "lon": res[1], **res[2], "n_units": res[3]})
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("interrupted: writing partial macrostrat_grid.csv")
    pool.shutdown(wait=False, cancel_futures=True)
    empty = len(points) - len(rows) - failed
    print(f"macrostrat: {len(rows)} points with units, {empty} with no units (offshore coverage unverified), {failed} failed")
    if not rows:
        return "manual (no data retrieved; check network)"
    pd.DataFrame(rows).to_csv(path, index=False)
    return "ok" if failed == 0 and len(rows) + empty == len(points) else "ok (partial)"


def _grid_step(spacing_deg: float) -> int:
    """Subsampling stride that gives about TARGET_SPACING_KM from a grid with the given spacing."""
    return max(1, round((TARGET_SPACING_KM / 111.0) / max(spacing_deg, 1e-9)))


def _write_bathymetry(lats: np.ndarray, lons: np.ndarray, elev: np.ndarray, path: Path) -> None:
    """Write thinned (lat, lon, depth_m) rows where depth is positive under water."""
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    pd.DataFrame({"lat": la.ravel().round(5), "lon": lo.ravel().round(5), "depth_m": (-elev).ravel().astype(np.float32)}).to_csv(path, index=False)


def _read_netcdf(file: Path):
    """Read a GEBCO-style NetCDF subset over the Gulf box."""
    import netCDF4
    with netCDF4.Dataset(file) as ds:
        lat, lon = ds.variables["lat"][:], ds.variables["lon"][:]
        elev = ds.variables["elevation"]
        ia = np.flatnonzero((lat >= GULF_BOX["lat"][0]) & (lat <= GULF_BOX["lat"][1]))
        ja = np.flatnonzero((lon >= GULF_BOX["lon"][0]) & (lon <= GULF_BOX["lon"][1]))
        si, sj = _grid_step(abs(float(lat[1] - lat[0]))), _grid_step(abs(float(lon[1] - lon[0])))
        rows, cols = ia[::si], ja[::sj]
        return np.asarray(lat[rows]), np.asarray(lon[cols]), np.asarray(elev[rows.min():rows.max() + 1:si, cols.min():cols.max() + 1:sj])


def _read_geotiff(file: Path):
    """Read a north-up GeoTIFF using its ModelPixelScale and ModelTiepoint tags (loads the whole raster)."""
    import tifffile
    with tifffile.TiffFile(file) as tif:
        page = tif.pages[0]
        scale, tie = page.tags["ModelPixelScaleTag"].value, page.tags["ModelTiepointTag"].value
        data = page.asarray()
    lats = tie[4] - (np.arange(data.shape[0]) + 0.5) * scale[1]
    lons = tie[3] + (np.arange(data.shape[1]) + 0.5) * scale[0]
    ia = np.flatnonzero((lats >= GULF_BOX["lat"][0]) & (lats <= GULF_BOX["lat"][1]))[::_grid_step(scale[1])]
    ja = np.flatnonzero((lons >= GULF_BOX["lon"][0]) & (lons <= GULF_BOX["lon"][1]))[::_grid_step(scale[0])]
    return lats[ia], lons[ja], data[np.ix_(ia, ja)]


def step_bathymetry(path: Path, file: str | None, force: bool) -> str:
    """Convert a user-supplied GEBCO/NOAA grid, or print manual download instructions."""
    if path.exists() and not force:
        return "skipped (exists)"
    if not file:
        print(f"bathymetry: download a GEBCO grid for the Gulf box (lat {GULF_BOX['lat']}, lon {GULF_BOX['lon']}).\n"
              f"  Official page: {GEBCO_PAGE}\n  Download app (choose a user-defined area, NetCDF or GeoTIFF): {GEBCO_APP}\n"
              f"  Then run: python scripts/fetch_reference_data.py --only bathymetry --bathymetry-file PATH")
        return "manual"
    src = Path(file)
    try:
        lats, lons, elev = _read_netcdf(src) if src.suffix.lower() in (".nc", ".nc4") else _read_geotiff(src)
    except ImportError as exc:
        print(f"bathymetry: missing optional dependency ({exc}). pip install netCDF4")
        return "manual"
    except (OSError, KeyError, ValueError) as exc:
        print(f"bathymetry: could not read {src}: {exc}")
        return "manual"
    _write_bathymetry(lats, lons, elev.astype(np.float32), path)
    return "ok"


def _normalise(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame | None:
    """Rename columns using alias lists; None when a needed column is missing."""
    lower = {c.lower().strip(): c for c in df.columns}
    out = {}
    for target, names in aliases.items():
        match = next((lower[n] for n in names if n in lower), None)
        if match is not None:
            out[target] = df[match]
    return pd.DataFrame(out) if {"lat", "lon"} <= set(out) else None


def _fix_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Western longitudes must be negative."""
    df["lon"] = -df["lon"].abs()
    return df


def step_boem_wells(path: Path, file: str | None, force: bool) -> str:
    """Normalise a BOEM borehole CSV into boem_wells.csv."""
    if path.exists() and not force:
        return "skipped (exists)"
    if not file:
        print(f"boem wells: download the borehole raw data (CSV) from the BOEM Data Center.\n  Borehole query/download page: {BOEM_BOREHOLE_PAGE}\n"
              f"  Data Center home: {BOEM_DATA_CENTER}\n  Then run: python scripts/fetch_reference_data.py --only boem --boem-wells-file PATH")
        return "manual"
    df = _normalise(pd.read_csv(file, low_memory=False), WELL_ALIASES)
    if df is None:
        print("boem wells: no latitude/longitude columns found; columns were:", list(pd.read_csv(file, nrows=0).columns))
        return "manual"
    for col in ("api", "water_depth_ft", "total_depth_ft"):
        if col not in df:
            df[col] = np.nan
    _fix_lon(df)[["api", "lat", "lon", "water_depth_ft", "total_depth_ft"]].dropna(subset=["lat", "lon"]).to_csv(path, index=False)
    return "ok"


def step_boem_blocks(path: Path, file: str | None, force: bool) -> str:
    """Normalise a BOEM lease-block centroid CSV into boem_blocks.csv."""
    if path.exists() and not force:
        return "skipped (exists)"
    if not file or Path(file).suffix.lower() in (".shp", ".zip", ".gdb"):
        print(f"boem blocks: BOEM publishes block grids as GIS shapefiles; open them in QGIS, compute centroids and export a CSV "
              f"(columns for area code, block number, centroid lat/lon).\n  Mapping and data page (see 'Gulf' under GIS Data/Shapefiles): {BOEM_MAPPING_PAGE}\n"
              f"  Then run: python scripts/fetch_reference_data.py --only boem --boem-blocks-file PATH.csv")
        return "manual"
    df = _normalise(pd.read_csv(file, low_memory=False), BLOCK_ALIASES)
    if df is None or not {"area_code", "block"} <= set(df.columns):
        print("boem blocks: need area code, block and lat/lon columns; found:", list(pd.read_csv(file, nrows=0).columns))
        return "manual"
    df["area_code"] = df["area_code"].astype(str).str.strip().str.upper()
    if "area_name" not in df:
        df["area_name"] = df["area_code"].map(AREA_NAMES).fillna(df["area_code"])
    _fix_lon(df)[["area_code", "area_name", "block", "lat", "lon"]].to_csv(path, index=False)
    return "ok"


def main(argv: list[str] | None = None) -> int:
    """Run the selected steps and print a summary table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["macrostrat", "bathymetry", "boem"])
    ap.add_argument("--bathymetry-file")
    ap.add_argument("--boem-wells-file")
    ap.add_argument("--boem-blocks-file")
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = {
        "macrostrat_grid.csv": ("macrostrat", lambda p: step_macrostrat(p, args.force)),
        "bathymetry_gulf.csv": ("bathymetry", lambda p: step_bathymetry(p, args.bathymetry_file, args.force)),
        "boem_wells.csv": ("boem", lambda p: step_boem_wells(p, args.boem_wells_file, args.force)),
        "boem_blocks.csv": ("boem", lambda p: step_boem_blocks(p, args.boem_blocks_file, args.force)),
    }
    summary = {}
    for name, (group, run) in steps.items():
        if args.only and args.only != group:
            continue
        try:
            summary[name] = run(OUT_DIR / name)
        except Exception as exc:  # each step is independent; report and continue
            print(f"warning: {name} failed: {type(exc).__name__}: {exc}")
            summary[name] = "manual (failed)"
    print("\nSummary")
    for name, status in summary.items():
        print(f"  {name:24} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
