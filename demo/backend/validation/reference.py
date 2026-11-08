"""Lazy, cached loaders for reference data. Every loader returns None when its file is missing."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from validation.config import BATHYMETRY_MAX_GAP_DEG

EARTH_RADIUS_KM = 6371.0088
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"
_state = {"dir": Path(os.environ.get("VALIDATION_REFERENCE_DIR", _DEFAULT_DIR))}


def set_reference_dir(path: str | Path | None) -> None:
    """Point the loaders at another folder (or back to the default) and clear all caches."""
    _state["dir"] = Path(path) if path else _DEFAULT_DIR
    for fn in (load_macrostrat_grid, load_boem_wells, load_bathymetry, load_transitions, _bathymetry_grid):
        fn.cache_clear()


def reference_dir() -> Path:
    """Return the folder currently used for reference files."""
    return _state["dir"]


def _read_csv(name: str) -> pd.DataFrame | None:
    """Read one reference CSV and add radian columns, or None if absent or unreadable."""
    path = reference_dir() / name
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if df.empty or not {"lat", "lon"} <= set(df.columns):
        return None
    df["lat_rad"], df["lon_rad"] = np.radians(df["lat"].to_numpy(float)), np.radians(df["lon"].to_numpy(float))
    return df


@lru_cache(maxsize=1)
def load_macrostrat_grid() -> pd.DataFrame | None:
    """Macrostrat grid: lat, lon, one share column per coarse class, n_units."""
    return _read_csv("macrostrat_grid.csv")


@lru_cache(maxsize=1)
def load_boem_wells() -> pd.DataFrame | None:
    """BOEM wells: api, lat, lon, water_depth_ft, total_depth_ft."""
    return _read_csv("boem_wells.csv")


@lru_cache(maxsize=1)
def load_bathymetry() -> pd.DataFrame | None:
    """Bathymetry points: lat, lon, depth_m (positive under water)."""
    return _read_csv("bathymetry_gulf.csv")


@lru_cache(maxsize=1)
def load_transitions() -> dict | None:
    """Coarse-class transition data: {'classes': [...], 'counts': {a: {b: n}}, 'probs': {a: {b: p}}}."""
    path = reference_dir() / "transitions.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and ("counts" in data or "probs" in data) else None


def haversine_km(lat_rad: float, lon_rad: float, lats_rad: np.ndarray, lons_rad: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to arrays of points (inputs in radians)."""
    dlat, dlon = lats_rad - lat_rad, lons_rad - lon_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distances_to(df: pd.DataFrame, lat: float, lon: float) -> np.ndarray:
    """Distances in km from (lat, lon) in degrees to every row of a loaded reference table."""
    return haversine_km(np.radians(lat), np.radians(lon), df["lat_rad"].to_numpy(), df["lon_rad"].to_numpy())


@lru_cache(maxsize=1)
def _bathymetry_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pivot the bathymetry points into a (lats, lons, depth[lat, lon]) grid for O(1) lookups."""
    df = load_bathymetry()
    if df is None or "depth_m" not in df.columns:
        return None
    lats, li = np.unique(df["lat"].to_numpy(float), return_inverse=True)
    lons, lj = np.unique(df["lon"].to_numpy(float), return_inverse=True)
    grid = np.full((lats.size, lons.size), np.nan)
    grid[li, lj] = df["depth_m"].to_numpy(float)
    return lats, lons, grid


def _nearest_index(axis: np.ndarray, value: float) -> int:
    """Index of the axis value closest to `value`."""
    i = int(np.searchsorted(axis, value))
    if i == 0:
        return 0
    if i == axis.size:
        return axis.size - 1
    return i if axis[i] - value < value - axis[i - 1] else i - 1


def bathymetry_at(lat: float, lon: float) -> float | None:
    """Water depth in metres at a point (positive = under water), or None when unknown."""
    packed = _bathymetry_grid()
    if packed is None:
        return None
    lats, lons, grid = packed
    i, j = _nearest_index(lats, lat), _nearest_index(lons, lon)
    if abs(lats[i] - lat) > BATHYMETRY_MAX_GAP_DEG or abs(lons[j] - lon) > BATHYMETRY_MAX_GAP_DEG:
        return None
    value = grid[i, j]
    return None if np.isnan(value) else float(value)
