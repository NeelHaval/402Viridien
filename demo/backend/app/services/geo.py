"""Geometry helpers: point-in-polygon and great-circle distance."""
from __future__ import annotations

import math

EARTH_KM = 6371.0088
EDGE_TOLERANCE = 1e-10


def point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray casting over (lon, lat) vertices; points on an edge or vertex count as inside."""
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < EDGE_TOLERANCE and min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points in degrees."""
    p, q = math.radians(lat1), math.radians(lat2)
    d = math.radians(lon2 - lon1)
    h = math.sin((q - p) / 2) ** 2 + math.cos(p) * math.cos(q) * math.sin(d / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(min(1.0, h)))
