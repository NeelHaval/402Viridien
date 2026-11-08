"""Repository functions for the wells table. Rows come back as plain dicts with JSON columns decoded."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

from app.store.db import Database

JSON_COLUMNS = {"validation_json": "validation", "scores_json": "scores", "warnings_json": "warnings", "timings_json": "timings"}


def _finite(value, name, minimum=None, maximum=None):
    try:
        if isinstance(value,bool) or not math.isfinite(value):raise ValueError()
        if minimum is not None and value<minimum or maximum is not None and value>maximum:raise ValueError()
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(f"{name} must be finite and within its allowed range") from exc


def _scores_json(scores):
    if not isinstance(scores,dict) or not isinstance(scores.get("values"),dict):raise ValueError("scores must contain a values object")
    for name in ("drilling","cost","price","co2"):
        value=scores["values"].get(name)
        if value is not None and (type(value) is not int or not 0<=value<=100):raise ValueError(f"score {name} must be an integer from 0 to 100 or null")
    _finite(scores.get("logged_ft"),"logged_ft",minimum=0)
    return json.dumps(scores,allow_nan=False)


def now() -> str:
    """UTC timestamp, ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decode(row: sqlite3.Row | None) -> dict | None:
    """Turn a row into a dict and decode its JSON columns."""
    if row is None:
        return None
    out = dict(row)
    for column, key in JSON_COLUMNS.items():
        raw = out.pop(column)
        out[key] = json.loads(raw) if raw else None
    return out


def insert_queued(db: Database, *, well_id: str, file_hash: str, filename: str, name: str, scan_path: str) -> None:
    """Add a new well in the `queued` state."""
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO wells (id, name, status, created_at, source_filename, scan_path, file_hash) VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (well_id, name, now(), filename, scan_path, file_hash),
        )


def get(db: Database, well_id: str) -> dict | None:
    """One well, or None."""
    with db.connect() as conn:
        return _decode(conn.execute("SELECT * FROM wells WHERE id = ?", (well_id,)).fetchone())


def get_by_hash(db: Database, file_hash: str) -> dict | None:
    """The well ingested from a file with this hash, or None."""
    with db.connect() as conn:
        return _decode(conn.execute("SELECT * FROM wells WHERE file_hash = ?", (file_hash,)).fetchone())


def list_wells(db: Database, status: str | None = None) -> list[dict]:
    """All wells (optionally one status), oldest first."""
    with db.connect() as conn:
        sql, args = ("SELECT * FROM wells WHERE status = ? ORDER BY created_at, id", (status,)) if status else \
            ("SELECT * FROM wells ORDER BY created_at, id", ())
        return [_decode(r) for r in conn.execute(sql, args)]


def list_ready_located(db: Database) -> list[dict]:
    """Ready wells that have a position."""
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM wells WHERE status = 'ready' AND lat IS NOT NULL AND lon IS NOT NULL ORDER BY created_at, id")
        return [_decode(r) for r in rows]


def requeue(db: Database, well_id: str, scan_path: str, filename: str) -> None:
    """Reset a failed well so it can run again."""
    with db.connect() as conn:
        conn.execute(
            "UPDATE wells SET status='queued', error=NULL, finished_at=NULL, scan_path=?, source_filename=?, created_at=? WHERE id=?",
            (scan_path, filename, now(), well_id),
        )


def mark_processing(db: Database, well_id: str) -> None:
    """Move a queued well to `processing`."""
    with db.connect() as conn:
        conn.execute("UPDATE wells SET status='processing' WHERE id=?", (well_id,))


def mark_failed(db: Database, well_id: str, error: str, timings: dict | None = None) -> None:
    """Record a failure."""
    try:stored_timings=json.dumps(timings,allow_nan=False) if timings else None
    except (TypeError,ValueError,OverflowError,RecursionError):stored_timings=None
    with db.connect() as conn:
        conn.execute(
            "UPDATE wells SET status='failed', error=?, finished_at=?, timings_json=?, scores_json=NULL WHERE id=?",
            (error, now(), stored_timings, well_id),
        )


def mark_ready(db: Database, well_id: str, *, name: str, lat: float | None, lon: float | None, block: str | None,
               total_depth_ft: float, csv_path: str, validation_confidence: float, validation: dict, scores: dict,
               warnings: list[str], timings: dict) -> None:
    """Store a finished well."""
    _finite(total_depth_ft,"total_depth_ft")
    _finite(validation_confidence,"validation_confidence",minimum=0,maximum=1)
    if lat is not None:_finite(lat,"lat",minimum=-90,maximum=90)
    if lon is not None:_finite(lon,"lon",minimum=-180,maximum=180)
    stored_scores=_scores_json(scores)
    stored_validation=json.dumps(validation,allow_nan=False)
    stored_warnings=json.dumps(warnings,allow_nan=False)
    stored_timings=json.dumps(timings,allow_nan=False)
    with db.connect() as conn:
        conn.execute(
            """UPDATE wells SET status='ready', error=NULL, name=?, lat=?, lon=?, block=?, total_depth_ft=?, finished_at=?,
               csv_path=?, validation_confidence=?, validation_json=?, scores_json=?, warnings_json=?, timings_json=? WHERE id=?""",
            (name, lat, lon, block, total_depth_ft, now(), csv_path, validation_confidence, stored_validation,
             stored_scores, stored_warnings, stored_timings, well_id),
        )


def set_scores(db: Database, well_id: str, scores: dict) -> None:
    """Replace the stored scores of a well."""
    with db.connect() as conn:
        conn.execute("UPDATE wells SET scores_json=? WHERE id=?", (_scores_json(scores), well_id))


def fail_interrupted(db: Database) -> int:
    """Mark wells left `queued` or `processing` by a previous run as failed; returns how many."""
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE wells SET status='failed', error='interrupted by restart', finished_at=? WHERE status IN ('queued', 'processing')",
            (now(),),
        )
        return cur.rowcount
