"""Repository functions for the wells table. Rows come back as plain dicts with JSON columns decoded."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app.store.db import Database

JSON_COLUMNS = {"validation_json": "validation", "scores_json": "scores", "warnings_json": "warnings", "timings_json": "timings"}


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
        sql, args = ("SELECT * FROM wells WHERE archived_at IS NULL AND status = ? ORDER BY created_at, id", (status,)) if status else \
            ("SELECT * FROM wells WHERE archived_at IS NULL ORDER BY created_at, id", ())
        return [_decode(r) for r in conn.execute(sql, args)]


def list_ready_located(db: Database) -> list[dict]:
    """Ready wells that have a position."""
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM wells WHERE archived_at IS NULL AND status = 'ready' AND lat IS NOT NULL AND lon IS NOT NULL ORDER BY created_at, id")
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
    with db.connect() as conn:
        conn.execute(
            "UPDATE wells SET status='failed', error=?, finished_at=?, timings_json=?, scores_json=NULL WHERE id=?",
            (error, now(), json.dumps(timings) if timings else None, well_id),
        )


def mark_ready(db: Database, well_id: str, *, name: str, lat: float | None, lon: float | None, block: str | None,
               total_depth_ft: float, csv_path: str, validation_confidence: float, validation: dict, scores: dict,
               warnings: list[str], timings: dict, location_source: str | None = None, api_number: str | None = None) -> None:
    """Store a finished well. `located_at` keeps the moment the position was first stored."""
    with db.connect() as conn:
        conn.execute(
            """UPDATE wells SET status='ready', error=NULL, name=?, lat=?, lon=?, block=?, total_depth_ft=?, finished_at=?,
               csv_path=?, validation_confidence=?, validation_json=?, scores_json=?, warnings_json=?, timings_json=?,
               location_source=COALESCE(?, location_source), api_number=COALESCE(?, api_number),
               located_at=CASE WHEN ? IS NOT NULL AND located_at IS NULL THEN ? ELSE located_at END WHERE id=?""",
            (name, lat, lon, block, total_depth_ft, now(), csv_path, validation_confidence, json.dumps(validation),
             json.dumps(scores), json.dumps(warnings), json.dumps(timings), location_source, api_number, lat, now(), well_id),
        )


def set_location(db: Database, well_id: str, *, lat: float, lon: float, source: str, block: str | None = None,
                 api_number: str | None = None, name: str | None = None) -> None:
    """Store a found position at once (the job may still be running); `located_at` is the moment it was found."""
    with db.connect() as conn:
        conn.execute("UPDATE wells SET lat=?, lon=?, location_source=?, located_at=?, block=COALESCE(?, block), "
                     "api_number=COALESCE(?, api_number), name=COALESCE(?, name) WHERE id=?",
                     (lat, lon, source, now(), block, api_number, name, well_id))


def set_timings(db: Database, well_id: str, timings: dict) -> None:
    """Replace the stored timings (flat totals plus the span list) of a well."""
    with db.connect() as conn:
        conn.execute("UPDATE wells SET timings_json=? WHERE id=?", (json.dumps(timings), well_id))


def set_scores(db: Database, well_id: str, scores: dict) -> None:
    """Replace the stored scores of a well."""
    with db.connect() as conn:
        conn.execute("UPDATE wells SET scores_json=? WHERE id=?", (json.dumps(scores), well_id))


def fail_interrupted(db: Database) -> int:
    """Mark wells left `queued` or `processing` by a previous run as failed; returns how many."""
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE wells SET status='failed', error='interrupted by restart', finished_at=? WHERE status IN ('queued', 'processing')",
            (now(),),
        )
        return cur.rowcount
