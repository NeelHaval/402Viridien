"""SQLite access with stdlib sqlite3. Schema changes are plain SQL in MIGRATIONS, applied in order."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

MIGRATIONS = [
    """
    CREATE TABLE wells (
        id TEXT PRIMARY KEY,
        name TEXT,
        lat REAL,
        lon REAL,
        block TEXT,
        status TEXT NOT NULL,
        error TEXT,
        total_depth_ft REAL,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        source_filename TEXT,
        scan_path TEXT,
        csv_path TEXT,
        validation_confidence REAL,
        validation_json TEXT,
        scores_json TEXT,
        file_hash TEXT NOT NULL,
        warnings_json TEXT,
        timings_json TEXT
    );
    CREATE UNIQUE INDEX wells_file_hash ON wells (file_hash);
    CREATE INDEX wells_status ON wells (status);
    """,
    """
    ALTER TABLE wells ADD COLUMN location_source TEXT;
    ALTER TABLE wells ADD COLUMN located_at TEXT;
    ALTER TABLE wells ADD COLUMN api_number TEXT;
    """,
]


class Database:
    """A SQLite file; every call opens its own short-lived connection, so threads never share one."""

    def __init__(self, path: Path) -> None:
        """Remember the file location."""
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with row access by name; commit on success, roll back on error."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        """Create the file and apply the migrations that have not run yet."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for number, sql in enumerate(MIGRATIONS[version:], start=version + 1):
                conn.executescript(sql)
                conn.execute(f"PRAGMA user_version = {number}")

    def ok(self) -> bool:
        """True when the database answers a trivial query."""
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1 FROM wells LIMIT 1")
            return True
        except sqlite3.Error:
            return False
