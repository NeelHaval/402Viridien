import sqlite3

import pytest

from app.store import wells as repo
from app.store.db import MIGRATIONS, Database


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "t.db")
    database.migrate()
    return database


def add(db, well_id="a", file_hash=None):
    repo.insert_queued(db, well_id=well_id, file_hash=file_hash or well_id, filename=f"{well_id}.tif", name=well_id, scan_path="p")


def test_migrate_is_idempotent_and_sets_the_version(db) -> None:
    db.migrate()
    with db.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert db.ok()


def test_table_has_the_documented_columns(db) -> None:
    with db.connect() as conn:
        names = {r["name"] for r in conn.execute("PRAGMA table_info(wells)")}
    assert {"id", "name", "lat", "lon", "block", "status", "error", "total_depth_ft", "created_at", "finished_at", "source_filename",
            "scan_path", "csv_path", "validation_confidence", "validation_json", "scores_json"} <= names


def test_file_hash_is_unique(db) -> None:
    add(db, "a", "same")
    with pytest.raises(sqlite3.IntegrityError):
        add(db, "b", "same")


def test_lifecycle_and_json_decoding(db) -> None:
    add(db)
    assert repo.get(db, "a")["status"] == "queued"
    repo.mark_processing(db, "a")
    repo.mark_ready(db, "a", name="W", lat=1.0, lon=2.0, block="B", total_depth_ft=10.0, csv_path="c.csv", validation_confidence=0.5,
                    validation={"flags": ["x"]}, scores={"values": {"drilling": 1}, "logged_ft": 5, "flags": []}, warnings=["w"], timings={"total": 1.0})
    row = repo.get(db, "a")
    assert row["status"] == "ready" and row["scores"]["values"] == {"drilling": 1} and row["warnings"] == ["w"] and row["finished_at"]
    assert [w["id"] for w in repo.list_ready_located(db)] == ["a"]
    repo.mark_failed(db, "a", "boom")
    row = repo.get(db, "a")
    assert row["status"] == "failed" and row["error"] == "boom" and row["scores"] is None and repo.list_ready_located(db) == []


def test_unlocated_ready_well_is_not_in_the_located_list(db) -> None:
    add(db)
    repo.mark_ready(db, "a", name="W", lat=None, lon=None, block=None, total_depth_ft=1.0, csv_path="c", validation_confidence=1.0,
                    validation={}, scores={"values": {}, "logged_ft": 1, "flags": []}, warnings=[], timings={})
    assert repo.list_ready_located(db) == [] and len(repo.list_wells(db, "ready")) == 1


def test_fail_interrupted_only_touches_unfinished_wells(db) -> None:
    for i in "abc":
        add(db, i)
    repo.mark_processing(db, "b")
    repo.mark_failed(db, "c", "old error")
    assert repo.fail_interrupted(db) == 2
    assert [repo.get(db, i)["error"] for i in "abc"] == ["interrupted by restart", "interrupted by restart", "old error"]


def test_unknown_ids(db) -> None:
    assert repo.get(db, "nope") is None and repo.get_by_hash(db, "nope") is None
