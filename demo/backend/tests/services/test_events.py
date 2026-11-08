"""The SSE stream: an upload produces events in order; a client that connects late reads current state from GET /wells."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.services.events import Broadcaster
from app.services.ingest import IngestService
from app.api.events import frame
from app.pipeline.model import ModelState
from app.store.db import Database
from app.config import Settings
from tests import synth


async def collect(queue, until: str, timeout=20.0) -> list[dict]:
    seen: list[dict] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        event = await asyncio.wait_for(queue.get(), max(0.1, deadline - time.time()))
        seen.append(event)
        if event["type"] == until:
            return seen
    raise AssertionError(f"never saw {until}: {seen}")


def test_frame_is_valid_sse() -> None:
    text = frame({"type": "well_located", "id": "a", "lat": 1})
    assert text.startswith("event: well_located\ndata: ") and text.endswith("\n\n")
    assert json.loads(text.split("data: ")[1])["lat"] == 1


def test_upload_emits_events_in_order(tmp_path, stub_ocr_engine, stub_layer1) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.migrate()

    async def scenario() -> list[dict]:
        events = Broadcaster()
        queue = events.subscribe()
        svc = IngestService(settings, db, lambda: ModelState(True, None, "scripted"), events)
        try:
            await asyncio.to_thread(svc.submit, synth.write_scan(tmp_path / "a.tif"), "a.tif", synth.bbox_json())
            return await collect(queue, "well_scored")
        finally:
            await asyncio.to_thread(svc.shutdown)

    seen = asyncio.run(scenario())
    types = [e["type"] for e in seen]
    assert types[0] == "well_created"
    assert types.index("well_located") < types.index("well_ready") < types.index("well_scored")
    located = next(e for e in seen if e["type"] == "well_located")
    assert (located["lat"], located["lon"]) == (27.5, -90.0)
    assert set(seen[-1]["scores"]) == {"drilling", "cost", "price", "co2"}
    assert len({e["id"] for e in seen}) == 1


def test_failed_job_emits_well_failed(tmp_path, stub_ocr_engine, stub_layer1) -> None:
    stub_layer1["calibration"] = None
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.migrate()

    async def scenario() -> list[dict]:
        events = Broadcaster()
        queue = events.subscribe()
        svc = IngestService(settings, db, lambda: ModelState(True, None, "scripted"), events)
        try:
            await asyncio.to_thread(svc.submit, synth.write_scan(tmp_path / "a.tif"), "a.tif", synth.bbox_json())
            return await collect(queue, "well_failed")
        finally:
            await asyncio.to_thread(svc.shutdown)

    seen = asyncio.run(scenario())
    assert "depth calibration failed" in seen[-1]["error"]


def test_late_client_gets_current_state_from_the_list(client, tmp_path) -> None:
    from tests.api.test_api import upload_fixture, wait_ready
    well = wait_ready(client, upload_fixture(client, tmp_path).json()["id"])
    # nothing is replayed to a client that connects now; the list endpoint is the source of current state
    listed = client.get("/wells").json()
    assert [w["id"] for w in listed] == [well["id"]]
    assert listed[0]["status"] == "ready" and listed[0]["located"] is True and listed[0]["scores"]["cost"] is not None


def test_duplicate_upload_returns_stored_result_without_processing(client, tmp_path) -> None:
    from tests.api.test_api import upload_fixture, wait_ready
    first = upload_fixture(client, tmp_path).json()
    done = wait_ready(client, first["id"])
    jobs = []
    client.app.state.ingest.jobs.enqueue = lambda wid: jobs.append(wid)
    again = upload_fixture(client, tmp_path).json()
    assert again["id"] == first["id"] and again["status"] == "ready"
    assert jobs == []                                                       # no new job
    after = client.get(f"/wells/{first['id']}").json()
    assert (after["lat"], after["lon"], after["scores"]) == (done["lat"], done["lon"], done["scores"])
