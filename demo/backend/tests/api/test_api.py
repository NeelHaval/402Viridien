import io
import json
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline.model import ModelState
from app.services.ingest import compute_scores
from app.store import wells as repo
from app.store.db import Database
from tests import synth

COLUMNS = ["top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth"]
AROUND_THE_WELL = [[-91, 26], [-89, 26], [-89, 29], [-91, 29]]      # contains lon -90, lat 27.5
ELSEWHERE = [[-80, 20], [-79, 20], [-79, 21]]


def upload_fixture(client, tmp_path, name="scan.tif", tweak=0, with_bbox=True):
    path = synth.write_scan(tmp_path / name, tweak)
    files = {"file": (name, path.read_bytes(), "image/tiff")}
    if with_bbox:
        files["bbox_json"] = ("scan.json", json.dumps(synth.bbox_json()).encode(), "application/json")
    return client.post("/upload", files=files)


def wait_ready(client, well_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/wells/{well_id}").json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("well did not finish in time")


def test_end_to_end(client, tmp_path) -> None:
    resp = upload_fixture(client, tmp_path)
    assert resp.status_code == 200 and resp.json()["status"] in ("queued", "processing")
    well_id = resp.json()["id"]

    well = wait_ready(client, well_id)
    assert well["status"] == "ready" and well["located"] and (well["lat"], well["lon"]) == (27.5, -90.0)
    assert set(well["scores"]) == {"drilling", "cost", "price", "co2"}
    assert len(well["column"]) == 2 and well["total_depth_ft"] == 5325.0       # 300 ft logged, pooled in 250 ft bins
    assert set(well["validation"]) == {"confidence", "components", "flags"} and well["error"] is None

    csv = client.get(f"/wells/{well_id}/csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert f'{well_id}.notebook.csv' in csv.headers["content-disposition"]
    table = pd.read_csv(io.StringIO(csv.text))
    assert set(COLUMNS) - {'base_depth'} <= set(table.columns)
    assert 'bottom_depth' in table.columns and len(table) >= 3
    assert table.depth_status.eq('available').all()
    assert table.source_image_sha256.notna().all() and table.result_sha256.notna().all()
    table = table.rename(columns={'bottom_depth': 'base_depth'})
    assert not table[COLUMNS].isna().any().any()
    assert (table.base_depth > table.top_depth).all() and table.top_depth.is_monotonic_increasing

    recomputed = compute_scores(table, well["lat"], well["lon"], []).values       # stored scores == recomputed scores
    assert well["scores"] == recomputed

    listing = client.get("/wells").json()
    assert [w["id"] for w in listing] == [well_id] and listing[0]["scores"] == well["scores"]

    area = client.post("/report", json={"polygon": AROUND_THE_WELL}).json()
    assert [w["id"] for w in area["wells"]] == [well_id]
    assert area["scores"] == well["scores"]                                          # one well: aggregate == the well's scores
    assert area["column"] == well["column"] and "water_depth_unavailable" in area["data_quality"]

    empty = client.post("/report", json={"polygon": ELSEWHERE}).json()
    assert empty == {"wells": [], "scores": None, "column": [], "data_quality": []}


def test_empty_database(client) -> None:
    assert client.get("/wells").json() == []
    assert client.post("/report", json={"polygon": AROUND_THE_WELL}).json()["wells"] == []


def test_routes_also_live_under_api_prefix(client) -> None:
    assert client.get("/api/wells").json() == [] and client.get("/api/health").status_code == 200


def test_status_filter(client, tmp_path) -> None:
    well_id = upload_fixture(client, tmp_path).json()["id"]
    wait_ready(client, well_id)
    assert len(client.get("/wells?status=ready").json()) == 1
    assert client.get("/wells?status=failed").json() == []
    assert client.get("/wells?status=bogus").status_code == 422


def test_unlocated_well_is_listed_but_not_reported(client, tmp_path, stub_layer1) -> None:
    from app.tools.location import Location
    stub_layer1["location"] = Location()
    well = wait_ready(client, upload_fixture(client, tmp_path).json()["id"])
    assert well["status"] == "ready" and well["located"] is False and well["scores"]["cost"] is None
    assert client.get("/wells").json()[0]["located"] is False
    assert client.post("/report", json={"polygon": [[-180, -90], [180, -90], [180, 90], [-180, 90]]}).json()["wells"] == []


def test_failed_well_reports_its_error(client, tmp_path, stub_layer1) -> None:
    stub_layer1["calibration"] = None
    well = wait_ready(client, upload_fixture(client, tmp_path).json()["id"])
    assert well["status"] == "failed" and "depth calibration" in well["error"] and well["scores"] is None and well["column"] == []
    assert client.get(f"/wells/{well['id']}/csv").status_code == 404


def test_concurrent_uploads_queue_and_all_finish(client, tmp_path) -> None:
    ids = [upload_fixture(client, tmp_path, f"s{i}.tif", tweak=i + 1).json()["id"] for i in range(3)]
    assert len(set(ids)) == 3
    assert [wait_ready(client, i)["status"] for i in ids] == ["ready"] * 3


@pytest.mark.parametrize("name,content,status,code", [
    ("fake.tif", b"hello, not an image", 415, "unsupported_type"),
    ("scan.tif", b"%PDF-1.4 rest of file", 415, "pdf_unsupported"),
    ("empty.tif", b"", 400, None),
])
def test_rejected_files(client, name, content, status, code) -> None:
    resp = client.post("/upload", files={"file": (name, content, "image/tiff")})
    assert resp.status_code == status
    body = resp.json()["error"]
    assert body["message"] and (code is None or body["code"] == code)
    assert client.get("/wells").json() == []


def test_oversize_file(settings, stub_ocr_engine, stub_layer1, tmp_path) -> None:
    from dataclasses import replace
    path = synth.write_scan(tmp_path / "big.tif")
    with TestClient(create_app(replace(settings, max_upload_bytes=1000))) as c:
        resp = c.post("/upload", files={"file": ("big.tif", path.read_bytes(), "image/tiff")})
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "too_large"


def test_bad_bbox_json(client, tmp_path) -> None:
    path = synth.write_scan(tmp_path / "s.tif")
    resp = client.post("/upload", files={"file": ("s.tif", path.read_bytes(), "image/tiff"), "bbox_json": ("b.json", b"{nope", "application/json")})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_bbox"


def test_upload_without_file(client) -> None:
    resp = client.post("/upload")
    assert resp.status_code == 422 and "error" in resp.json()


def test_unknown_id_is_404(client) -> None:
    for path in ("/wells/nope", "/wells/nope/csv"):
        resp = client.get(path)
        assert resp.status_code == 404 and resp.json()["error"]["code"] == "unknown_well"


@pytest.mark.parametrize("polygon", [[[0, 0], [1, 1]], [], [[0, 0], [1, 91], [1, 1]], [[0, 0], [181, 1], [1, 1]], [[0, 0], [1, 1], [2]]])
def test_bad_polygon_is_422(client, polygon) -> None:
    resp = client.post("/report", json={"polygon": polygon})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "invalid_request"


def test_non_finite_polygon_is_422(client) -> None:
    resp = client.post("/report", content=b'{"polygon": [[0, 0], [1, NaN], [1, 1]]}', headers={"content-type": "application/json"})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "invalid_request"


def test_health_with_model(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["model_error"] is None and body["db_ok"] is True
    assert set(body["reference_data"]) == {"bathymetry_gulf", "macrostrat_grid", "boem_wells", "transitions"}
    assert body["reference_data"]["bathymetry_gulf"] is False


def test_health_without_model_and_uploads_fail_clearly(settings, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.main.load_model", lambda: ModelState(False, "RuntimeError: weights missing"))
    with TestClient(create_app(settings)) as c:
        body = c.get("/health").json()
        assert body["model_loaded"] is False and "weights missing" in body["model_error"] and body["db_ok"] is True
        resp = upload_fixture(c, tmp_path)
        assert resp.status_code == 503 and resp.json()["error"]["code"] == "model_unavailable"
        assert "weights missing" in resp.json()["error"]["message"]
        assert c.get("/wells").json() == []


def test_restart_marks_interrupted_jobs_failed(settings) -> None:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.migrate()
    for well_id, state in (("stale-processing", "processing"), ("stale-queued", "queued")):
        repo.insert_queued(db, well_id=well_id, file_hash=well_id, filename="x.tif", name="x", scan_path="x")
        if state == "processing":
            repo.mark_processing(db, well_id)
    with TestClient(create_app(settings)) as c:
        for well_id in ("stale-processing", "stale-queued"):
            body = c.get(f"/wells/{well_id}").json()
            assert body["status"] == "failed" and body["error"] == "interrupted by restart"


def test_data_survives_a_restart(settings, stub_ocr_engine, stub_layer1, tmp_path) -> None:
    with TestClient(create_app(settings)) as c:
        well_id = upload_fixture(c, tmp_path).json()["id"]
        before = wait_ready(c, well_id)
    with TestClient(create_app(settings)) as c:
        after = c.get(f"/wells/{well_id}").json()
        assert after == before and c.get(f"/wells/{well_id}/csv").status_code == 200
