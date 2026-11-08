"""Demo API boundary regressions using owned bytes and an in-process ASGI app.

No OCR engines, network clients, external services or real credentials are used.
Run with the demo backend environment, which supplies FastAPI and its dependencies.
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "demo" / "backend"
sys.path.insert(0, str(BACKEND))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api import deps, errors, report, upload
from app.config import Settings
from app.tools.errors import InputConflict, QueueBusy


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "owned-data", max_upload_bytes=32)
    settings.ensure_dirs()
    submitted = []

    def submit(path, filename, bbox):
        submitted.append((filename, bbox))
        path.unlink()
        return SimpleNamespace(id="owned-fixture", status="queued")

    service = SimpleNamespace(submit=submit)
    app = FastAPI()
    errors.install(app, settings)
    app.include_router(upload.router)
    app.include_router(upload.router, prefix="/api")
    app.include_router(report.router)
    app.dependency_overrides[deps.settings] = lambda: settings
    app.dependency_overrides[deps.ingest] = lambda: service
    app.dependency_overrides[deps.db] = lambda: None
    monkeypatch.setattr(report.repo, "list_ready_located", lambda db: [])
    return app, service, submitted, settings


@pytest.mark.parametrize("raw", [
    b'[]', b'null', b'{broken', b'{"bounding_boxes": []}',
    b'{"a":' + b'[' * 5000 + b'0' + b']' * 5000 + b'}',
    b'{"metadata": NaN}', b'{"metadata": Infinity}',
    b'{"interpreted_lithology_column": [0, 0, true, 8]}',
    b'{"interpreted_lithology_column": [0, 0, 1.5, 8]}',
    b'{"interpreted_lithology_column": [0, 0, 4]}',
    b'{"interpreted_lithology_column": [4, 0, 0, 8]}',
    b'{"depth_column": [-1, 0, 4, 8]}',
])
def test_invalid_bbox_is_400_before_submission_or_scan_write(api, raw):
    app, _, submitted, settings = api
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/upload", files={"file": ("owned.png", b"owned"),
                                                "bbox_json": ("owned.json", raw, "application/json")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_bbox"
    assert not submitted and list(settings.scans_dir.iterdir()) == []


def test_valid_bbox_is_preserved_and_queued(api):
    app, _, submitted, _ = api
    bbox = {"width": 20, "height": 30, "bounding_boxes": {"interpreted_lithology_column": [1, 2, 10, 25]}}
    with TestClient(app) as client:
        response = client.post("/api/upload", files={"file": ("owned.png", b"owned"),
                            "bbox_json": ("owned.json", json.dumps(bbox).encode(), "application/json")})
    assert response.status_code == 200
    assert submitted == [("owned.png", bbox)]


def test_input_conflict_is_409_and_incoming_file_is_removed(api):
    app, service, submitted, settings = api

    def conflict(*args):
        raise InputConflict("This scan already has different bounding-box inputs")

    service.submit = conflict
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("owned.png", b"owned")})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "input_conflict"
    assert not submitted and list(settings.scans_dir.iterdir()) == []


def test_busy_queue_is_503_and_incoming_file_is_removed(api):
    app, service, _, settings = api

    def busy(*args):
        raise QueueBusy("Processing queue is full; try again later")

    service.submit = busy
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("owned.png", b"owned")})
    assert response.status_code == 503 and response.json()["error"]["code"] == "queue_busy"
    assert list(settings.scans_dir.iterdir()) == []


def test_client_suffix_never_becomes_a_filesystem_path(api):
    app, service, _, settings = api
    observed = []

    def submit(path, filename, bbox):
        observed.append((path.name, filename))
        assert path.parent == settings.scans_dir
        assert path.name.startswith(".incoming-") and ":" not in path.name
        path.unlink()
        return SimpleNamespace(id="owned-fixture", status="queued")

    service.submit = submit
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("owned.png:CANARY_STREAM", b"owned")})
    assert response.status_code == 200 and observed[0][1] == "owned.png:CANARY_STREAM"
    assert list(settings.scans_dir.iterdir()) == []


def test_file_budget_remains_stricter_than_total_multipart_budget(api):
    with TestClient(api[0]) as client:
        response = client.post("/upload", files={"file": ("owned.png", b"X" * 33)})
    assert response.status_code == 413 and response.json()["error"]["code"] == "too_large"
    assert api[2] == [] and list(api[3].scans_dir.iterdir()) == []


def test_partial_incoming_file_is_removed_on_read_failure(tmp_path):
    class FailingStream:
        reads = 0

        def read(self, size):
            self.reads += 1
            if self.reads == 1:
                return b"OWNED_PARTIAL_DATA"
            raise OSError("Generated read failure; no real disk exhaustion")

    target = tmp_path / "owned-incoming"
    with pytest.raises(OSError, match="Generated read failure"):
        upload._save(SimpleNamespace(file=FailingStream()), target, 100)
    assert not target.exists()


def test_incoming_name_collision_preserves_existing_bytes(tmp_path):
    target = tmp_path / "owned-existing"
    target.write_bytes(b"OWNED_KEEP_CANARY")
    with pytest.raises(FileExistsError):
        upload._save(SimpleNamespace(file=io.BytesIO(b"replacement")), target, 100)
    assert target.read_bytes() == b"OWNED_KEEP_CANARY"


def test_real_app_body_limit_keeps_cors_error_visible(api, monkeypatch):
    from app.main import create_app
    from app.pipeline.model import ModelState
    monkeypatch.setattr("app.main.load_model", lambda: ModelState(False, "No OCR used in this boundary test"))
    with TestClient(create_app(api[3])) as client:
        response = client.post("/api/upload", content=b"owned", headers={
            "Content-Length": "2000000", "Origin": "http://localhost:5173"})
    assert response.status_code == 413 and response.json()["error"]["code"] == "too_large"
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.mark.parametrize("polygon", [
    [[0, 0], [0, 0], [0, 0]],
    [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, 0], [0, 1]] * 3334,
])
def test_degenerate_or_overbudget_polygons_are_rejected(api, polygon):
    with TestClient(api[0]) as client:
        response = client.post("/report", json={"polygon": polygon})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("polygon", [
    [[-91, 26], [-89, 26], [-89, 29], [-91, 29]],
    [[-180, -90], [180, -90], [180, 90], [-180, 90]],
    [[179, 1], [-179, 1], [-179, 2], [179, 2]],
    [[1, 1], [1 + 1e-9, 1], [1, 1 + 1e-9]],
])
def test_valid_small_world_and_dateline_polygons_are_not_false_rejected(api, polygon):
    with TestClient(api[0]) as client:
        response = client.post("/report", json={"polygon": polygon})
    assert response.status_code == 200
    assert response.json()["wells"] == []


def _asgi_exchange(app, chunks, headers=(), path="/upload"):
    messages, reads = [], []
    iterator = iter(chunks)

    async def receive():
        try:
            body = next(iterator)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        reads.append(len(body))
        return {"type": "http.request", "body": body, "more_body": True}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": list(headers), "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
    asyncio.run(app(scope, receive, send))
    starts = [m for m in messages if m["type"] == "http.response.start"]
    assert len(starts) == 1, messages
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return starts[0]["status"], json.loads(payload), reads


def test_advertised_oversize_is_rejected_before_any_body_read(api):
    status, payload, reads = _asgi_exchange(api[0], [b"NEVER_READ"], [(b"content-length", b"2000000")])
    assert status == 413 and payload["error"]["code"] == "too_large"
    assert reads == [] and api[2] == []


@pytest.mark.parametrize("header", [b"9" * 5000, b"-1", b"not-a-length"])
def test_untrusted_length_headers_never_crash_or_consume_body(api, header):
    status, payload, reads = _asgi_exchange(api[0], [b"NEVER_READ"], [(b"content-length", header)])
    assert status in (400, 413) and payload["error"]["code"] in ("invalid_request", "too_large")
    assert not reads


@pytest.mark.parametrize("extra_headers", [[], [(b"content-length", b"1")]], ids=["no-length", "understated-length"])
def test_streamed_multipart_is_bounded_and_partial_spools_are_closed(api, monkeypatch, extra_headers):
    import starlette.formparsers as forms
    opened = []
    original = forms.SpooledTemporaryFile

    def tracked(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(forms, "SpooledTemporaryFile", tracked)
    head = b'--OWNED\r\nContent-Disposition: form-data; name="file"; filename="owned.png"\r\nContent-Type: image/png\r\n\r\n'
    # A bounded 1.25MiB fixture crosses the configured request limit by chunks;
    # no declared Content-Length and no external HTTP socket are involved.
    chunks = [head, *([b"X" * 65536] * 20), b'\r\n--OWNED--\r\n']
    status, payload, reads = _asgi_exchange(api[0], chunks, [(b"content-type", b"multipart/form-data; boundary=OWNED"), *extra_headers])
    limit = api[3].max_upload_bytes + errors.JSON_BODY_LIMIT + errors.MULTIPART_OVERHEAD
    assert status == 413 and payload["error"]["code"] == "too_large"
    assert sum(reads) <= limit + 65536 and len(reads) < len(chunks)
    assert opened and all(file.closed for file in opened)
    assert api[2] == [] and list(api[3].scans_dir.iterdir()) == []


def test_streamed_json_uses_the_separate_one_megabyte_budget(api):
    status, payload, reads = _asgi_exchange(api[0], [b" " * 65536] * 20,
                                          [(b"content-type", b"application/json")], path="/report")
    assert status == 413 and payload["error"]["code"] == "too_large"
    assert sum(reads) <= errors.JSON_BODY_LIMIT + 65536 and len(reads) < 20
