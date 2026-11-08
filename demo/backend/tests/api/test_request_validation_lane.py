"""Bounded request validation checks with generated fixtures only."""
import json

import pytest

from tests import synth


@pytest.fixture(autouse=True)
def no_background_jobs(monkeypatch):
    """Exercise real request and file validation without running OCR jobs."""
    monkeypatch.setattr("app.services.jobs.JobQueue.enqueue", lambda self, well_id: None)


@pytest.mark.parametrize("bbox", [
    {"bounding_boxes": []},
    {"depth_column": [1, 2, 3]},
    {"depth_column": [1, 2, 3, 4, 5]},
    {"depth_column": [True, 2, 3, 4]},
    {"depth_column": ["1", 2, 3, 4]},
    {"depth_column": [float("nan"), 2, 3, 4]},
    {"depth_column": [0, 0, float("inf"), 4]},
    {"depth_column": [3, 2, 1, 4]},
    {"depth_column": [-1, 0, 5, 5]},
    {"depth_column": [0, 0, 601, 5]},
    {"depth_column": [0, 0, 5, 1701]},
    {"depth_column": [0.1, 0, 0.9, 5]},
])
def test_invalid_box_rejected_before_queue(client, tmp_path, bbox):
    scan = synth.write_scan(tmp_path / "scan.tif")
    response = client.post("/upload", files={"file": ("scan.tif", scan.read_bytes(), "image/tiff"),
        "bbox_json": ("bbox.json", json.dumps(bbox).encode(), "application/json")})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_bbox"
    assert client.get("/wells").json() == []


@pytest.mark.parametrize("bbox", [
    {"bounding_boxes": {"depth_column": [20, 100, 90, 1300]}, "width": 600, "label": "ok"},
    {"depth_column": [0, 0, 600, 1700], "note": {"arbitrary": True}},
    {"bounding_boxes": {}, "extra": ["annotation"]},
])
def test_valid_box_and_metadata_accepted(client, tmp_path, bbox):
    scan = synth.write_scan(tmp_path / "scan.tif")
    response = client.post("/upload", files={"file": ("scan.tif", scan.read_bytes(), "image/tiff"),
        "bbox_json": ("bbox.json", json.dumps(bbox).encode(), "application/json")})
    assert response.status_code == 200, response.text


def test_report_vertex_limit_and_closed_polygon(client):
    closed = [[0, 0], [1, 0], [1, 1], [0, 0]]
    assert client.post("/report", json={"polygon": closed}).status_code == 200
    response = client.post("/report", json={"polygon": [[0, 0]] * 10001})
    assert response.status_code == 422
    assert "10000" in response.json()["error"]["message"]


@pytest.mark.parametrize("name,content,status,code", [
    ("empty.tif", b"", 400, "empty_file"),
    ("bad.tif", b"II*\x00broken", 400, "no_pages"),
    ("bad.png", b"\x89PNG\r\n\x1a\ninvalid", 400, "unreadable"),
    ("bad.jpg", b"\xff\xd8\xffinvalid", 400, "unreadable"),
    ("fake.tif", b"not an image", 415, "unsupported_type"),
    ("doc.pdf", b"%PDF-1.4", 415, "pdf_unsupported"),
])
def test_invalid_image_signatures(client, name, content, status, code):
    response = client.post("/upload", files={"file": (name, content, "application/octet-stream")})
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert client.get("/wells").json() == []
