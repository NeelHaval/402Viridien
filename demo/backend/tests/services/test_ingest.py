import json
import time

import pandas as pd
import pytest

from app.config import Settings
from app.pipeline.model import ModelState
from app.services.ingest import IngestService, ModelUnavailable
from app.store import wells as repo
from app.store.db import Database
from app.tools.errors import UnsupportedFile
from tests import synth


@pytest.fixture
def service(tmp_path, stub_ocr_engine, stub_layer1):
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.migrate()
    svc = IngestService(settings, db, lambda: ModelState(True, None, "scripted"))
    yield svc
    svc.shutdown()


def wait(service, well_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = repo.get(service.db, well_id)
        if row["status"] in ("ready", "failed"):
            return row
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def submit(service, tmp_path, name="a.tif", tweak=0):
    return service.submit(synth.write_scan(tmp_path / name, tweak), name, synth.bbox_json())


def test_success_path_stores_everything(service, tmp_path) -> None:
    sub = submit(service, tmp_path)
    assert sub.created and sub.status == "queued" and len(sub.id) == 16
    row = wait(service, sub.id)
    assert row["status"] == "ready" and row["error"] is None
    assert (row["name"], row["lat"], row["lon"], row["block"]) == ("Test Well", 27.5, -90.0, "GC 100")
    assert row["total_depth_ft"] == pytest.approx(5000 + 0.25 * (synth.LITH_BOX[1] + 1200), abs=1)
    assert set(row["scores"]["values"]) == {"drilling", "cost", "price", "co2"}
    assert row["scores"]["logged_ft"] > 0 and "water_depth_unavailable" in row["scores"]["flags"]
    assert row["timings"]["total"] > 0 and "pipeline.naming" in row["timings"] and "validation" in row["timings"]
    assert list(pd.read_csv(row["csv_path"]).columns) == ["top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth"]
    assert (service.settings.scans_dir / f"{sub.id}.tif").is_file()


def test_pipeline_exception_marks_failed_with_the_message(service, tmp_path, stub_layer1) -> None:
    stub_layer1["calibration"] = None
    row = wait(service, submit(service, tmp_path).id)
    assert row["status"] == "failed" and "depth calibration failed" in row["error"]
    assert row["scores"] is None and row["finished_at"]


def test_unexpected_exception_is_contained(service, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.ingest.validate", lambda *a, **k: 1 / 0)
    row = wait(service, submit(service, tmp_path).id)
    assert row["status"] == "failed" and "division by zero" in row["error"]
    ok = wait(service, submit(service, tmp_path, "b.tif", tweak=2).id)       # the worker is still alive
    assert ok["status"] == "failed"


def test_no_location_is_stored_unlocated(service, tmp_path, stub_layer1) -> None:
    from app.tools.location import Location
    stub_layer1["location"] = Location()
    row = wait(service, submit(service, tmp_path).id)
    assert row["status"] == "ready" and row["lat"] is None and row["lon"] is None
    assert row["scores"]["values"]["cost"] is None and row["scores"]["values"]["drilling"] is not None
    assert row["name"] == "a"


def test_validation_correction_is_applied_and_raw_csv_preserved(service, tmp_path, monkeypatch) -> None:
    import app.services.ingest as ingest

    def relabel(df, result):
        out = df.copy()
        out.loc[0, "lithology"] = "Corrected"
        return out

    monkeypatch.setattr(ingest, "apply_corrections", relabel)
    row = wait(service, submit(service, tmp_path).id)
    final = pd.read_csv(row["csv_path"])
    raw = pd.read_csv(row["csv_path"].replace(".csv", ".raw.csv"))
    assert final.lithology[0] == "Corrected" and raw.lithology[0] == "Shale grey"
    assert list(raw.columns) == list(final.columns)


def test_same_file_is_not_ingested_twice(service, tmp_path) -> None:
    first = submit(service, tmp_path)
    wait(service, first.id)
    again = submit(service, tmp_path, "copy.tif")
    assert (again.id, again.created, again.status) == (first.id, False, "ready")
    assert len(repo.list_wells(service.db)) == 1


def test_failed_well_is_queued_again_on_resubmit(service, tmp_path, stub_layer1) -> None:
    stub_layer1["calibration"] = None
    first = wait(service, submit(service, tmp_path).id)
    assert first["status"] == "failed"
    from app.tools.preprocess import Calibration
    stub_layer1["calibration"] = Calibration(0.25, 5000.0, 0.1, 10, 12)
    again = submit(service, tmp_path)
    assert again.created and again.id == first["id"]
    assert wait(service, again.id)["status"] == "ready"


def test_rejects_a_non_image(service, tmp_path) -> None:
    path = tmp_path / "x.tif"
    path.write_text("not an image")
    with pytest.raises(UnsupportedFile):
        service.submit(path, "x.tif")
    assert repo.list_wells(service.db) == []


def test_refuses_to_run_without_a_model(service, tmp_path) -> None:
    service._model_state = lambda: ModelState(False, "weights missing")
    with pytest.raises(ModelUnavailable, match="weights missing"):
        submit(service, tmp_path)


def test_neighbours_in_the_database_change_the_cost_score(service, tmp_path, stub_layer1) -> None:
    from app.tools.location import Location
    wait(service, submit(service, tmp_path).id)
    stub_layer1["location"] = Location(lat=27.51, lon=-90.0, well_name="Next", source="header", confidence=0.9)
    near = wait(service, submit(service, tmp_path, "b.tif", tweak=9).id)
    stub_layer1["location"] = Location(lat=25.0, lon=-86.0, well_name="Far", source="header", confidence=0.9)
    far = wait(service, submit(service, tmp_path, "c.tif", tweak=10).id)
    assert near["status"] == far["status"] == "ready"
    assert near["scores"]["values"]["cost"] > far["scores"]["values"]["cost"]   # isolation raises the development burden


def test_bbox_is_kept_next_to_the_scan(service, tmp_path) -> None:
    sub = submit(service, tmp_path)
    wait(service, sub.id)
    saved = json.loads((service.settings.scans_dir / f"{sub.id}.bbox.json").read_text())
    assert saved["bounding_boxes"]["lithology_legend"] == list(synth.LEGEND_BOX)


def test_drain_runs_every_queued_job(service, tmp_path) -> None:
    ids = [submit(service, tmp_path, f"d{i}.tif", tweak=i + 20).id for i in range(3)]
    service.shutdown(drain=True)
    assert [repo.get(service.db, i)["status"] for i in ids] == ["ready"] * 3
