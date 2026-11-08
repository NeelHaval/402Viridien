"""Ingest: scan file -> tools (layer 1) -> pipeline -> validation -> store. Used by the upload endpoint and the dataset script."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from app import pipeline
from app.config import Settings
from app.pipeline.model import ModelState, configure_tools
from app.pipeline.schema import check_output_schema
from app.services import scoring
from app.services.events import Broadcaster
from app.services.jobs import JobQueue
from app.services.locator import locate_well
from app.store import wells as repo
from app.store.db import Database
from app.timing import Trace, attach, span
from app.tools.location import Location
from app.tools.preprocess import finish_calibration, preprocess
from app.tools.read_upload import read_upload
from app.validation import apply_corrections, reference, submission_df, validate

log = logging.getLogger("ingest")
HASH_CHUNK = 1024 * 1024


class ModelUnavailable(Exception):
    """The OCR model is not loaded, so scans cannot be processed."""


@dataclass
class Submitted:
    """Outcome of submitting a scan: the well id, its status, and whether a new job was created."""

    id: str
    status: str
    created: bool


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_scores(df: pd.DataFrame, lat: float | None, lon: float | None, others: list[dict]) -> scoring.WellScores:
    """Scores with the water depth taken from the bathymetry reference data (None when it has no value)."""
    water = reference.bathymetry_at(lat, lon) if lat is not None and lon is not None else None
    return scoring.score_well(df, lat, lon, others, water)


class IngestService:
    """Accepts scans, queues them and processes them one at a time. Location runs as its own task from the moment a scan is saved."""

    def __init__(self, settings: Settings, db: Database, model_state: Callable[[], ModelState], events: Broadcaster | None = None) -> None:
        """`model_state()` returns the current model status; the queue starts immediately."""
        self.settings, self.db, self._model_state = settings, db, model_state
        self.events = events or Broadcaster()
        self._traces: dict[str, Trace] = {}
        self._locations: dict[str, Future] = {}
        self._enqueued_at: dict[str, float] = {}
        self._loc_pool = ThreadPoolExecutor(2, thread_name_prefix="locate")
        configure_tools(settings)
        self.jobs = JobQueue(self.process)
        self._submit_lock = threading.RLock()

    def submit(self, *args, **kwargs):
        with self._submit_lock:
            return self._submit(*args, **kwargs)

    def _submit(self, src: Path, filename: str, bbox: dict | None = None, *, keep_source: bool = False,
               trace: Trace | None = None) -> Submitted:
        """Validate a scan, store it under data/scans, start locating it and queue the job. Raises ToolError for bad files,
        ModelUnavailable without a model.

        Files already ingested (same hash) return the existing well with its stored location and results and do no work;
        a failed one is queued again.
        """
        state = self._model_state()
        if not state.loaded:
            raise ModelUnavailable(state.error or "OCR model is not loaded")
        trace = trace or Trace()
        with attach(trace):
            with span("upload.validate"):
                read_upload(str(src)).close()
            with span("upload.hash"):
                source_hash = sha256_of(src)
                identity = {"image_sha256": source_hash, "bbox": bbox, "engine": "S19" if os.environ.get("S19_ADAPTER_ENABLED") == "1" else "native", "release": os.environ.get("APP_RELEASE_ID", "combined-v1")}
                file_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            well_id = file_hash[:16]
            existing = repo.get_by_hash(self.db, file_hash)
            if (not existing or existing["status"] == "failed") and sum(w["status"] in ("queued", "processing") for w in repo.list_wells(self.db)) >= 16:
                from app.api.errors import ApiError
                raise ApiError(429, "queue_full", "The queue is full. Try again after a job completes.")
            if existing and existing["status"] != "failed":
                if not keep_source:
                    src.unlink(missing_ok=True)
                return Submitted(existing["id"], existing["status"], False)
            # The content reader determines the format. Bound untrusted filename suffixes.
            suffix = Path(filename).suffix.lower()
            suffix = suffix if suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg") else ".scan"
            dest = self.settings.scans_dir / f"{well_id}{suffix}"
            with span("upload.move"):
                (shutil.copyfile if keep_source else shutil.move)(src, dest)
            bbox_path = dest.with_suffix(".bbox.json")
            bbox_path.unlink(missing_ok=True)
            if bbox:
                bbox_path.write_text(json.dumps(bbox))
            with span("db.insert"):
                if existing:
                    repo.requeue(self.db, existing["id"], str(dest), filename)
                else:
                    repo.insert_queued(self.db, well_id=well_id, file_hash=file_hash, filename=filename, name=Path(filename).stem, scan_path=str(dest))
            trace.mark("upload_saved")
            self._traces[well_id] = trace
            with span("event_publish"):
                self.events.publish("well_created", well_id, status="queued", name=Path(filename).stem)
            self._locations[well_id] = self._loc_pool.submit(self._locate, well_id, dest, filename, bbox, trace)
            self._enqueued_at[well_id] = time.perf_counter()
            self.jobs.enqueue(well_id)
        return Submitted(well_id, "queued", True)

    def _locate(self, well_id: str, path: Path, filename: str, bbox: dict | None, trace: Trace) -> Location:
        """The location task: runs at once, publishes the pin as soon as a position is known, never raises."""
        with attach(trace):
            try:
                return locate_well(path, filename, bbox, on_located=lambda loc: self._pin(well_id, loc))
            except Exception:
                log.exception("well=%s location failed", well_id)
                return Location(warnings=["not_located:location_error"])

    def _pin(self, well_id: str, loc: Location) -> None:
        """Store a found position immediately and tell the UI."""
        with span("location.db_write"):
            repo.set_location(self.db, well_id, lat=loc.lat, lon=loc.lon, source=loc.source or "", block=loc.block, api_number=loc.api_number)
        with span("location.event_publish"):
            self.events.publish("well_located", well_id, lat=loc.lat, lon=loc.lon, source=loc.source, confidence=loc.confidence, block=loc.block)

    def process(self, well_id: str) -> None:
        """Run one queued well to `ready` or `failed`. Called on the worker thread."""
        well = repo.get(self.db, well_id)
        if well is None:
            return
        trace = self._traces.get(well_id) or Trace()
        started = time.perf_counter()
        timings: dict = {}
        with attach(trace):
            trace.add("queue_wait", self._enqueued_at.pop(well_id, started), started - self._enqueued_at.get(well_id, started))
            with span("db.mark_processing"):
                repo.mark_processing(self.db, well_id)
            try:
                self._process(well, timings, trace)
            except Exception as exc:
                log.exception("well=%s failed", well_id)
                timings["total"] = time.perf_counter() - started
                timings["spans"] = trace.snapshot()
                repo.mark_failed(self.db, well_id, str(exc) or type(exc).__name__, timings)
                self.events.publish("well_failed", well_id, status="failed", error=str(exc) or type(exc).__name__)
            finally:
                self._traces.pop(well_id, None)
                self._locations.pop(well_id, None)

    def _process(self, well: dict, timings: dict, trace: Trace) -> None:
        """The actual work for one well; raises on any failure."""
        well_id, t0 = well["id"], time.perf_counter()
        scan_path = Path(well["scan_path"])
        bbox_path = scan_path.with_suffix(".bbox.json")
        bbox = json.loads(bbox_path.read_text()) if bbox_path.is_file() else None

        if os.environ.get("S19_ADAPTER_ENABLED") == "1":
            from app.services.s19_adapter import app_output
            out = app_output(scan_path, bbox, well_id, self.settings)
        else:
            out = self._legacy_pipeline(scan_path, bbox, well_id, timings)
        if getattr(out, "partial_result", None):
            self._save_partial(well, out.partial_result, timings)
            return
        timings.update({f"pipeline.{k}": v for k, v in out.timings.items()})

        with span("wait_location"):  # normally already finished: location takes about a second, the rest of the job much longer
            location = self._locations[well_id].result() if well_id in self._locations else Location()
        out.lat, out.lon, out.name, out.block = location.lat, location.lon, location.well_name, location.block
        out.warnings = list(out.warnings) + [w for w in location.warnings if w.startswith("not_located")]

        t1 = time.perf_counter()
        with span("validation"):
            checked = validate(out.df, location)
        with span("correction"):
            final = submission_df(out.df) if out.timings.get("s19") else submission_df(apply_corrections(out.df, checked))
        with span("schema_check"):
            check_output_schema(final)
        timings["validation"] = time.perf_counter() - t1
        log.info("well=%s stage=validation elapsed=%.2fs confidence=%.2f", well_id, timings["validation"], checked.confidence)

        csv_path = self.settings.csv_dir / f"{well_id}.csv"
        with span("csv_write"):
            out.df.to_csv(csv_path.with_suffix(".raw.csv"), index=False)
            final.to_csv(csv_path, index=False)
            from app.services.result_exports import save_result
            save_result(self.settings, well_id, scan_path, final.to_dict("records"), "S19" if out.timings.get("s19") else "native", "available", bbox)

        t2 = time.perf_counter()
        with span("scoring.neighbours"):
            others = [w for w in repo.list_ready_located(self.db) if w["id"] != well_id]
        with span("scoring"):
            scores = compute_scores(final, out.lat, out.lon, others)
        timings["scoring"] = time.perf_counter() - t2
        timings["total"] = time.perf_counter() - t0
        log.info("well=%s stage=scoring elapsed=%.2fs total=%.2fs", well_id, timings["scoring"], timings["total"])
        timings["spans"] = trace.snapshot()
        with span("db.mark_ready"):
            repo.mark_ready(
                self.db, well_id, name=out.name or Path(well["source_filename"] or well_id).stem, lat=out.lat, lon=out.lon,
                block=out.block, total_depth_ft=float(final.base_depth.max()), csv_path=str(csv_path),
                validation_confidence=checked.confidence,
                validation={"components": checked.components, "flags": checked.flags, "needs_review": checked.needs_review},
                scores=scores.to_json(), warnings=out.warnings, timings=timings,
                location_source=location.source, api_number=location.api_number)
        with span("event_publish"):
            self.events.publish("well_ready", well_id, status="ready", name=out.name, total_depth_ft=float(final.base_depth.max()),
                                validation_confidence=checked.confidence)
            self.events.publish("well_scored", well_id, scores=scores.to_json()["values"])
        timings["spans"] = trace.snapshot()
        repo.set_timings(self.db, well_id, timings)  # second write so the DB-write and publish spans are included


    def _save_partial(self, well, result, timings):
        from app.services.result_exports import save_result
        well_id = well["id"]
        save_result(self.settings, well_id, Path(well["scan_path"]), result["rows"], "S19", "unavailable", None, result)
        with self.db.connect() as conn:
            conn.execute("UPDATE wells SET status='partial', error=NULL, total_depth_ft=NULL, scores_json=NULL, warnings_json=?, timings_json=? WHERE id=?", (json.dumps(["depth_unavailable: printed depth units and calibration are not verified"]), json.dumps(timings), well_id))
        self.events.publish("well_ready", well_id, status="partial")

    def _legacy_pipeline(self, scan_path, bbox, well_id, timings):
        with span("scan_open"):
            scan = read_upload(str(scan_path))
        try:
            layer1 = preprocess(scan, bbox, with_location=False, calibrate=False)
            log.info("well=%s stage=layer1 elapsed=%.2fs", well_id, layer1.timings.get("total", 0.0))
            out = pipeline.run(scan, layer1, settings=self.settings, well_id=well_id,
                               finish_layer1=lambda: finish_calibration(scan, layer1))
            timings.update({f"layer1.{k}": v for k, v in layer1.timings.items()})
        finally:
            scan.close()
        return out

    def shutdown(self, drain: bool = False) -> None:
        """Stop the worker after the running job; `drain` also runs everything still queued."""
        self.jobs.shutdown(drain=drain)
        self._loc_pool.shutdown(wait=True)
