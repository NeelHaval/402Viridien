"""Ingest: scan file -> tools (layer 1) -> pipeline -> validation -> store. Used by the upload endpoint and the dataset script."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from app import pipeline
from app.config import Settings
from app.pipeline.model import ModelState, configure_tools
from app.pipeline.schema import check_output_schema
from app.services import scoring
from app.services.jobs import JobQueue
from app.store import wells as repo
from app.store.db import Database
from app.tools.preprocess import preprocess
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
    """Accepts scans, queues them and processes them one at a time."""

    def __init__(self, settings: Settings, db: Database, model_state: Callable[[], ModelState]) -> None:
        """`model_state()` returns the current model status; the queue starts immediately."""
        self.settings, self.db, self._model_state = settings, db, model_state
        configure_tools(settings)
        self.jobs = JobQueue(self.process)

    def submit(self, src: Path, filename: str, bbox: dict | None = None, *, keep_source: bool = False) -> Submitted:
        """Validate a scan, store it under data/scans and queue it. Raises ToolError for bad files, ModelUnavailable without a model.

        Files already ingested (same hash) return the existing well; a failed one is queued again.
        """
        state = self._model_state()
        if not state.loaded:
            raise ModelUnavailable(state.error or "OCR model is not loaded")
        read_upload(str(src)).close()
        file_hash = sha256_of(src)
        well_id = file_hash[:16]
        existing = repo.get_by_hash(self.db, file_hash)
        if existing and existing["status"] != "failed":
            if not keep_source:
                src.unlink(missing_ok=True)
            return Submitted(existing["id"], existing["status"], False)
        dest = self.settings.scans_dir / f"{well_id}{Path(filename).suffix.lower()}"
        (shutil.copyfile if keep_source else shutil.move)(src, dest)
        bbox_path = dest.with_suffix(".bbox.json")
        bbox_path.unlink(missing_ok=True)
        if bbox:
            bbox_path.write_text(json.dumps(bbox))
        if existing:
            repo.requeue(self.db, existing["id"], str(dest), filename)
        else:
            repo.insert_queued(self.db, well_id=well_id, file_hash=file_hash, filename=filename, name=Path(filename).stem, scan_path=str(dest))
        self.jobs.enqueue(well_id)
        return Submitted(well_id, "queued", True)

    def process(self, well_id: str) -> None:
        """Run one queued well to `ready` or `failed`. Called on the worker thread."""
        well = repo.get(self.db, well_id)
        if well is None:
            return
        repo.mark_processing(self.db, well_id)
        started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            self._process(well, timings)
        except Exception as exc:
            log.exception("well=%s failed", well_id)
            timings["total"] = time.perf_counter() - started
            repo.mark_failed(self.db, well_id, str(exc) or type(exc).__name__, timings)

    def _process(self, well: dict, timings: dict[str, float]) -> None:
        """The actual work for one well; raises on any failure."""
        well_id, t0 = well["id"], time.perf_counter()
        scan_path = Path(well["scan_path"])
        bbox_path = scan_path.with_suffix(".bbox.json")
        bbox = json.loads(bbox_path.read_text()) if bbox_path.is_file() else None

        scan = read_upload(str(scan_path))
        try:
            layer1 = preprocess(scan, bbox)
            timings.update({f"layer1.{k}": v for k, v in layer1.timings.items()})
            log.info("well=%s stage=layer1 elapsed=%.2fs", well_id, layer1.timings.get("total", 0.0))
            out = pipeline.run(scan, layer1, settings=self.settings, well_id=well_id)
        finally:
            scan.close()
        timings.update({f"pipeline.{k}": v for k, v in out.timings.items()})

        t1 = time.perf_counter()
        checked = validate(out.df, layer1.location)
        final = submission_df(apply_corrections(out.df, checked))
        check_output_schema(final)
        timings["validation"] = time.perf_counter() - t1
        log.info("well=%s stage=validation elapsed=%.2fs confidence=%.2f", well_id, timings["validation"], checked.confidence)

        csv_path = self.settings.csv_dir / f"{well_id}.csv"
        out.df.to_csv(csv_path.with_suffix(".raw.csv"), index=False)
        final.to_csv(csv_path, index=False)

        t2 = time.perf_counter()
        others = [w for w in repo.list_ready_located(self.db) if w["id"] != well_id]
        scores = compute_scores(final, out.lat, out.lon, others)
        timings["scoring"] = time.perf_counter() - t2
        timings["total"] = time.perf_counter() - t0
        log.info("well=%s stage=scoring elapsed=%.2fs total=%.2fs", well_id, timings["scoring"], timings["total"])
        repo.mark_ready(
            self.db, well_id, name=out.name or Path(well["source_filename"] or well_id).stem, lat=out.lat, lon=out.lon,
            block=out.block, total_depth_ft=float(final.base_depth.max()), csv_path=str(csv_path),
            validation_confidence=checked.confidence,
            validation={"components": checked.components, "flags": checked.flags, "needs_review": checked.needs_review},
            scores=scores.to_json(), warnings=out.warnings, timings=timings)

    def shutdown(self, drain: bool = False) -> None:
        """Stop the worker after the running job; `drain` also runs everything still queued."""
        self.jobs.shutdown(drain=drain)
