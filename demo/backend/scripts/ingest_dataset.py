"""Run every scan of a dataset folder through the same ingest service the upload endpoint uses.

    python scripts/ingest_dataset.py DATASET_DIR

DATASET_DIR holds `images/` and (optionally) `bbox_annotation/` with one `<scan stem>.json` per scan. Scans already
ingested (same file hash) are skipped. Prints a summary at the end.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.pipeline.model import load_model  # noqa: E402
from app.services.ingest import IngestService, ModelUnavailable  # noqa: E402
from app.store import wells as repo  # noqa: E402
from app.store.db import Database  # noqa: E402
from app.tools.errors import ToolError  # noqa: E402
from app.validation import reference  # noqa: E402

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def main() -> int:
    """Submit all scans, wait for the queue to drain and print the outcome."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    images = args.dataset / "images"
    if not images.is_dir():
        print(f"error: {images} does not exist", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = Settings()
    settings.ensure_dirs()
    reference.set_reference_dir(settings.reference_dir)
    db = Database(settings.db_path)
    db.migrate()
    repo.fail_interrupted(db)
    state = load_model()
    service = IngestService(settings, db, lambda: state)
    started = time.perf_counter()
    outcome: Counter[str] = Counter()
    ids: list[str] = []
    for path in sorted(images.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            outcome["not an image"] += 1
            continue
        bbox_file = args.dataset / "bbox_annotation" / f"{path.stem}.json"
        bbox = json.loads(bbox_file.read_text()) if bbox_file.is_file() else None
        try:
            submitted = service.submit(path, path.name, bbox, keep_source=True)
        except ModelUnavailable as exc:
            print(f"error: OCR model not loaded: {exc}", file=sys.stderr)
            service.shutdown()
            return 1
        except ToolError as exc:
            outcome["rejected"] += 1
            print(f"rejected {path.name}: {exc}")
            continue
        outcome["queued" if submitted.created else "already ingested"] += 1
        ids.append(submitted.id)
    service.shutdown(drain=True)  # the single worker runs the whole queue first
    rows = [repo.get(db, i) for i in ids]
    status = Counter(r["status"] for r in rows)
    located = sum(1 for r in rows if r["status"] == "ready" and r["lat"] is not None)
    for r in rows:
        if r["status"] == "failed":
            print(f"failed {r['source_filename']}: {r['error']}")
    print(f"\nsummary: {dict(outcome)}")
    print(f"status of the {len(rows)} wells in this run: {dict(status)}; located: {located}; unlocated: {status['ready'] - located}")
    print(f"elapsed: {time.perf_counter() - started:.0f}s; database: {settings.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
