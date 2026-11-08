"""Recompute the stored scores of every ready well (run after the scoring rules or reference data change)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.ingest import compute_scores  # noqa: E402
from app.store import wells as repo  # noqa: E402
from app.store.db import Database  # noqa: E402
from app.validation import reference  # noqa: E402


def rescore(db: Database) -> int:
    """Rewrite scores_json for all ready wells; returns how many were updated."""
    ready = [w for w in repo.list_wells(db, "ready") if w["csv_path"] and Path(w["csv_path"]).is_file()]
    located = [w for w in ready if w["lat"] is not None and w["lon"] is not None]
    for well in ready:
        others = [w for w in located if w["id"] != well["id"]]
        scores = compute_scores(pd.read_csv(well["csv_path"]), well["lat"], well["lon"], others)
        repo.set_scores(db, well["id"], scores.to_json())
    return len(ready)


if __name__ == "__main__":
    settings = Settings()
    reference.set_reference_dir(settings.reference_dir)
    print(f"rescored {rescore(Database(settings.db_path))} wells")
