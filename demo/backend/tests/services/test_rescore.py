import sys
import time
from pathlib import Path

import pytest

from app.store import wells as repo
from tests import synth
from tests.services.test_ingest import service, submit, wait  # noqa: F401  (fixtures and helpers)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import rescore  # noqa: E402


def test_rescore_recomputes_stored_scores(service, tmp_path) -> None:  # noqa: F811
    well_id = submit(service, tmp_path).id
    row = wait(service, well_id)
    original = row["scores"]
    repo.set_scores(service.db, well_id, {"values": {"drilling": 0, "cost": 0, "price": 0, "co2": 0}, "logged_ft": 1, "flags": []})
    assert rescore.rescore(service.db) == 1
    assert repo.get(service.db, well_id)["scores"] == original
