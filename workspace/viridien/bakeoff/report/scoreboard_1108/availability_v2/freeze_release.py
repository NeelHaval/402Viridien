"""Freeze the cached-only score release. Do not open dataset payloads."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BAKEOFF = REPORT.parent.parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    preservation = read(REPORT / "baseline_v1_preservation_receipt.json")
    original_pins = {
        REPORT / "scores.json": preservation["scores_json_sha256"],
        REPORT / "raw_results.json": preservation["raw_results_json_sha256"],
        REPORT / "index_baseline_v1.html": preservation["original_html_sha256"],
        BAKEOFF / "harness/score.py": preservation["shared_scorer_sha256"],
    }
    for path, expected in original_pins.items():
        assert sha(path) == expected, f"Original changed: {path.name}"
    release = read(REPORT / "SCORE_RELEASE.json")
    assert sha(REPORT / release["active_data"]) == release["active_data_sha256"]
    assert sha(REPORT / release["jev_checkpoint"]["data"]) == release["jev_checkpoint"]["sha256"]
    assert sha(REPORT / release["external_version_notice"]) == release["external_version_notice_sha256"]
    checks = read(HERE / "checks_receipt.json")
    audit = read(HERE / "parent_audit.json")
    jev = REPORT / "jev_comparison/availability_v2"
    jev_audit = read(jev / "parent_checkpoint_audit.json")
    assert checks["exit_code"] == 0
    assert not audit["errors"] and audit["records"] == 272
    assert not jev_audit["errors"] and jev_audit["paired_cases"] == 29
    assert audit["derived_sha256"] == release["active_data_sha256"]
    assert jev_audit["scores_sha256"] == release["jev_checkpoint"]["sha256"]
    assert release["selection_promoted"] is False
    assert release["new_dataset_inference_allowed"] is False
    assert release["new_JEV_calls_allowed"] is False
    sources = list(HERE.glob("*.py")) + list(jev.glob("*.py"))
    sources += [BAKEOFF / "strategies" / name for name in (
        "grading_policy.py", "test_grading_policy.py", "grading_availability_v2.py",
        "test_grading_availability_v2.py", "SCORING_CONTRACT_V2.md", "AGENTS.md")]
    sources += [BAKEOFF / "harness" / name for name in ("score.py", "run.py")]
    now = datetime.now(timezone.utc).isoformat()
    source_pins = {str(p.relative_to(BAKEOFF)).replace("\\", "/"): sha(p)
                   for p in sorted(set(sources))}
    write(HERE / "source_freeze.json", {
        "created_utc": now, "scope": "Availability V2 and cached JEV checkpoint only",
        "sources": source_pins, "historical_source_freeze_replaced": False,
        "Astra_review": "Earlier baseline and S23 only; no new review of Availability V2",
    })
    artifacts = [REPORT / "SCORE_RELEASE.json", REPORT / "index.html",
                 REPORT / "baseline_v1_preservation_receipt.json", HERE / "source_freeze.json"]
    artifacts += [p for folder in (HERE, jev) for p in folder.iterdir()
                  if p.is_file() and p.suffix in (".json", ".html", ".md", ".log")
                  and p.name not in ("release_receipt.json", "source_freeze.json")]
    artifact_pins = {str(p.relative_to(BAKEOFF)).replace("\\", "/"): sha(p)
                     for p in sorted(set(artifacts))}
    write(HERE / "release_receipt.json", {
        "version": "availability-v2-final-cached-freeze", "created_utc": now,
        "sources": len(source_pins), "artifacts": artifact_pins,
        "original_hashes_unchanged": True, "records": 272,
        "JEV_paired_cases": 29, "JEV_saved_cases": 30, "JEV_expected_cases": 68,
        "focused_checks": "58 passed in 6.27s; exit 0",
        "parent_audit": "5440 metric cells; 2240 summary metrics; 72 paired CIs; 0 errors",
        "full_harness": "Known PNG fixture/TIFF loader failure remains; full suite not passing",
        "cache_pilot": "Separate source packet; 27 targeted checks passed in 5.33s",
        "selection_promoted": False, "new_inference_or_API_started": False,
        "dataset_hold_lifted": False, "full_JEV_run_complete": False,
    })
    print(f"RELEASE_FREEZE: {len(source_pins)} sources; {len(artifact_pins)} artifacts; "
          "272 baseline records; 29 JEV pairs; originals unchanged; no selection promoted")


if __name__ == "__main__":
    main()
