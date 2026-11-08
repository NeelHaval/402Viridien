"""Bounded cached-output contract check. No model, image, truth or API calls."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BAKEOFF = REPORT.parent.parent
sys.path.insert(0, str(BAKEOFF))
from strategies.challenge_export import export, read_rows, INPUT_FIELDS
from report.scoreboard_1108.scan_readiness import validated_prediction_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def timing(records, field):
    values = sorted(float(r[field]) for r in records
                    if isinstance(r.get(field), (int, float)) and math.isfinite(r[field]))
    return {"n": len(values), "missing": len(records) - len(values),
            "median_seconds": statistics.median(values) if values else None,
            "p95_seconds": values[math.ceil(.95 * len(values)) - 1] if values else None,
            "max_seconds": max(values) if values else None}


def main():
    started = time.perf_counter()
    source = REPORT / "scores.json"
    original_pin = "f3b2f61b983815cc41350680dd241b5e9a7236bbe6aa1e969443614d6d6b3245"
    assert sha(source) == original_pin
    release = REPORT / "availability_v2/release_receipt.json"
    release_pin = sha(release)
    records = load(source)["records"]
    readiness_path = REPORT / "readiness_preview.json"
    readiness = load(readiness_path)
    ready = {(r["case_id"], r["strategy"]): r for r in readiness["records"]}
    assert len(records) == len(ready) == 272
    cases = {}
    details = []
    for record in records:
        if time.perf_counter() - started > 150:
            raise TimeoutError("150-second cached export budget exceeded")
        key = (record["case_id"], record["strategy"])
        item = ready[key]
        identity = (record["image_sha256"], record["annotation_sha256"],
                    tuple(record["window"]), record["truth_sha256"])
        prior = cases.setdefault(record["case_id"], identity)
        assert identity == prior, f"Unequal case inputs: {key}"
        source_csv = validated_prediction_path(record["prediction_csv"])
        assert sha(source_csv) == item["prediction_sha256"], f"CSV changed: {key}"
        rows = read_rows(source_csv, image_height=item["image_height"])
        y0, y1 = record["window"]
        assert all(y0 <= int(row["top_y"]) < int(row["bottom_y"]) <= y1 for row in rows)
        result = {"case_id": key[0], "strategy": key[1], "rows": len(rows),
                  "source_csv_sha256": sha(source_csv), "extent": item["extent"],
                  "unknown_label_rows": sum(row["lithology"].strip().lower() == "unknown" for row in rows)}
        if not rows or record.get("error") or record.get("failure"):
            result["status"] = "failed_or_empty_unverified; not exported as success"
        else:
            destination = BAKEOFF / "out/quick_contract_20261108" / key[1] / key[0]
            manifest = export(source_csv, destination, item["image_height"])
            for variant in ("notebook", "slide"):
                with (destination / manifest[variant]["file"]).open(encoding="utf-8", newline="") as stream:
                    actual = list(csv.DictReader(stream))
                if variant == "slide":
                    for row in actual:
                        row["bottom_depth"] = row.pop("base_depth")
                assert actual == rows, f"Export changed raw predictions: {key} {variant}"
            result.update(status="both_schemas_round_trip", export_manifest_sha256=sha(
                destination / (source_csv.stem + ".manifest.json")))
        details.append(result)
    assert len(cases) == 68
    command = [sys.executable, "-X", "utf8", "-B", "-m", "pytest", "-q",
               "strategies/test_challenge_export.py",
               "report/scoreboard_1108/test_readiness_paths.py", "--tb=short",
               "-o", "cache_dir=out/quick_contract_pytest_cache"]
    check = subprocess.run(command, cwd=BAKEOFF, capture_output=True, text=True,
                           encoding="utf-8", timeout=30)
    (HERE / "tests.log").write_text(check.stdout + check.stderr, encoding="utf-8")
    assert check.returncode == 0, "Contract tests failed; see tests.log"
    assert sha(source) == original_pin and sha(release) == release_pin
    strategies = sorted({r["strategy"] for r in records})
    summary = {
        "version": "quick-cached-contract-v1", "elapsed_seconds": time.perf_counter() - started,
        "cached_cases": 68, "records": 272, "unique_source_images": 61,
        "successful_dual_schema_exports": sum(r["status"] == "both_schemas_round_trip" for r in details),
        "non_success_records": [r for r in details if r["status"] != "both_schemas_round_trip"],
        "input_identity_checks": "Same frozen image, annotation, window and truth hashes across four strategies",
        "schema": list(INPUT_FIELDS), "raw_labels_contacts_clusters_depths_preserved": True,
        "historical_runtime_by_strategy": {
            strategy: {field: timing([r for r in records if r["strategy"] == strategy], field)
                       for field in ("seconds", "preprocess_seconds", "end_to_end_seconds")}
            for strategy in strategies},
        "runtime_limit": "Historical window times on mixed hardware/cache states; no full-run forecast",
        "readiness_by_strategy": readiness["by_strategy"],
        "test_command": command, "test_exit": check.returncode,
        "source_sha256": sha(Path(__file__)), "scores_sha256": original_pin,
        "readiness_sha256": sha(readiness_path), "frozen_release_sha256": release_pin,
        "submission_ready": False, "official_numeric_grade_verified": False,
        "new_model_inference": False, "truth_or_source_images_opened": False,
        "API_calls": 0, "owner_hold_lifted": False, "records_detail": details,
    }
    (HERE / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"QUICK_CONTRACT: {summary['successful_dual_schema_exports']}/272 dual-schema exports; "
          f"68 identical case contexts; {summary['elapsed_seconds']:.3f}s; "
          "raw outputs preserved; no inference; submission readiness unverified")
    print(check.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
