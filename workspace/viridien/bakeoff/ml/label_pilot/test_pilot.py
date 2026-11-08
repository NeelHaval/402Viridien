"""Check the split guard, label coverage, names, and training mask."""
import copy
import csv
import json

import pytest

from prepare import HOME, eval_names, require_non_eval, sha256
from build import parser_name, validate_review


def read(name):
    return json.loads((HOME / name).read_text())


def test_all_eval_logs_are_blocked():
    for name in eval_names():
        with pytest.raises(ValueError, match="Eval log is blocked"):
            require_non_eval(name)


def test_unknown_log_is_blocked():
    with pytest.raises(ValueError, match="outside the two-log pilot"):
        require_non_eval("not-a-pilot-log")


def test_review_covers_only_allowed_windows():
    review = read("review.json")
    validate_review(review)
    assert len(review["logs"]) == 2
    for log in review["logs"]:
        assert log["log"] not in eval_names()


def test_overlapping_rows_are_rejected():
    review = copy.deepcopy(read("review.json"))
    review["logs"][1]["rows"][1]["top_y"] -= 1
    with pytest.raises(ValueError, match="overlapping"):
        validate_review(review)


def test_changed_evidence_is_detectable():
    for path, digest in read("artifact_hashes.json").items():
        assert sha256(HOME / path) == digest, path


def test_harness_names_and_training_mask():
    names = read("legend_names.json")
    for log in read("review.json")["logs"]:
        filename = log["log"] + ".csv"
        with (HOME / "labels" / "harness" / filename).open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(log["rows"])
        for actual, source in zip(rows, log["rows"]):
            assert actual["lithology"] == parser_name(source["lithology"], names[log["key"]])
            assert actual["top_depth"] == actual["bottom_depth"] == ""
        expected = {(r["top_y"]+2, r["bottom_y"]-2,
                     parser_name(r["lithology"], names[log["key"]]))
                    for r in log["rows"] if r["confidence"] == "high" and
                    parser_name(r["lithology"], names[log["key"]]) != "Unknown"}
        with (HOME / "labels" / "training" / filename).open(newline="") as f:
            actual = {(int(r["top_y"]), int(r["bottom_y"]), r["lithology"])
                      for r in csv.DictReader(f)}
        assert actual == expected


def test_reference_agreement_is_not_claimed():
    audit = read("reference_audit.json")
    assert audit["files_checked"] == audit["eval_files_excluded"] == 26
    assert audit["eligible_references"] == []
    assert not audit["truth_csv_contents_read"]
    assert read("summary.json")["agreement"] is None
