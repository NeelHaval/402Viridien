"""Visual evidence uses existing demo pixels, not generated screenshots."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from viridian402.data import load_intervals, load_manifest, sha256
from viridian402.evaluate import evaluate
from viridian402.metrics import score
from viridian402.report import load_run, write_visual_report
from viridian402.visual_report import find_issues


DEMO = Path(__file__).resolve().parents[1]/"examples"/"demo"/"manifest.json"


@pytest.fixture(scope="module")
def visual_run(tmp_path_factory):
    run, _ = evaluate(DEMO, output=tmp_path_factory.mktemp("visual")/"runs",
                      model="builtin:baseline", mode="supplied_legend")
    return run


def read_evidence(run):
    return json.loads((run/"visual-evidence.json").read_text(encoding="utf-8"))


def test_actual_fixture_pixels_boxes_legends_and_report(visual_run):
    evidence = read_evidence(visual_run)
    manifest = load_manifest(DEMO)
    records = {r["id"]: r for r in manifest["records"]}
    assert evidence["evidence_complete"]
    assert evidence["promotion_supported"] is False
    assert len(evidence["cases"]) == 3
    assert any(c["issues"] for c in evidence["cases"])
    for case in evidence["cases"]:
        record = records[case["id"]]
        with Image.open(visual_run/case["legend"]["image"]) as legend, Image.open(record["image"]) as source:
            assert np.array_equal(np.asarray(legend), np.asarray(source.crop(case["legend"]["original_box"])))
        assert 0 <= len(case["issues"]) <= 5
        assert case["issues"] == sorted(case["issues"], key=lambda i: (-i["severity"], i["category"], i["expected_box"] or i["predicted_box"]))
        for issue in case["issues"]:
            assert issue["source_sha256"] == sha256(record["image"])
            assert issue["crop_transform"]["scale_x"] == issue["crop_transform"]["scale_y"] == 1
            with Image.open(visual_run/issue["crop"]) as crop, Image.open(record["image"]) as source:
                assert np.array_equal(np.asarray(crop), np.asarray(source.crop(issue["crop_box"])))
            with Image.open(visual_run/issue["overlay"]) as overlay:
                colors = set(overlay.getdata())
                if issue["expected_box"]:
                    assert (0, 114, 178) in colors
                if issue["predicted_box"]:
                    assert (213, 94, 0) in colors
            assert issue["original_pixels"]["expected"] == issue["expected_box"]
            assert issue["legend"] == case["legend"]
    document = (visual_run/"report.html").read_text(encoding="utf-8")
    assert "Blue: truth" in document and "Orange: prediction" in document
    assert all(c["id"] in document for c in evidence["cases"])
    assert "<script" not in document and "https://" not in document
    before = {name: sha256(visual_run/name) for name in ("metrics.json", "per_case.jsonl", "manifest.snapshot.json")}
    assert write_visual_report(visual_run) == visual_run/"report.html"
    assert before == {name: sha256(visual_run/name) for name in before}


def row(a, b, label):
    return {"top_y": a, "bottom_y": b, "lithology": label}


def test_categories_null_boxes_tolerance_and_original_coordinates():
    gt = [row(0, 50, "Shale"), row(50, 100, "Sandstone")]
    issues, counts = find_issues([row(0, 60, "Shale"), row(60, 100, "Sandstone")], gt, [0, 100], [32, 0, 128, 100], 4)
    shifted = next(i for i in issues if i["category"] == "shift_over_tolerance")
    assert shifted["expected_box"] == [32, 50, 128, 51]
    assert shifted["predicted_box"] == [32, 60, 128, 61]
    assert shifted["rows_affected"] == 10 and shifted["tolerance_px"] == 4
    assert counts["boundary_matches"] == 0
    assert any(i["category"] == "wrong_lithology" for i in issues)
    missed, _ = find_issues([], gt, [0, 100], [32, 0, 128, 100], 4)
    assert any(i["category"] == "missed_boundary" and i["predicted_box"] is None for i in missed)
    false, _ = find_issues(gt, [row(0, 100, "Shale")], [0, 100], [32, 0, 128, 100], 4)
    assert any(i["category"] == "false_boundary" and i["expected_box"] is None for i in false)
    matched, counts = find_issues([row(0, 54, "Shale"), row(54, 100, "Sandstone")], gt, [0, 100], [32, 0, 128, 100], 4)
    assert counts["boundary_matches"] == 1
    assert not any(i["category"] == "shift_over_tolerance" for i in matched)


def test_matching_parity_with_actual_fixtures_and_partial_truth():
    for record in load_manifest(DEMO)["records"]:
        gt = load_intervals(record["truth"])
        y0, y1 = record["window"]
        for offset in (0, 1, 4, 7, 30):
            pred = [{**r, "top_y": max(y0, r["top_y"]-offset), "bottom_y": max(y0, r["bottom_y"]-offset)} for r in gt]
            pred = [r for r in pred if r["bottom_y"] > r["top_y"]]
            for truth in (gt, [gt[0], *gt[2:]]):
                _, counts = find_issues(pred, truth, record["window"], [32, y0, 128, y1], 4)
                metrics = score(pred, truth, record["window"], boundary_tolerance=4)
                assert counts == {key: metrics[key] for key in counts}


def test_all_statuses_and_render_failure_stay_visible(tmp_path, monkeypatch):
    import viridian402.evaluate as evaluator
    manifest = load_manifest(DEMO)
    records = manifest["records"]
    truth = {r["id"]: load_intervals(r["truth"]) for r in records}
    records[2].update(truth=None, source="external_unlabelled")
    records[2].pop("truth_sha256", None)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    def predict(model, request, config, timeout, max_worker_rss_mb=None):
        if request["id"] == records[1]["id"]:
            return None, "fixture crash <unsafe>", .01, 1
        return truth[request["id"]], None, .01, 1
    monkeypatch.setattr(evaluator, "_predict", predict)
    run, _ = evaluate(path, output=tmp_path/"runs")
    evidence = read_evidence(run)
    assert [c["status"] for c in evidence["cases"]] == ["pass", "crash", "unlabelled"]
    assert evidence["cases"][2]["accuracy"] is None
    assert "fixture crash &lt;unsafe&gt;" in (run/"report.html").read_text(encoding="utf-8")
    _, cases = load_run(run)
    (run/cases[0]["preview"]).unlink()
    write_visual_report(run)
    evidence = read_evidence(run)
    assert len(evidence["cases"]) == 3
    assert evidence["cases"][0]["render_status"] == "render_failure"
    assert not evidence["evidence_complete"] and not evidence["promotion_supported"]


def test_source_change_is_visible_and_no_preview_is_incomplete(tmp_path, monkeypatch):
    import viridian402.visual_report as visual
    run, _ = evaluate(DEMO, output=tmp_path/"runs", model="builtin:empty", previews=False)
    assert not read_evidence(run)["evidence_complete"]
    original = visual.sha256
    monkeypatch.setattr(visual, "sha256", lambda path: "0"*64 if Path(path).suffix == ".png" else original(path))
    write_visual_report(run)
    evidence = read_evidence(run)
    assert all(c["render_status"] == "render_failure" for c in evidence["cases"])
    assert "checksum" in str(evidence)


def test_crop_decoder_failure_retains_required_image_warning(tmp_path, monkeypatch):
    import viridian402.visual_report as visual
    run, _ = evaluate(DEMO, output=tmp_path/"runs", model="builtin:empty")
    def refuse(*args, **kwargs):
        raise MemoryError("fixture decoder budget")
    monkeypatch.setattr(visual, "read_rgb", refuse)
    write_visual_report(run)
    evidence = read_evidence(run)
    assert not evidence["evidence_complete"]
    assert all(c["issues"] for c in evidence["cases"])
    assert all(i["crop"] is None and i["render_error"] for c in evidence["cases"] for i in c["issues"])
    document = (run/"report.html").read_text(encoding="utf-8")
    assert "Required crop image unavailable" in document
    assert "fixture decoder budget" in document
