"""Bounded local report adversaries: owned files, no browser/network/model calls."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import struct
import time
import zlib

import pytest

from viridian402 import evaluate as evaluator, report, visual_report as visual
from viridian402.data import load_intervals, load_manifest, save_intervals, sha256

DEMO = Path(__file__).resolve().parents[2] / "examples/demo/manifest.json"
IMMUTABLE = ("metrics.json", "per_case.jsonl", "manifest.snapshot.json", "provenance.json")


@pytest.fixture(scope="module")
def perfect_run(tmp_path_factory):
    truth = {r["id"]: load_intervals(r["truth"]) for r in load_manifest(DEMO)["records"]}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluator, "_predict", lambda model, request, *args, **kwargs:
                      (deepcopy(truth[request["id"]]), None, 0., 0.))
        run, _ = evaluator.evaluate(DEMO, output=tmp_path_factory.mktemp("visual-source"))
    assert json.loads((run / "visual-evidence.json").read_text())["evidence_complete"]
    return run


@pytest.fixture
def run(tmp_path, perfect_run):
    target = tmp_path / "owned-run"
    shutil.copytree(perfect_run, target)
    return target


def _cases(run):
    return [json.loads(line) for line in (run / "per_case.jsonl").read_text().splitlines()]


def _save_cases(run, cases):
    (run / "per_case.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    summary = json.loads((run / "metrics.json").read_text())
    summary["per_case_sha256"] = sha256(run / "per_case.jsonl")
    (run / "metrics.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize("relative", ["report.html", "visual-evidence.json", "evidence/00000/legend.png"])
def test_hardlinked_outputs_cannot_rewrite_immutable_scores(run, relative):
    target = run / relative
    target.unlink(missing_ok=True)
    try:
        target.hardlink_to(run / "metrics.json")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Owned hardlink creation unavailable: {exc}")
    before = {name: sha256(run / name) for name in IMMUTABLE}
    if relative.startswith("evidence/"):
        report.write_visual_report(run)
        assert not json.loads((run / "visual-evidence.json").read_text())["evidence_complete"]
    else:
        with pytest.raises(ValueError, match="hardlinks"):
            report.write_visual_report(run)
    assert before == {name: sha256(run / name) for name in IMMUTABLE}
    assert not list(run.rglob(".visual-*"))


def test_symlinked_evidence_parent_cannot_write_outside_owned_run(run, tmp_path):
    outside = tmp_path / "owned-outside"
    outside.mkdir()
    (outside / "canary").write_bytes(b"OWNED_KEEP")
    original = run / "evidence"
    assert original.resolve().is_relative_to(tmp_path.resolve())
    original.rename(run / "old-evidence")
    try:
        original.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Owned directory symlink unavailable: {exc}")
    report.write_visual_report(run)
    assert {p.name: p.read_bytes() for p in outside.iterdir()} == {"canary": b"OWNED_KEEP"}
    assert not json.loads((run / "visual-evidence.json").read_text())["evidence_complete"]


def test_atomic_failure_preserves_prior_complete_artifact(tmp_path):
    target = tmp_path / "report.html"
    target.write_bytes(b"OWNED_COMPLETE_REPORT")

    def failing_writer(stream):
        stream.write(b"PARTIAL")
        raise OSError("generated interrupted write")

    with pytest.raises(OSError, match="interrupted"):
        visual.atomic_write(tmp_path, "report.html", failing_writer)
    assert target.read_bytes() == b"OWNED_COMPLETE_REPORT"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.html"]


@pytest.mark.parametrize("path", ["https://example.invalid/canary.png", "javascript:CANARY.png", "//example.invalid/a.png",
    "../outside.png", "C:/outside.png", "..\\outside.png", "image.svg", "image.png?query", "NUL.png", "a./image.png"])
def test_image_urls_reject_schemes_traversal_devices_and_non_png(path):
    with pytest.raises(ValueError):
        visual.image_url(path)


def test_real_preview_path_with_fragment_and_percent_is_url_encoded(run):
    cases = _cases(run)
    path = "previews/owned #1 %.png"
    shutil.copyfile(run / cases[0]["preview"], run / path)
    cases[0]["preview"] = path
    _save_cases(run, cases)
    report.write_visual_report(run)
    assert json.loads((run / "visual-evidence.json").read_text())["evidence_complete"]
    document = (run / "report.html").read_text()
    assert 'src="previews/owned%20%231%20%25.png"' in document
    assert 'src="previews/owned #1 %.png"' not in document


def test_raw_metadata_cannot_inject_html_or_svg(run):
    summary = json.loads((run / "metrics.json").read_text())
    evidence = json.loads((run / "visual-evidence.json").read_text())
    payload = '<svg onload="CANARY">&<script>CANARY</script>'
    summary["model"] = payload
    evidence["cases"][0].update(id=payload, render_status=payload, model_error=payload)
    document = visual.render_html(summary, evidence)
    assert "<svg" not in document and "<script>" not in document
    assert "&lt;svg" in document and "Content-Security-Policy" in document


@pytest.mark.parametrize("kind", ["prediction-label", "prediction-cluster", "truth-label"])
def test_rewritten_checksums_cannot_hide_changed_scoring_inputs(run, tmp_path, kind):
    cases = _cases(run)
    if kind == "truth-label":
        snapshot = json.loads((run / "manifest.snapshot.json").read_text())
        record = snapshot["records"][0]
        rows = load_intervals(record["truth"])
        rows[0]["lithology"] = "OWNED_WRONG_LABEL"
        target = tmp_path / "changed-owned-truth.csv"
        save_intervals(target, rows)
        record.update(truth=str(target), truth_sha256=sha256(target))
        (run / "manifest.snapshot.json").write_text(json.dumps(snapshot))
        summary = json.loads((run / "metrics.json").read_text())
        summary["manifest_snapshot_sha256"] = sha256(run / "manifest.snapshot.json")
        (run / "metrics.json").write_text(json.dumps(summary))
    else:
        target = run / cases[0]["prediction"]
        rows = load_intervals(target)
        if kind == "prediction-label":
            rows[0]["lithology"] = "OWNED_WRONG_LABEL"
        else:
            for row in rows:
                row["cluster_id"] = 0
        save_intervals(target, rows)
        cases[0]["prediction_sha256"] = sha256(target)
        _save_cases(run, cases)
    # Hash consistency alone accepts this internally contradictory run.
    report.load_run(run)
    before = {name: sha256(run / name) for name in IMMUTABLE}
    report.write_visual_report(run)
    evidence = json.loads((run / "visual-evidence.json").read_text())
    assert evidence["cases"][0]["status"] == "inconsistent"
    assert evidence["cases"][0]["accuracy"] is None and not evidence["evidence_complete"]
    assert "metrics differ" in " ".join(evidence["cases"][0]["errors"])
    assert before == {name: sha256(run / name) for name in IMMUTABLE}


def test_no_previews_does_not_decode_new_evidence_and_timing_includes_report(tmp_path, monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(True)
        raise AssertionError("image decode forbidden for disabled previews")

    monkeypatch.setattr(visual, "read_rgb", refuse)
    monkeypatch.setattr(evaluator, "_predict", lambda *a, **k: ([], None, 0., 0.))
    original = report.write_visual_report

    def delayed(*args, **kwargs):
        time.sleep(.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "write_visual_report", delayed)
    run, summary = evaluator.evaluate(DEMO, output=tmp_path / "runs", previews=False)
    evidence = json.loads((run / "visual-evidence.json").read_text())
    assert calls == [] and not (run / "evidence").exists()
    assert not evidence["image_generation_enabled"] and not evidence["evidence_complete"]
    assert all("disabled" in " ".join(c["errors"]) for c in evidence["cases"])
    assert summary["timings"]["visual_report_seconds"] >= .02
    assert summary["timings"]["wall_seconds"] >= summary["timings"]["visual_report_seconds"]
    assert json.loads((run / "metrics.json").read_text())["timings"] == summary["timings"]


def test_heap_pairing_matches_independent_cartesian_oracle(stress_seed, stress_iterations):
    rng = random.Random(stress_seed ^ 0xA57A)
    for index in range(stress_iterations):
        pred = set(rng.sample(range(-40, 40), rng.randrange(15)))
        truth = set(rng.sample(range(-40, 40), rng.randrange(15)))
        p, g = set(pred), set(truth)
        expected = []
        while p and g:
            pair = min((abs(a-b), a, b) for a in p for b in g)
            expected.append(pair)
            p.remove(pair[1]); g.remove(pair[2])
        assert list(visual._nearest_pairs(pred, truth)) == expected, (stress_seed, index)


def test_thousands_of_unmatched_boundaries_have_bounded_runtime():
    count = 6000
    truth = [{"top_y": i*10, "bottom_y": (i+1)*10, "lithology": "Shale"} for i in range(count)]
    pred = [{**row, "top_y": row["top_y"]+5, "bottom_y": row["bottom_y"]+5} for row in truth]
    started = time.monotonic()
    issues, counts = visual.find_issues(pred, truth, [0, count*10+5], [0, 0, 4, count*10+5], 0)
    elapsed = time.monotonic()-started
    assert counts == {"boundary_matches": 0, "predicted_boundaries": count, "truth_boundaries": count}
    assert sum(i["category"] == "shift_over_tolerance" for i in issues) == count
    assert elapsed < 5, f"6000-boundary diagnostic exceeded 5 seconds: {elapsed:.3f}s"


def test_large_one_sided_boundary_set_uses_correct_adjacent_spans():
    count = 12_000
    pred = [{"top_y": i*10, "bottom_y": (i+1)*10, "lithology": "Shale"} for i in range(count)]
    truth = [{"top_y": 0, "bottom_y": count*10, "lithology": "Shale"}]
    started = time.monotonic()
    issues, counts = visual.find_issues(pred, truth, [0, count*10], [0, 0, 4, count*10], 4)
    elapsed = time.monotonic()-started
    assert counts == {"boundary_matches": 0, "predicted_boundaries": count-1, "truth_boundaries": 0}
    assert len(issues) == count-1
    assert all(item["category"] == "false_boundary" and item["rows_affected"] == 20 for item in issues)
    assert elapsed < 5, f"12000-interval one-sided diagnostic exceeded 5 seconds: {elapsed:.3f}s"


def test_preview_pixel_budget_is_checked_before_verification_or_decode(run, monkeypatch):
    from PIL import PngImagePlugin
    target = run / _cases(run)[0]["preview"]
    data = bytearray(target.read_bytes())
    assert data[12:16] == b"IHDR"
    struct.pack_into(">II", data, 16, 50_000, 50_000)
    struct.pack_into(">I", data, 29, zlib.crc32(data[12:29]) & 0xffffffff)
    target.write_bytes(data)  # tiny existing compressed bytes; no large image is created
    original = PngImagePlugin.PngImageFile.verify
    attempted = []

    def guarded(image, *args, **kwargs):
        if image.width == 50_000:
            attempted.append(True)
            pytest.fail("overbudget preview reached PNG verification")
        return original(image, *args, **kwargs)

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "verify", guarded)
    report.write_visual_report(run)
    evidence = json.loads((run / "visual-evidence.json").read_text())
    assert not attempted and not evidence["cases"][0]["evidence_complete"]
    assert "bounded PNG" in " ".join(evidence["cases"][0]["errors"])


def test_large_tolerance_candidate_explosion_is_refused():
    rows = [{"top_y": i, "bottom_y": i+1, "lithology": "Shale"} for i in range(1000)]
    with pytest.raises(ValueError, match="candidate budget"):
        visual.find_issues(rows, rows, [0, 1000], [0, 0, 4, 1000], 10_000)


def test_csv_byte_budget_refuses_before_csv_parsing(tmp_path, monkeypatch):
    target = tmp_path / "owned-sparse.csv"
    with target.open("wb") as stream:
        stream.seek(visual.MAX_EVIDENCE_CSV_BYTES)
        stream.write(b"X")
    monkeypatch.setattr(visual, "load_intervals", lambda *a, **k: pytest.fail("oversized CSV parsed"))
    with pytest.raises(ValueError, match="CSV byte budget"):
        visual._evidence_rows(target)
