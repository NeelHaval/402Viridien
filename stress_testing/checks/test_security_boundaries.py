"""Local failure-containment checks, not a claim that Python plugins are sandboxed.

Only generated CANARY values/files are used. Worker payloads write at most
128 KiB of stdout or 32 KiB of JSON, never call external services, and cannot
run longer than their short hard timeout. No real credentials are printed/read.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import random
import sys

from PIL import Image
import pytest

from viridian402 import evaluate as evaluator
from viridian402.metrics import aggregate, grouped_aggregates, quality_gate_failures, score
from viridian402.report import render_summary


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    module = tmp_path / "stress_security_canary_plugin.py"
    module.write_text('''
import os, sys, time
from pathlib import Path

def denied_env(request, config):
    leaked = [name for name in config['names'] if name in os.environ]
    if leaked:
        raise RuntimeError('Generated CANARY variable names leaked: ' + ','.join(leaked))
    return []

def noisy_exit(request, config):
    for _ in range(2):
        os.write(1, b'S' * 65500)
    os.write(1, b'\\nSTRESS_LOG_TAIL_CANARY\\n')
    os._exit(23)

def limited_noise(request, config):
    for _ in range(2):
        os.write(1, b'N' * 65500)
    time.sleep(.3)
    return []

def large_result(request, config):
    return ['J' * 32768]

def small_result(request, config):
    return ['STRESS_SMALL_RESULT']

def malformed_protocol(request, config):
    Path(sys.argv[-1]).write_text(config['wire_json'], encoding='utf-8')
    os._exit(0)

def explicit_local_read(request, config):
    return [Path(config['canary_path']).read_text(encoding='utf-8')]
''', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return module.stem


def test_generated_credential_canaries_do_not_reach_worker(plugin, monkeypatch):
    names = ["STRESS_SECRET_CANARY", "GITHUB_TOKEN", "GH_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY"]
    for number, name in enumerate(names):
        monkeypatch.setenv(name, f"GENERATED_STRESS_CANARY_{number}_NOT_A_REAL_CREDENTIAL")
    result, error, seconds, rss = evaluator._predict(f"{plugin}:denied_env", {}, {"names": names}, 5)
    assert error is None
    assert result == []
    assert seconds < 8 and rss >= 0


def test_crashed_worker_log_is_read_as_a_bounded_tail(plugin, monkeypatch):
    original = Path.read_text
    def disallow_whole_log(path, *args, **kwargs):
        if path.name == "worker.log":
            pytest.fail("Crash handling attempted a whole-file worker log read")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", disallow_whole_log)
    result, error, _, _ = evaluator._predict(f"{plugin}:noisy_exit", {}, {}, 5)
    assert result is None and isinstance(error, str)
    assert "23" in error and "STRESS_LOG_TAIL_CANARY" in error
    assert len(error) < 6500


def test_noisy_worker_fails_configured_log_budget(plugin):
    result, error, seconds, _ = evaluator._predict(
        f"{plugin}:limited_noise", {}, {}, 5, max_worker_log_bytes=8192)
    assert result is None and isinstance(error, str)
    assert "log" in error.lower() and any(word in error.lower() for word in ("limit", "budget", "exceed"))
    assert seconds < 8


def test_oversized_worker_result_rejected_before_full_parse(plugin):
    result, error, seconds, _ = evaluator._predict(
        f"{plugin}:large_result", {}, {}, 5, max_worker_output_bytes=4096)
    assert result is None and isinstance(error, str)
    assert any(word in error.lower() for word in ("output", "result"))
    assert any(word in error.lower() for word in ("limit", "budget", "exceed"))
    assert seconds < 8


def test_small_worker_result_survives_tight_nonzero_budgets(plugin):
    result, error, _, _ = evaluator._predict(
        f"{plugin}:small_result", {}, {}, 5,
        max_worker_log_bytes=4096, max_worker_output_bytes=4096)
    assert error is None and result == ["STRESS_SMALL_RESULT"]


@pytest.mark.parametrize("wire", ["[]", "null", '"not-an-object"', '{"rows":"not-a-list"}', '{"error":42}', '{"error":""}',
    pytest.param('{"error":' + '[' * 5000 + '0' + ']' * 5000 + '}', id="deeply-nested-json-under-16KiB")])
def test_malformed_worker_protocol_is_a_case_error_not_parent_exception(plugin, wire):
    result, error, _, _ = evaluator._predict(f"{plugin}:malformed_protocol", {}, {"wire_json": wire}, 5)
    assert result is None and isinstance(error, str) and error


def test_valid_large_worker_error_is_bounded_before_report_propagation(plugin):
    wire = json.dumps({"error": "STRESS_ERROR_CANARY:" + "E" * 20000})
    result, error, _, _ = evaluator._predict(f"{plugin}:malformed_protocol", {}, {"wire_json": wire}, 5)
    assert result is None and isinstance(error, str)
    assert len(error) <= 12000
    assert "STRESS_ERROR_CANARY" in error or error.endswith("E" * 100)


def test_model_identifiers_are_import_data_not_shell_or_eval_code(tmp_path):
    marker = tmp_path / "STRESS_MUST_NOT_EXECUTE"
    candidates = [
        f"not_a_module;echo STRESS>{marker}:predict",
        f"os:system('echo STRESS>{marker}')",
        f"__import__('pathlib').Path({str(marker)!r}).write_text('STRESS'):predict",
    ]
    for model in candidates:
        result, error, _, _ = evaluator._predict(model, {}, {}, 5)
        assert result is None and isinstance(error, str) and error
        assert not marker.exists()


def test_seeded_private_metadata_never_enters_full_image_request(tmp_path, stress_seed, stress_iterations):
    rng = random.Random(stress_seed ^ 0x5EC)
    image, annotation = tmp_path / "public.png", tmp_path / "public-boxes.json"
    Image.new("RGB", (32, 64), (80, 120, 170)).save(image)
    boxes = {"interpreted_lithology_column": [4, 10, 20, 60], "lithology_legend": [1, 1, 31, 9]}
    for index in range(stress_iterations):
        canary = f"STRESS_PRIVATE_LABEL_{rng.getrandbits(96):024x}"
        annotation.write_text(json.dumps({"width": 32, "height": 64, "bounding_boxes": boxes,
                                         "layers": [{"target": canary}], "answers": canary, "truth": canary}), encoding="utf-8")
        record = {"id": f"public-{index}", "image": str(image), "annotation": str(annotation), "window": [10, 60],
                  "truth": canary, "split": "test", "source": canary, "tags": [canary],
                  "label_provenance": {"answer": canary}, "legend": [{"label": canary}],
                  "depth_calibration": {"private": canary}}
        request = evaluator.public_request(record, "full_image")
        assert canary not in json.dumps(request)
        assert request["annotation"] == {"bounding_boxes": boxes}
        assert request["image_size"] == [32, 64]
        assisted = evaluator.public_request(record, "supplied_legend")
        assert assisted["legend"] == record["legend"]
        assert not {"truth", "split", "source", "tags", "label_provenance"}.intersection(assisted)


def test_metadata_and_output_directories_are_not_implicitly_scanned(tmp_path):
    private = tmp_path / "generated-canary-directory"
    private.mkdir()
    (private / "generated-canary.txt").write_text("STRESS_NOT_A_REAL_SECRET", encoding="utf-8")
    config = {"description": str(private), "cache_dir": str(private), "output_dir": str(private),
              "save_dir": str(private), "log_dir": str(private), "training_groups": [str(private)]}
    assert evaluator.discover_config_assets(config) == []
    explicit = {"model": {"checkpoint": str(private / "generated-canary.txt")}}
    assert evaluator.discover_config_assets(explicit) == [(private / "generated-canary.txt").resolve()]


def test_trusted_plugin_is_not_a_filesystem_sandbox(plugin, tmp_path):
    """Positive control documents the threat boundary; it reads an owned file only."""
    canary = tmp_path / "explicitly-authorized-canary.txt"
    canary.write_text("STRESS_LOCAL_ACCESS_IS_EXPECTED", encoding="utf-8")
    result, error, _, _ = evaluator._predict(f"{plugin}:explicit_local_read", {}, {"canary_path": str(canary)}, 5)
    assert error is None and result == ["STRESS_LOCAL_ACCESS_IS_EXPECTED"]


@pytest.mark.parametrize("location", ["id", "group", "tags", "lithology", "model"])
def test_report_metadata_cannot_inject_headings_html_or_images(location):
    untrusted = "STRESSMETADATA`\n## STRESS_FORGED_RESULT\n<svg onload=STRESS_CANARY> ![STRESS_IMAGE](CANARY_IMAGE)"
    label = untrusted if location == "lithology" else "shale"
    truth = [{"top_y": 0, "bottom_y": 20, "lithology": label, "cluster_id": 0}]
    case = {"id": untrusted if location == "id" else "case", "source": "human", "status": "ok", "split": "dev",
            "group": untrusted if location == "group" else "well:fixture", "tags": [untrusted] if location == "tags" else [],
            "has_truth": True, "empty_prediction": True, "seconds": 0., "peak_rss_mb": 0.,
            "metrics": score([], truth, (0, 20)), "prediction": "predictions/safe.csv", "preview": None}
    overall = aggregate([case])
    failures = quality_gate_failures([case], overall)
    summary = {"run_id": "fixture", "model": untrusted if location == "model" else "builtin:empty", "mode": "full_image",
               "split": "dev", "overall": overall, "by": grouped_aggregates([case]), "notes": [],
               "quality_gates": {"passed": False, "failures": failures, "min_score": None}}
    rendered = render_summary(summary, [case])
    assert "stressmetadata" in rendered.casefold()
    assert "\n## stress_forged_result" not in rendered.casefold()
    assert "<svg" not in rendered.casefold()
    assert "![stress_image](canary_image)" not in rendered.casefold()


@pytest.mark.parametrize("document", [
    pytest.param('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="2"/></testsuites>', id="all-skipped"),
    pytest.param('<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0"/></testsuites>', id="failure-with-zero-exit"),
    pytest.param('<testsuites><testsuite tests="2" failures="0" errors="1" skipped="0"/></testsuites>', id="error-with-zero-exit"),
    pytest.param('<testsuites><testsuite', id="truncated-junit"),
    pytest.param(None, id="missing-junit"),
])
def test_runner_never_reports_success_without_executed_checks(tmp_path, monkeypatch, document):
    """Fault injection uses a real tiny child, not an invented successful result."""
    script = Path(__file__).resolve().parents[1] / "run_stress.py"
    spec = importlib.util.spec_from_file_location("stress_runner_failure_review", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    real_popen = runner.subprocess.Popen
    def controlled_pytest_child(command, *args, **kwargs):
        if len(command) > 2 and command[1:3] == ['-m', 'pytest']:
            destination = command[command.index('--junitxml') + 1]
            code = "from pathlib import Path\n"
            if document is not None:
                code += f"Path({destination!r}).write_text({document!r}, encoding='utf-8')\n"
            command = [sys.executable, '-c', code]
        return real_popen(command, *args, **kwargs)
    monkeypatch.setattr(runner.subprocess, "Popen", controlled_pytest_child)
    args = runner.parser().parse_args(['--suite', 'security', '--iterations', '1', '--timeout', '5', '--output', str(tmp_path / 'reports')])
    assert runner.run(args) != 0
    summary_path = next((tmp_path / 'reports').glob('*/summary.json'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['passed'] is False
    if document is not None and ('failures="1"' in document or 'errors="1"' in document):
        assert summary['status'] == 'failed'
        assert summary['counts']['failures'] + summary['counts']['errors'] == 1
    elif document is not None and document.endswith('</testsuites>'):
        assert summary['status'] == 'no_completed_tests'
        assert summary['counts']['passed'] == 0 and summary['counts']['skipped'] == 2
    else:
        assert summary['status'] == 'no_completed_tests'
        assert summary['counts'] is None
