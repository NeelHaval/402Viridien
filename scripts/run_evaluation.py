"""Workflow entry point. Inputs travel as environment data, never shell code."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from viridian402.evaluate import evaluate
from viridian402.synthetic import generate_demo, generate_synthetic

SUITES = ("demo", "human", "organizer", "external-dev", "external-diagnostic",
          "challenge-unlabelled", "synthetic", "stress", "all")


def main():
    suite = os.environ.get("EVAL_SUITE", "demo")
    if suite not in SUITES:
        raise ValueError("Unknown evaluation suite")
    model = os.environ.get("EVAL_MODEL", "builtin:baseline")
    config_path = os.environ.get("EVAL_CONFIG", "configs/baseline.json")
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    if not config_file.is_relative_to(root) or not config_file.is_file():
        raise ValueError("Config must be a JSON file inside this checkout")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Model config must be a JSON object")
    timeout = float(os.environ.get("EVAL_TIMEOUT", "60"))
    if not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError("Per-case timeout must be between 1 and 600 seconds")
    mode = os.environ.get("EVAL_MODE", "full_image")
    if mode not in ("full_image", "supplied_legend"):
        raise ValueError("Invalid input mode")
    minimum = os.environ.get("EVAL_MIN_SCORE", "").strip()
    minimum = float(minimum) if minimum else None
    if minimum is not None and (not math.isfinite(minimum) or not 0 <= minimum <= 1):
        raise ValueError("Minimum score must be between 0 and 1")
    selected = ["human", "organizer", "external-dev", "external-diagnostic"] if suite == "all" else [suite]
    summaries = []
    passed = True
    for name in selected:
        if name == "demo":
            manifest = generate_demo("data/local/demo", seed=2026, n=3)
        elif name in ("synthetic", "stress"):
            manifest = generate_synthetic("data/local/human.json", f"data/local/{name}",
                                          seed=2026, n=3, height=1600,
                                          stress="noise" if name == "stress" else "clean")
        else:
            manifest = Path("data/local") / f"{name}.json"
        run, summary = evaluate(manifest, model=model, config=config,
                                output=Path("runs") / name, mode=mode, timeout=timeout,
                                min_score=minimum, min_completion=1, previews=True)
        passed = passed and summary["quality_gates"]["passed"]
        summaries.append(f"## {name}\n\n" + (run / "summary.md").read_text(encoding="utf-8"))
        print(f"{name}: {run}")
    combined = "\n\n".join(summaries)
    Path("runs/workflow-summary.md").write_text(combined, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(combined)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
