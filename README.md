# viridian402

Reproducible model experiments for the Viridien “Decoding the Subsurface” challenge. Run the same evaluator locally or in GitHub Actions, add data through manifests, and compare immutable runs with accuracy and resource measurements.

## Run an experiment

Python 3.12 is the tested environment. From this checkout:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -c requirements.lock -e ".[test]"
python -m viridian402 synthetic --output data/local/demo --seed 2026
python -m viridian402 validate --manifest data/local/demo/manifest.json
python -m viridian402 evaluate --manifest data/local/demo/manifest.json --model builtin:baseline --config configs/baseline.json --output runs
python -m viridian402 runs --output runs
python -m viridian402 report runs/REPLACE_WITH_RUN_ID
```

Each evaluation prints its run directory. It contains `summary.md`, `metrics.json`, `provenance.json`, `per_case.jsonl`, prediction CSVs, challenge submission CSVs (`base_depth`), and annotated previews. Null accuracy means the case has no truth. Baseline naming remains `Unknown` without a supplied legend or an OCR model.

## Evaluate the existing model and real labels

```sh
python scripts/datasets.py fetch evaluation
python -m viridian402 evaluate --manifest data/local/human.json --model builtin:baseline --config configs/baseline.json --output runs
python -m pip install -c requirements.lock -c requirements-legacy.lock -e ".[legacy]"
python -m viridian402 evaluate --manifest data/local/human.json --model builtin:legacy-s11 --config configs/legacy-s11.json --timeout 120 --output runs
```

The fetch command uses `GH_TOKEN`/`GITHUB_TOKEN` or local Git Credential Manager. Release assets are private, versioned and checksum-pinned in `data/datasets.lock.json`. Challenge TIFFs and model weights stay out of Git. The separate `challenge` bundle adds the remaining challenge scans for unlabelled operational checks.

## GitHub Actions

Open **Actions → Evaluate model → Run workflow**. Choose the code branch, dataset suite, model, config, input mode and timeout. Enable **legacy dependencies** for S11. Every run publishes an Actions summary and downloadable results, including failed cases. Collaborators need write access to dispatch workflows. Accuracy gates are optional because the initial baseline has substantial known limitations.

| Workflow | Trigger | Result |
|---|---|---|
| CI | Push or pull request | Windows/Linux tests, package build and demo end-to-end run |
| Evaluate model | Manual | Selected real/synthetic/stress suite, metrics, CSVs and previews |
| Release CLI | `v*` tag or manual build | Tested wheel/source archive; version tags publish a private GitHub Release |

Model evaluation produces reviewable artifacts. It does not automatically replace a winning checkpoint. Update the tracked model config in a pull request after reviewing paired results.

## Current evidence

| Suite | Cases | Interpretation |
|---|---:|---|
| `human` | 10 | Two manually labelled challenge pages; Shenzi has partial coverage |
| `organizer` | 1 | Supplied log06 example, repeatedly used for development |
| `external-dev` | 17 | Vector-derived public logs from three wells; visible development data |
| `external-diagnostic` | 8 | ForsythB2-17; used in prior tuning, so not fresh held-out evidence |
| `challenge-unlabelled` | Remaining available pages | Completion, coverage, unknown rate and resources; no accuracy |
| `synthetic`, `stress` | Generated | Real labelled interval crops with lineage; report separately from real data |

The original sealed outside tests are not in the source repo. This repository does not recreate or claim access to them.

Read [dataset and labelling instructions](docs/datasets.md), [model interface](docs/model-interface.md), [metrics and comparisons](docs/metrics.md), [CI/CD and releases](docs/ci.md), [challenge requirements and pitfalls](docs/challenge.md), and [review findings](docs/review.md).
