# viridian402

Reproducible model experiments for the Viridien “Decoding the Subsurface” challenge. Run the same evaluator locally or in GitHub Actions, add data through manifests, and compare immutable runs with accuracy and resource measurements.

The [local source workspace](workspace/README.md) contains the original labellers,
generator fixes, strategy experiments, outside-source tools and saved reports.
It has a source manifest and fixed release bundles in this repository.

## Start locally

Use Python 3.12 and GitHub CLI with repository access. Get the code without
downloading the large scans and OCR weights in the full repository:

```sh
gh repo clone NeelHaval/viridian402 -- --filter=blob:none --sparse
cd viridian402
git sparse-checkout set src scripts configs docs tests examples stress_testing .github notebooks data/provenance
python scripts/local_test.py
```

The last command creates an isolated environment and checks one manufactured
example. It runs no model. Download and check two fixed outside DEV windows:

```sh
python scripts/local_test.py --suite quick-dev --check
```

To request a local evaluation, add `--run` in place of `--check`. Reports are
written under `runs/local`. Use `--software-tests` for regression and stress
checks, or `--download all --data-dir YOUR_LARGE_DATA_DIRECTORY` for the frozen
datasets and checked notices. Held-out reference labels require a separate
scorer directory and explicit opt-in.

Read [the local guide](docs/local-testing.md) for models, full DEV runs, download
sizes and permissions. The downloads are immutable and checksum-pinned. The
human diagnostic selects 15 pages and 422 boxes after the protected-well screen.
The original 27-page bundle remains exposed history. DEV auto-labelling is
disabled; no JEV or model-API authority is granted. DJS depth limits and the
130-page depth-unit/synthetic-v2 gates remain open. These checks do not establish
independent accuracy.

The older import, crop-training and dataset examples below retain their original
experiment context. Their split names do not establish current training
eligibility. Use the checked exclusion policy before any new use.

Contributing? Start with [CONTRIBUTING.md](CONTRIBUTING.md). Check readiness with `python -m viridian402 doctor --suite demo --model builtin:baseline`, copy the working model template, and use `--case-id` or `--tag` for targeted checks. `python scripts/benchmark.py --help` describes repeatable suite comparisons.

Latest: [v0.2 benchmark statistics](docs/benchmarks/2026-11-08.md) and [deep review evidence](docs/review-loop.md). The review grew the local suite from 65 to 172 passing tests and fixed eight S11 human-window crashes. Full-scan resource limits remain visible in the benchmark report.

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

PR #54's fixed real crop snapshot has its own classification experiment:

```sh
python scripts/datasets.py fetch crops --destination data/local/pr54
python -m viridian402 classify --data-root data/local/pr54 --model builtin:prototype --output runs/crops
```

This fits a small baseline using only the 907 eligible training crops and scores all 112 fixed test crops, excluding 9 quality holds. Test results retain 60 source-prospective and 52 retrospective records separately. Crop classification measures a different task from interval boundaries/depths. See [PR54 integration and crop interface](docs/pr54.md).

Each interval evaluation also writes a static [visual evidence report](docs/visual-evidence.md). Keep its relative image directories with the HTML file.

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
| Stress testing | Manual | Bounded adversarial checks, JUnit/JSON reports and optional procedural degradation probes |
| Release CLI | `v*` tag or manual build | Tested wheel/source archive; version tags publish a private GitHub Release |

Model evaluation produces reviewable artifacts. It does not automatically replace a winning checkpoint. Update the tracked model config in a pull request after reviewing paired results.

## Current evidence

For reproducible adversarial checks and model degradation runs, see the isolated [stress-testing guide](stress_testing/README.md). Start with `python stress_testing/run_stress.py --profile quick`; generated reports stay in `stress_testing/.runs/`.

| Suite | Cases | Interpretation |
|---|---:|---|
| `human` | 10 | Two manually labelled challenge pages; Shenzi has partial coverage |
| `organizer` | 1 | Supplied log06 example, repeatedly used for development |
| `external-dev` | 17 | Vector-derived public logs from three wells; visible development data |
| `external-diagnostic` | 8 | ForsythB2-17; used in prior tuning, so not fresh held-out evidence |
| `challenge-unlabelled` | Remaining available pages | Completion, coverage, unknown rate and resources; no accuracy |
| `real-crops` | 112 | PR54 fixed crop test; 907 train, 9 holds; inherited prior-exposure status |
| `synthetic`, `stress` | Generated | Real labelled interval crops with lineage; report separately from real data |

The original sealed outside tests are not in the source repo. This repository does not recreate or claim access to them.

Read [dataset and labelling instructions](docs/datasets.md), [model interface](docs/model-interface.md), [metrics and comparisons](docs/metrics.md), [CI/CD and releases](docs/ci.md), [challenge requirements and pitfalls](docs/challenge.md), [review findings](docs/review.md), and [executed verification](docs/verification.md).

For the bounded notebook rule, see [candidate adapters](docs/candidate-adapters.md). For local reviewed-window exports, see [window exchange](docs/review-window-exchange.md). The five exposed windows are development evidence. Their local bundle remains unlabelled and has an unresolved parent split conflict.
