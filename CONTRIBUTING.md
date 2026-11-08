# Contributing experiments

Use one branch for one model or data change. Keep source code, configuration and the evidence needed to reproduce the result together.

## Clean checkout and fast checks

```sh
git clone https://github.com/NeelHaval/viridian402.git
cd viridian402
git switch -c model/my-experiment
```

Create the tested Python 3.12 environment. In PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

Then, in either shell:

```sh
python -m pip install -c requirements.lock -e ".[test]"
python -m viridian402 doctor --suite demo --model builtin:baseline --config configs/baseline.json
python -m pytest -q
python -m viridian402 evaluate --manifest examples/demo/manifest.json --model builtin:baseline --output runs/demo
```

`doctor` uses local files and dependency imports. It neither contacts services nor executes your model. Missing optional real-data bundles are informational; selecting a missing suite makes readiness fail and prints its fetch command. Exit codes are 0 for ready, 1 for missing prerequisites and 2 for invalid arguments. Other commands use 1 for a failed evaluation gate and 2 for invalid input.

## Add an interval model

Start with a working model template. These commands work in both shells:

```sh
python -c "from pathlib import Path; import shutil; Path('src/my_model').mkdir(); shutil.copyfile('examples/models/template_model.py', 'src/my_model/__init__.py')"
python -m pip install -c requirements.lock -e ".[test]"
python -m viridian402 doctor --suite demo --model my_model:predict --config configs/baseline.json
python -m viridian402 evaluate --manifest examples/demo/manifest.json --model my_model:predict --config configs/baseline.json --output runs/my-model
```

Replace the template's `predict(request, config)` implementation. The request contains an image path, public boxes, dimensions and a full-image row window. Return sorted, disjoint positive-height rows with `top_y`, `bottom_y`, `lithology` and integer `cluster_id`. Both depth endpoints are optional together. Use the page legend's label or `Unknown`. See the complete [model contract](docs/model-interface.md).

Put options in a new tracked `configs/my-model.json`. Record all training/tuning well groups as `training_groups`, including PR54 crop groups if used. Checkpoint bytes stay outside Git; pin their revision/checksum and retain the training-data history. Add tests for the behavior you changed, then run `python -m pytest -q`. A model can access local files because plugins are trusted team code; never read truth or fit on evaluation cases inside `predict`.

## Test the real data and compare

Use local Git Credential Manager, or an existing `GH_TOKEN` with read access to this private repository. The downloader verifies the versioned release size and SHA-256.

```sh
python scripts/datasets.py fetch evaluation
python -m viridian402 doctor --suite human --model my_model:predict --config configs/my-model.json
python -m viridian402 evaluate --manifest data/local/human.json --model builtin:baseline --config configs/baseline.json --output runs/human
python -m viridian402 evaluate --manifest data/local/human.json --model my_model:predict --config configs/my-model.json --output runs/human
```

For fast debugging, repeat `--case-id EXACT_RECORD_ID` or `--tag TAG`. Values within a filter are ORed; case/tag/split filters are combined with AND. For example, `--tag partial_truth` selects the Shenzi windows. Unknown values and an empty selection fail before inference. The complete manifest still undergoes leakage validation. `--no-previews` skips rendering; `--max-worker-rss-mb 2048` stops workers whose sampled process-tree RSS exceeds that MiB threshold. Sampling can miss short peaks, so this is not an allocator cap.

Each evaluation prints its saved run path. Substitute the two paths below:

```sh
python -m viridian402 compare runs/human/BASELINE_ID runs/human/CANDIDATE_ID --output runs/paired.json
python -m viridian402 report runs/human/CANDIDATE_ID
```

Use identical selected cases and input modes for comparisons. Inspect failures, class/group results, runtime and overlays as well as the headline score. A default passing gate confirms execution and nonempty output; set `--min-score` when you want an accuracy threshold. The published external diagnostic subset was previously tuned and is not a fresh holdout.

Crop classification has a separate fixed split and output contract:

```sh
python scripts/datasets.py fetch crops --destination data/local/pr54
python -m viridian402 doctor --suite real-crops --model builtin:prototype
python -m viridian402 classify --data-root data/local/pr54 --model builtin:prototype --output runs/crops
```

Keep its 907 usable training crops, 112 test crops and 9 quality holds separate. A crop model returns exact label-key strings, including multiple labels when appropriate. Crop classification scores do not measure interval boundaries, depths or bed thickness. Read [PR54 integration](docs/pr54.md) before changing that task.

## Add data

Prepare a new directory such as `data/local/my-suite/` containing its own portable manifest, images, bounding-box annotations and truth CSVs. Follow [the dataset schema and labelling rules](docs/datasets.md). Each record needs a stable well `group`, split, source, explicit scoring window and label provenance. Labelled records also declare `truth_window`; null truth means no accuracy. Synthetic derivatives inherit the parent split/group. Keep incomplete and disputed labels explicit.

```sh
python -m viridian402 validate --manifest data/local/my-suite/manifest.json
python -m viridian402 doctor --manifest data/local/my-suite/manifest.json
python -m viridian402 evaluate --manifest data/local/my-suite/manifest.json --model my_model:predict --config configs/my-model.json --output runs/my-suite
python scripts/datasets.py pack data/local/my-suite/manifest.json --output data/downloads/my-suite-v1.zip
```

Packing requires all referenced files under the manifest directory. It prints the archive checksum and size. Preserve attribution and source metadata there. Ask the dataset maintainer to publish a versioned private release asset and add its immutable ID, size and SHA-256 to `data/datasets.lock.json`. Keep existing snapshots unchanged. Small deterministic fixtures can go in `examples/`; source scans, downloaded archives and weights stay ignored.

## Run the same experiment in Actions

Open **Actions â†’ Evaluate model â†’ Run workflow** and choose your branch, model, config and suite. Optional **manifest** points to a tracked or bundle-provided JSON inside the checkout. Optional **selection** accepts JSON such as `{"tags":["partial_truth"],"max_worker_rss_mb":2048}`. For a new pinned bundle, set **data_bundle** to its lock key and **manifest** to its extracted manifest; no new workflow code is needed. These overrides apply to interval suites. The fixed crop suite retains all test records.

Download the result artifact and use the same CLI comparison locally. In your pull request, include the problem/result, exact commands, test status on Windows/Linux, data exposure, paired run IDs/metrics and remaining limits. The [PR template](.github/pull_request_template.md) lists the evidence to report. CI checks source/demo behavior without private-data downloads; real evaluations are deliberate manual runs.
