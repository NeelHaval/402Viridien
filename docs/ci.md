# CI/CD and GitHub operation

GitHub Actions uses the same Python entry points as local evaluation. Action implementations are SHA-pinned, default token permissions are read-only, and checkout does not persist credentials. PR CI fetches no private datasets/secrets. Manual evaluation downloads only pinned private release assets with the job's read token; token scope is limited to the fetch step.

## Run a selected suite

On the Actions **Evaluate model** page choose a branch with your model code, then select a suite. `all` covers the four supervised suites. `synthetic` and `stress` derive cases from reviewed human interval crops. `challenge-unlabelled` runs the larger remaining-scan suite for robustness. Choose `builtin:legacy-s11` and `configs/legacy-s11.json` and enable legacy dependencies to exercise the original model. Custom `module:function` models must be importable from the installed checkout, with dependencies already declared.

Inputs are passed through environment variables and parsed as data in `scripts/run_evaluation.py`. They are not inserted into shell commands. Per-case timeout is restricted to 1–600 seconds and the whole job to 90 minutes. Runtime depends on source image decoding and OCR. A per-case failure makes the default completion gate fail but still saves diagnostics. The artifact upload uses `always()` so completed cases survive quality-gate failures; cancelling the entire runner can still interrupt artifact generation.

Use the optional **manifest** input for a tracked or bundle-provided interval manifest inside the checkout. The **suite** still chooses the default data download. Use **selection** for a JSON object such as `{"tags":["partial_truth"],"max_worker_rss_mb":2048}`. Supported keys are `case_ids`, `tags`, `split` and `max_worker_rss_mb`. The arrays use exact strings; OR applies within each array and AND between filters. Unknown IDs/tags and empty selections fail before inference. Full-manifest leakage checks still run, and saved fingerprints identify the selected records. The RSS limit observes the process tree at approximately 25 ms intervals; it is not a hard memory allocator cap.

For a new versioned data bundle, add its immutable release asset entry to `data/datasets.lock.json`, then set **data_bundle** to that key and **manifest** to its extracted manifest under `data/local/`. The input selects trusted lock entries, never an arbitrary download URL. Overrides apply to one interval suite; `all` cannot take one manifest override, and `real-crops` keeps its fixed classification test set intact. Model/config inputs remain checkout-local. There are no extra repository tokens in model workers.

Download `evaluation-RUN_ID-ATTEMPT` from the run's **Artifacts** section. The Actions summary has the metrics; open downloaded `summary.md` for relative links to CSVs and images. Results are retained for 30 days. Download important runs for permanent comparison. The CLI can compare two downloaded run directories directly. The `real-crops` suite uses the separate PR54 fixed classification split and metrics, described in [PR54 integration](pr54.md).

The equivalent local workflow command is:

```sh
python scripts/datasets.py fetch evaluation
# Set EVAL_SUITE, EVAL_MODEL, EVAL_CONFIG, EVAL_MODE, EVAL_TIMEOUT if changing defaults.
python scripts/run_evaluation.py
```

With the same environment inputs, `python scripts/run_evaluation.py --fetch-data` performs just the pinned data download. Actions runs that command in a separate step containing the read token; the evaluation step receives no token. Default downloads are none for `demo`, `crops` for `real-crops`, `evaluation` plus `challenge` for `challenge-unlabelled`, and `evaluation` for the other suites.

Examples in PowerShell:

```powershell
$env:EVAL_SUITE = 'human'
$env:EVAL_MODEL = 'builtin:legacy-s11'
$env:EVAL_CONFIG = 'configs/legacy-s11.json'
python scripts/run_evaluation.py
```

For a targeted interval debug run:

```powershell
$env:EVAL_SELECTION = '{"tags":["partial_truth"],"max_worker_rss_mb":2048}'
python scripts/run_evaluation.py
```

The equivalent CLI supports repeated `--case-id` and `--tag` flags, plus `--max-worker-rss-mb`. Remove stale environment overrides before switching suites. Run `python -m viridian402 doctor` for environment readiness or add `--suite human --model your_package.model:predict --config configs/experiment.json` to audit the selected dataset locally. Doctor uses no network and does not execute model plugins. See the complete [contributor path](../CONTRIBUTING.md).

## Release and update models

CI tests on Windows and Linux, builds wheel/source archives and runs the synthetic smoke path. A `v0.1.0` style tag runs the same gates and then publishes a private GitHub Release with the tested distributions. Manual **Release CLI** runs produce a distribution artifact without publishing a release. No PyPI token, production server or deployment secret is needed.

Review a candidate model's same-manifest paired comparison, failures and overlays. Commit its code/config and checkpoint reference through a PR, then dispatch evaluation against that branch. Merge after reviewing the result. The pipeline deploys versioned evaluation software and result artifacts; it does not automatically promote trained weights into another service.

## Efficient iteration

Use demo/targeted window manifests on PRs, and manual real-data evaluations after model changes. Python dependency wheels are cached; expensive paid/GPU models are opt-in extensions. Separate evaluation and challenge scan bundles avoid a gigabyte-scale download for every smoke test. Run outputs never pollute caches used for model answers. Fingerprints include dataset contents and model/config assets so old results remain attributable.

For more performance, profile decode/OCR/startup costs first, then batch model initialization or cache immutable image-derived features by image hash, code version, preprocessing config and checkpoint. Add tests proving cache invalidation. Preserve identical windows and hardware when comparing speed. Self-hosted GPU runners need an explicitly reviewed workflow and resource/cost instrumentation before use.

New interval data can use the existing manifest/selection/data-bundle inputs without adding a pipeline path. Add a named suite choice only when a repeated team workflow merits a convenient preset. A custom local manifest works immediately with `viridian402 evaluate`. Keep existing frozen manifests and release archives unchanged when publishing a new dataset version.

Manual dispatch and permission details follow the [GitHub workflow syntax documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
