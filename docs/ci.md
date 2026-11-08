# CI/CD and GitHub operation

GitHub Actions uses the same Python entry points as local evaluation. Action implementations are SHA-pinned, default token permissions are read-only, and checkout does not persist credentials. PR CI fetches no private datasets/secrets. Manual evaluation downloads only pinned private release assets with the job's read token; token scope is limited to the fetch step.

## Run a selected suite

On the Actions **Evaluate model** page choose a branch with your model code, then select a suite. `all` covers the four supervised suites. `synthetic` and `stress` derive cases from reviewed human interval crops. `challenge-unlabelled` runs the larger remaining-scan suite for robustness. Choose `builtin:legacy-s11` and `configs/legacy-s11.json` and enable legacy dependencies to exercise the original model. Custom `module:function` models must be importable from the installed checkout, with dependencies already declared.

Inputs are passed through environment variables and parsed as data in `scripts/run_evaluation.py`. They are not inserted into shell commands. Per-case timeout is restricted to 1–600 seconds and the whole job to 90 minutes. Runtime depends on source image decoding and OCR. A per-case failure makes the default completion gate fail but still saves diagnostics. The artifact upload uses `always()` so completed cases survive quality-gate failures; cancelling the entire runner can still interrupt artifact generation.

Download `evaluation-RUN_ID-ATTEMPT` from the run's **Artifacts** section. The Actions summary has the metrics; open downloaded `summary.md` for relative links to CSVs and images. Results are retained for 30 days. Download important runs for permanent comparison. The CLI can compare two downloaded run directories directly. The `real-crops` suite uses the separate PR54 fixed classification split and metrics, described in [PR54 integration](pr54.md).

The equivalent local workflow command is:

```sh
python scripts/datasets.py fetch evaluation
# Set EVAL_SUITE, EVAL_MODEL, EVAL_CONFIG, EVAL_MODE, EVAL_TIMEOUT if changing defaults.
python scripts/run_evaluation.py
```

Examples in PowerShell:

```powershell
$env:EVAL_SUITE = 'human'
$env:EVAL_MODEL = 'builtin:legacy-s11'
$env:EVAL_CONFIG = 'configs/legacy-s11.json'
python scripts/run_evaluation.py
```

## Release and update models

CI tests on Windows and Linux, builds wheel/source archives and runs the synthetic smoke path. A `v0.1.0` style tag runs the same gates and then publishes a private GitHub Release with the tested distributions. Manual **Release CLI** runs produce a distribution artifact without publishing a release. No PyPI token, production server or deployment secret is needed.

Review a candidate model's same-manifest paired comparison, failures and overlays. Commit its code/config and checkpoint reference through a PR, then dispatch evaluation against that branch. Merge after reviewing the result. The pipeline deploys versioned evaluation software and result artifacts; it does not automatically promote trained weights into another service.

## Efficient iteration

Use demo/targeted window manifests on PRs, and manual real-data evaluations after model changes. Python dependency wheels are cached; expensive paid/GPU models are opt-in extensions. Separate evaluation and challenge scan bundles avoid a gigabyte-scale download for every smoke test. Run outputs never pollute caches used for model answers. Fingerprints include dataset contents and model/config assets so old results remain attributable.

For more performance, profile decode/OCR/startup costs first, then batch model initialization or cache immutable image-derived features by image hash, code version, preprocessing config and checkpoint. Add tests proving cache invalidation. Preserve identical windows and hardware when comparing speed. Self-hosted GPU runners need an explicitly reviewed workflow and resource/cost instrumentation before use.

Adding a suite requires a new manifest, versioned bundle/lock entry if its assets are private, and the allowlist/choice in `scripts/run_evaluation.py` and `.github/workflows/evaluate.yml`. A custom local manifest works immediately with `viridian402 evaluate`; this avoids building a second orchestration system.

Manual dispatch and permission details follow the [GitHub workflow syntax documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
