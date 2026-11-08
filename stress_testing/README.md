# Stress testing Viridian402

For recorded results and discovered defects, see [executed verification](VERIFICATION.md)
and its [machine-readable evidence](verification.json).
The application review began with [main-0072114](RETEST-0072114.md) and continues
with [the newly merged b790a6b code](RETEST-b790a6b.md).

This directory contains opt-in tests for data validation, scoring, saved results,
worker failures, archive handling, and notebook helper contracts. Two Python
entry points separate **project robustness** from **model quality under image
degradation**. Every invocation creates a new report directory. Nothing here
trains a model, fetches datasets, or automatically changes a checkpoint.

## Start here

Use a checkout of this repository and Python 3.12, matching GitHub Actions.
Python 3.13 also works in the local verification environment. From the repo root:

```sh
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -c requirements.lock -c requirements-legacy.lock -e ".[test,review]"
python stress_testing/run_stress.py --profile quick
python stress_testing/run_model_stress.py --profile quick
```

The first command tests the software's contracts with generated local fixtures.
The second evaluates the baseline on nine constructed image degradations. A
passing robustness run means assertions passed; a passing model run without a
score threshold means operational checks passed. Neither establishes accuracy
on unseen real scans.

Core stress checks need only `.[test]`; `.[review]` enables the additional sklearn
oracle and TIFF checks in the normal regression suite. Missing optional
capabilities are recorded as skips, including unavailable OS symlink privileges.
The runners never install dependencies themselves. Installation requires network
access; subsequent provided checks use local fixtures only.

## Defined location and isolation

```text
stress_testing/
  README.md                     this guide and interpretation rules
  run_stress.py                 bounded pytest process-tree runner
  run_model_stress.py           bounded nine-degradation experiment runner
  run_frontend_checks.cjs       offline browser/API-contract probes
  requirements-demo.lock       resolved Python 3.12 demo/stress environment
  checks/
    conftest.py                 seeded loop controls
    test_contracts.py           manifests, truth, boxes, leakage and lineage
    test_metrics.py             independent scoring oracles and large coordinates
    test_results.py             saved-run integrity and paired comparison
    test_workers.py             crash, timeout, memory and child cleanup
    test_archives.py            hostile ZIP metadata and safe packaging
    test_security_boundaries.py protocol, logs, canaries and report spoofing
    test_notebook.py            static cells and selected pure helper checks
    test_harness.py             runner reporting and command-line regressions
    test_adapter_hardening.py   finite component features and candidate limits
    test_visual_hardening.py    report paths, integrity, budgets and pairing
    test_exchange_hardening.py  reviewed labels, provenance and download policy
    test_demo_config.py         budgets and bounded pooled reports
    test_demo_pipeline.py       strict output, thin beds and feature edge cases
    test_demo_services.py       concurrency, immutable inputs, queue and storage
    test_demo_security.py       HTTP bodies, bbox, polygons and cleanup
  .runs/                        ignored, generated artifacts; never commit these
```

Normal `python -m pytest -q` still collects `tests/` only. Production code does
not import this directory. You can omit/remove `stress_testing/` to isolate the
optional harness; retain production validation fixes and their normal tests.
Source archives include the scripts and guide, excluding `.runs/`. Use a source
checkout/archive for these commands; a wheel contains the application package.
The demo checks require a full Git checkout containing `demo/`.

Both entry points accept an absolute script path from another working directory.
`--manifest`, `--config`, and explicit relative `--output` paths resolve against
your current working directory. The default output always lives beside these
scripts. The project runner executes pytest from the repository root.

## Software robustness runs

```sh
# All isolated checks, reproducible defaults
python stress_testing/run_stress.py --profile quick --seed 20261108

# More randomized examples, plus the normal regression suite
python stress_testing/run_stress.py --profile extended --include-regressions

# Investigate one area without rerunning everything
python stress_testing/run_stress.py --suite archives --suite security --iterations 100
python stress_testing/run_stress.py --suite metrics --suite results --seed 9122026 --profile soak

# Bound a local investigation; stop at the first failed assertion
python stress_testing/run_stress.py --profile extended --timeout 120 --max-rss-mb 768 --fail-fast
```

| Profile | Iterations per seeded property loop | Whole pytest deadline |
| --- | ---: | ---: |
| `quick` (default) | 50 | 180 seconds |
| `extended` | 500 | 600 seconds |
| `soak` | 3,000 | 1,200 seconds |

Iterations apply to randomized contract/oracle loops, not to every test. Worker
and archive regression probes run once per invocation. `--iterations` accepts
1–10,000, `--seed` 0–4,294,967,295, and `--timeout` 1–1,800 seconds. Default
sampled pytest process-tree RSS ceiling is 1,536 MiB (`--max-rss-mb` 64–8,192).
The pytest log ceiling is 16 MiB. Exceeding a runner budget fails the run and
retains partial artifacts; it is not counted as a successful stress result.

Repeat `--suite` to combine `data`, `metrics`, `results`, `workers`, `archives`,
`security`, `notebook`, `harness`, `adapter`, `visual`, or `exchange`.
Omit it, or select `all`, for all core checks. The adapter suite requires SciPy;
use the review extra or the demo environment to execute it rather than skip it.
The runner records the interpreter, package versions, seed, settings, elapsed
time, sampled peak RSS, Git commit/dirty status, and source SHA-256 fingerprints.
Changing source files during a run invalidates it (`source_changed`); rerun after
finishing edits. Unconfirmed descendant cleanup also prevents a passing status.
Use these together when reproducing a result from uncommitted source changes.

| Area | What a passing test demonstrates |
| --- | --- |
| Data | Malformed shapes/types, overlapping/reversed intervals, invalid boxes, image/annotation mismatch, duplicate bytes across splits and synthetic lineage are handled as asserted. |
| Metrics | Interval scores agree with independent dense-pixel and enumerated-pair oracles on small bounded cases; coordinates above 2^53 retain integer boundaries; gaps/unknown labels do not receive invented accuracy. |
| Results | Tampered aggregates, removed class tables, changed artifacts, escaped paths and incompatible paired comparisons are rejected; depth support changes remain visible. |
| Workers | Import failures, exceptions, nonzero exits, timeout and bounded memory pressure become case errors; observed child processes are cleaned up. |
| Archives | Traversal, absolute/device names, duplicate destinations, corrupt/truncated members, symlinks and oversized declarations are rejected without committing staged content. |
| Security boundaries | Invalid/deep protocol JSON, oversized logs/results, report metadata injection and accidental private-label forwarding are exercised using generated canaries. A positive control confirms plugins can read explicitly supplied local files. |
| Notebook | Cells parse and selected pure geometry/classification/text helpers meet their checked examples; training and top-level notebook code are never executed. |
| Harness | Bad settings, malformed JUnit and the actual metrics-only CLI produce consistent results without recursive full-suite execution. |

Hostile inputs are **expected** in these tests. For example, a traversal ZIP
test passes when extraction refuses the archive and the owned outside canary
remains untouched. It does not pass by successfully extracting hostile content.
There are no deliberately hidden expected failures used to make the suite green.

### Reading robustness artifacts

Each `stress_testing/.runs/<UTC timestamp>-<unique suffix>/` contains:

| Artifact | Purpose |
| --- | --- |
| `summary.md` | Quick human-readable status, counts and resource use. |
| `summary.json` | Machine-readable pass/fail, settings, counts, versions and limits. |
| `provenance.json` | Git commit/dirty state and source-file fingerprints before/after execution. |
| `junit.xml` | Individual assertion results, failure tracebacks and skip reasons. May be incomplete after forced termination. |
| `pytest.log` | Collection, assertions and diagnostic output. |
| `work/` | Isolated temporary fixtures for investigation; may contain intentional malformed files. |

`counts` is null if no valid complete JUnit report exists. A zero-test or all-skip
run fails; a partial set of skips is visible alongside executed passes. Check
`skipped` and its reasons before claiming coverage of a particular OS feature.
RSS is a sampled sum of the pytest process tree, not a precise allocator maximum.

Exit codes: **0** assertions passed with at least one passing check; **1** test,
collection, log or RSS failure; **2** invalid CLI/setup; **124** runner timeout;
**130** interruption. The summary distinguishes these statuses. Reports are
retained intentionally. To reclaim disk space, remove only the generated `.runs/`
directory after saving evidence you need; the scripts do not delete past runs.

## Model degradation experiments

```sh
# One case for each degradation, using constructed procedural patterns
python stress_testing/run_model_stress.py --profile quick

# Three per degradation, assisted legend diagnostic (report separately)
python stress_testing/run_model_stress.py --profile extended --mode supplied_legend --height 800

# Existing labelled real interval crops, without fetching data
python stress_testing/run_model_stress.py --profile quick --manifest data/local/human.json --model builtin:baseline --config configs/baseline.json

# Example experiment gate: choose the threshold from your frozen baseline
python stress_testing/run_model_stress.py --profile quick --mode supplied_legend --min-score 0.20 --previews
```

The nine degradations are `clean`, `blur`, `noise`, `low-contrast`, `jpeg`,
`horizontal-shear`, `downsample`, `occlusion`, and `mixed`. The same source and
seed across degradations aid comparison. Output height defaults to 800 pixels;
64–1,600 is allowed. Quick/extended/soak generate 1/3/10 cases **per degradation**
(9/27/90 total). `--cases` overrides this with a hard total cap of 100 (maximum
11 per degradation). This is bounded experimental stress, not an unlimited load
generator.

Defaults are `builtin:baseline`, `full_image`, seed 2026, 10-second worker timeout
and 512 MiB sampled worker RSS limit. `--timeout` accepts 0.1–60 seconds;
`--max-worker-rss-mb` must be finite and positive. Seed validation reserves room
for per-case offsets. Use `--model your_module:predict --config your_config.json`
for a trusted experiment adapter; see [the model interface](../docs/model-interface.md).
Install optional legacy dependencies separately before selecting `builtin:legacy-s11`.

Without `--manifest`, data is explicitly marked `procedural_fixture`. Labels come
from its construction. The unaided baseline normally emits `Unknown`, so its
supervised score can be zero even when all operational checks pass. With a real
manifest, existing labelled nonsynthetic parents supply interval patterns; their
image/truth fingerprints, parent ID, well group and split lineage are retained.
Derivatives belong with their parents and must not leak into a different split.
Unlabelled-only manifests are rejected for this labelled degradation experiment.
Use the normal evaluator for unlabelled completion/resource checks instead.

Each run writes `request.json`, incremental `degradations.json`, `summary.json`,
`summary.md`, generated `datasets/` and normal `evaluations/` artifacts. Per-case
JSONL, prediction CSVs, metric tables and optional previews link back to each
degradation. The normal evaluator records model/scorer/config provenance too.
Operational status and optional accuracy status are separate:

- `operational_passed`: all requested cases execute and satisfy the evaluator's
  completion/nonempty-output checks.
- `accuracy_passed`: null unless `--min-score` is supplied; otherwise every
  degradation must meet its mean interval score threshold (0–1).
- `passed`: operational success and no configured accuracy failure. CLI exit 0
means this combined condition, 1 means experiment/generation/evaluation failure,
and 2 means argument or configuration parsing error. Evaluation preflight failures
are retained per degradation and exit 1.

Compare row accuracy, boundary F1, score, per-class support, coverage, unknown
rate, latency and sampled RSS together. A fast model returning `Unknown` is not
an accurate classifier. `full_image` and `supplied_legend` are distinct tasks;
do not mix them in a leaderboard. Construction tests do not replace frozen human
labels or independent held-out wells. See [dataset guidance](../docs/datasets.md)
and [metric definitions](../docs/metrics.md).

## Test the current backend and frontend

The core `all` selection covers the original evaluation package. Add
`--include-demo` to include the application's normal backend tests and all four
demo adversarial suites. This is explicit because the app requires NumPy <2,
while the core CPU lock uses NumPy 2. Use a **separate Python 3.12 environment**:

```sh
python3.12 -m venv .venv-demo
# PowerShell: .venv-demo\Scripts\Activate.ps1
# Linux/macOS: source .venv-demo/bin/activate
python -m pip install -r stress_testing/requirements-demo.lock
python stress_testing/run_stress.py --profile quick --include-regressions --include-demo
python stress_testing/run_stress.py --profile extended --include-regressions --include-demo

# Focus on one app boundary without running the whole project
python stress_testing/run_stress.py --suite demo-security --iterations 50
python stress_testing/run_stress.py --suite demo-services --profile soak
```

`demo-config`, `demo-pipeline`, `demo-services` and `demo-security` can be
selected individually. Backend package versions are recorded alongside the
other run metadata. The normal backend tests use scripted OCR and synthetic
scans; they do not establish learned-model accuracy. Their real-OCR test is
excluded by default. To opt into the existing real-data crop smoke test:

```sh
python stress_testing/run_stress.py --profile extended --include-regressions --include-demo --include-real-model
```

That flag requires `--include-demo`. The test can skip when local model/data
assets are unavailable; inspect JUnit before claiming it ran. It exercises one
real crop, not every challenge scan or a held-out accuracy dataset. The runner's
whole-process deadline and sampled RSS limits still apply. Some restricted
environments block asyncio's internal socket pair: API TestClient may then hang
before app startup. Enable local loopback for the test process in that case.

The browser probe uses a built frontend, a local ephemeral HTTP server and a
mocked API. All external requests are intercepted; the map gets an empty local
style. It checks empty state, dialog keyboard behavior, connection/upload
errors, queued-to-ready polling, CSV links, metadata escaping and response races:

```sh
npm ci --prefix demo/frontend
npm run typecheck --prefix demo/frontend
npm run build --prefix demo/frontend
npm install --prefix .local/browser --no-save --package-lock=false --ignore-scripts playwright@1.62.1
node .local/browser/node_modules/playwright/cli.js install chromium
# Replace /absolute/path/to/repo with your checkout's absolute path.
node stress_testing/run_frontend_checks.cjs --playwright-module /absolute/path/to/repo/.local/browser/node_modules/playwright
```

You can use an installed Chrome with `--browser-executable /absolute/path/to/chrome`
instead of downloading Chromium. JSON results, built/source fingerprints and
screenshots stay under `.runs/frontend-*`. This browser test mocks backend
responses; the separate Python suites exercise the real API. Browser timings
are diagnostics, not production performance benchmarks.

The CI `demo-checks` job now runs the combined quick suite, TypeScript/build and
browser probes. The manual stress workflow has an **include demo** option.
Neither workflow runs the real-OCR test or trains a model by default.

### Application contracts exercised

- Identical scan bytes and identical bbox input are idempotent. A different bbox
  for an existing scan returns **409 input_conflict**, preserving the original
  job. Concurrent submissions cannot overwrite its inputs.
- At most **32 running/queued jobs** are admitted by default; set
  `APP_MAX_PENDING_JOBS` to another positive integer. Capacity rejection is
  **503 queue_busy** before persistent publication. Completion, cancellation,
  preparation failures and shutdown release reservations correctly.
- Oversized streamed request bodies return **413** before multipart spooling
  grows beyond the configured total; individual file/bbox limits also apply.
  Malformed/deep/nonfinite bbox JSON and invalid geometry return **400**.
- Pixels and cluster IDs must be finite signed-64-bit integers; pixels must be
  nonnegative, labels nonempty text, and beds strictly positive/nonoverlapping.
  Thin beds use calibrated precision when decimal rounding would erase them.
- Stored report data must be finite. Pooled columns are limited to 10,000 bins
  and 5 million row/bin pairs; excessive or invalid stored reports return
  **409 invalid_result** with instructions to select fewer wells or reprocess.
  Polygon input is finite, nondegenerate and capped at 10,000 vertices.

Queue admission bounds job count, not all application memory. Image validation
still precedes admission; native image readers retain their existing large
pixel/cache caps. The service has per-stage OCR budgets, but no hard whole-job
memory/timeout sandbox. The documented single-worker deployment remains required.
The app tests do not establish safe public hosting or complete vulnerability
coverage.

## Adding a focused test or dataset

1. Put a new adversarial assertion in the appropriate `checks/test_*.py` file.
   Use pytest's `tmp_path`, tiny generated images/archives and synthetic canaries.
   Never use real credentials, other users' files or remote targets as fixtures.
2. For randomized checks, use `random.Random(stress_seed)` or an independent
   bounded NumPy generator, and loop over `stress_iterations`. Keep allocation
   and image sizes bounded regardless of profile. Include the seed in failures.
3. Assert both the expected exception/error and absence of unintended side
   effects. For scoring changes, use an independent small oracle rather than
   copying the implementation being tested. Do not silently `xfail` a defect.
4. Run the narrow suite, then extended checks and normal regressions. Include
   exact commands, report status/counts/skips, seed, provenance and a small
   reproducer in the PR. A bug report should state expected vs observed behavior
   and distinguish correctness, resource exhaustion, leakage and formatting.
5. For new labelled data, follow [the manifest/CSV guide](../docs/datasets.md).
   Keep well groups intact, record label provenance, resolve disagreements and
   version the manifest/bundle before drawing improvement claims. Add hard cases
   to a diagnostic split first; tuning on them makes them development evidence.

Minimal deterministic property-test structure:

```python
import random

def test_my_contract(tmp_path, stress_seed, stress_iterations):
    rng = random.Random(stress_seed ^ 0x402)
    for index in range(stress_iterations):
        height = rng.randint(8, 64)
        # Construct an owned fixture and assert its independent contract here.
        assert 8 <= height <= 64, (stress_seed, index)
```

For GitHub Actions, **Stress testing → Run workflow** runs these same scripts
without private data or repository secrets. Pick quick or extended for review;
soak is intended for deliberate longer investigations. Artifacts upload even
when checks fail. Large real/legacy model experiments remain an explicit local
or Evaluate model workflow choice.

## Findings and limits

This review added reproductions and fixes for malformed manifest/annotation
top-level JSON, silently removed per-class result tables, symlinked asset bundles
whose manifests could become broken, an archive destination redirected after
staging, unbounded worker logs/results, malformed worker response fields/deep
JSON, and Markdown metadata capable of forging report structure. Consult the
individual assertions and generated evidence for exact coverage.

These are robustness checks, not a proof that the project has no vulnerabilities.
In particular:

- Model plugins and notebook source are trusted Python. The subprocess worker
  restricts forwarded environment variables but is **not an OS/filesystem/network
  sandbox**. It can access files and network allowed to your account. Never run
  an unknown model plugin with secrets present. Canary tests use only owned fake
  data; they do not read actual credentials.
- Provided fixtures make no network calls. `network_requested: false` describes
  runner intent, not an enforced firewall. A custom plugin may do otherwise.
- Time, RSS and output sizes are sampled. Short peaks and bursts can exceed a
  threshold between polls. Use an OS/container resource boundary for untrusted
  workloads. Parent image generation/scoring is outside the model worker RSS cap.
- Symlink rechecks reject a deterministic redirect between staging and commit;
  they do not provide race-free filesystem isolation against a concurrent local
  attacker. Archive destinations must be under exclusive caller control.
- Notebook checks exercise selected pure helpers and AST syntax only. They do
  not validate training, GPU execution, arbitrary notebook side effects, or every
  malformed notebook record. Its legacy `record_errors` helper still has weaker
  nonfinite/proportion validation than the production manifest interface.
- Routine test execution does not query dependency advisories. The latest review
  separately audited the public lockfiles and recorded its snapshot in
  [the retest](RETEST-b790a6b.md). This is not a remote service penetration test
  or a full-data throughput benchmark. Existing image decoder
  budget tests run with `--include-regressions`. Optional and OS-specific skips
  reduce the coverage actually obtained on that machine.

Keep these limits in reports. Record failures before fixing them, retain the
reproducer, and compare accuracy on frozen real data before promoting a model.
