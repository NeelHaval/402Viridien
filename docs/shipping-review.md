# Independent shipping review

Reviewed on 2026-11-08 by a separately delegated GPT-6 Astra reviewer. This is a local verification snapshot, not a claim that remote CI, release publishing or a fresh authenticated download succeeded.

## Executed checks

- Windows, Python **3.13.4**, existing project virtual environment: `python -m pytest -q` passed **48 tests**.
- `python scripts/run_evaluation.py` completed the three-case demo and wrote predictions, previews and summaries. Baseline names were all Unknown; score was zero. Successful completion is not model accuracy evidence.
- `python -m build` built a wheel and source archive. Installing that wheel into a separate target, verifying its import path, and running demo generation/evaluation there completed **3/3 cases**.
- Inspected the locally prepared evaluation ZIP: **86 members, 36 supervised windows**, no missing manifest-referenced assets; expanded size **38,598,413 bytes**. Extracted it into a fresh directory and validated its human manifest.
- Evaluated the extracted human bundle with the baseline/config: **10/10 completed**, zero errors, row accuracy and score **0**, boundary F1 **0.591225**, ARI **0.781470**, coverage **1**, Unknown rate **1**. Run: `.local/shipping-review-runs/20261108T040846Z-8621fae110` (local ignored evidence).
- Inspected the locally prepared challenge ZIP: **110 members, 48 unlabelled cases**, no missing manifest-referenced assets; expanded size **1,009,383,982 bytes**, below the extractor's 3 GB cap. Did not run all challenge cases.
- Workflow inputs reach Python through environment variables, not interpolated shell fragments. Fetch credentials are scoped to fetching, worker environments use an allowlist, and release redirects remove Authorization. Existing archive traversal/symlink/size tests passed.
- Confirmed the pinned [checkout](https://github.com/actions/checkout/commit/11d5960a326750d5838078e36cf38b85af677262), [setup-python](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065), and [upload-artifact](https://github.com/actions/upload-artifact/commit/ea165f8d65b6e75b540449e92b4886f43607fa02) commits exist upstream. This does not substitute for an actual Actions run.

## Findings at review time

1. **Source archive incomplete.** The built source archive includes `tests/test_bundles.py` but omits `scripts/datasets.py`, which that test imports. It also omits configs, dependency constraint files and linked documentation. Include the checkout resources needed by packaged tests and README commands, then rebuild and test the extracted source archive.
2. **Gate wording can overstate evidence.** `summary.md` prints `Quality gates: PASS` with no configured score threshold, including zero-score labelled runs and unlabelled runs. Distinguish operational/completion checks from supervised quality, and explicitly state when accuracy is unassessed or no accuracy threshold was configured.
3. **Legacy dependency conflict.** At inspection, the optional extra required `rapidocr-onnxruntime>=1.3` while its constraint was `==1.2.3`; that combination cannot install. Resolve and verify the documented constrained installation. The legacy file does not pin all transitive dependencies, and isolated builds use an unconstrained `setuptools>=68`; describe these files as constraints unless full environment locking is added.
4. **Evidence/documentation alignment.** Local verified Python is 3.13.4; the workflows request 3.12. Do not describe 3.12 as tested until its execution is observed. `docs/metrics.md` promises grouping by tags, but the reviewed evaluator groups only by source, group and split.
5. **Resource semantics.** Aggregate `seconds` sums worker inference wall time, including worker startup/loading; `peak_rss_mb` is the maximum sampled worker/descendant RSS across cases. Parent manifest loading, input hashing, scoring and preview generation are excluded. Describe this scope explicitly when comparing costs or end-to-end run duration.

## Verdict and limits

The basic local CLI, worker execution, supervised data bundle and built wheel work. The source archive and documentation need the corrections above before describing the release as reproducibly tested. The private dataset lock was not yet present at initial review; release asset IDs, remote checksum verification, GitHub Actions execution, legacy OCR runtime and full unlabelled robustness remain for the release owner to verify. No prediction-quality or unseen-generalisation claim is supported by this shipping review.
