# Independent shipping review

Reviewed and rechecked on 2026-11-08 by a separately delegated GPT-6 Astra reviewer. This distinguishes independently executed local checks from release-owner reports; it does not independently certify remote publishing or authenticated downloads.

## Executed checks

- Windows, Python **3.13.4**, existing project virtual environment: initial **48 tests**, then **62 tests** including crop classification passed.
- `python scripts/run_evaluation.py` completed the three-case demo and wrote predictions, previews and summaries. Baseline names were all Unknown; score was zero. Successful completion is not model accuracy evidence.
- `python -m build` built a wheel and source archive. Installing that wheel into a separate target, verifying its import path, and running demo generation/evaluation there completed **3/3 cases**.
- After the packaging fix, rebuilt into `.local/shipping-dist2`, extracted the source archive into a fresh directory, and ran its bundled tests: **62 passed**. Installed its wheel into a new separate target, verified the import path, and repeated the **3/3** demo. These checks preceded the final crop CLI/workflow integration.
- Inspected the locally prepared evaluation ZIP: **86 members, 36 supervised windows**, no missing manifest-referenced assets; expanded size **38,598,413 bytes**. Extracted it into a fresh directory and validated its human manifest.
- Evaluated the extracted human bundle with the baseline/config: **10/10 completed**, zero errors, row accuracy and score **0**, boundary F1 **0.591225**, ARI **0.781470**, coverage **1**, Unknown rate **1**. Run: `.local/shipping-review-runs/20261108T040846Z-8621fae110` (local ignored evidence).
- Inspected the locally prepared challenge ZIP: **110 members, 48 unlabelled cases**, no missing manifest-referenced assets; expanded size **1,009,383,982 bytes**, below the extractor's 3 GB cap. Did not run all challenge cases.
- Workflow inputs reach Python through environment variables, not interpolated shell fragments. Fetch credentials are scoped to fetching, worker environments use an allowlist, and release redirects remove Authorization. Existing archive traversal/symlink/size tests passed.
- Confirmed the pinned [checkout](https://github.com/actions/checkout/commit/11d5960a326750d5838078e36cf38b85af677262), [setup-python](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065), and [upload-artifact](https://github.com/actions/upload-artifact/commit/ea165f8d65b6e75b540449e92b4886f43607fa02) commits exist upstream. This does not substitute for an actual Actions run.

## Findings and resolution

1. **Resolved: source archive completeness.** `MANIFEST.in` now includes scripts, configs, constraints, docs, examples and the dataset lock. The rebuilt extracted archive passes all 62 bundled tests.
2. **Resolved: misleading PASS wording.** Interval reports now say “Configured checks” and explicitly distinguish missing truth or absence of an accuracy threshold from measured model quality.
3. **Resolved conflict; reproducibility qualification remains.** The optional RapidOCR lower bound now agrees with `==1.2.3`. The release owner reports successful constrained installation and an organizer S11 run. Transitive legacy dependencies and isolated build dependencies are not fully locked; these files constrain versions rather than define a completely reproducible environment.
4. **Resolved: tag grouping; remote evidence attributed.** The evaluator now stores tags and groups their metrics. The release owner reports hosted Ubuntu/Python 3.12 CI success; this reviewer independently ran Windows/Python 3.13.4 checks.
5. **Resolved: resource scope documentation.** The model-interface documentation now states that interval inference timing/RSS includes worker startup/loading but excludes parent validation, hashing, scoring and previews. RSS remains sampled rather than an exact peak.
6. **Downloader inspected.** New 4 MiB range requests require exact HTTP 206, Content-Range and length, retry failures three times, and verify the complete SHA-256 before cache replacement. Redirected requests strip Authorization. No blocker found by inspection; real storage/network behavior needs the release owner's download evidence.

## Crop classification extension

The real PR54 snapshot independently audits as **1,028 accepted records: 907 usable training crops, 112 fixed test crops, 9 quality holds**. The test set has **52 retrospective and 60 prospective** records. The prototype fits descriptors, scale and candidate labels solely from usable training rows. Quality holds are excluded from fitting/scoring; group hashes, export membership, exact native pixels and mask geometry are checked. Native object paths are content hashes. Test labels are used for scoring, not candidate generation or fitting. This is trusted local code with accessible files, not a sealed security boundary.

An independent complete prototype run classified **112/112** cases without errors or empty predictions: exact-set accuracy **0.089286**, macro F1 **0.091283**; prospective accuracy **0.133333**, retrospective accuracy **0.038462**. Run: `.local/shipping-review-crops/20261108T041511Z-502f9ad44c`. These low baseline scores support harness operation, not competitive geological performance. Tests exercise multilabel denominators, unseen labels, holds, invalid evidence, invalid output and timeout paths.

Requested installed package provenance was added to current crop source after review. Crop timing includes prototype fitting and classification after audit; its worker RSS excludes parent fitting/auditing. Prospective status comes from the upstream policy and does not establish a new blinded evaluation. Crop classification does not measure interval boundaries or depth.

## Verdict and limits

No unresolved blocker was found in the rechecked local packaging, interval execution, crop train/test separation or downloader validation logic. The final crop CLI/workflow integration and release publication were still in progress during these checks. The evaluation asset lock is present; remote release/download evidence, final hosted workflow results and full unlabelled robustness remain the release owner's verification scope. No prediction-quality or unseen-generalisation claim is supported by this shipping review.
