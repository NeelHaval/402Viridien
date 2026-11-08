# Astra review: local fixes and final checks

One GPT-6 Astra agent reviewed the baseline method and code at xhigh. Its original review is preserved in `astra_review.md`. The owner then authorized one Astra recheck. That review is complete in `astra_recheck.md`. It accepted the S23 distinct-caption change for development comparison, with final integrity checks still required. Neither review approved a champion from incomplete results.

The owner later clarified a different grading requirement: Clay, Shale, Clay/Shale and Clay-Shale must receive full label credit against each other in both directions. The shared scorer stays unchanged. `strategies/grading_policy.py` applies this local rule to copies of prediction and truth rows. Raw labels, contacts, depths and cluster IDs are preserved. Saved predictions are rescored; inference is not repeated for this change. Strict scores remain available. This later policy has local tests and an independent CSV replay gate. The Astra recheck did not cover the later policy; do not claim that it did.

The official challenge slides specify correctness and efficiency but give no numerical formula or Clay/Shale rule. Both numerical series are local proxies. See `CHALLENGE_ALIGNMENT.md` for the official sources and the required export formats.

The parent and the Sol workers made the following local fixes after that review. These fixes do not change the scorer, s19, v4, or DINO inference.

| Finding | Local fix | Evidence |
| --- | --- | --- |
| M1: well groups differ and omit Idaho page dependence | One explicit map covers all allowed DEV pages in five provisional source groups. Both summaries use that map. Equal-group means are separate from the primary equal-log means. The map does not certify geological independence. | `reviewed_dev_wells.json`, `report_inputs.py`, runner source guard |
| M2: missing source dependency and weak input/resume guards | Add s02 and the dataset path resolver to the precise dependency guard. Pin DINO weights, config and software. Check original and copied truth, image and annotation hashes. Require checkpoint records to match the manifest and record its digest. Preserve earlier manifests and checkpoints. | `score_all.py`, `manifest_1108.json`, preserved `manifest_pre_m2_*` and `checkpoint_pre_m2_*` files |
| M3: incomplete or disjoint cases can look paired | Require the exact 68 cases times the explicit named candidates in the manifest. Reject duplicate, stale, extra or missing records. Pair exact cases before log means. Require a successful independent CSV audit bound to the manifest and raw records. | `test_report.py`, `test_report_gate.py`, `audit_saved_predictions.py` |
| M4: s19 output depth can differ from the common input calibration | Name and report the common initial calibration. State that unchanged s19 can refine it. Depth MAE applies only to matched contacts with truth depths. | Runner records and report method text |
| M5: runtime and cache state need limits | Retain observed first-call, process and resume markers. Missing inference time is unknown. Show timed-case, diagnostic and budget counts. Do not use these observations as a controlled cold/warm speed test. | Runner records and report timing section |

Checks after the local fixes:

- Report and gate tests: `python.exe -X utf8 -m pytest report/scoreboard_1108/test_report.py report/scoreboard_1108/test_report_gate.py -q` — `15 passed in 6.24s`.
- Runner guard tests: `python.exe -X utf8 -m pytest report/scoreboard_1108/test_scoreboard.py -q` — `5 passed in 13.19s`.
- Harness checks after the strategy edits: `python.exe -X utf8 -m pytest harness -q --tb=short` — `1 failed, 89 passed in 37.12s`. The existing PNG/TIFF loader failure remains. All 68 source images in this comparison are TIFF files.

The first eight retained records were checked against the current manifest and all declared input hashes before migration. Their records state that the original read-time guard was absent. These are retained results with validated current identity; they are not described as results from the later guarded process.

The complete score audits, final report build and visual checks have finished. Their machine-readable evidence is saved beside this file. A practical choice from these development data does not establish a held-out or universal champion. An unscored component hybrid remains unverified.

## Owner-requested follow-up

After the baseline log06 result and visual comparison, the owner requested a Clay/Shale fix and reevaluation. The fourth candidate, `s23_v4dino_identity`, preserves separately printed captions rather than replacing them with visual twin groups. The three original candidates and the scorer stay frozen. The updated exact matrix has 68 cases times four candidates, or 272 records. This is an authorized development follow-up, prompted by observed baseline errors. It is not an independently validated experiment.

The first Astra review does not cover this new strategy code. The owner answered the review question with “Run one Astra recheck after the fix.” The completed recheck covers S23 and the method, guards and report at that point. Its R1 integrity finding was fixed locally: final report gates bind the manifest, exact record content and primary/pixel DINO model pins to two audits. Its R2 cache-history limit remains: later snapshots cannot prove the state of mutable OCR/depth caches at every earlier read. These are observed development results, not a completely frozen end-to-end evaluation. See `CLAY_SHALE_FIX.md` for the next-PR handoff and `CLAY_SHALE_FIX_MEASURED.md` for the first measured result and source audit.

After the S23 edit, the full harness check reported `1 failed, 90 passed in 50.55s`. The same PNG fixture fails. S23 CPU tests reported four tests, OK. The synthetic GPU smoke returned Shale in 57.61 seconds cold. Under the historical strict grade, the first real log06 result changed from 0.083250 to 0.092555. Under the owner's later equivalent-label grade, the same saved baseline DINO output scores 0.480353 and S23 scores 0.452097. Both have row accuracy 0.768565. Baseline boundary F1 is 0.625; S23 is 0.588235. The grading correction therefore removes the apparent label gain on this example. It does not establish a broad winner.

After the equivalent-label and audit edits, this command reported `42 passed, 2 warnings in 11.24s`:

```text
python -X utf8 -B -m pytest strategies/test_grading_policy.py strategies/test_legend_identity.py strategies/test_s23_identity.py strategies/test_challenge_export.py report/scoreboard_1108/test_saved_audit_guards.py -q --tb=short
```

The final combined check includes the whole harness and all relevant strategy, policy, export and report tests. Run `python -X utf8 -B report/scoreboard_1108/run_final_checks.py`. Its saved command and exit record are in `final_checks_exit.json`; output is in `final_checks.log`. The last summary is `1 failed, 188 passed, 2 warnings in 55.73s`. The one failure is the existing PNG fixture passed to the TIFF reader, `harness/test_bench.py::test_notebook_callable_with_spawn_and_cached_second_run`. The shared loader was not changed. This is not a passing full harness.

The full 272-record replay and final report are complete. No sealed test or verify split was opened or scored.


## Final local completion

The final matrix has exactly 272 unique candidate/case records. Numerical replay checked strict and owner-equivalent grades from 271 successful primary CSVs and zero pixel diagnostics, with zero errors. The integrity audit checked the same manifest and record digests, exact keys, source/input identities and model evidence, with zero errors. The final report gate also checks the exact audited prediction-CSV path set and file bytes before rendering. See [final_audit_commands.json](final_audit_commands.json), [audit_saved_predictions.json](audit_saved_predictions.json), [result_audit.json](result_audit.json) and [decision_gate.json](decision_gate.json).

The formal S19 Shenzi attempt reached the common whole-attempt cap. Its receipt records 600.705492 seconds for the attempt and unknown inference seconds. It remains a zero-score failure. The earlier manual interruption is historical evidence, not the final timeout measurement.

Final focused command: python -X utf8 -B report/scoreboard_1108/run_final_checks.py --without-harness. Its last summary was **123 passed, 2 warnings in 11.76s**, exit zero. Later readiness path checks ended with **2 passed in 2.45s**; the final report/diagnostic check ended with **24 passed in 6.06s**. Historical full-harness failure records above remain valid and are not replaced by these focused checks.

The final build reported: Report built: 272 windows, 252 log results, 12 diagnostic images, 0 render failures. The four additional log06 grading/identity images also rendered. Two labelled contact sheets cover all 16 images; both were inspected at no more than 1024 pixels and showed aligned source columns, interval bars and error strips. Submission readiness remains unverified even though record coverage is complete.

Cache first/end-state evidence records 53 changed depth-cache JSON files and 26 new run-owned S19 OCR JSON files, with zero capture errors. No exact per-case read interception was installed. These snapshots cannot prove historical cache bytes, so the evaluation is observed development evidence rather than a fully frozen end-to-end run. Astra accepted the S23 implementation and the reviewed method fixes within its recheck scope; it did not review the later owner grading configuration or approve a final champion. The parent verified the local fixes and final data.

S19 is the recorded development accuracy/coverage default, with the missing-legend, truth-quality, timeout and source-history limits shown in the report. This choice does not certify the later 1–2 minute evaluation goal, production readiness, an intrinsic matcher advantage, a held-out champion or an untested hybrid. The live cache pilot and 120-second speed gate remain separate.
