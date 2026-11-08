# Astra recheck: S23 and the development comparison

Date: 8 November 2026. Reviewer: Codex GPT-6 Astra, xhigh.

Verdict: accept the S23 caption identity fix for this development reevaluation. The code implements the requested change. It does not establish an accuracy gain or approve a final practical choice. M1, M3, M4, and M5 are resolved within the limits below. M2 is partly resolved. Two controls still need work: the report does not require the run integrity audit, and the freeze does not cover OCR cache contents.

The owner authorized this one recheck. No child agent was used. No GPU inference was run. No worker file, baseline, scorer, or dataset was changed. All review writes are new files below this report directory. The original `astra_review.md` is intact. This is a review of specified files and their first-review hashes, not a Git diff review. The code-review skill was used for separate standards and specification checks. The supplied scope takes the place of a Git base.

## Standards

No new blocking standards defect was found.

The first review's S1 duplication remains a maintenance concern. `score_all.py:137-185` and `build_report.py:51-88` still implement statistics separately. The demonstrated pairing defect is fixed: the report now pairs exact cases before it computes log means. A reviewer probe with the 51 DEV case identities gave five source groups in both implementations. The mean deltas agreed after their declared signs were reversed. This probe does not certify exact equality of every bootstrap draw. Different group order can give different finite-sample bootstrap endpoints.

S23 also copies the S21 two-pass run structure. This is a reasonable choice for a frozen baseline comparison. Do not refactor the baseline during this run. In a later pipeline change, put the caption policy at one classifier construction point and retain tests for the two-pass path.

## Specification and method

### R1 — P1: the required report gate can bypass run integrity

The first review required exact record/manifest agreement and final artifact checks. The runner now checks each record's manifest digest (`score_all.py:52-64`). The separate integrity audit checks that digest, the top-level digest, and primary DINO model hashes (`audit_run_integrity.py:10-31`).

However, `build_report.py:205-208` requires only `report_inputs.validate_records` and the saved-prediction audit. The first check does not inspect manifest digests or model pins (`report_inputs.py:14-37`). The second audit checks case inputs, scores, and source files, but does not inspect record manifest digests or model pins (`audit_saved_predictions.py:43-66,111-124`). The report never reads `result_audit.json`.

Concrete CPU probe: four synthetic records had correct case metadata and correct CSV scores, but a stale manifest digest and wrong model hashes. Saved-prediction replay returned `4/4 records; 4 predictions and 0 pixel diagnostics re-scored; 0 errors`. Both required report checks returned `valid`. The runner rejected the same records with `Resume manifest digest mismatch`. This is a synthetic guard test. It is not evidence of corrupt real results.

The probe uses one synthetic case and sets the record gate's `expected_case_count` to 1. The production count stays 68. This fixture adjustment does not change the missing digest and model checks.

The depth-failure diagnostic has another path around the model check. `score_all.py:262-264` checks primary DINO diagnostics. The call at lines 277-283 stores pixel-only model diagnostics without the same check. The integrity audit at lines 23-26 checks only the primary field. Thus the diagnostic table can use a model that the audit did not validate.

Local fix: make one final report gate require both CSV replay and run integrity. Bind both audit records to the exact manifest and result-record digests. Add those bindings to the integrity audit output. Check model hashes in primary and pixel-only diagnostics. Require model evidence when embedding chunks were used; allow no model when no embedding was needed. Reject a failed integrity audit before writing selection text. The parent can apply these report/audit fixes without changing S19, S20, S21, S23, or the scorer. Until then, the parent's explicit final integrity check is mandatory; a green report gate alone is insufficient.

### R2 — P2: OCR caches remain outside the freeze

The first M2 fix request explicitly included preprocessing caches that can affect execution. `score_all.py:30-37,46-50` now covers more source files and all declared input assets. It still does not cover the mutable caches used by the actual inference path:

- `harness/legend.py:183-191` accepts cached entries and swatch files by log name and parser version.
- `harness/depth.py:171-182` accepts a stored calibration when its input signature matches. The signature protects the input pixels, not the stored calibration bytes.
- `ml/combine/runtime.py:35-52` copies raw OCR evidence from `ml/legend_names/cache` when needed and uses the runtime legend cache.
- `harness/legend_v3.py:155-162` selects that evidence by an image hash, then reads its unpinned JSON contents.

The reviewer changed only an isolated synthetic cache. With the same input pixels and the same cache filename, the real v3 parser returned Clay, then Shale, after the cache text changed. No OCR or model call was needed. The manifest has no entries for these cache paths. This demonstrates a missing control. It does not show that the active run's cache changed.

Local fix: preserve and hash the cache files consumed by each case, including cached swatch images, v3 evidence, and depth calibration. Use a run-owned snapshot or record the exact bytes at use time. Validate them on resume and in the final integrity audit. Put these controls in the local runner; keep the shared parser and candidates unchanged. Record OCR model identity for any cache miss that causes fresh OCR. Package version text alone does not pin model bytes.

The present run cannot gain a past read-time guarantee from a hash collected later. Keep the historical caveat. If the parent retains these results, describe them as an observed development comparison with this cache provenance limit. Do not claim a fully frozen, reproducible end-to-end run. This limit is stronger for S19 and shared preprocessing than for S21 versus S23: the latter use the same hash-pinned native swatch snapshot. It does not reject the S23 identity fix.

## M1–M5 status

| First finding | Recheck status | Evidence and remaining limit |
| --- | --- | --- |
| M1: wrong DEV groups | Resolved for this comparison | `reviewed_dev_wells.json:2-56` covers 46 DEV log IDs and all 51 windows. Counts are KGS 2, Utah 21, Idaho Barlow 6, Flat Bay 9, Storm 13. The runner applies this map at `score_all.py:148-150`. The report builds the same map at `build_report.py:214-217`. The file and generated map match. Equal-group results stay separate at lines 238-247. These are five provisional source groups, not five certified independent geological wells. |
| M2: source, input, and resume guards | Partly resolved | S02 and the path resolver are added at `score_all.py:31`. Input hashes are checked before use, on resume, and at completion, at lines 46-64, 241, and 297-299. Records carry a manifest digest. DINO pins and package versions are declared at lines 72-79. All 264 current asset checks and 53 source checks matched. R1 and R2 remain. |
| M3: incomplete/disjoint pairing | Resolved for case coverage and pairing | `report_inputs.py:14-37` requires exact keys, named candidates, complete status, and full case identity. Lines 41-54 reject disjoint windows and unmatched keys. The report pairs cases before log means at `build_report.py:72-78`. Runner completion uses exact keys at `score_all.py:301`. Current manifest: 68 cases, four named candidates, 272 expected records. R1 is a separate integrity requirement. |
| M4: initial versus S19 output depth | Resolved for interpretation | New records use `initial_shared_depth_calibration` and an explicit note at `score_all.py:252-253`. `build_report.py:321` says S19 can refine output depths and defines conditional depth MAE. The old `depth_calibration` alias remains at runner line 271 and in legacy records. Treat it as initial context, never as proof of identical output calibration. A later schema cleanup can remove that ambiguity. |
| M5: runtime and cache limits | Resolved for descriptive timing | Runner lines 220, 242, 260, and 287-293 retain process, first-call, resume, attempted wall time, and unknown-time values. Report lines 283-284 and 305 show coverage and explain the limits. First S23 call can use a model already loaded by S21. Diagnostic calls can warm it. The machine is shared. No controlled cold/warm or speed tie-break claim is supported. |

## S23 identity fix

`strategies/s23_v4dino_identity.py:15-23` replaces `twins` only on a fresh local classifier. `scoreboard_v4.py:40-50` creates that separate namespace. Both passes use it, at S23 lines 32-33 and 50-51. No baseline module or shared classifier namespace is modified.

The effect is generic to all distinct captions. There is no Clay-to-Shale rule. Separate Clay and Shale captions remain separate candidates. Repeated swatches with the exact same caption still use the minimum hand distance and maximum DINO similarity (`s21_v4dino.py:129-148`). A genuine single `Clay/Shale` caption remains that caption. Case, whitespace, or geological synonyms are not added to this policy. The unchanged scorer still decides label equality.

S23 calls the unchanged S21 model, canvas, swatch, state, embedding, and fusion helpers. The canvas tiles native pixels to 224 by 224 and crops oversized content at the top left. The embedding joins CLS and mean patch tokens. Swatches use nearest-neighbour scales 0.5, 1, and 2. Fusion minimizes z(hand distance) minus z(DINO similarity). The confidence gap, blank gate, unknown gate, and bed code are retained (`s21_v4dino.py:27-34,69-104,107-161`). Removing groups changes the candidate set, so it can change z-scores, confidence, bed votes, and final boundaries. It is not only an output spelling change.

The four S23 CPU tests passed in the reviewer run. They cover the actual log06 native captions, repeated caption similarity, a genuine slash caption, hand and DINO ranking, and injection before both passes. The separate S21 CPU checks passed for fusion direction, rejection, blanks, and native tiling. These checks support implementation correctness. They do not establish classification accuracy or robustness to visually identical distinct classes.

`CLAY_SHALE_FIX.md:5-27` states the scope and the pipeline handoff correctly. `variant_protocol_addendum.json:2-14` states the observed-error trigger and the four-candidate design. The owner requested this change after seeing development truth. It is an authorized development follow-up, not an independent validation experiment. No alias or scoring remap was found.

The synthetic GPU smoke was read, not rerun. `out/s23_smoke.out.log` reports Shale, five chunks, and 57.61 seconds cold with the declared model/config hashes. It is not a gold score or a controlled runtime comparison.

## Test and hash evidence

Reviewer command, from bakeoff with `PYTHONPATH=.`:

```text
C:/Users/hssil/anaconda3/python.exe -X utf8 -B report/scoreboard_1108/astra_recheck_checks.py
```

The script runs only bounded CPU checks. Test temporary files and synthetic audit files stay in `astra_recheck_sandbox`. It uses no candidate GPU inference. The final run ended with:

```text
40 passed, 2 warnings in 7.46s
PASS: fusion ranking, rejection, blank state, and native tiling
4/4 records; 4 predictions and 0 pixel diagnostics re-scored; 0 errors
ASTRA RECHECK EXIT 0
```

The four-record line is the isolated R1 probe, not the live comparison. The two warnings concern Paramiko's TripleDES deprecation. The first CPU run also passed: 40 tests in 15.27 seconds. It is retained as `astra_recheck_checks_initial.log`. The final run replaced the raw cache probe with a valid parser example, which returned different labels from the same pixels. No production source was edited between these runs.

`astra_recheck_checks.json` contains the test status, 264 asset checks, 53 source checks, map checks, synthetic probes, and preserved-checkpoint evidence. Twenty-two previously reviewed strategy, harness, and model-source files match the first review. This includes S19, S20, S21, the v4 adapter/snapshot, and the shared scorer. All four entries in `report_source_freeze.json` also match.

Full harness: not run by this reviewer. The existing post-S23 log ends with `1 failed, 90 passed in 50.55s`. Its failure is `harness/test_bench.py::test_notebook_callable_with_spawn_and_cached_second_run`, where the PNG fixture reaches tifffile. All 68 declared comparison images have `.tif` or `.tiff` suffixes. Do not report the full harness as passing. No loader change is required for this review.

## Preserved history and result cutoff

The preserved checkpoint/manifest pairs contain 7, 8, and 13 records. Their recorded digests, where present, match their corresponding manifests. Their case identities agree with the current cases. `migrate_manifest.py:7-21` checks record identity and current inputs before migration, checks unchanged sources apart from the revised runner, and preserves prior files. This is present-state evidence, not proof of all earlier read-time bytes.

One live result snapshot was read at 2026-11-08 06:25:35 UTC. It had 21 unique records: six S19, five S20, five S21, and five S23, all in `viridien_eval`. No final `scores.json`, report data, CSV audit, or run integrity audit existed at that cutoff. No polling or wait for the full run was done.

The 21 records matched their manifest identities and digests. Every model hash present in their primary or pixel-only diagnostics matched the declared pins. All 13 retained baseline records kept their score parts, inference times, prediction paths, and error values. Eight carry the explicit absent-original-read-guard caveat; five carry the later migration note. This supports retention with the stated history. It is not a final CSV replay.

The snapshot and its hashes are saved in `astra_recheck_results_observed.json` and `astra_recheck_snapshot_raw_results.json`. The parent must independently replay all saved CSVs, check exact 272-record coverage, verify both primary and diagnostic provenance, and validate the final data before naming a practical choice. No champion is approved here.

## Local completion steps and claim limits

1. Complete the existing serial run. Keep the four candidate identities and scorer fixed.
2. Apply R1 in the local audit/report code. Require both final audits and exact 272-record coverage. Check failure zeros and all saved CSVs.
3. Preserve available cache evidence. Apply R2 for future guarded reads, and state the limit for earlier reads. Do not silently relabel historical records as fully frozen.
4. Build the final report with the reviewed DEV map and matched cases. Update the stale review statement at `build_report.py:321` to cite this recheck and its limits. Version any changed report-source freeze. The original protocol remains historical; the addendum and this authorized recheck govern the fourth candidate and review scope.
5. State suite trade-offs, native DEV cache failures, overlap, truth quality, and runtime coverage. S23 does not repair missing DEV native legends, thin-bed absorption, or same-label bed merging. An unscored hybrid remains unverified.

A completed, audited comparison can support a current practical choice on these development inputs. It cannot support a universal, held-out, or new-well champion claim. The remaining cache history limit must remain explicit. A successful synthetic smoke or one improved log is not sufficient selection evidence. No further Astra review is requested by this note; these are local fixes and parent checks within the authorized work.

## Reviewed identity

`astra_recheck_hashes.json` contains full SHA-256 values for the reviewed files and evidence. Key inspected sources:

```text
strategies/s23_v4dino_identity.py
9ea876ef8761b0bf6399e61947df473e33d0cfc61dac9a1b2fe856f6c394bdbb
report/scoreboard_1108/score_all.py
26e32c403a3d9677bed4906142619d8456b942d6469ed17c289120a4f230971e
report/scoreboard_1108/build_report.py
674b04b160483c5bc6aacd632ba955a71a8f156cdfe694e2fad7153d74727e79
report/scoreboard_1108/report_inputs.py
ccfa9a023421b398007b64183667b338f4209c62af26a72ea6ac39628df6a28f
report/scoreboard_1108/audit_saved_predictions.py
4799f4694ec61610cb0a68dde4ff7b2109c3b11ef46733dd4ad765118b365dad
report/scoreboard_1108/audit_run_integrity.py
31b599a206675d58d73801522c27f38b194556b575c63d0b77f8e9cfa807098b
report/scoreboard_1108/manifest_1108.json
b352bde5f9b34aa86d875d1c75aa882b4cfc24bbb5fa0213afef53ed7f784205
```

Standards: no new blocker; the earlier duplication concern remains. Specification/method: two remaining control findings, R1 P1 and R2 P2. The S23 identity implementation is accepted for this development experiment. Final selection remains with the parent after the stated checks.
