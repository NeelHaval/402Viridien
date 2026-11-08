# Give the approved Clay/Shale family full label credit

The owner requests full label credit for every pairing of **Clay, Shale, Clay/Shale and Clay-Shale**, in either direction. This rule is now implemented in `strategies/grading_policy.py`. Use `score_equivalent(predictions, truth, window)` at the evaluation boundary. It copies both lists and calls the unchanged harness scorer. Preserve raw captions, every bed contact, cluster IDs and depths. The result schema keeps strict scalars and adds `accepted_metrics`; `rescore_policy.py` replays saved CSVs without another inference run.

This is a local grading rule. The official slides give no numerical formula or Clay/Shale equivalence. See [CHALLENGE_ALIGNMENT.md](CHALLENGE_ALIGNMENT.md). The optional S23 distinct-caption experiment changes matching and segmentation, so it must remain separate from this grading correction.

## Measured effect on the same saved window

For D-582 log06, rows 4404–11195:

| Strategy | Strict product | Approved product | Approved row accuracy | Boundary F1, unchanged |
|---|---:|---:|---:|---:|
| s19 | 0.008285 | 0.029907 | 0.158675 | 0.188482 |
| v4 | 0.082129 | 0.106365 | 0.272561 | 0.390244 |
| Baseline DINO | 0.083250 | 0.480353 | 0.768565 | 0.625000 |
| S23 distinct captions | 0.092555 | 0.452097 | 0.768565 | 0.588235 |

These are local window scores, not an official grade or a broad winner. S23 has no extra row credit here under the approved rule. The [saved prediction view](log06_dino_equivalent_detail.html) shows the raw labels and policy-consistent error colors.

Keep the approved names exact. In the returned MC765 case, v4 predicts `Clay-M / Clay-S / Clay-SACS` against truth `Clay-S`. Both strict and approved products are zero. Those qualified captions are outside this family. Removing their qualifiers or using a `startswith('clay')` shortcut would conceal another false merge.

## Problem and evidence

The frozen v4 matcher groups visually similar swatches into a twins label. In the D-582 log06 native cache, Clay and Shale are separate entries, each with a 104 × 148 pixel swatch. The old matcher maps both to Clay / Shale. In the saved S21 log06 prediction, rows 4429–4814 carry Clay / Shale, while the supplied gold has Shale from rows 4409–4809. The strict scorer does not treat those strings as equal. This historical strict-grade example isolated the grouping change; it did not predict the S23 outcome. The measured strict and equivalent results above show that outcome.

## Candidate and scope

The optional S23 experiment preserves every distinct printed caption as a separate matching identity. It disables visual twin grouping for the local classifier instance before both segmentation passes. Multiple swatches with the same exact caption remain one candidate, using the lowest hand distance and highest DINO similarity as in S21. A single genuine caption such as Clay/Shale stays exactly that caption. The change is generic: it affects **all** groups of distinct captions that v4 would join as visual twins, not just Clay and Shale. Separate candidate identity is not necessary to satisfy the owner's grading rule.

S23 uses the S21 frozen model and helper functions: native-scale 224-pixel tiled canvas, three swatch scales, hand distance and NCC, z-score fusion, blank and unknown gates, fusion-gap voting, and unchanged bed segmentation. It changes neither the scorer nor label spelling or truth. The S20 and S21 baselines, original labeltool code, and native swatch cache remain unchanged. The new strategy records its own source hash, the S21 helper hash, the frozen adapter source hashes, and model hashes in diagnostics.

S21 and S23 can share an in-process model cache: report whether each measurement is cold or warm and do not compare such times as equal conditions.

## Next pipeline PR handoff

The current matcher callsite is labeltool/legend_classify.py, function classify: it gets tw = twins(log) and uses tw.get(lb, lb) as the key in the hand-distance candidate map. The same substitution occurs in both the hand and embedding candidate maps in strategies/s21_v4dino.py, function _classify. Preserve the original predicted captions and boundaries in saved outputs. At the grading boundary, apply the owner's symmetric equivalence to the four approved names before invoking the existing scorer. Use the shared policy helper, rather than a separate alias table in each caller. Visual twins and entry IDs may be retained for diagnostics.

S23 is a reviewable experiment using the frozen adapter; it does not edit the live labeltool. The grading change does not require consumers to migrate away from slash-joined raw labels. Keep the raw CSV schema and existing depth, blank, and bed behavior unchanged. Capture any proposed change to those behaviors separately.

## Evidence to collect before a pipeline PR

1. Run the CPU identity checks from bakeoff: python -X utf8 -m strategies.test_s23_identity. They check the real log06 cache, independent hand and DINO rankings, a repeated caption, a genuine slash caption, and both segmentation passes.
2. Re-score saved outputs with the policy helper. If separately evaluating S23, run that distinct model candidate with the same approved windows and preprocessing, and save its own predictions and provenance. A grading-only change does not justify rerunning S21 or S23.
3. Compare strict scores separately from the owner's equivalent-grade scores, per window and per label. Keep the original CSVs and report the grading policy and source hash with each series. All 16 approved spelling pairings must receive full label credit; other class names must not gain credit.
4. Inspect cases where distinct swatches are visually indistinguishable. Report ties and instability; preserve a genuinely combined printed caption rather than guessing its component.
5. For the next pipeline PR, implement and test the grading equivalence at the boundary, including all directions and the genuine compound caption. Keep the strict series available for comparison and include paired results and regression checks. S23 can be evaluated independently as an optional experiment.

Run the regression commands in [strategies/LEGEND_IDENTITY.md](../../strategies/LEGEND_IDENTITY.md). The combined check is recorded in `final_checks_exit.json` and `final_checks.log`: `1 failed, 188 passed, 2 warnings in 55.73s`. The one failure is the existing harness PNG fixture passed to a TIFF reader. After adding the qualified-caption guards, the focused grading, identity, CSV-audit and policy-gate checks reported `61 passed in 21.92s`; output is `out/final_grade_regression.log`. The shared loader and scorer were not edited. The full 272-record comparison and both bound final audits are complete; see the final status below.


## Final status

All 272 records passed exact coverage and identity checks. Independent strict and owner-equivalent CSV replay reported 271 successful primary predictions, no pixel diagnostics and zero errors. The run integrity audit also reported zero errors. The completed [scoreboard](index.html) retains both grades and all four candidates.

The final focused check recorded **123 passed, 2 warnings in 11.76s**. Later readiness path checks recorded **2 passed in 2.45s**; the final report/diagnostic check after the provenance-size correction recorded **24 passed in 6.06s**. These runs overlap and their counts must not be added. The historical full-harness PNG/TIFF fixture failure remains; this is not a passing full-harness claim.

S19 is the recorded development accuracy/coverage default. Missing native legends on all 51 DEV cases / 46 logs dominate this pipeline choice. S23 leads the Viridien suite, but its paired margin over baseline DINO remains uncertain. The later 1–2 minute latency goal and separate cache/speed work are not certified by this matrix. The grading equivalence is an owner-configured local proxy; the official challenge rule remains unverified.
