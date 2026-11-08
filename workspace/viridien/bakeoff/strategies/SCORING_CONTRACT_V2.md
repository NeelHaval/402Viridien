# Score availability and label credit

This is the handoff for the next pipeline PR. The shared legacy scorer stays unchanged for the saved baseline. New reports must use the version 2 availability layer. These are local measures. The challenge material does not confirm this formula or the Clay/Shale rule.

Read `report/scoreboard_1108/SCORE_RELEASE.json` for the active derived data path and hash. The old `scores.json` is a preserved version 1 artifact. Its presence does not make its metrics available for a current summary.

## Apply these rules in this order

1. Check the case, window, prediction, truth and source hashes. Keep the raw labels, contacts, cluster IDs and depths.
2. Check truth status and supervision. `grading_availability_v2.truth_structure` counts labelled rows and valid depth contacts. It does not change truth.
3. Use the shared scorer for the strict reference. Use `grading_policy.score_equivalent` for the approved label family. Then apply `grading_availability_v2.metric_view` with an explicit series and truth status.
4. Read each metric's `available`, `eligible`, `value` and `reasons` fields. A null value is unavailable. Do not convert null to zero or one. Do not use a default value when a metric is missing.
5. Average available window products within a source log, then give each log equal weight. Do not multiply the table's mean row accuracy by its mean boundary F1. The two means can have different denominators.
6. Show total, available, excluded and eligible windows and logs for every metric. Show zero supervision, missing truth, invalid legend vocabulary and missing depth as separate counts. Pair only identical cases with the same available metric.

## Empty truth and failed predictions are different

The frozen Neptune window has no labelled rock rows. The old scorer returns row accuracy 1 for that input. Version 2 gives it no lithology accuracy or product. Keep it as a blank diagnostic. Do not move its window or make up labels.

A model failure on a supervised case remains a zero in the quality mean. Its valid truth does not disappear because the prediction failed. An unknown inference time stays unknown.

## Keep the approved Clay/Shale rule exact

Clay, Shale, Clay/Shale and Clay-Shale receive full label credit against each other, in both directions. The helper handles case and space differences. It changes copies for grading only. It preserves every contact, depth and cluster ID.

Do not add Claystone, Clay-S, Clay-M, Clay-SACS or joined subtype names to this family. Do not use a prefix match. Do not give Unknown credit. `ClayShale` without a separator is not a global member of this family. The reviewed Shell source-B spelling issue belongs in a separate approved truth release, with the original source kept.

## Do not turn unreviewed sources into truth

The 12 silver evaluation rasters equal source A. A nonblank A/B tie selected A. Their status is `unreviewed-two-source-A-tie-break`; they are not AI consensus.

The `reference_diagnostic` series can show the valid parts of that source reference. The `qualified_development` series excludes silver lithology accuracy, ARI and product. Silver contact measurements remain source-A diagnostics. Hadrian has no QA release and stays separate. Supplied DEV references are development evidence; their presence does not prove human review or performance on unseen wells.

Source A Allegheny and Rio have labels outside their source legends. Their lithology metrics stay unavailable until an approved human truth release. Do not guess a class mapping from a model or colour. A correct Clay/Shale grading rule cannot repair invalid truth vocabulary.

## Depth needs depth truth

A truth file path does not establish depth supervision. Require a finite depth on a labelled truth contact inside the scored window. If no valid matched prediction contact has depth, report depth error as unavailable. Never report zero feet as a substitute for missing truth or a failed prediction.

The 12 silver evaluation truth files have no depth columns. Report their depth exclusion count. Finite model depths do not create depth truth.

A calibration can be available and wrong. Keep a valid large depth error in the accuracy result. Do not mark a wrong fit unavailable to remove the error. Bind a corrected data version to its own manifest and receipt. Do not pool data versions or apply a held-out case fix to code. Develop a repair on DEV before an independent test.

## Keep every release distinct

Keep the original 272 records, CSVs, metric values and shared scorer hash. Store version 2 results in a new derived directory. Save input, source, policy and output hashes and a bound audit. Keep an exact copy of the old report before adding a correction notice. Do not promote a new champion automatically.

The split registry and the API launch hold are separate controls. A successful cached rescore does not release training, tuning, new label calls, dataset inference or a worker move. Do not put truth overlays in a training or review gallery. Keep scorer-only evidence separate.

## Required checks after the last edit

Run `python -X utf8 -B -m pytest -q strategies/test_grading_policy.py strategies/test_grading_availability_v2.py` from `bakeoff`, with `PYTHONPATH=.`. Run the versioned report and audit checks too. Report the real last summary line. Do not add counts from overlapping test runs.

The checks must reject empty truth as a perfect score, unknown series or truth status, depth on blank rows, invalid legend vocabulary and aliases outside the approved family. They must keep supervised prediction failures in the denominator and use the same available cases for paired results.

No new inference is needed for a grading or availability change. Reuse the saved predictions.
