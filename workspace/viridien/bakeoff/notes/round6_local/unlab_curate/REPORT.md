# Unlabelled TEST results

This set has no rock truth. These checks cannot establish rock-type accuracy. Never train or tune on the set, predictions or diagnostics.

The frozen set has 130 pages and windows, 36 well-name groups, and 61 source URLs. Pages from one well are related evidence. Well names are grouping proxies.

| Layout guess | Pages | Well-name groups |
|---|---:|---:|
| Gulf 3-panel | 60 | 6 |
| Texas 'Rock Types' box | 13 | 3 |
| MC765 colour grid | 25 | 3 |
| One-off layout | 5 | 3 |
| Geoservices grid | 8 | 2 |
| D-582 1-bit grid | 12 | 12 |
| MC761 mineral/lithology | 7 | 7 |

No outside source is a confirmed score-3 match. Gulf pages include historic onshore logs, an offshore Canadian log and vendor samples. Much of MC765 is geothermal colour-grid material. Geoservices includes four Norwegian pages with its printed key and four DHI analogues. The seven MC761 mineral-volume pages are a separate stress subset.

## Clean output and scan stability

Failure includes execution errors and empty named output. Coverage counts named, assigned rows. Scan agreement uses only rows assigned on both copies. ARI 1 can also mean one constant class. The multiclass column excludes pairs with only one class on either side. Numeric depth values are unavailable when calibration fails. Depth check % (all) counts a missing fit as a failed check; Depth check % (available) uses only pages with a fit. Other means omit missing values. Legend means use observed pages only; the count in parentheses gives that support. Available shared-parser evidence remains when preprocessing fails before the strategy runs.

| Strategy | Failure % | Coverage % | Depth check % (all) | Depth check % (available, pages) | Named / found keys (pages) | Short beds % | Beds / 1,000 px | Redundant boundaries % | Scan ARI | Eligible scan pairs | Multiclass scan ARI (pairs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s11_fft_legend | 43.8 | 47.1 | 70.8 | 93.9 (98) | 13.5 / 13.8 (130) | 0.0 | 2.0 | 10.8 | 0.603 | 70/130 | 0.636 (49) |
| s17_synth_gulf | 41.5 | 56.7 | 70.8 | 93.9 (98) | 13.5 / 13.8 (130) | 0.8 | 5.8 | 0.0 | 0.542 | 74/130 | 0.577 (66) |
| s19_combine | 37.7 | 61.3 | 72.3 | 95.9 (98) | 14.6 / 14.7 (130) | 0.0 | 8.9 | 7.7 | 0.501 | 80/130 | 0.521 (73) |
| s22_boundary_v2 | 41.5 | 48.9 | 70.8 | 93.9 (98) | 13.5 / 13.8 (130) | 0.7 | 6.4 | 0.0 | 0.548 | 73/130 | 0.571 (63) |

## Denominators (clean / medium scan)

Every attempted page stays in Failure %. Other means skip missing values. This table gives their support. Numeric depth values are unavailable when calibration fails. Depth check % (all) counts a missing fit as a failed check; Depth check % (available) uses only pages with a fit. Other means omit missing values. These are TEST diagnostics, not an accuracy result. Per-family detail: `FAILURE_AUDIT.md`. Per-case flags, error category and availability: `runs/*/derived/corrected_records.csv`.

| Strategy | Attempted | Execution failures (category) | Completed | Empty | Scorable | Depth available |
|---|---:|---|---:|---:|---:|---:|
| s11_fft_legend | 130 / 130 | 32 / 36 (depth_calibration_unavailable 32 / depth_calibration_unavailable 36) | 98 / 94 | 25 / 20 | 73 / 74 | 98 / 94 |
| s17_synth_gulf | 130 / 130 | 32 / 36 (depth_calibration_unavailable 32 / depth_calibration_unavailable 36) | 98 / 94 | 22 / 20 | 76 / 74 | 98 / 94 |
| s19_combine | 130 / 130 | 32 / 36 (depth_calibration_unavailable 32 / depth_calibration_unavailable 36) | 98 / 94 | 17 / 8 | 81 / 86 | 98 / 94 |
| s22_boundary_v2 | 130 / 130 | 32 / 36 (depth_calibration_unavailable 32 / depth_calibration_unavailable 36) | 98 / 94 | 22 / 20 | 76 / 74 | 98 / 94 |

## Per-family checks

| Layout guess | Strategy | Failure % | Coverage % | Depth check % (all) | Depth check % (available, pages) | Named / found keys (pages) | Short beds % | Beds / 1,000 px | Redundant boundaries % | Scan ARI (eligible pairs) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gulf 3-panel | s11_fft_legend | 53.3 | 34.2 | 63.3 | 100.0 (38) | 19.5 / 19.5 (60) | 0.0 | 1.3 | 3.0 | 0.764 (27/60) |
| Gulf 3-panel | s17_synth_gulf | 50.0 | 49.2 | 63.3 | 100.0 (38) | 19.5 / 19.5 (60) | 1.8 | 7.5 | 0.0 | 0.757 (30/60) |
| Gulf 3-panel | s19_combine | 50.0 | 49.2 | 63.3 | 100.0 (38) | 19.7 / 19.7 (60) | 0.0 | 8.0 | 2.3 | 0.724 (30/60) |
| Gulf 3-panel | s22_boundary_v2 | 50.0 | 38.4 | 63.3 | 100.0 (38) | 19.5 / 19.5 (60) | 1.8 | 9.2 | 0.0 | 0.724 (30/60) |
| Texas 'Rock Types' box | s11_fft_legend | 30.8 | 58.4 | 69.2 | 100.0 (9) | 5.3 / 5.3 (13) | 0.0 | 1.9 | 4.2 | 0.567 (8/13) |
| Texas 'Rock Types' box | s17_synth_gulf | 30.8 | 68.7 | 69.2 | 100.0 (9) | 5.3 / 5.3 (13) | 0.6 | 4.6 | 0.0 | 0.265 (8/13) |
| Texas 'Rock Types' box | s19_combine | 30.8 | 68.1 | 69.2 | 100.0 (9) | 9.4 / 9.8 (13) | 0.0 | 10.6 | 0.2 | 0.603 (8/13) |
| Texas 'Rock Types' box | s22_boundary_v2 | 30.8 | 58.6 | 69.2 | 100.0 (9) | 5.3 / 5.3 (13) | 0.0 | 5.1 | 0.0 | 0.460 (8/13) |
| MC765 colour grid | s11_fft_legend | 16.0 | 82.8 | 60.0 | 71.4 (21) | 13.3 / 14.3 (25) | 0.0 | 2.5 | 0.0 | 0.430 (21/25) |
| MC765 colour grid | s17_synth_gulf | 16.0 | 82.5 | 60.0 | 71.4 (21) | 13.3 / 14.3 (25) | 0.2 | 6.5 | 0.0 | 0.390 (21/25) |
| MC765 colour grid | s19_combine | 16.0 | 84.0 | 68.0 | 81.0 (21) | 15.7 / 15.9 (25) | 0.0 | 11.9 | 0.0 | 0.271 (21/25) |
| MC765 colour grid | s22_boundary_v2 | 16.0 | 82.7 | 60.0 | 71.4 (21) | 13.3 / 14.3 (25) | 0.0 | 4.4 | 0.0 | 0.306 (21/25) |
| One-off layout | s11_fft_legend | 80.0 | 20.0 | 80.0 | 100.0 (4) | 1.4 / 1.8 (5) | 0.0 | 0.7 | 33.3 | 0.000 (1/5) |
| One-off layout | s17_synth_gulf | 80.0 | 20.0 | 80.0 | 100.0 (4) | 1.4 / 1.8 (5) | 0.0 | 2.8 | 0.0 | 0.135 (1/5) |
| One-off layout | s19_combine | 20.0 | 80.0 | 80.0 | 100.0 (4) | 2.6 / 2.6 (5) | 0.0 | 7.2 | 0.8 | 0.263 (4/5) |
| One-off layout | s22_boundary_v2 | 80.0 | 16.6 | 80.0 | 100.0 (4) | 1.4 / 1.8 (5) | 0.0 | 4.6 | 0.0 | 0.015 (1/5) |
| Geoservices grid | s11_fft_legend | 0.0 | 90.9 | 100.0 | 100.0 (8) | 20.0 / 20.0 (8) | 0.0 | 2.2 | 0.0 | 0.750 (8/8) |
| Geoservices grid | s17_synth_gulf | 0.0 | 92.6 | 100.0 | 100.0 (8) | 20.0 / 20.0 (8) | 0.0 | 6.1 | 0.0 | 0.747 (8/8) |
| Geoservices grid | s19_combine | 0.0 | 92.6 | 100.0 | 100.0 (8) | 18.5 / 18.5 (8) | 0.0 | 8.2 | 0.0 | 0.774 (8/8) |
| Geoservices grid | s22_boundary_v2 | 0.0 | 89.4 | 100.0 | 100.0 (8) | 20.0 / 20.0 (8) | 0.0 | 7.6 | 0.0 | 0.769 (8/8) |
| D-582 1-bit grid | s11_fft_legend | 50.0 | 34.0 | 91.7 | 100.0 (11) | 1.8 / 2.2 (12) | 0.0 | 6.4 | 74.7 | 0.400 (5/12) |
| D-582 1-bit grid | s17_synth_gulf | 41.7 | 52.0 | 91.7 | 100.0 (11) | 1.8 / 2.2 (12) | 0.0 | 1.6 | 0.0 | 0.160 (6/12) |
| D-582 1-bit grid | s19_combine | 25.0 | 74.2 | 91.7 | 100.0 (11) | 3.7 / 3.7 (12) | 0.0 | 11.5 | 57.7 | 0.064 (9/12) |
| D-582 1-bit grid | s22_boundary_v2 | 41.7 | 35.7 | 91.7 | 100.0 (11) | 1.8 / 2.2 (12) | 0.0 | 1.6 | 0.0 | 0.399 (5/12) |
| MC761 mineral/lithology | s11_fft_legend | 100.0 | 0.0 | 100.0 | 100.0 (7) | 0.0 / 0.0 (7) | missing | 0.0 | missing | missing (0/7) |
| MC761 mineral/lithology | s17_synth_gulf | 100.0 | 0.0 | 100.0 | 100.0 (7) | 0.0 / 0.0 (7) | missing | 0.0 | missing | missing (0/7) |
| MC761 mineral/lithology | s19_combine | 100.0 | 0.0 | 100.0 | 100.0 (7) | 0.0 / 0.0 (7) | missing | 0.0 | missing | missing (0/7) |
| MC761 mineral/lithology | s22_boundary_v2 | 100.0 | 0.0 | 100.0 | 100.0 (7) | 0.0 / 0.0 (7) | missing | 0.0 | missing | missing (0/7) |

## Agreement with s19

ARI compares partitions and ignores class names. It is not a rock-name accuracy score. Missing or failed pairs are excluded from its mean; the eligible count gives its support.

| Layout guess | Pair | Mean ARI | Eligible / pages | Common assigned % | Multiclass ARI (pairs) |
|---|---|---:|---:|---:|---:|
| D-582 1-bit grid | s19_combine / s17_synth_gulf | 0.152 | 7/12 | 51.6 | 0.022 (3) |
| D-582 1-bit grid | s19_combine / s22_boundary_v2 | 0.336 | 7/12 | 35.5 | 0.117 (3) |
| Geoservices grid | s19_combine / s17_synth_gulf | 0.841 | 8/8 | 92.6 | 0.818 (7) |
| Geoservices grid | s19_combine / s22_boundary_v2 | 0.730 | 8/8 | 89.4 | 0.834 (7) |
| Gulf 3-panel | s19_combine / s17_synth_gulf | 0.565 | 30/60 | 49.2 | 0.565 (30) |
| Gulf 3-panel | s19_combine / s22_boundary_v2 | 0.445 | 30/60 | 38.4 | 0.445 (30) |
| MC761 mineral/lithology | s19_combine / s17_synth_gulf | missing | 0/7 | 0.0 | missing (0) |
| MC761 mineral/lithology | s19_combine / s22_boundary_v2 | missing | 0/7 | 0.0 | missing (0) |
| MC765 colour grid | s19_combine / s17_synth_gulf | 0.420 | 21/25 | 82.5 | 0.441 (20) |
| MC765 colour grid | s19_combine / s22_boundary_v2 | 0.214 | 21/25 | 82.7 | 0.224 (20) |
| One-off layout | s19_combine / s17_synth_gulf | 0.075 | 1/5 | 20.0 | 0.075 (1) |
| One-off layout | s19_combine / s22_boundary_v2 | 0.024 | 1/5 | 16.6 | 0.024 (1) |
| Texas 'Rock Types' box | s19_combine / s17_synth_gulf | 0.360 | 9/13 | 68.1 | 0.177 (7) |
| Texas 'Rock Types' box | s19_combine / s22_boundary_v2 | 0.359 | 9/13 | 57.5 | 0.176 (7) |

## Execution problems

| Copy | Strategy | Category | Error | Pages |
|---|---|---|---|---:|
| clean | s11_fft_legend | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 25 |
| clean | s17_synth_gulf | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 25 |
| clean | s19_combine | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 25 |
| clean | s22_boundary_v2 | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 25 |
| clean | s11_fft_legend | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 6 |
| clean | s17_synth_gulf | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 6 |
| clean | s19_combine | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 6 |
| clean | s22_boundary_v2 | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 6 |
| clean | s11_fft_legend | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| clean | s17_synth_gulf | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| clean | s19_combine | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| clean | s22_boundary_v2 | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| medium scan | s11_fft_legend | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 26 |
| medium scan | s17_synth_gulf | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 26 |
| medium scan | s19_combine | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 26 |
| medium scan | s22_boundary_v2 | depth_calibration_unavailable | ValueError: No consistent increasing depth calibration | 26 |
| medium scan | s11_fft_legend | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 9 |
| medium scan | s17_synth_gulf | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 9 |
| medium scan | s19_combine | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 9 |
| medium scan | s22_boundary_v2 | depth_calibration_unavailable | ValueError: Depth calibration extrapolates above zero at column top | 9 |
| medium scan | s11_fft_legend | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| medium scan | s17_synth_gulf | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| medium scan | s19_combine | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |
| medium scan | s22_boundary_v2 | depth_calibration_unavailable | ValueError: Fewer than two numeric depth anchors found | 1 |

## Limits and checks

- The depth check requires a positive, plausible slope, at least three anchors and at most 16 px residual. Missing fits and two-anchor fits do not pass.
- The shared parser is measured for s11, s17 and s22. The runner captures s19’s actual v3 legend and adjusted depth. If shared preprocessing fails first, only its available evidence remains.
- Short beds are under eight pixels. Redundant same-name boundaries are an over-segmentation proxy. Neither measure gives a true bed error rate.
- The fixed medium scanify preset adds blur, noise, fade, colour change and JPEG loss. It does not change geometry, apply shear or rescale the page.
- Family means weight pages equally. Coverage, agreement and a readable key can all be high for a wrong rock assignment.
- All image regions were reviewed before inference. The source-page, crop, rotation and key placement are recorded in each annotation.
- Source reuse rights remain limited by the manifest licence notes. Public access alone is not a reuse licence.
- Clean and scan inference file hashes match. The files were unchanged through both runs. No model or threshold was fitted here.
- Unit check: `python -X utf8 -m pytest harness/test_unlabelled_metrics.py -q` — `18 passed in 11.82s`.
- Combined check: `not re-run`.

## Commands and files

The harness uses plain Python. Downloaded image processing uses `-I` with explicit paths. The runs use two workers with one compute thread each and fixed model settings. Deterministic depth-fit ValueErrors are cached for the exact image and annotation. All new caches stay in this lane.

From `bakeoff/`, the actual runner is:

```powershell
python -X utf8 notes/round6_local/unlab_curate/run_suite.py ../external_unlabelled_test notes/round6_local/unlab_curate/runs/clean --workers 2
python -I -X utf8 notes/round6_local/unlab_curate/make_degraded.py .. ../external_unlabelled_test
python -X utf8 notes/round6_local/unlab_curate/run_suite.py ../external_unlabelled_test/degraded notes/round6_local/unlab_curate/runs/medium --workers 2 --clean-dir notes/round6_local/unlab_curate/runs/clean
```

The registered suite is `unlab_test`, with `truth_mode=none`. The runner sets `LITHOLOGY_DATASET` and the three `BENCH_UNLAB_TEST_*` path variables, then calls `harness.bench.evaluate`.

- Dataset: `external_unlabelled_test/`; flat gallery: `index.html`.
- Page scores and rejection reasons: `page_scores.csv`, `selection_decisions.json`.
- Box checks: `visual_review.json`, `set_validation.json`, `details/depth_zoom_*.jpg`.
- Method and metric limits: `CURATION_METHOD.md`, `METRICS.md`.
- Raw outputs: `runs/clean/`, `runs/medium/`.
- Per-page and family tables: `unlabelled_metrics.csv`, `unlabelled_by_family.csv` in each run.
- Strategy comparisons: `strategy_agreement.csv`, `strategy_agreement_by_family.csv`.
- Full machine-readable diagnostics: `unlabelled_report.json` in each run.
- Bench backup: `bakeoff/harness/bench.py.bak_r6`. No commit or push was made.
