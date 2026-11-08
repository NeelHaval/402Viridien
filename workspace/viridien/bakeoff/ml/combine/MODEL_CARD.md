# S19 model card

S19 uses the S18 joint decoder, the v3 legend parser, and the S17 model design.
All S19 model fits use the new data split. The final benchmark is pending.
Use `--strategy s19` in the shared harness. Only the team lead can run the final benchmark.

## Data limits

The dev set has 51 windows from five wells. ForsythB2-17 is outside this set.
This lane did not open its images or truth, or use its queries for a fit.
This lane did not search for the three sealed verify wells.
This lane did not run ext_test, ext_shared_verify or viridien_eval.
This lane did not open their truth files.
Earlier reports contain old final scores. This lane read those reports as requested.
Those scores did not set the new grid or select S19.

The S17 synthetic set has 1,920 train windows and 480 held windows.
The check uses 84 held windows, with 12 windows per family. The split uses source logs.
No source log occurs in both parts. No source is one of the 13 eval logs.
The check supplies the native legend swatches and names. It does not test legend OCR.
The D-582 and MC765 sources share legend art across logs. Their check does not test new art.
There is no human interval truth for non-eval in-domain logs in this mirror.
Thus, the in-domain selection check uses synthetic data from non-eval source logs.

The dev list SHA-256 is `84c8a6b57359e5e3f471a559d91ecdaf48def8167d12e2f1612684ba1e4b5631`.
The fit files list the five wells and each excluded query well.
The code checks that the dev list still has 51 windows before each main step.
It does not use eval_set_dev_full.json.

## Old model limits

The old S13, S14, S17 and S18 fits used ForsythB2-17 before the split changed.
Their weights are contaminated under the new split rule.
S19 does not load those weights or the old confidence model.
The boundary fits start from random weights. The matcher fits start from new estimates.
This lane also did not load the old selected parser file.
The parser uses explicit defaults: `{"cleanup": true, "easy": true, "easy_weight": 0.9}`.
S19 reuses model code, synthetic images, and raw OCR evidence from earlier lanes.
The inherited code and parser vocabulary came from earlier work on the old split.
The clean refit removes weight contamination. It cannot remove the history of that code.

## Method and selection

The boundary net has six dilated convolution layers and 24 hidden channels.
It uses 32 row features. The pair matcher uses 27 visual features and logistic regression.
There are two training choices: real data only, or 35% synthetic data.
Each choice has five fits that exclude one well, and one fit for all five wells.
The matcher also excludes the query well and all its swatch copies.
The synthetic train sources stay separate from the held source logs.
The selected net and matcher use only allowed train data. They do not use held synthetic data.

The grid has 28 settings. 15 settings keep the requested joint decoder
and a nonzero contribution from both synthetic-mixed models.
Only these settings can be selected for S19. The other settings are comparison checks.
This limit follows the lane task.
The first grid had 22 settings. The mixed models lost dev accuracy in this check.
We then checked six blends of the real-only and synthetic-mixed model outputs.
The boundary and matcher use the same blend weight. S19 gives the mixed models
weight 0.5. It gives the real-only models the remaining weight.
Both grids use the same CV results. This extra selection can increase optimism.
The score for each window is row accuracy times boundary F1.
The selection value is 75% dev mean by well plus 25% synthetic mean by family.
The same CV results select the settings. These values are selection estimates.
They are not scores from a new blind test. The reported well means give each well equal weight.
The small dev set can give weak estimates for a new well or scan type.

| Method | Dev row | Dev boundary | Dev ARI | Dev score | Synthetic score | Selection value |
|---|---:|---:|---:|---:|---:|---:|
| Real models, joint decoder, v3 | 0.8943 | 0.5660 | 0.8058 | 0.5092 | 0.4731 | 0.5002 |
| Synthetic models, CNN cuts, v3 | 0.7935 | 0.5077 | 0.7653 | 0.4146 | 0.6202 | 0.4660 |
| Synthetic models, joint decoder, v3 | 0.7988 | 0.5395 | 0.7817 | 0.4499 | 0.6102 | 0.4900 |
| Real boundary, synthetic matcher, joint | 0.8071 | 0.5553 | 0.7900 | 0.4676 | 0.4830 | 0.4715 |
| Synthetic boundary, real matcher, joint | 0.8744 | 0.5308 | 0.8024 | 0.4674 | 0.5882 | 0.4976 |
| Selected S19 | 0.8503 | 0.5741 | 0.8306 | 0.5107 | 0.5158 | 0.5119 |

These comparisons use new fits and the same allowed data. They are not reruns of old final scores.
The first row keeps the S18 joint rule with new real-only models and v3 names.
The second row uses CNN cuts with new S17-style synthetic-mixed models and v3 names.
It keeps the same drawn-rule policy as the other grid settings.

| Dev well | Windows | Row accuracy | Boundary F1 | ARI | Score |
|---|---:|---:|---:|---:|---:|
| 386825 | 2 | 0.8757 | 0.9354 | 0.6866 | 0.8203 |
| 4304150011 | 21 | 0.9296 | 0.7018 | 0.8612 | 0.6569 |
| barlow3-14 | 6 | 0.6973 | 0.3591 | 0.8040 | 0.2747 |
| flat-bay-2 | 9 | 0.7671 | 0.3768 | 0.8469 | 0.3154 |
| storm | 13 | 0.9818 | 0.4973 | 0.9544 | 0.4861 |

The selected settings are:

```json
{
  "blank_ink": 0.006,
  "blend": 1.0,
  "boundary_fraction": 0.35,
  "boundary_weight": 0.0,
  "gap": 16,
  "matcher_fraction": 0.35,
  "min_bed": 8,
  "mode": "joint",
  "model_mix": 0.5,
  "penalty": 0.0,
  "preserve": "rules",
  "reject": 2.0,
  "rule_tolerance": 8,
  "threshold": 0.4
}
```

## Parser check

A fixed check replaces v3 with v1 on the 51 dev windows after selection.
It keeps the selected model fits and all decoder settings. It does not refit the matcher.
Thus, it measures the parser change with a matcher trained with v3 entries.
The dev mean by well changes from row accuracy 0.4454 and score 0.2745
with v1 to row accuracy 0.8503 and score 0.5107 with v3.
The fraction of filled dev rows with an exact name in the parsed legend changes
from 0.4755 to 0.9873, with equal weight per well.
Scores use the shared exact-name rule. They do not use expanded rock-name aliases.

## Depth repair

The old report lists a 384.4 ft mean boundary error for ext_kgs_EmmaWard1_p002.
This lane inspected its saved OCR anchors and calibration. It did not open its image or truth.
The old fit chose 3.66429 ft per pixel. It ranked anchor span before residual error.
Three printed anchors, 4000, 4050 and 4100, support about 0.13323 ft per pixel.
The new fit ranks equal support by pixel residual before span.
The regression check uses the saved OCR anchors only. No truth sets the correction.
S19 reads small depth-track crops when the input calibration has few anchors,
a large residual, or a negative top depth. It accepts a better fit with at least three anchors.
It uses full-image rows and keeps the unit conversion. It returns explicit depths,
so the harness keeps the repaired scale in the six-column output.
There are no log-name rules or fixed depth offsets.
The final benchmark must confirm the depth error on that held window.

The fixed dev audit checked 46 unique logs and changed 0 calibrations.
Mean absolute error at all exact dev interval endpoints was 0.2010 ft before
and 0.2010 ft after the check. Each log has equal weight in these two means.
This is a depth check, not the harness metric over matched boundaries.

## Checks and limits

`tests.log` records checks for split limits, source separation, the depth failure,
several depth scales, duplicate OCR, metre conversion, blank gaps, drawn rules,
absolute pixel rows, and excluded-well fit records.
All 18 checks passed.
The deployment check uses two allowed dev windows. Its predictions match the saved-data path.
Both return ordered, non-overlapping intervals inside the requested window.
The larger runtime in this check was 1.050 seconds.
This is a smoke check on dev. It is not a runtime bound for all logs.
The model exports use numeric arrays and CPU torch weights. All checks use CPU.
The lane limits torch, BLAS, OMP and MKL to three compute threads. OpenCV uses one.
The code derives paths from the repo or the dataset environment variable.
There are no fixed laptop or VM paths in the code.
All lane files are under ml/combine or the s19 strategy file. There were no commits or pushes.

| Step | Seconds | Peak RSS GiB |
|---|---:|---:|
| check | 2.15 | 0.402 |
| compare | 19.31 | 0.390 |
| depth_check | 6.91 | 0.635 |
| fit | 0.01 | 0.367 |
| lock_selection | 0.00 | 0.367 |
| prepare | 67.68 | 0.450 |
| select_blends | 104.10 | 0.386 |
| select | 348.38 | 0.389 |
| train | 41.50 | 1.025 |

S19 returns visual match costs and margins as extra fields. They are not calibrated probabilities.
The harness writes its normal six CSV columns. S19 does not use the old confidence weights.
Short beds, incorrect OCR names, mixtures, and missing swatches remain possible failure modes.
The held synthetic check uses artificial scan effects. Real scan results can differ.
No final-set score is claimed here. The team lead will run the one final benchmark.

## Reproduce

Set LITHOLOGY_DATASET to the local dataset. Use the project Python environment.
Run these commands from bakeoff. Set OMP_NUM_THREADS, MKL_NUM_THREADS and
OPENBLAS_NUM_THREADS to 3. Set PYTHONDONTWRITEBYTECODE to 1 to keep shared code read-only.

```sh
python -m ml.combine.experiment prepare
python -m ml.combine.experiment train --steps 450
python -m ml.combine.experiment select
python -m ml.combine.experiment select_blends
python -m ml.combine.experiment fit
python -m ml.combine.experiment check
python -m ml.combine.experiment compare
python -m ml.combine.experiment depth_check
python -m pytest ml/combine/test_combine.py -q -o cache_dir=ml/combine/.pytest_cache
python -m ml.combine.report
```

The experiment has no command for final sets. Do not use the normal harness on them in this lane.
READY.json records code and model hashes for the team lead.
