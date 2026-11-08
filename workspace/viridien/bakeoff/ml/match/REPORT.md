Learned legend matching lane

Selected on six external dev wells only, using leave-one-well-out validation. Every real and synthetic query from a held-out well is excluded from that fold's training. The 20 external test windows and 13 in-domain evaluation windows were opened only in the locked final run.

The selected scorer is logistic on 27 visual pair distances, trained with balanced well weights. Features include multiscale FFT, Lab colour, ink density, stroke run lengths, gradient orientation, and local pixel comparisons. No filenames or label vocabulary enter the model. The numeric matcher.npz checkpoint runs without scikit-learn pickle compatibility requirements.

S14 uses the original S11 PELT and drawn-rule boundary proposals; S15 uses the published S13 checkpoint. Matching training uses exact dev legend swatches and merged truth intervals, plus phase/scale-augmented dev swatch tiles. A linear model, boosted trees, FFT, multiscale FFT, and stroke-run matching were compared. No CNN embedding was trained. Rejection is disabled because an Unknown prediction cannot improve exact-name row accuracy on this truth.

| Dev row accuracy | S11 | Selected matcher (out of fold) |
|---|---:|---:|
| Macro over six wells | 35.75% | 39.76% |
| Macro over 59 windows | 43.88% | 46.61% |

| Held-out dev well | S11 | Selected matcher |
|---|---:|---:|
| 386825 | 13.33% | 19.00% |
| 4304150011 | 54.67% | 54.49% |
| ForsythB2-17 | 44.05% | 63.72% |
| barlow3-14 | 8.99% | 7.56% |
| flat-bay-2 | 51.69% | 51.69% |
| storm | 41.76% | 42.12% |

Parser versus matcher (59 dev-window means; scorer remains out of fold):

| Legend entries | True intervals | S11 boundary proposals |
|---|---:|---:|
| Shared parser boxes/names | 53.47% | 46.61% |
| Oracle boxes/names | 87.34% | 76.18% |

Parsed-name availability limits perfect matching to 54.77% row accuracy. Missing, excluded, or misread names therefore account for at least 45.23 percentage points of error. With true intervals, 1.30 further points are lost in choosing among available names. Oracle boxes/names improve true-interval accuracy by 33.87 points; the remaining 12.66 points reflect matching/visual ambiguity even with oracle legends. These are complementary diagnostics, not disjoint additive attributions: correcting names also changes the set and quality of swatches.

Single final evaluation (window-macro means; shared parser and depth calibration; in-domain truth is gold-first silver consensus):

| Split | Strategy | Row acc | Boundary F1 | Cluster ARI | Depth MAE (ft) | Seconds/window | Peak RSS (GB) | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ext_test | S11 | 10.21% | 0.216 | 0.539 | 0.6 | 0.18 | 0.57 | 0 |
| ext_test | S12 | 0.00% | 0.012 | 0.286 | 0.7 | 0.11 | 0.57 | 0 |
| ext_test | S13 | 15.57% | 0.324 | 0.430 | 0.5 | 0.15 | 0.57 | 0 |
| ext_test | S14 | 22.65% | 0.248 | 0.480 | 13.7 | 0.38 | 0.57 | 0 |
| ext_test | S15 | 17.11% | 0.301 | 0.532 | 0.5 | 0.78 | 0.57 | 0 |
| viridien_eval | S11 | 36.47% | 0.546 | 0.632 | 0.3 | 0.31 | 1.45 | 0 |
| viridien_eval | S12 | 14.68% | 0.175 | 0.198 | n/a | 0.21 | 0.71 | 12 |
| viridien_eval | S13 | 39.67% | 0.569 | 0.677 | 0.3 | 0.17 | 1.45 | 0 |
| viridien_eval | S14 | 49.11% | 0.428 | 0.550 | 0.3 | 0.47 | 1.45 | 0 |
| viridien_eval | S15 | 53.42% | 0.549 | 0.632 | 0.0 | 0.56 | 1.45 | 0 |

Recorded runtime failures:

- viridien_eval, s12_emil: URLError: <urlopen error [Errno -2] Name or service not known>

S12's missing EasyOCR weights and failed network download make its affected in-domain baseline incomplete. Those windows retain zero predictions and were not rerun.

Error details and per-window timings are retained in final_metrics.json. Preprocessing failures produce empty predictions for every strategy on that window, consistent with the harness. One in-domain consensus window has no filled truth rows: the shared scorer returns row_acc=1 for empty truth, even for an empty prediction. Reported window means retain that convention; S12 therefore receives this credit despite its OCR error on that window. Runtime measurements exclude shared OCR/calibration. S12 uses its strategy-default Drive checkpoint; the exact hash is recorded. CPU thread requests are capped at three; no GPU or paid APIs are used.

Reproduce from bakeoff with the repository Python environment:

```sh
python -m ml.match.experiment prepare
python -m ml.match.experiment select
python -m ml.match.experiment baseline
python -m pytest ml/match/test_match.py -q
# final is guarded: it refuses a second run when FINAL_LOCK.json exists
python -m ml.match.experiment final
python -m ml.match.report
```

Use LITHOLOGY_DATASET for the in-domain dataset location. External data paths are derived from the repository. For normal harness inference use --strategy s14 or --strategy s15. All dev data, selection results, final CSVs and caches for this lane are under bakeoff/ml/match/. The runtime strategies never import this experiment module or access truth.

Environment: Python 3.12.3, NumPy 2.5.4, scikit-learn 1.9.1. Validation: six focused matching tests pass; exporting the selected numeric checkpoint changed zero dev decisions.
