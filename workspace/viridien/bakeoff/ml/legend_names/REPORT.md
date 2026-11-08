Legend names: opt-in v3 and S16

Selected using six external development wells only. Outer leave-one-well-out selection excludes the held-out well from parser setting selection and matcher fitting. Boundary probabilities are the boundary lane's saved out-of-fold rich_real probabilities. S16 uses the same boundaries and matcher as S15; only the parser changes. Test and in-domain eval were run once after the source/configuration hash lock. Existing S11/S13/S14/S15 final baseline measurements are reused, without another test run.

| Legend truth | Logs | Entries | Parser | Detected (IoU >= .5) | Exact name | Exact + usable | Canonical + usable |
|---|---:|---:|---|---:|---:|---:|---:|
| ext_dev (outer CV) | 54 | 1650 | v1 | 98.73% | 38.00% | 38.00% | 38.12% |
| ext_dev (outer CV) | 54 | 1650 | v3 | 98.73% | 94.18% | 94.18% | 94.18% |
| in-domain non-eval | 1 | 18 | v1 | 94.44% | 88.89% | 88.89% | 88.89% |
| in-domain non-eval | 1 | 18 | v3 | 100.00% | 94.44% | 94.44% | 94.44% |
| ext_test | 16 | 524 | v1 | 87.98% | 29.77% | 29.77% | 29.96% |
| ext_test | 16 | 524 | v3 | 87.98% | 73.09% | 73.09% | 73.09% |
| viridien_eval | 1 | 17 | v1 | 100.00% | 100.00% | 58.82% | 58.82% |
| viridien_eval | 1 | 17 | v3 | 100.00% | 88.24% | 52.94% | 58.82% |

All entry rates use all truth swatches as denominator, including missed detections. Exact + usable also requires is_lithology. A one-to-one IoU association prevents names from choosing their own match. Canonical comparison symmetrically expands a general abbreviation/colour lexicon; it is diagnostic only. Strategy row accuracy always uses the unmodified shared exact-name scorer. Non-lithology truth entries remain in the denominator where supplied by the annotations. In-domain legend truth is sparse and hand-labelled; interval truth is gold-first silver consensus.

| Split | Strategy | Row acc | Boundary F1 | Cluster ARI |
|---|---|---:|---:|---:|
| ext_dev CV (window macro) | S11 | 43.88% | 0.352 | 0.656 |
| ext_dev CV (window macro) | S13 | 46.17% | 0.555 | 0.768 |
| ext_dev CV (window macro) | S14 | 46.61% | 0.356 | 0.652 |
| ext_dev CV (window macro) | S15 | 50.34% | 0.553 | 0.813 |
| ext_dev CV (window macro) | S16 | 81.31% | 0.553 | 0.813 |
| ext_dev CV (well macro) | S11 | 35.75% | 0.292 | 0.539 |
| ext_dev CV (well macro) | S13 | 37.23% | 0.499 | 0.638 |
| ext_dev CV (well macro) | S14 | 39.76% | 0.288 | 0.543 |
| ext_dev CV (well macro) | S15 | 43.48% | 0.489 | 0.693 |
| ext_dev CV (well macro) | S16 | 72.35% | 0.489 | 0.693 |
| ext_test | S11 | 10.21% | 0.216 | 0.539 |
| ext_test | S13 | 15.57% | 0.324 | 0.430 |
| ext_test | S14 | 22.65% | 0.248 | 0.480 |
| ext_test | S15 | 17.11% | 0.301 | 0.532 |
| ext_test | S16 | 60.69% | 0.301 | 0.532 |
| viridien_eval | S11 | 36.47% | 0.546 | 0.632 |
| viridien_eval | S13 | 39.67% | 0.569 | 0.677 |
| viridien_eval | S14 | 49.11% | 0.428 | 0.550 |
| viridien_eval | S15 | 53.42% | 0.549 | 0.632 |
| viridien_eval | S16 | 53.43% | 0.549 | 0.632 |

Selected ensemble: {"easy": true, "cleanup": true, "easy_weight": 1.15}. Six parser regression tests pass (tests.log).

Dev parser name ceilings:
- S15: 54.77% of truth-filled rows have an available exact legend name.
- S16: 98.93% of truth-filled rows have an available exact legend name.

Final resource measurements (shared OCR included in peak RSS; seconds/window below excludes OCR):
- ext_test: 0.672 seconds/window; 0.291 seconds/window legend preparation (including cache reuse); peak RSS 0.64 GiB; 0 errors.
- viridien_eval: 0.452 seconds/window; 0.722 seconds/window legend preparation (including cache reuse); peak RSS 1.41 GiB; 0 errors.

Parser changes: full label cells retain descenders beyond short swatches; tight RapidOCR and CPU EasyOCR recognition; conservative vocabulary repair; left/right/below label candidates; physical frames supplement v2 detection; v2 panel classification; exact-pixel content cache with relative coordinates. No network downloads, filenames, per-log rules, or truth at inference.

Artifacts: harness/legend_v3.py; strategies/s16_legend_v3.py; ml/legend_names/selected.json; dev_names.json; dev_metrics.json; final_names.json; final_metrics.json; FINAL_LOCK.json; dev_predictions/; final_predictions/; cache/.

Reproduce from bakeoff: `python -m ml.legend_names.experiment dev`, `python -m pytest ml/legend_names/test_legend_v3.py -q`, then the guarded `python -m ml.legend_names.experiment final`, and `python -m ml.legend_names.experiment report`. Set LITHOLOGY_DATASET to the local in-domain dataset. All generated files stay under ml/legend_names.

The in-domain eval naming regression is retained as observed: `Sandy Clay/Shale` becomes `Sand Clay/Shale`, and `LOT/FIT` becomes `LOTFIT` and evades the existing exclusion rule. The parser was not changed after the final evaluation. See [FINAL_NOTES.md](FINAL_NOTES.md) for denominator details, sparse in-domain coverage, the empty-truth convention, and the completed artifact audit.
