# Legend v4 and S20

Legend v4 keeps printed rock names. S20 calls S18 with the new legend entries. It keeps the S18 boundary, match, decode, and confidence code. The shared parser does not change.

## Result

| Data | Parser | Pages | Printed entries | Detection | Exact name | Exact name and usable |
|---|---|---:|---:|---:|---:|---:|
| External dev, grouped CV | v1 | 46 | 1490 | 98.59% | 35.64% | 35.64% |
| External dev, grouped CV | v3 | 46 | 1490 | 98.59% | 93.62% | 93.62% |
| External dev, grouped CV | v4 | 46 | 1490 | 98.59% | 93.42% | 93.42% |
| In-domain non-eval, development check | v1 | 7 | 127 | 73.23% | 70.87% | 58.27% |
| In-domain non-eval, development check | v3 | 7 | 127 | 82.68% | 70.08% | 69.29% |
| In-domain non-eval, development check | v4 | 7 | 127 | 100.00% | 99.21% | 97.64% |

Exact agreement ignores case and spaces. It does not expand synonyms. Each truth sample can match one predicted sample at IoU 0.5 or more. Missed samples count as errors. The in-domain labels include two No Sample cells. Those cells can have a correct name but are not usable rocks. Page totals repeat some identical legends. They are not independent samples.

## Data limits

The current external dev file has 51 windows from five wells and 46 pages. Only these external pages were opened. ForsythB2-17 was excluded. The full old dev file was not read. No protected benchmark was run. No protected image or truth file was opened. The sealed wells were not searched for or restored.

The fixed in-domain eval list was read only to exclude its 13 log names. Six new legend label files come from manual reads of non-eval images. One existing label file supplies the Hadrian legend. The manual labels and source legend crops are in this folder. No real non-eval interval truth is present in the harness truth folder. The in-domain strategy results below therefore use synthetic columns.

## Changes

- Keep valid geological adjectives and plural rock names. Do not reduce Dolomites to Dolomite or Volcanics to Volcanic.
- Keep readable page text on the right when tight OCR fails. This stops a label on the left from taking its place.
- Keep printed separators when the letters agree across OCR engines.
- Apply section headings for mineral, modifier, fossil, and technical cells. Reject technical names when OCR removes spaces or punctuation.
- Split fused rows in a shallow Rock Types grid. This rule requires wide samples and unequal sample heights. Intact grids retain their geometry. Read each label inside its own row and column.
- Cache raw OCR by exact pixels. Read an inherited cache only for the current image digest. Store new evidence inside this lane. No filename or well rule changes a runtime label.

## Legend names by well and family

| Group | Parser | Entries | Detection | Exact name | Exact name and usable |
|---|---|---:|---:|---:|---:|
| 386825 | v1 | 20 | 100.00% | 40.00% | 40.00% |
| 386825 | v3 | 20 | 100.00% | 100.00% | 100.00% |
| 386825 | v4 | 20 | 100.00% | 100.00% | 100.00% |
| 4304150011 | v1 | 903 | 97.67% | 25.58% | 25.58% |
| 4304150011 | v3 | 903 | 97.67% | 90.70% | 90.70% |
| 4304150011 | v4 | 903 | 97.67% | 90.70% | 90.70% |
| barlow3-14 | v1 | 118 | 100.00% | 18.64% | 18.64% |
| barlow3-14 | v3 | 118 | 100.00% | 98.31% | 98.31% |
| barlow3-14 | v4 | 118 | 100.00% | 96.61% | 96.61% |
| flat-bay-2 | v1 | 189 | 100.00% | 68.78% | 68.78% |
| flat-bay-2 | v3 | 189 | 100.00% | 95.24% | 95.24% |
| flat-bay-2 | v4 | 189 | 100.00% | 95.24% | 95.24% |
| storm | v1 | 260 | 100.00% | 53.85% | 53.85% |
| storm | v3 | 260 | 100.00% | 100.00% | 100.00% |
| storm | v4 | 260 | 100.00% | 99.62% | 99.62% |
| D-582 1-bit grid | v1 | 10 | 100.00% | 80.00% | 80.00% |
| D-582 1-bit grid | v3 | 10 | 100.00% | 80.00% | 80.00% |
| D-582 1-bit grid | v4 | 10 | 100.00% | 90.00% | 90.00% |
| Geoservices grid | v1 | 18 | 94.44% | 88.89% | 88.89% |
| Geoservices grid | v3 | 18 | 100.00% | 94.44% | 94.44% |
| Geoservices grid | v4 | 18 | 100.00% | 100.00% | 100.00% |
| Gulf 3-panel | v1 | 15 | 100.00% | 100.00% | 0.00% |
| Gulf 3-panel | v3 | 15 | 100.00% | 86.67% | 86.67% |
| Gulf 3-panel | v4 | 15 | 100.00% | 100.00% | 100.00% |
| MC761 mineral/lithology | v1 | 12 | 100.00% | 100.00% | 91.67% |
| MC761 mineral/lithology | v3 | 12 | 100.00% | 100.00% | 91.67% |
| MC761 mineral/lithology | v4 | 12 | 100.00% | 100.00% | 91.67% |
| MC765 colour grid | v1 | 23 | 100.00% | 100.00% | 100.00% |
| MC765 colour grid | v3 | 23 | 100.00% | 86.96% | 86.96% |
| MC765 colour grid | v4 | 23 | 100.00% | 100.00% | 100.00% |
| One-off layout | v1 | 16 | 100.00% | 100.00% | 100.00% |
| One-off layout | v3 | 16 | 100.00% | 93.75% | 93.75% |
| One-off layout | v4 | 16 | 100.00% | 100.00% | 100.00% |
| Texas 'Rock Types' box | v1 | 33 | 0.00% | 0.00% | 0.00% |
| Texas 'Rock Types' box | v3 | 33 | 33.33% | 12.12% | 12.12% |
| Texas 'Rock Types' box | v4 | 33 | 100.00% | 100.00% | 96.97% |

The v4 OCR weight is selected without the held well. The objective gives each training well equal weight. The three candidate EasyOCR weights are 0.9, 1.0, and 1.15. Cleanup is on for all three. The deployed weight is 1.15. The v3 reference keeps its inherited weight of 1.15. Thus the v4 grouped result includes the cost of setting selection. These are development estimates. Parser code changes used these development labels.

## Strategy metrics on external dev

Each held well is excluded from fresh boundary and matcher fits. Each CNN starts from new random weights. It uses rich features, 450 steps, and seed 100 plus the fold number. The matcher is a fresh logistic fit. Training pairs use only the other current dev wells and synthetic samples made from those wells. No old training-pair file or saved boundary probabilities are used.

S18 below uses parser v1. S18_v3 uses parser v3. S20 uses parser v4. The three variants share each fold model, proposals, and fixed S18 decoder. Confidence does not change labels and is not part of these CV scores.

| Group | Strategy | Windows | Row accuracy | Boundary F1 | ARI | Score |
|---|---|---:|---:|---:|---:|---:|
| 386825 | s18 | 2 | 38.30% | 0.897 | 0.575 | 0.343 |
| 386825 | s18_v3 | 2 | 80.18% | 0.897 | 0.575 | 0.720 |
| 386825 | s20 | 2 | 80.18% | 0.897 | 0.575 | 0.720 |
| 4304150011 | s18 | 21 | 56.85% | 0.720 | 0.857 | 0.373 |
| 4304150011 | s18_v3 | 21 | 93.15% | 0.720 | 0.857 | 0.676 |
| 4304150011 | s20 | 21 | 93.15% | 0.720 | 0.857 | 0.676 |
| barlow3-14 | s18 | 6 | 12.67% | 0.399 | 0.638 | 0.061 |
| barlow3-14 | s18_v3 | 6 | 50.22% | 0.399 | 0.638 | 0.217 |
| barlow3-14 | s20 | 6 | 50.22% | 0.399 | 0.638 | 0.217 |
| flat-bay-2 | s18 | 9 | 53.08% | 0.445 | 0.813 | 0.278 |
| flat-bay-2 | s18_v3 | 9 | 68.00% | 0.445 | 0.813 | 0.299 |
| flat-bay-2 | s20 | 9 | 68.00% | 0.445 | 0.813 | 0.299 |
| storm | s18 | 13 | 50.63% | 0.474 | 0.931 | 0.246 |
| storm | s18_v3 | 13 | 95.94% | 0.474 | 0.931 | 0.452 |
| storm | s20 | 13 | 95.94% | 0.474 | 0.931 | 0.452 |

| Mean | Strategy | Row accuracy | Boundary F1 | ARI | Score |
|---|---|---:|---:|---:|---:|
| window | s18 | 48.67% | 0.578 | 0.831 | 0.286 |
| well | s18 | 42.31% | 0.587 | 0.763 | 0.260 |
| window | s18_v3 | 83.86% | 0.578 | 0.831 | 0.500 |
| well | s18_v3 | 77.50% | 0.587 | 0.763 | 0.473 |
| window | s20 | 83.86% | 0.578 | 0.831 | 0.500 |
| well | s20 | 77.50% | 0.587 | 0.763 | 0.473 |

## Strategy metrics by non-eval family

These are synthetic diagnostics, not real log scores or independent CV. Three fixed seeds make 3200-row columns for each labelled non-eval log. Each bed has 200 rows. Truth names come from the manual legend read. Patterns come from the inner sample pixels. The CNN and matcher are fitted on the 51 external dev windows only. The same models serve all families. The synthetic results did not select a model or threshold.

| Group | Strategy | Windows | Row accuracy | Boundary F1 | ARI | Score |
|---|---|---:|---:|---:|---:|---:|
| D-582 1-bit grid | s18 | 3 | 35.72% | 0.341 | 0.446 | 0.124 |
| D-582 1-bit grid | s18_v3 | 3 | 40.33% | 0.341 | 0.446 | 0.139 |
| D-582 1-bit grid | s20 | 3 | 44.05% | 0.341 | 0.446 | 0.151 |
| Geoservices grid | s18 | 3 | 79.89% | 0.615 | 0.798 | 0.498 |
| Geoservices grid | s18_v3 | 3 | 86.46% | 0.651 | 0.818 | 0.560 |
| Geoservices grid | s20 | 3 | 88.11% | 0.626 | 0.823 | 0.552 |
| Gulf 3-panel | s18 | 3 | 0.00% | 0.000 | 0.000 | 0.000 |
| Gulf 3-panel | s18_v3 | 3 | 77.18% | 0.839 | 0.862 | 0.658 |
| Gulf 3-panel | s20 | 3 | 89.50% | 0.839 | 0.862 | 0.771 |
| MC761 mineral/lithology | s18 | 3 | 98.15% | 0.939 | 0.965 | 0.922 |
| MC761 mineral/lithology | s18_v3 | 3 | 98.15% | 0.939 | 0.965 | 0.922 |
| MC761 mineral/lithology | s20 | 3 | 98.15% | 0.939 | 0.965 | 0.922 |
| MC765 colour grid | s18 | 3 | 95.86% | 0.938 | 0.931 | 0.900 |
| MC765 colour grid | s18_v3 | 3 | 84.01% | 0.938 | 0.931 | 0.792 |
| MC765 colour grid | s20 | 3 | 95.86% | 0.938 | 0.931 | 0.900 |
| One-off layout | s18 | 3 | 90.95% | 0.728 | 0.887 | 0.672 |
| One-off layout | s18_v3 | 3 | 84.77% | 0.728 | 0.887 | 0.630 |
| One-off layout | s20 | 3 | 90.95% | 0.728 | 0.887 | 0.672 |
| Texas 'Rock Types' box | s18 | 3 | 0.00% | 0.732 | 0.272 | 0.000 |
| Texas 'Rock Types' box | s18_v3 | 3 | 16.65% | 0.888 | 0.667 | 0.149 |
| Texas 'Rock Types' box | s20 | 3 | 91.09% | 0.989 | 0.930 | 0.902 |

Depth MAE is not available. These checks do not fit a depth calibration. The saved per-window records retain the null depth field.

## Inherited model history

The deployed S20 wrapper inherits the S18 weights. The old boundary training record lists 59 windows and includes ForsythB2-17. The old matcher preparation also used the former dev split. The S18 decoder settings and confidence fit were selected on that former split. These inherited assets have historical exposure to the new shared_verify well. S20 is not a clean retrain of the full S18 system.

The new CV fits exclude that well. However, the decoder settings, model form, logistic strength, and training schedule are fixed inherited choices. Their earlier selection included the former dev split. Treat the new CV scores as controlled parser comparisons with this history. Do not call them an independent clean estimate of the entire system. No evidence in the earlier training code shows test or in-domain eval truth in the boundary or matcher fit. This is a code review statement, not proof of all past runs.

Fresh full-dev weights are saved as clean_boundary.pt and clean_matcher.npz for review. They are not loaded by S20. No protected final run was used to select between old and new weights. The orchestrator owns the final benchmark.

## Checks and limits

Nine input and contract tests pass. They cover printed words, right-cell ownership, separators, technical cells, section headings, translated cache coordinates, the S20 call, intact wide grids, and fused wide rows.

The deployed S20 wrapper also passed a structure check on one allowed dev window: 63 intervals, 0.99 seconds, and 0.40 GiB peak RSS. This check read no truth.

- names: 9.3 seconds; peak RSS 1.54 GiB; at most three compute threads.
- cv: 83.5 seconds; peak RSS 0.60 GiB; at most three compute threads.
- synthetic: 34.5 seconds; peak RSS 1.70 GiB; at most three compute threads.

Limits: the manual sample has one log per family. Some non-eval logs can share a well or template with an excluded log. The manual name check is development data. Synthetic columns do not represent all real mixtures, blank zones, scale changes, or scan faults. The small grouped external name loss is retained as measured.

## Run again

From bakeoff, set LITHOLOGY_DATASET to the local dataset path. Use the local Python environment. Set OMP_NUM_THREADS, MKL_NUM_THREADS, and OPENBLAS_NUM_THREADS to 3. The code derives all output paths from this folder.

```text
python -m ml.legend_v4.experiment names
python -m ml.legend_v4.experiment prepare
python -m ml.legend_v4.experiment cv
python -m ml.legend_v4.experiment synthetic
python -m pytest ml/legend_v4/test_legend_v4.py -q -o cache_dir=ml/legend_v4/pytest_cache
python -m ml.legend_v4.report
```

The experiment has no final action. All writes stay in the owned lane paths. No commits, pushes, child agents, external downloads, or team service calls were made.
