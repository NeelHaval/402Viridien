# S17 style-matched synthetic training

Frozen models are selected using six outside dev wells and held-source synthetic windows only. S17 was evaluated once on 20 external test windows and 13 in-domain evaluation windows. S13/S15 values reuse the matching lane’s archived final predictions and metrics; those baselines were not rerun.

| Split | Strategy | Row acc | Boundary F1 | ARI | Score | Depth MAE ft | s/window | Max RSS GB | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ext_test | s13 | 15.57% | 0.324 | 0.430 | 0.071 | 0.49 | 0.155 | 0.57 | 0 |
| ext_test | s15 | 17.11% | 0.301 | 0.532 | 0.073 | 0.48 | 0.777 | 0.57 | 0 |
| ext_test | s17 | 18.45% | 0.268 | 0.525 | 0.059 | 0.39 | 0.651 | 0.41 | 0 |
| viridien_eval | s13 | 39.67% | 0.569 | 0.677 | 0.245 | 0.32 | 0.165 | 1.45 | 0 |
| viridien_eval | s15 | 53.42% | 0.549 | 0.632 | 0.367 | 0.00 | 0.561 | 1.45 | 0 |
| viridien_eval | s17 | 63.75% | 0.656 | 0.727 | 0.432 | 0.31 | 0.364 | 1.19 | 0 |

## In-domain families

Each cell is **row accuracy / boundary F1 / ARI**; windows are weighted equally within family. Families come directly from `analysis/label_shares_per_log.csv`.

| Family | Windows | S13 | S15 | S17 |
|---|---:|---:|---:|---:|
| Gulf 3-panel | 2 | 67.71% / 0.673 / 0.656 | 94.67% / 0.757 / 0.839 | 92.88% / 0.743 / 0.790 |
| Geoservices grid | 2 | 51.36% / 0.686 / 0.495 | 54.92% / 0.647 / 0.558 | 51.19% / 0.736 / 0.620 |
| D-582 1-bit grid | 3 | 24.34% / 0.136 / 0.546 | 0.62% / 0.062 / 0.166 | 49.28% / 0.430 / 0.570 |
| MC765 colour grid | 1 | 0.00% / 1.000 / 1.000 | 100.00% / 1.000 / 1.000 | 100.00% / 1.000 / 1.000 |
| MC761 mineral/lithology | 1 | 93.62% / 0.409 / 0.773 | 94.27% / 0.373 / 0.778 | 94.12% / 0.420 / 0.783 |
| Texas 'Rock Types' box | 2 | 46.22% / 0.862 / 0.857 | 46.37% / 0.848 / 0.858 | 47.42% / 0.869 / 0.884 |
| One-off layout | 2 | 9.28% / 0.572 / 0.689 | 53.24% / 0.535 / 0.713 | 51.92% / 0.562 / 0.688 |

## Selection and ablations

Selection objective: 75% mean of the six dev-well composite scores plus 25% mean of the seven synthetic-family composite scores. Composite score is per-window row accuracy × boundary F1. These pooled CV predictions also select hyperparameters; selection values are optimistic, not a second unbiased test. The synthetic full-pipeline validation subset is fixed before CV: 12 windows per family (84 total). All 480 held windows have cached probabilities, and held query matching uses all available held queries.

| Boundary synth fraction | Threshold | Matcher style weight | Dev well score | Synthetic family score | Objective |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.70 | 0.00 | 0.2390 | 0.5571 | 0.3185 |
| 0.15 | 0.55 | 0.15 | 0.2323 | 0.5975 | 0.3236 |
| 0.35 | 0.55 | 0.35 | 0.2437 | 0.6343 | 0.3414 |

| Matcher style weight | Dev well row acc on S11 proposals | Held synthetic pair accuracy (family mean) |
|---:|---:|---:|
| 0.00 | 39.76% | 76.64% |
| 0.15 | 39.22% | 80.06% |
| 0.35 | 39.54% | 79.46% |

## What helped

Style-augmented pair training changed held synthetic Gulf matching from 59.38% to 66.67%. Its dev well-macro row accuracy on S11 proposals changed from 39.76% to 39.54%; synthetic improvement did not imply a dev improvement.

The selected boundary retraining samples synthetic crops with probability 0.35; the ablation table records the tradeoff with real-only boundaries.

On the single final in-domain run, Gulf row accuracy changed by -1.79 percentage points versus S15 (94.67% → 92.88%). This is a held-out result, retained without further tuning.

The largest in-domain family improvement is D-582: row accuracy rises from 0.62% to 49.28%, with boundary F1 0.062 → 0.430. This is the combined training pipeline’s result; there was no final-set factorial ablation of individual scan augmentations.

External-test row accuracy rises from 17.11% to 18.45%, but boundary F1 falls from 0.301 to 0.268 and composite score falls from 0.073 to 0.059. S17 is not an across-the-board replacement for S15.

## Data and generator

Generated 2,400 windows: 1,920 training and 480 held windows, each 1,536 rows. Gulf has 960 windows; each of the other six families has 240. PNGs, exact CSVs, and compressed 32-channel boundary features/targets are in `data/`. `synthetic_index.json` records every source, seed, split, and interval. No in-domain human truth is consumed during training.

Usable native tiles came from 27 of 38 allowed non-eval logs. The remaining 11 had fewer than two usable lithology entries under the shared parser and were omitted. Each family reserves one source log before generation; source log IDs do not cross the synthetic split. D-582 and MC765 have identical legend pixels across their non-eval source logs, so their validation checks new windows and artefacts, not unseen legend artwork. Other families have distinct held-source legend hashes.

Colours and hatch symbols are native legend crops, tiled with random phase and 0.85/1/1.15 scale. Source-measured vertical-rule positions and native text crops are reused. Gulf columns have a central primary interpretation track and percentage mixtures on side tracks. Other grid families add independent horizontal grid lines, with blank intervals and optional real boundary rules in all families. Synthetic legend pages reuse each source’s box layout and shuffle the entries. The matcher receives swatches directly, as the shared strategy contract requires; this does not train the legend parser.

Scan augmentations are explicitly chosen ranges, not fitted distributions: fade 0–18%, noise sigma 0–3.5, Gaussian blur sigma 0.35–0.85, JPEG quality 45–95, horizontal shear ±0.004, and ordered 1-bit dithering. Horizontal shear leaves every row boundary unchanged; no vertical warp invalidates the exact truth. Procedural text placement and panel proportions are approximations; this is not a reconstruction of entire source pages.

## Models and limits

Boundary architecture and features are unchanged from S13: six dilated Conv1d layers, 24 hidden channels, 32 panel/colour/rule/change features. Each augmented candidate is retrained from scratch in each fold; the published all-dev S13 checkpoint is never used to initialize a held-well model. Each fold excludes every real and swatch-augmented query from its held well. S14’s 27 visual pair features feed a newly fitted logistic model with balanced positive/negative and well/family weights. No filenames or family IDs are inference features.

Frozen selection: boundary synthetic sampling fraction 0.35, threshold 0.55, gap 24, merge same-label intervals; matcher synthetic weight 0.35. The selected all-dev boundary candidate is reused exactly, without further training on held synthetic data. Portable matcher export is checked against scikit-learn decisions.

Weights total 50,859 bytes. Training/CV runtime is 0.69 minutes, including all 14 boundary fits; selection takes 2.72 minutes including matcher fits and scoring. Peak process RSS across preparation/CV/selection/fit/final is 1.20 GB. Torch and BLAS are capped at three threads and OpenCV at one. No GPU is used.

In-domain truth is gold-first silver consensus. One in-domain window has no filled consensus truth; the shared scorer awards row accuracy 1 for empty truth. Family means preserve that convention. External truth is exact. Runtime excludes shared legend/depth preprocessing. Missing legend names remain an upstream limitation: improving textures cannot recover a label the parser omitted.

Validation: 11 focused pytest checks pass, covering source/eval exclusion, held-source separation, deterministic generation and interval geometry in all seven families, held-well exclusion, and agreement between selection-time and deployed interval naming.

## Reproduction

Set `LITHOLOGY_DATASET` to the local dataset directory, use the project Python environment, and run from `bakeoff/`. No paths are tied to the VM or Windows. All generated writes stay in this lane.

```sh
python -c "from ml.synth_gulf.generator import assets; assets()"
python -m ml.synth_gulf.experiment prepare
python -m ml.synth_gulf.experiment cv --steps 450
python -m ml.synth_gulf.experiment select
python -m ml.synth_gulf.experiment fit
python -m pytest ml/synth_gulf/test_synth_gulf.py -q -o cache_dir=ml/synth_gulf/.pytest_cache
# One final pass only; refuses to rerun once FINAL_LOCK.json exists.
python -m ml.synth_gulf.experiment final
python -m ml.synth_gulf.experiment report
python -m harness.run --strategy s17 --gt consensus
```

The last command is normal inference usage; do not use it to rescore the frozen holdout during selection. Prerequisites are the existing dev-only boundary feature cache and matching query cache. `assets.json`, `preparation.json`, `selection.json`, `fit.json`, `FINAL_LOCK.json`, and `final_metrics.json` retain provenance, ablations, hashes, and all final per-window results.
