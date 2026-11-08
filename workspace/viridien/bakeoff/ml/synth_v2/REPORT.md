# S21 synthetic model report

S21 is ready for the team benchmark. The team must run the final benchmark.
This lane did not run the final benchmark.

## Data limits

The current dev set has 51 windows from five wells. ForsythB2-17 is not in this set.
No sealed verify well was opened or rebuilt. No external test truth was opened.
No truth from the 13 in-domain eval logs was opened.
The eval list was used only to exclude source logs.
The old full dev list was not used.

All real data were read again from the current dev list. Old real training caches were not used.
Each CV fold leaves out one whole well. Boundary and mixed matcher fits use the same well split.
The final boundary fit uses the five allowed dev wells and training synthetic sources.
The selected matcher uses synthetic pairs only.
The mixed matcher candidates use the same well folds.

## Old model data

S17 used the old six-well dev set. That set included ForsythB2-17.
Thus, the old S17 boundary and matcher weights are not clean for this split.
S21 does not load these weights. All boundary fits start with random weights.
All matcher fits start again from visual pair features.
S21 uses S17 code and visual tiles from allowed non-eval source logs.
The old S17 report was read as a code reference. It contains old final scores.
Those scores are not used in the S21 selection rule. No old final prediction file was read.
The inherited design has prior benchmark exposure. A new fit does not remove that design history.

## Synthetic data

There are 10,000 exact synthetic windows. Each window has 1024 rows.
There are 8,000 training windows and 2,000 held windows.
The source log sets do not cross the training and held split.
The held set supplies 20 fixed selection windows per family, for 140 windows in total.
The other held windows supply pair checks or remain unused.

| Family | Total windows | Training source logs | Held source logs |
|---|---:|---:|---:|
| Gulf 3-panel | 2000 | 5 | 1 |
| Geoservices grid | 1800 | 1 | 1 |
| D-582 1-bit grid | 1800 | 5 | 1 |
| MC765 colour grid | 1400 | 2 | 1 |
| MC761 mineral/lithology | 1400 | 1 | 1 |
| Texas 'Rock Types' box | 800 | 4 | 1 |
| One-off layout | 800 | 2 | 1 |

The generator adds grid rules to Geoservices, MC765, and D-582 columns.
Grid rows do not have to match a class change. D-582 output has only black and white pixels.
MC761 columns have side panels with mineral textures. The centre panel keeps the true class.
The generator keeps Gulf panels, blank spans, text, fade, blur, JPEG loss, and horizontal shear.
Horizontal shear does not change the row truth. No vertical warp is used.
The source tiles and exact interval rows remain available in this lane.
The images can be made again from the source record and seed.
D-582 and MC765 source logs can share the same legend art.
A held source log does not always mean new legend art.
The layouts are approximations. They are not full source pages.

## Selection checks

The selection score is 0.75 times the mean dev well score plus 0.25 times the mean synthetic family score.
The window score is row accuracy times boundary F1.
CV results choose the settings. Thus, these results are selection results, not an unbiased final test.
Two boundary mixes, four probability thresholds, and three matcher mixes were checked.
The boundary gap is 24 rows. Adjacent equal labels are merged.

Selected boundary synthetic fraction: 0.70.
Selected boundary threshold: 0.40.
Selected matcher synthetic fraction: 1.00.

| Selection set | Row accuracy | Boundary F1 | ARI | Score |
|---|---:|---:|---:|---:|
| dev group mean | 0.4218 | 0.4990 | 0.7872 | 0.2063 |
| synthetic group mean | 0.7320 | 0.8158 | 0.8566 | 0.6216 |

| Dev well | Windows | Row accuracy | Boundary F1 | Score |
|---|---:|---:|---:|---:|
| 386825 | 2 | 0.3684 | 0.7148 | 0.2622 |
| 4304150011 | 21 | 0.5613 | 0.7079 | 0.3601 |
| barlow3-14 | 6 | 0.1301 | 0.3475 | 0.0559 |
| flat-bay-2 | 9 | 0.5253 | 0.2433 | 0.1336 |
| storm | 13 | 0.5240 | 0.4815 | 0.2198 |

| Held synthetic family | Row accuracy | Boundary F1 | Score |
|---|---:|---:|---:|
| Gulf 3-panel | 0.7970 | 0.8587 | 0.6884 |
| Geoservices grid | 0.4762 | 0.8260 | 0.3972 |
| D-582 1-bit grid | 0.5457 | 0.3846 | 0.2263 |
| MC765 colour grid | 0.6875 | 0.8717 | 0.5950 |
| MC761 mineral/lithology | 0.8752 | 0.9153 | 0.8249 |
| Texas 'Rock Types' box | 0.8287 | 0.9464 | 0.7913 |
| One-off layout | 0.9139 | 0.9081 | 0.8283 |

The weakest synthetic boundary result is D-582 1-bit grid, with F1 0.3846.

## Model size and time

The boundary model has six dilated Conv1d layers with 24 hidden channels.
It uses the same 32 row features as S13. The matcher uses 27 visual pair features.
The matcher is a scaled linear logistic model. The stored arrays do not need a pickle loader.
These fits train boundaries and class matching. They do not train the legend parser.
The two deployed weight files total 50,859 bytes. The limit is 10 MB.
Boundary CV and training took 2.33 minutes. The limit is 60 minutes.
Preparation took 5.18 minutes. Selection took 4.18 minutes.
Peak measured RAM was 1.68 GB. The limit is 4 GB.
Torch and BLAS use at most three threads. OpenCV uses one thread. No GPU is used.
The corpus uses memory maps. Training reads only the requested feature windows.

A speed check used seven new synthetic columns with 6,000 rows each. The slowest run took 0.513 seconds.
The speed check peaked at 0.45 GB RAM. It did not use a benchmark log or choose settings.

## Checks and files

The test command checks source exclusion, source splits, seeded output, row geometry,
the 51-window list, well folds, model size, and strategy use without truth access.
It also checks that selection and deployed naming give the same rows.
The fit step checks exported matcher probabilities against the training model.
See test_results.txt for the test result.

The strategy entry point is strategies/s21_synth_v2.py.
The source records are in assets.json. The window records are in synthetic_index.json.
Each window record gives its source, seed, split, and exact truth.
The best settings are in selected.json. The full selection results are in selection.json.
The deployed models are boundary_net.pt and matcher.npz. Their hashes are in fit.json.
READY.json gives the final code, report, data record, and model hashes.
The preview image is synthetic_preview.png.

The shared legend parser can omit or misname a class. This model cannot restore a missing label.
No gain on the final benchmark is claimed.

## Run again

Set LITHOLOGY_DATASET to the local dataset folder. Run these commands from bakeoff.
Use the project Python environment. The code has no fixed host paths.

```sh
python -m ml.synth_v2.experiment prepare
python -m ml.synth_v2.experiment train --steps 1500
python -m ml.synth_v2.experiment select
python -m pytest ml/synth_v2/test_synth_v2.py -q -o cache_dir=ml/synth_v2/.pytest_cache
python -m ml.synth_v2.smoke
python -m ml.synth_v2.report
```

Set PYTHONDONTWRITEBYTECODE=1 to keep imports from writing shared bytecode caches.
No final benchmark command is part of this lane.
