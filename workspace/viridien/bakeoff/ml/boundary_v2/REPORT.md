# S22 boundary model

S22 is ready for the one final team benchmark. This lane did not run a final benchmark.

## Data control

Training and selection use the current 51 dev windows from five wells. ForsythB2-17 is excluded. The sealed wells were not searched or opened. External test data and the 13 in-domain eval logs were not used.

The old S13 and S17 weights used the old 59-window dev set. That set included ForsythB2-17. Those weights have historical contamination under the new rule. S22 uses new random weights for every fit. It reuses only model code, feature code, S11 naming code, exact-pixel OCR caches, and allowed synthetic images. No inherited checkpoint or old dev truth cache was used.

The existing synth_gulf set supplies 336 training images and 84 held images. There are seven layout families. Each family supplies 48 training images and 12 held images. Source logs do not cross this split. All source log IDs were checked against the 13-log eval list before image reads. synth_v2 was absent at the first check and was not used. Some source logs have the same legend art. A separate source log does not always mean new pattern art.

Scan changes include fade, noise, blur, JPEG loss, vertical streaks, horizontal row jitter, and binary scans. They do not move rows in the vertical direction. Each image has one fixed scan copy. The scan variant samples that copy with probability 0.6. A second scan variant uses the copy for 20 percent of synthetic crops only. Real crops stay clean in that variant. This lighter variant was added after the strong scan variant reduced dev F1. Wells are sampled with equal probability.

## Dev selection

Cross-validation (CV) holds out one entire well per fold. All fold models start from random weights. Five thresholds and three minimum gaps are tested on held-well predictions. The two best proposal settings per model enter full S11 naming checks, with and without boundary retention. Selection uses the mean boundary F1 across wells after naming. Composite score breaks ties. These CV results also select settings. They can be optimistic. They are not an independent test.

| Training | Boundary F1 | Row accuracy | ARI | Composite score |
|---|---:|---:|---:|---:|
| real | 0.5385 | 0.3703 | 0.8013 | 0.1791 |
| synth | 0.5441 | 0.3691 | 0.7769 | 0.1839 |
| scan | 0.4976 | 0.3627 | 0.7423 | 0.1706 |
| scan_light | 0.5056 | 0.3533 | 0.7656 | 0.1645 |
| S11 fixed baseline | 0.2555 | 0.3409 | 0.5534 | 0.1041 |

Selected settings: `{"gap": 12, "preserve": false, "threshold": 0.45, "variant": "synth"}`.

Both added scan settings reduced dev boundary F1 against the synthetic model. The selected model uses the existing synth_gulf scan variation. It does not use the added scan copies. The scan models remain in their fold folders for review.

| Held dev well | Windows | Boundary F1 | Row accuracy |
|---|---:|---:|---:|
| 386825 | 2 | 0.8673 | 0.1661 |
| 4304150011 | 21 | 0.6488 | 0.5701 |
| barlow3-14 | 6 | 0.3934 | 0.1203 |
| flat-bay-2 | 9 | 0.3045 | 0.5293 |
| storm | 13 | 0.5068 | 0.4594 |

## Held synthetic check

This check uses the final model and held source images. It measures proposals before naming. It was run after settings were fixed. No setting was changed from this result.

| Family | Clean boundary F1 | Scan boundary F1 |
|---|---:|---:|
| D-582 1-bit grid | 0.3974 | 0.3952 |
| Geoservices grid | 0.8414 | 0.8634 |
| Gulf 3-panel | 0.8199 | 0.8584 |
| MC761 mineral/lithology | 0.8540 | 0.7943 |
| MC765 colour grid | 0.8561 | 0.8175 |
| One-off layout | 0.8344 | 0.8318 |
| Texas 'Rock Types' box | 0.9331 | 0.8382 |

## Model and limits

The model has six dilated Conv1d layers with 24 hidden channels. It uses 32 row features. The loss uses soft boundary targets with a four-pixel standard deviation. Each fit uses eight crops of 768 rows per step. AdamW uses a learning rate of 0.0015. Inference uses CPU chunks of 8192 rows with 128 context rows on each side. S11 descriptors assign legend names. Missing legend entries can still limit row accuracy.

Each fit used 650 steps. All CV fits took 110.6 seconds. The final fit took 5.1 seconds. Weights use 50,077 bytes. OMP, MKL, OpenBLAS, and torch use at most three threads. CV logs and resource records are in this folder.

The largest measured process memory use was 0.665 GB. Four tests passed. They check boundary scoring near blanks, chunk edges, fold data control, and the strategy output format on a generated image.

No claim is made for external test or in-domain eval performance. The team must run that benchmark once.

## Reproduction

From bakeoff, use the project Python environment. Set LITHOLOGY_DATASET to the local dataset if needed.

```text
python -m ml.boundary_v2.experiment prepare
python -m ml.boundary_v2.experiment baseline
python -m ml.boundary_v2.experiment cv --steps 650
python -m ml.boundary_v2.experiment select
python -m ml.boundary_v2.experiment fit --steps 650
python -m pytest ml/boundary_v2/test_boundary_v2.py -o cache_dir=ml/boundary_v2/.pytest_cache
python -m ml.boundary_v2.experiment report
```

Dev list SHA256: `84c8a6b57359e5e3f471a559d91ecdaf48def8167d12e2f1612684ba1e4b5631`.

Weight SHA256: `4b714c193b14c9208030d7017119d7d7c89f12e053a13f8d69190c5a33f4e9da`.
