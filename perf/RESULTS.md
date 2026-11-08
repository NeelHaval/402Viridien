# Performance results

Branch `perf`. Machine: Apple M3 (8 cores: 4 performance + 4 efficiency), 8 GB RAM, no GPU. Python 3.12.5, numpy 1.26.4, pandas 2.2.3,
OpenCV 4.11, onnxruntime 1.31.1 (CPU), RapidOCR 1.2.3. Every timing below comes from `scripts/bench_e2e.py` (HTTP upload, pin from the
SSE stream, CSV by polling every 100 ms) and the per-stage numbers from the spans stored in `timings_json['spans']`. Raw data: `perf/bench_*.json`.
Times are from **upload saved** (the upload response). Run-to-run noise is about ±0.3 s.

## Benchmark scans

| key | pixels (h x w) | type |
|---|---|---|
| shell08852_9k | 9,356 x 1,653 | RGB, small |
| mc761_15k_pal | 15,400 x 1,700 | palette |
| hadrian_50k | 50,000 x 1,700 | RGB |
| shenzi_73k_pal | 72,600 x 2,550 | palette, widest |
| mdn1_188k | 187,600 x 1,534 | RGB, tallest |
| bw_g4_10k | 10,007 x 4,864 | 1-bit CCITT G4 (failed to decode before; no baseline) |

## Headline numbers (warm medians of 3 runs; first request after server start in the last column)

| scan | baseline total (s) | **time-to-CSV** (s) | **time-to-pin** (s) | first request: CSV / pin |
|---|---|---|---|---|
| shell08852_9k | 11.5 | **2.04** | 0.67 | 2.57 / 1.12 |
| mc761_15k_pal | 18.0 | **7.65** | 7.60 | 8.51 / 8.41 |
| hadrian_50k | 30.4 | **2.54** | 0.70 | 2.60 / 0.87 |
| shenzi_73k_pal | 51.3 | **2.98** | 1.22 | 3.18 / 1.42 |
| mdn1_188k | 85.4 | **3.23** | 0.60 | 3.35 / 0.84 |
| bw_g4_10k | decode error | **2.16** | 1.65 | 2.34 / 1.81 |

Baseline had no separate pin: location appeared only when the job finished, so its time-to-pin equalled its total.

**Against the targets**
- Time-to-CSV under 3 s for scans up to 75k px: **met** for 9k, 50k, 73k (2.98 s, borderline) and B&W. **Not met** for the 15k palette scan (7.65 s, see below).
- Time-to-pin under 1 s: **met** for 9k, 50k, 188k. **Missed** for 73k (1.22 s), B&W (1.65 s) and the 15k scan (7.6 s).
- Time-to-pin of about 0.1 s from a filename API number: **not measured on real data.** The BOEM tables (`boem_wells.csv`, `boem_blocks.csv`) are not in the repo. The code path and its unit tests (`tests/services/test_locator.py`) use fixture tables only.

## Per-stage before and after (warm run, seconds)

Baseline flat timings vs spans of the final code. Stages now run concurrently and compete for CPU, so a stage's wall time can be higher than its
standalone cost (features: 0.57 s alone on 50k, 1.4 s while OCR runs beside it).

| scan | crop | depth calibration | features | legend | naming | wait for location |
|---|---|---|---|---|---|---|
| 9k | 0.63 -> 0.24 | 9.76 -> 1.78 | 0.06 -> 0.28 | 0.32 -> 0.44 | 0.34 -> 0.02 | 0 -> 0 |
| 50k | 0.92 -> 0.24 | 6.94 -> 1.83 | 0.88 -> 1.40 | 0.73 -> 1.10 | 19.05 -> 0.20 | 1.5 -> 0 |
| 73k | 0.80 -> 0.38 | 4.35 -> 1.59 | 1.32 -> 1.64 | 1.53 -> 1.73 | 40.45 -> 0.55 | 0 -> 0 |
| 188k | 2.18 -> 0.71 | 5.90 -> 1.12 | 1.37 -> 1.34 | 1.19 -> 1.03 | 71.36 -> 0.83 | 2.23 -> 0 |

Upload receive and save: 0.02-0.12 s. Hash, validation, scoring, CSV write and DB writes are each under 0.05 s on every scan.

## What was changed (one commit each; all in `git log`)

| commit | change | gate |
|---|---|---|
| `0ef7ba5` | `describe()` once per window instead of once per legend prototype | pass (4 scans bit-identical; 9k flake, see below) |
| G4 decode | 1-bit CCITT G3/G4 TIFFs decoded strip by strip through libtiff; test with generated G3/G4 files | n/a (new capability), pixel-identical to PIL |
| location + events + spans | location as its own task on a dedicated engine; filename API, filename block, header API, header coordinates, staged fallback; SSE `GET /events`; spans in every stage | pass, 5/5 bit-identical |
| frontend | EventSource client, reconnect with backoff, poll only while disconnected; typecheck and build pass | not run in a browser |
| naming batch | all window and bed descriptors in one vectorised pass (run lengths by `nonzero`+`bincount`, medians from exact histograms) | pass, bit-identical; unit tests compare to the old code with exact equality |
| parallel stages | legend OCR and depth calibration run beside features and boundaries; OCR time limits raised so they never bind | pass |
| OCR threads | onnxruntime intra-op 2, 4 windows in parallel, 1 recognition thread (measured grid in the commit message) | pass |

Where the big win came from, with evidence: naming was 63-84% of the time on tall pages (pyinstrument and spans). Depth calibration fell from 5-10 s
to 1-2 s once location OCR left the shared OCR pool: the old code ran both on the same 4 workers and calibration kept hitting its time limit.

## Accuracy

Gate: lithology agreement at least 99%, boundary F1 at least 0.98 at 5 ft, calibration within 1 ft, scores within 2 points, labelled scores not down by more than 0.5 points.

| scan | lithology agreement | F1 at 2 ft / 5 ft | calibration diff | score diff | confidence diff |
|---|---|---|---|---|---|
| 9k, 15k, 50k, 73k, 188k (final code) | 100% | 1.0 / 1.0 | 0 ft | 0 | 0 |

Labelled pages (identical before and after): hadrian per-foot accuracy 76.6%, F1 at 5 ft 0.49; shenzi 93.5% and 0.588. The gate passed after every step.

**Calibration stability:** the 9k scan gave an identical fit (slope 0.4999, intercept 16729.5, 16 inliers) in 10 of 10 consecutive runs with no OCR time-limit warning.
In the baseline code, 1 of 6 runs gave a wrong fit (slope 0.15, 3 inliers, depths near 65 ft instead of 17,900 ft): a load-dependent time limit.

Differences that are not in the gate: header-derived well names are the same as the baseline except one character on `mdn1` (`$T00BP01` vs `ST00BP01`),
because the header is read in a different batch. Names are display metadata; no score or CSV uses them.

## Tried and not adopted, or not done

- **Threads inside features:** slower (50k: 0.50 s serial, 0.68 s with 2 threads, 0.96 s with 8). The work holds the GIL. Left serial.
- **One OCR worker (`intra=6, workers=1`):** 6-9 s on the same scans. Kept 4 windows in parallel, so the "remove the 4-worker pool" item was measured and rejected.
- **Step 4, depth calibration from blobs and 6 crops: not implemented.** The strips in the dataset mix horizontal text (4 of 5 scans), rotated text (15k), bold rotated text (B&W),
  secondary decimal series beside the depth labels (10050 and 10048.4), arrows and tick marks. A blob heuristic needs per-layout tuning that I could not validate here, and
  calibration now passes the gate and is stable with the existing method. Its cost is 1.1-1.8 s of the job.
- **Legend names in one recognition call: not done.** Changing the recognition batch composition changes padding and can change characters (seen on `mdn1`); the legend names feed lithology labels, so it needs its own gate run.
- **Real-sized warm-up of every engine, and reading depth strip and column in one pass: not done.** The warm-up is still one tiny call per engine; first-request cost is 0.5 s above warm.
- **Vectorising `stripe_features`: not done.** Cost 0.28 s (9k) to 1.4 s (188k under load).

## Still above target

1. **mc761_15k_pal: 7.65 s to CSV and 7.6 s to pin.** Its coordinates are not in the header rows above the legend, so the location falls back to the old staged reader (1,500 / 3,000 / 6,000 px). The job waits for it (`wait_location` 4 s).
   The position it finds (lat 13.13, lon -17.72) is outside the Gulf in the baseline too, so this scan is probably not worth optimising before its location is fixed.
2. **shenzi 73k: pin 1.22 s, CSV 2.98 s.** Everything runs at once on 8 cores (4 performance); the header recognition competes with calibration, legend and features.
3. **B&W: pin 1.65 s.** The 4,864 px wide page downsamples to 1,600 and the header has more lines before coordinates.
4. **188k: 3.23 s.** Biggest remaining stages: features 1.34 s, calibration 1.12 s, legend 1.03 s, naming 0.83 s, crop 0.71 s. (Above the 75k target range anyway.)

Experiment only, not adopted (changes results): the 188k lithology column resized to a fixed 4 px per foot (74,399 rows instead of 186,001): features + naming 1.75 s -> 0.67 s, layers 130 -> 111.

## Needs your decision or data

- The BOEM wells and blocks CSVs. Without them the 0.1 s filename path cannot be measured, and header API lookup is skipped (with a `boem_wells_missing` warning).
- Whether to accept that the job waits for the whole header read (name, block) before validating. Validation only needs position, source and API, so deferring the rest would save about 0.8 s on the 9k scan, but would make the API number race with validation.
- Whether to pursue Step 4 (needs per-layout work and more labelled depth strips) or the stripe-feature vectorisation first (est. 0.3-0.7 s on tall pages).
- The frontend was verified by typecheck and build only, not clicked through in a browser.
