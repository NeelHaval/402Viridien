# Emil live notebook evaluation: dev and test

Main checkpoint: `emil/drive/boundary_net.pt`; SHA256 `45273902c94e642e0bc5edac8d51295f9e7106372b6b2f2bb137a54f9be40079`.

Hadrian reproduces Emil’s 95 beds → 47 segments, row accuracy 0.858, 85.57% classification PixAcc in both detect and labels, 0.989 naming accuracy with true segments, and 94.44% detected-legend text accuracy. Section 3’s default auto loads the GT legend. Live Section 4 selects detect explicitly.

1inch tvd mudlog is in splits.test and excluded by the current TRAIN_PAGES rule. The historical main checkpoint has no training-page manifest. gc743 has only two labelled layers and is excluded from complete aggregates.

## Same-window direct results

Macro means over 2 complete in-domain columns, external dev (59 windows), test (20), and dev+test (79). All strategies bypass depth OCR. detect uses page pixels and cached OCR; labels uses true legend boxes/names.

| Set | Method | Row acc | Boundary F1 | ARI | Failures | Emil coverage | Emil PixAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| viridien | detect | 0.8218 | 0.2494 | 0.5547 | 0/2 | 0.9861 | 0.8118 |
| viridien | labels | 0.8504 | 0.2610 | 0.5984 | 0/2 | 0.9956 | 0.8488 |
| viridien | s11 | 0.6903 | 0.1128 | 0.5837 | 0/2 | 0.9798 | 0.6759 |
| dev | detect | 0.0000 | 0.0169 | 0.1186 | 0/59 | N/A | N/A |
| dev | labels | 0.8454 | 0.4622 | 0.7372 | 0/59 | 0.9873 | 0.8393 |
| dev | s11 | 0.4391 | 0.3535 | 0.6574 | 0/59 | 0.9857 | 0.4351 |
| test | detect | 0.0000 | 0.0123 | 0.2860 | 0/20 | 0.9991 | 0.0367 |
| test | labels | 0.6648 | 0.1755 | 0.4245 | 0/20 | 0.9861 | 0.7022 |
| test | s11 | 0.1069 | 0.2191 | 0.5399 | 0/20 | 0.9425 | 0.0993 |
| dev_test | detect | 0.0000 | 0.0158 | 0.1610 | 0/79 | 0.9991 | 0.0367 |
| dev_test | labels | 0.7997 | 0.3897 | 0.6581 | 0/79 | 0.9870 | 0.8046 |
| dev_test | s11 | 0.3550 | 0.3195 | 0.6276 | 0/79 | 0.9748 | 0.3501 |

The unchanged team segmentation/classification evaluators reject empty predictions: external detect rejects 59/59 dev, 13/20 test, 72/79 combined. PixAcc is N/A with no accepted files; otherwise its mean uses accepted files only. Every harness metric still includes every window. Exact rejection counts and messages are retained in report.json.

## Shared harness including preprocessing failures

s12-detect and s11 reuse completed original harness outcomes. s12-labels uses CPU-replayed identical predict() with those recorded preprocessing outcomes. Five available windows fail depth calibration; failure rows remain in all means.

| Split | Method | Row acc | Boundary F1 | ARI | Failures |
|---|---|---:|---:|---:|---:|
| dev | detect | 0.0000 | 0.0169 | 0.1186 | 3/59 |
| dev | labels | 0.7982 | 0.4703 | 0.7429 | 3/59 |
| dev | s11 | 0.4078 | 0.3450 | 0.6430 | 3/59 |
| test | detect | 0.0000 | 0.0123 | 0.2860 | 2/20 |
| test | labels | 0.6233 | 0.1555 | 0.3927 | 2/20 |
| test | s11 | 0.0734 | 0.1541 | 0.4440 | 2/20 |
| dev_test | detect | 0.0000 | 0.0158 | 0.1610 | 5/79 |
| dev_test | labels | 0.7539 | 0.3906 | 0.6543 | 5/79 |
| dev_test | s11 | 0.3231 | 0.2967 | 0.5926 | 5/79 |

Gold log06, s12-detect: row 0.0000, F1 0.0789, ARI 0.0776. Existing s11: row 0.1292, F1 0.8485, ARI 0.8300. Only one gold page is scored; other eval pages are timing-only. No supplied true legend is available for log06 oracle scoring.

## Complete-page reproduction

| Page | Method | Beds | Segments | Row acc | Boundary F1 | ARI | Emil PixAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | detect | 95 | 47 | 0.8580 | 0.2110 | 0.7179 | 0.8557 |
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | labels | 95 | 47 | 0.8580 | 0.2110 | 0.7179 | 0.8557 |
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | s11 | — | — | 0.5406 | 0.1013 | 0.6771 | 0.5310 |

True-segment naming (60808400140000_hadrian 2_masterlog_2in_10000-21962): `{'row_accuracy': 0.989, 'blank_kept_blank': 1.0}`.

| 60811405120000_1inch tvd mudlog | detect | 62 | 61 | 0.7856 | 0.2877 | 0.3914 | 0.7678 |
| 60811405120000_1inch tvd mudlog | labels | 62 | 61 | 0.8427 | 0.3109 | 0.4790 | 0.8419 |
| 60811405120000_1inch tvd mudlog | s11 | — | — | 0.8400 | 0.1244 | 0.4904 | 0.8208 |

True-segment naming (60811405120000_1inch tvd mudlog): `{'row_accuracy': 0.936, 'blank_kept_blank': 1.0}`.

## Legend detection

| Set | Pages with any detections | Box precision | Box recall | Text accuracy |
|---|---:|---:|---:|---:|
| viridien | 2/2 | 0.7798 | 1.0000 | 0.9134 |
| dev | 0/54 | N/A | 0.0000 | 0.0000 |
| test | 7/16 | 1.0000 | 0.0230 | 0.0000 |
| dev_test | 7/70 | 1.0000 | 0.0053 | 0.0000 |

Unique physical pages are counted once; precision conditions on pages with detections (undefined otherwise). Recall and text include zero detections. Window-weighted legend scores and per-source dev/test/combined breakdowns are in report.json. Swatch-size filter eligibility is diagnostic only.

## CPU runtime and memory

| Set | Method | Seconds / window | Seconds / page | Peak process GiB | GPU GiB |
|---|---|---:|---:|---:|---:|
| viridien | detect | 0.9878 | 0.9878 | 0.923 | 0 |
| viridien | labels | 0.8936 | 0.8936 | 0.959 | 0 |
| dev | detect | 0.0687 | 0.0750 | 0.440 | 0 |
| dev | labels | 0.1478 | 0.1615 | 0.441 | 0 |
| test | detect | 0.0978 | 0.1223 | 0.440 | 0 |
| test | labels | 0.2497 | 0.3121 | 0.442 | 0 |
| dev_test | detect | 0.0761 | 0.0859 | 0.440 | 0 |
| dev_test | labels | 0.1736 | 0.1959 | 0.442 | 0 |

External page time is the sum of its sampled evaluation windows, not full original-page inference. CPU, six torch threads, warm content-addressed OCR, no tracemalloc. Times exclude shared depth/legend parsing and rendering. [Per-page timings](page_resources.csv).

## Secondary retrained model

| Set | Method | Row acc | Boundary F1 | ARI | Failures | Emil coverage | Emil PixAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| retrained / 2 pages | detect | 0.8676 | 0.5218 | 0.6305 | 0/2 | 0.9855 | 0.8572 |
| retrained / 2 pages | labels | 0.8866 | 0.5360 | 0.6669 | 0/2 | 0.9944 | 0.8867 |

Reused the completed 2000-step retrain (29 pages, NumPy/torch seed 0), checkpoint `50a5751d8c1e636de3177c2c8a596c295a980b02987b577a51b782566e7c214c`; reran both modes on both complete pages. Training loss and selection remain in ../training.log and ../training.json. The current live selection differs; validation.json records the differences.

## Incomplete gc743

Only two layers (342 labelled rows), one with unknown/null legend ID and one numeric type. Oracle scoring is separate; uncached detect cannot run because EasyOCR weights are absent and downloads fail DNS. This is an environment failure, not a measured zero-quality prediction.

detect: `{'row_acc': 0.0, 'boundary_f1': 0.0, 'cluster_ari': 0.0, 'failures': 1, 'n_windows': 1, 'emil_coverage': None, 'emil_pixacc': None, 'emil_segmentation_rejected': 1, 'emil_classification_rejected': 1, 'legend_precision': nan, 'legend_recall': 0.0, 'legend_text': 0.0}`
labels: `{'row_acc': 0.11695906432748537, 'boundary_f1': 0.6666666666666666, 'cluster_ari': 0.9477589792214567, 'failures': 0, 'n_windows': 1, 'emil_coverage': 0.021743276749952318, 'emil_pixacc': 0.0025430733040879903, 'emil_segmentation_rejected': 0, 'emil_classification_rejected': 0, 'legend_precision': 1.0, 'legend_recall': 1.0, 'legend_text': 1.0}`

## Five observed failure modes

- Most external legends yield no detected swatches, so detect mode cannot name the column. Example: `ext_utah_4304150011_p153`.
- Two detected boxes on EmmaWard1 do not recover its true legend; OCR/naming still gives zero row accuracy. Example: `ext_kgs_EmmaWard1_p002`.
- Oracle names do not solve texture confusion: flat-bay-2 has nearly correct partitions with wrong lithology names. Example: `ext_nl_flat-bay-2_p078`.
- Bed clustering misses thin-layer boundaries. Hadrian has 680 labelled layers but only 47 Section-2 segments. Example: `60808400140000_hadrian 2_masterlog_2in_10000-21962`.
- Shared depth calibration fails before lithology inference on five allowed windows; direct scoring exposes the separate inference result. Example: `ext_utah_4304150011_p155`.

## Reuse and verification

{
  "allowed_cohort": {
    "dev_windows": 59,
    "test_windows": 20,
    "combined_windows": 79,
    "physical_pages": 70
  },
  "main_checkpoint_sha256": "45273902c94e642e0bc5edac8d51295f9e7106372b6b2f2bb137a54f9be40079",
  "reused": [
    "Live 56-cell notebook port, main checkpoint, byte-identical team evaluators",
    "70-page / 79-window external conversion and cropped dataset view",
    "Both complete in-domain pages in detect/labels/s11 after exact CPU replay of s12",
    "79 main detect predictions, 74 s11 direct CSVs, cached page-pixel OCR",
    "Completed original s12-detect/s11 shared preprocessing outcomes, restricted to dev/test",
    "Existing completed 2000-step, 29-page retrained checkpoint and loss log"
  ],
  "rerun": [
    "79 external labels/oracle predictions on CPU (12 disagreed in first portability replay)",
    "Five s11 direct windows had empty CSVs from shared preprocessing failures; rerun strategy with fresh shared RapidOCR legend parsing",
    "All three unchanged team evaluators and shared scoring separately for dev/test/dev+test",
    "162 untraced CPU inference measurements, all exact matches to final predictions",
    "Both modes on two complete pages with retrained checkpoint",
    "Live experiment API smoke, port/algorithm/conversion validation, existing tests",
    "gc743 detect attempted; unavailable without EasyOCR weights; oracle kept separate",
    "Shared oracle pixel scores using replayed inference and recorded preprocessing outcomes",
    "All report HTML/JSON/Markdown/CSV and PNGs"
  ],
  "removed_wells": "Excluded by eval-set allowlist; stale out-of-scope artifacts retired by filename only; no reconstruction",
  "gc743_detect": "Unavailable: no EasyOCR weights and DNS prevents download",
  "cpu_replay": {
    "calls": 162,
    "exact_prediction_matches": 162
  },
  "test_limit": "Harness passes; full converter suite lacks original sample and hits existing render guard"
}

Evaluator files are byte-identical to the live Drive scripts; only CONFIG changes in memory. Data acceptance passed; the full existing test suite is not green. Harness: 9 passed. Full suite: 36 passed, 3 failed, 8 errors (missing original dev PDF and unchanged converter render-size guard). Details: ../acceptance.json, ../tests_live.log and ../tests_harness_vm.log.

## Limits

- Only two complete in-domain pages; external windows are correlated within wells.
- Removed verify wells were excluded throughout. No verify metrics are generated.
- Direct scores bypass shared depth OCR equally; shared tables retain recorded preprocessing failures.
- Oracle shared results combine CPU-replayed predictions with original preprocessing outcomes, not a fresh end-to-end depth-OCR run.
- s11 predictions and detect shared-harness outcomes are reused. CSV-loading time is not s11 inference runtime.
- CPU replay uses exact cached OCR. EasyOCR weights are absent and DNS blocks downloading them; uncached gc743 detection is unavailable.
- Coverage PixAcc is not boundary accuracy; classification PixAcc includes background. ARI ignores class-name correctness.
- An empty prediction can have ARI=1 on a single-class window; boundary F1 is also 1 when both boundary sets are empty.
- Unchanged team segmentation/classification evaluators reject empty predictions. Their PixAcc means condition on accepted files; rejection counts are explicit. Harness means include all windows.
- Runtime uses CPU, six torch threads, warm OCR and no tracemalloc; original laptop timings are retained separately in source records.
- Both labelled pages are excluded by reconstructed training rules. Historical Drive checkpoint training metadata is unavailable.
- The existing retrain used 29 pages; current live selection differs as recorded in validation.json. It is a secondary check.

## Files

- [Full report and figures](index.html)
- [All metrics and provenance](report.json)
- [Overall comparison](per_source.png)
- [Per-source dev](per_source_dev.png)
- [Per-source test](per_source_test.png)
- [Per-source dev+test](per_source_dev_test.png)
- [viridien 60808400140000_hadrian 2_masterlog_2in_10000-21962](panel_60808400140000_hadrian%202_masterlog_2in_10000-21962.png)
- [viridien 60811405120000_1inch tvd mudlog](panel_60811405120000_1inch%20tvd%20mudlog.png)
- [dev ext_idaho_barlow3-14_p002__y7726_13726](panel_ext_idaho_barlow3-14_p002__y7726_13726.png)
- [test ext_idaho_irvin1-19_p001__y15750_21750](panel_ext_idaho_irvin1-19_p001__y15750_21750.png)
- [test ext_kgs_EmmaWard1_p002](panel_ext_kgs_EmmaWard1_p002.png)
- [dev ext_kgs_ForsythB2-17_p020](panel_ext_kgs_ForsythB2-17_p020.png)
- [test ext_nl_flat-bay-3_p060](panel_ext_nl_flat-bay-3_p060.png)
- [dev ext_nl_flat-bay-2_p078](panel_ext_nl_flat-bay-2_p078.png)
- [dev ext_utah_4304150011_p153](panel_ext_utah_4304150011_p153.png)
- [dev ext_utah_4304150011_p155](panel_ext_utah_4304150011_p155.png)
- [dev ext_utah_4304150011_p155](panel_shared_ext_utah_4304150011_p155.png)
- [Legend overlay](legend_60808400140000_hadrian%202_masterlog_2in_10000-21962.png)
- [Legend overlay](legend_60811405120000_1inch%20tvd%20mudlog.png)
- [Legend overlay](legend_ext_idaho_barlow3-14_p002.png)
- [Legend overlay](legend_ext_kgs_EmmaWard1_p002.png)
- [Legend overlay](legend_ext_nl_flat-bay-3_p060.png)
- [Legend overlay](legend_ext_utah_4304150011_p153.png)
