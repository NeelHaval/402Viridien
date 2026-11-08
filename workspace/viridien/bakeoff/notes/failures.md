# Depth pipeline robustness — 2026-11-08

Scope: only `harness/depth.py`, `harness/test_depth.py`, and this note.
Strategy, legend parser, scorer, annotations, evaluation windows and truth were
not edited. All runs use S11 FFT legend, CPU, three Torch/OpenCV/BLAS threads.

## Diagnosis on dev only

The current dev split contains 59 windows from six wells. Re-running the
original depth implementation produced three preprocessing failures:

| Dev page | Cause | Recovery |
|---|---|---|
| `ext_utah_4304150011_p155` | The annotation ends at y=790. Its top depth is obscured by a printed header, leaving only one readable label (9400). The old code requires three labels and reads only inside the annotation. | Clean and pad the depth track and extend the read below the annotation, where the same printed scale continues with 9450 and 9500. No neighbouring page is consulted. |
| `ext_kgs_ForsythB2-17_p020` | OCR truncates the top 4350 label to 435, attaches a tick to `4400-`, and misses another label. Only two good labels survive the old parser, so no three-point increasing consensus exists. | Accept a trailing tick, reject the truncated value through consensus, and retry with long vertical rules removed and white padding. |
| `ext_nl_storm_p071` | The printed labels are 50–75 metres; the old parser excludes every one because it requires at least three digits. | Accept short numeric depths and infer metric units from printed `meters` / `ROP (min/m)` context. Convert both slope and intercept to feet. |

The large errors on otherwise successful NL pages have the same unit cause:
metres were returned as feet. On dev, flat-bay-2 had mean depth MAE 1211.51 ft
and storm had 1169.55 ft. Printed ROP units provide evidence on continuation
pages without a repeated depth-unit header. OCR also reads upside-down digits
and fragments; these must not outweigh independent, increasing depth anchors.

## Implementation

- Parse one- and two-digit depths, decimal metres, thousands separators, and
  trailing tick marks. Reject arbitrary text and negative tokens.
- Fit a positive linear depth scale using separated point-pair consensus and
  a fixed 16-pixel residual threshold, followed by least squares on inliers.
  Score distinct rows, deduplicate repeat OCR votes, and reject nonfinite,
  nonpositive, excessively steep, insufficient-span and implausibly negative
  extrapolations. There is no per-log or per-well constant.
- If a fit fails or has only two anchors, retry OCR with vertical rules removed,
  padded crops, and a bounded extension of the same depth track. Read strips
  of at most 1800 rows. A two-anchor fit is allowed only when those are the
  entire independent evidence, and is explicitly marked `two-anchor`.
- Infer units from image text around the depth/curve track, with explicit depth
  units taking priority over ROP units. Convert metres using `1 / 0.3048`.
  Missing evidence is marked `ft-assumed`; unknown scales are not fabricated.
- Version and geometry-check the calibration cache. Save numeric anchors,
  unit text, unit decision, residual and recovery method for inspection. Recover
  from malformed JSON and stale versions.

No model was trained, and no parameter grid, threshold search or model selection
was performed. The fixed geometric and text rules were diagnosed on dev images.
Results below are grouped by well; these are regression checks, not a claim of
cross-validated model performance. In-domain evaluation truth was used only by
the scorer for the requested before/after check.

## Dev results

| Well | Windows | Failures before → after | Depth MAE ft before → after |
|---|---:|---:|---:|
| 386825 | 2 | 0 → 0 | 0.7506 → 0.7506 |
| 4304150011 | 21 | 1 → 0 | 0.7500 → 0.7500 |
| ForsythB2-17 | 8 | 1 → 0 | 0.6848 → 0.6675 |
| barlow3-14 | 6 | 0 → 0 | 0.8232 → 0.8232 |
| flat-bay-2 | 9 | 0 → 0 | 1211.5065 → 0.7770 |
| storm | 13 | 1 → 0 | 1169.5520 → 0.7241 |
| **All dev** | **59** | **3 → 0** | **363.6835 → 0.7442** |

Dev row accuracy: 0.407531 → 0.438826; boundary F1: 0.343383 → 0.351857;
cluster ARI: 0.641936 → 0.656335; score: 0.146876 → 0.170998.
These improvements include windows that previously produced empty predictions
because preprocessing failed. Depth MAE follows the existing scorer: macro mean
over windows with matched depth boundaries, 49 before and 50 after; crashes and
unmatched windows have no depth MAE rather than an artificial zero.

Artifacts: [before](../out/s11_fft_legend/robust_dev_before/metrics.json),
[after](../out/s11_fft_legend/robust_dev_after/metrics.json).

## In-domain results

Re-ran all 13 original evaluation windows with `--gt consensus` before modifying
the code and after the fix. Consensus includes the gold log06 labels and the
available silver annotations. Every prediction CSV is byte-identical.

| Metric | Before | After |
|---|---:|---:|
| Failures / 13 | 0 | 0 |
| Row accuracy | 0.3647324623 | 0.3647324623 |
| Boundary F1 | 0.5462494261 | 0.5462494261 |
| Cluster ARI | 0.6321550735 | 0.6321550735 |
| Score | 0.2028929599 | 0.2028929599 |
| Depth MAE ft | 0.2714285714 | 0.2714285714 |

Only log06 has scored matched depth truth in this evaluation; the depth number
does not measure all thirteen calibrations independently.

Artifacts: [before](../out/s11_fft_legend/robust_indomain_before/metrics.json),
[after](../out/s11_fft_legend/robust_indomain_after/metrics.json).

## Held-out test: one final evaluation

The implementation was frozen after dev and in-domain checks, then evaluated
once on the current 20 test windows. The baseline is the saved S11 external
result filtered to exactly those windows, rather than a second test run.
Test images, truth and per-page errors were not used for diagnosis, unit-rule
design, parameter selection or subsequent changes. Removed split pages were
not accessed or reconstructed.

| Split | Windows | Failures before → after | Depth MAE ft before → after |
|---|---:|---:|---:|
| Dev | 59 | 3 → 0 | 363.6835 → 0.7442 |
| Test | 20 | 2 → 0 | 111.9585 → 0.6269 |
| **External total** | **79** | **5 → 0** | See separate split means above |

Test row accuracy: 0.073441 → 0.102084; boundary F1: 0.154085 → 0.216408;
cluster ARI: 0.443976 → 0.538804; score: 0.032296 → 0.058662.
Test depth MAE has 15 contributing windows before and 17 after. This is the
existing matched-boundary metric, not an unconditional calibration MAE over all
20 windows. Since the baseline is archived, its resource timings are not directly
comparable to this VM run.

The final test process was sampled throughout preprocessing, strategy execution
and rendering, including child processes: peak RSS **0.6895 GiB**. The in-domain
after run's largest strategy-stage sampled RSS was 1.2975 GiB. Torch, OpenCV
and BLAS were limited to three threads; no GPU or paid API was used.

Frozen `harness/depth.py` SHA-256, verified unchanged after test:
`2f84c4dc6433cee1bd348087634fb3351968d66183175e3f19f46453f0ac6ee5`.

Artifacts: [saved test baseline subset](../out/s11_fft_legend/robust_test_before/metrics.json),
[final test run](../out/s11_fft_legend/robust_test_after/metrics.json).

## Regression checks

`python -m pytest bakeoff/harness/test_depth.py bakeoff/harness/test_score.py bakeoff/harness/test_run.py -q`

Result: **44 passed, 1 subtest passed**. Coverage includes short and decimal
labels, unit priority, outlier and duplicate rejection, monotonicity, sparse
anchors, rotation coordinates with padding, reading past a short annotation,
cache invalidation, and refusing unsupported/implausible depth scales.

To reproduce strategy runs, set `LITHOLOGY_DATASET` to the desired dataset root
and run from `bakeoff/`: `python -m harness.run --strategy s11_fft_legend` with
`--gt consensus` for in-domain, or `--items ../external_tests/eval_set_dev.json
--gt-dir ../external_tests/truth` for dev. Set a distinct `--out-tag`, limit BLAS
threads through environment variables, and call `torch.set_num_threads(3)` and
`cv2.setNumThreads(3)` before invoking the runner. Paths are repository-relative
or environment-derived; there are no new machine-specific paths in the code.
