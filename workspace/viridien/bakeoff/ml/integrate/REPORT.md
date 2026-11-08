S18 integrates boundary and naming evidence selected exclusively on the six dev wells.

Frozen configuration: `{"blank_ink": 0.006, "blend": 1.0, "boundary_weight": 0.0, "gap": 16, "min_bed": 16, "mode": "joint", "penalty": 0.0, "preserve": "rules", "reject": 2.0, "rule_tolerance": 8, "threshold": 0.4, "variant": "joint"}`.

Selection uses out-of-fold CNN probabilities and logistic matcher scores, each excluding the entire query well. The decoder grid is ranked by the mean per-well composite score. These are selection estimates, not an independent test. No final-set truth selected any setting.

The final sets were evaluated once for S18 after the hash lock. S11/S13/S14/S15 results and timings are reused from ml/match/final_metrics.json; they were measured earlier and timing differences include shared-machine load. Timing excludes shared legend OCR/depth calibration. Each metric is a window mean; score is the mean of each window’s row_acc × boundary_f1.

ext_test

| Strategy | n | Row acc | Boundary F1 | ARI | Score | s/window | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s11 | 20 | 0.1021 | 0.2164 | 0.5388 | 0.0587 | 0.1833 | 0 |
| s13 | 20 | 0.1557 | 0.3242 | 0.4303 | 0.0712 | 0.1545 | 0 |
| s14 | 20 | 0.2265 | 0.2480 | 0.4803 | 0.0883 | 0.3777 | 0 |
| s15 | 20 | 0.1711 | 0.3013 | 0.5315 | 0.0728 | 0.7769 | 0 |
| s18 | 20 | 0.2458 | 0.4099 | 0.6021 | 0.1116 | 1.3500 | 0 |

viridien_eval

| Strategy | n | Row acc | Boundary F1 | ARI | Score | s/window | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s11 | 13 | 0.3647 | 0.5462 | 0.6322 | 0.2029 | 0.3072 | 0 |
| s13 | 13 | 0.3967 | 0.5694 | 0.6773 | 0.2450 | 0.1651 | 0 |
| s14 | 13 | 0.4911 | 0.4284 | 0.5503 | 0.1654 | 0.4668 | 0 |
| s15 | 13 | 0.5342 | 0.5487 | 0.6317 | 0.3674 | 0.5614 | 0 |
| s18 | 13 | 0.5281 | 0.6684 | 0.6790 | 0.3468 | 1.1112 | 0 |

Per source

| Group | Strategy | n | Row acc | Boundary F1 | ARI | Score | s/window | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| idaho | s11 | 6 | 0.0285 | 0.0971 | 0.2300 | 0.0026 | 0.4121 | 0 |
| kgs | s11 | 7 | 0.0000 | 0.1682 | 0.3790 | 0.0000 | 0.0958 | 0 |
| nl | s11 | 7 | 0.2672 | 0.3669 | 0.9633 | 0.1654 | 0.0747 | 0 |
| idaho | s13 | 6 | 0.0621 | 0.4608 | 0.3687 | 0.0288 | 0.3723 | 0 |
| kgs | s13 | 7 | 0.0000 | 0.1373 | 0.0626 | 0.0000 | 0.0629 | 0 |
| nl | s13 | 7 | 0.3916 | 0.3940 | 0.8507 | 0.1788 | 0.0595 | 0 |
| idaho | s14 | 6 | 0.0534 | 0.0819 | 0.2926 | 0.0049 | 0.8158 | 0 |
| kgs | s14 | 7 | 0.1911 | 0.2693 | 0.1592 | 0.0823 | 0.2266 | 0 |
| nl | s14 | 7 | 0.4101 | 0.3690 | 0.9622 | 0.1657 | 0.1534 | 0 |
| idaho | s15 | 6 | 0.0790 | 0.3466 | 0.6317 | 0.0285 | 2.1690 | 0 |
| kgs | s15 | 7 | 0.0099 | 0.1572 | 0.1266 | 0.0027 | 0.1949 | 0 |
| nl | s15 | 7 | 0.4112 | 0.4067 | 0.8506 | 0.1808 | 0.1656 | 0 |
| idaho | s18 | 6 | 0.0827 | 0.4481 | 0.6601 | 0.0383 | 3.6847 | 0 |
| kgs | s18 | 7 | 0.2177 | 0.3913 | 0.1609 | 0.1023 | 0.4113 | 0 |
| nl | s18 | 7 | 0.4138 | 0.3958 | 0.9935 | 0.1838 | 0.2874 | 0 |
| Viridien | s11 | 13 | 0.3647 | 0.5462 | 0.6322 | 0.2029 | 0.3072 | 0 |
| Viridien | s13 | 13 | 0.3967 | 0.5694 | 0.6773 | 0.2450 | 0.1651 | 0 |
| Viridien | s14 | 13 | 0.4911 | 0.4284 | 0.5503 | 0.1654 | 0.4668 | 0 |
| Viridien | s15 | 13 | 0.5342 | 0.5487 | 0.6317 | 0.3674 | 0.5614 | 0 |
| Viridien | s18 | 13 | 0.5281 | 0.6684 | 0.6790 | 0.3468 | 1.1112 | 0 |

Per family (Viridien only; analysis/label_shares_per_log.csv)

| Group | Strategy | n | Row acc | Boundary F1 | ARI | Score | s/window | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-582 1-bit grid | s11 | 3 | 0.3037 | 0.8273 | 0.8311 | 0.2487 | 0.2322 | 0 |
| Geoservices grid | s11 | 2 | 0.3429 | 0.2873 | 0.3299 | 0.1458 | 0.1735 | 0 |
| Gulf 3-panel | s11 | 2 | 0.6407 | 0.5795 | 0.6505 | 0.5224 | 0.8875 | 0 |
| MC761 mineral/lithology | s11 | 1 | 0.9040 | 0.1368 | 0.5936 | 0.1236 | 0.1810 | 0 |
| MC765 colour grid | s11 | 1 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.1867 | 0 |
| One-off layout | s11 | 2 | 0.0798 | 0.2317 | 0.4556 | 0.0262 | 0.2193 | 0 |
| Texas 'Rock Types' box | s11 | 2 | 0.3997 | 0.6428 | 0.6296 | 0.1895 | 0.1844 | 0 |
| D-582 1-bit grid | s13 | 3 | 0.2434 | 0.1357 | 0.5463 | 0.0335 | 0.2644 | 0 |
| Geoservices grid | s13 | 2 | 0.5136 | 0.6859 | 0.4946 | 0.4021 | 0.1880 | 0 |
| Gulf 3-panel | s13 | 2 | 0.6771 | 0.6733 | 0.6561 | 0.5614 | 0.1202 | 0 |
| MC761 mineral/lithology | s13 | 1 | 0.9362 | 0.4088 | 0.7732 | 0.3827 | 0.1503 | 0 |
| MC765 colour grid | s13 | 1 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0492 | 0 |
| One-off layout | s13 | 2 | 0.0928 | 0.5724 | 0.6885 | 0.0528 | 0.1177 | 0 |
| Texas 'Rock Types' box | s13 | 2 | 0.4622 | 0.8617 | 0.8571 | 0.3349 | 0.1511 | 0 |
| D-582 1-bit grid | s14 | 3 | 0.0000 | 0.6798 | 0.6510 | 0.0000 | 0.4432 | 0 |
| Geoservices grid | s14 | 2 | 0.4859 | 0.2853 | 0.3655 | 0.1714 | 0.3822 | 0 |
| Gulf 3-panel | s14 | 2 | 0.9470 | 0.5805 | 0.8132 | 0.5719 | 0.8453 | 0 |
| MC761 mineral/lithology | s14 | 1 | 0.9040 | 0.1368 | 0.5936 | 0.1236 | 0.2751 | 0 |
| MC765 colour grid | s14 | 1 | 0.8213 | 0.0000 | 0.0000 | 0.0000 | 0.4752 | 0 |
| One-off layout | s14 | 2 | 0.4774 | 0.2646 | 0.4871 | 0.1088 | 0.3782 | 0 |
| Texas 'Rock Types' box | s14 | 2 | 0.4191 | 0.5658 | 0.6379 | 0.1613 | 0.3887 | 0 |
| D-582 1-bit grid | s15 | 3 | 0.0062 | 0.0625 | 0.1659 | 0.0003 | 0.7074 | 0 |
| Geoservices grid | s15 | 2 | 0.5492 | 0.6466 | 0.5580 | 0.3791 | 0.7446 | 0 |
| Gulf 3-panel | s15 | 2 | 0.9467 | 0.7569 | 0.8388 | 0.7295 | 0.5162 | 0 |
| MC761 mineral/lithology | s15 | 1 | 0.9427 | 0.3731 | 0.7782 | 0.3517 | 0.5097 | 0 |
| MC765 colour grid | s15 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1022 | 0 |
| One-off layout | s15 | 2 | 0.5324 | 0.5348 | 0.7134 | 0.2799 | 0.3563 | 0 |
| Texas 'Rock Types' box | s15 | 2 | 0.4637 | 0.8478 | 0.8579 | 0.3232 | 0.6652 | 0 |
| D-582 1-bit grid | s18 | 3 | 0.0094 | 0.6249 | 0.6065 | 0.0058 | 1.0827 | 0 |
| Geoservices grid | s18 | 2 | 0.5724 | 0.7313 | 0.6002 | 0.4276 | 1.5453 | 0 |
| Gulf 3-panel | s18 | 2 | 0.9610 | 0.8440 | 0.8760 | 0.8172 | 1.5486 | 0 |
| MC761 mineral/lithology | s18 | 1 | 0.9438 | 0.4861 | 0.7936 | 0.4588 | 0.8344 | 0 |
| MC765 colour grid | s18 | 1 | 0.8213 | 0.0000 | 0.0000 | 0.0000 | 0.6353 | 0 |
| One-off layout | s18 | 2 | 0.5238 | 0.6418 | 0.7296 | 0.3428 | 0.6467 | 0 |
| Texas 'Rock Types' box | s18 | 2 | 0.4791 | 0.9472 | 0.9010 | 0.4287 | 1.1233 | 0 |

Dev baselines and selected S18 (well-grouped out-of-fold; selection estimates)

| Strategy | Well-macro row | Well-macro boundary | Well-macro ARI | Well-macro score | Window-macro score |
| --- | ---: | ---: | ---: | ---: | ---: |
| s11 | 0.3575 | 0.2915 | 0.5392 | 0.1304 | 0.1710 |
| s13 | 0.3723 | 0.4989 | 0.6377 | 0.2025 | 0.2692 |
| s14 | 0.3976 | 0.2879 | 0.5433 | 0.1452 | 0.1849 |
| s15 | 0.4348 | 0.4895 | 0.6934 | 0.2390 | 0.2941 |
| s18 | 0.4634 | 0.6005 | 0.7866 | 0.3060 | 0.3302 |

The selected DP uses zero extra boundary penalty and zero boundary-log-odds weight: dev preferred visual labels over the union of CNN and S11 candidates, with a 16-pixel duration constraint and explicit drawn-rule splits. The best independent candidate scored 0.2790 by well; the best nonzero boundary prior scored 0.2799.

Keeping all S14 proposals changed well-macro boundary F1 from 0.2879 to 0.2954, while its score fell from 0.1452 to 0.1430. Keeping all S15 proposals also reduced dev score. The eight-pixel drawn-rule tolerance made no difference on these dev baselines. These checks support selective merging and richer candidates; they do not establish that line-coordinate fusion caused the in-domain regression.

Dev selection: {"well_macro": {"row_acc": 0.4634119441174451, "boundary_f1": 0.6005183771343215, "cluster_ari": 0.7865924835573711, "score": 0.3060491870187076}, "window_macro": {"row_acc": 0.512532384037783, "boundary_f1": 0.6038384853421632, "cluster_ari": 0.8510389532929321, "score": 0.3302372898729867}}

Blank is determined from column-interior ink; a no-match threshold and margin were tested on dev. Confidence is a separate logistic estimate of the fraction of filled interval rows with the exact parsed legend name. It is fitted only on dev out-of-fold decisions; confidence_validation.json reports grouped out-of-fold Brier score and bins. Confidence does not alter submission names.

Confidence regularization and shrinkage toward 0.5 are selected against a constant baseline by grouped dev Brier loss. Across-well calibration remains weak; the bins should be read before using confidence as a probability. Decoder-selection diagnostics and calibration scores reuse dev for selection and are not independent validation.

The rule-tolerance fix preserves explicit drawn separators when proposal fusion shifts their coordinate by up to eight pixels. Ordinary adjacent equal-label rows merge. Dev ablations in dev_selection.json compare preserving all proposals, drawn rules only, strong CNN boundaries, or none; joint DP compares minimum bed lengths and boundary log-odds weights.

Failures and lowest-scoring S18 windows (descriptive final analysis only):
- ext_test: `ext_idaho_irvin1-19_p001` [2510, 8510): row=0.0000, boundary=0.2645, ARI=0.9355, score=0.0000
- ext_test: `ext_idaho_irvin1-19_p001` [15750, 21750): row=0.0000, boundary=0.5911, ARI=0.4831, score=0.0000
- ext_test: `ext_kgs_EmmaWard1_p004` [330, 1980): row=0.0000, boundary=0.1395, ARI=0.0955, score=0.0000
- ext_test: `ext_nl_flat-bay-3_p055` [177, 1814): row=0.0000, boundary=0.5000, ARI=0.9878, score=0.0000
- ext_test: `ext_nl_flat-bay-3_p057` [177, 1814): row=0.0000, boundary=0.0000, ARI=1.0000, score=0.0000
- viridien_eval: `MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft` [7623, 13623): row=0.8213, boundary=0.0000, ARI=0.0000, score=0.0000
- viridien_eval: `Critter Creek 201-1807H_Mud_PDF_Vertical.pdf` [5480, 11480): row=0.0042, boundary=1.0000, ARI=0.9797, score=0.0042
- viridien_eval: `0000000009_d-582_0000020435-0001_log06_fe_5in_md` [4404, 11195): row=0.0082, boundary=0.6667, ARI=0.6811, score=0.0054
- viridien_eval: `0000000009_d-582_0000020435-0001_log04_fe_5in_md` [4384, 10384): row=0.0098, boundary=0.5581, ARI=0.4878, score=0.0055
- viridien_eval: `0000000009_d-582_0000020435-0001_log02_fe_5in_md` [4387, 10387): row=0.0102, boundary=0.6500, ARI=0.6508, score=0.0066

Resource limit: {"completed": 1794102895.7664695, "peak_rss_gb": 1.2201728820800781, "hashes_unchanged": true, "thread_limit": 3, "device": "cpu", "n_windows": 33}

Submission preview: 13 exact six-column CSVs in submission_preview/, validated against SPEC.md because the example notebook is absent from the dataset. Pixel coordinates are full-image, bottom-exclusive; depths are rounded to 0.1. Coverage is the fixed eval window per log. Additional confidence is under submission_preview/confidence/, outside the CSV schema.

Reproduce from bakeoff with LITHOLOGY_DATASET set: `python -m ml.integrate.experiment prepare`, then `select`; `final` refuses a second run. `report` reads saved final results without rerunning inference. Behavioral checks: `python -m pytest ml/integrate/test_integrate.py -q`.

Post-evaluation failure analysis: [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md). This reads saved predictions and truth; it does not rerun inference or change the frozen pipeline.
