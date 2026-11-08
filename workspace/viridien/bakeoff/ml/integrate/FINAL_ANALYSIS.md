S18 won dev selection and the external test composite; S15 retains the higher in-domain composite.

External test: S18 score 0.1116 versus S14 0.0883 (the strongest archived external-test composite). In-domain: S18 score 0.3468 versus S15 0.3674. S18 has the highest window-mean boundary F1 and ARI on both final sets. No pipeline setting was changed after either final result.

The extra boundaries on MC765 matter: S18 predicts 13 internal boundaries, while the current consensus truth has 0. Its row accuracy is 0.8213, boundary F1 is zero, and composite score is zero. This window alone contributes -1/13 to S18 versus S15’s composite difference; improvements on other families offset most of that loss. The fixed drawn-rule policy was selected on dev and was retained.

D-582 remains a naming failure: mean row accuracy is 0.0094 despite boundary F1 0.6249. This is descriptive evidence of name errors, not proof that the matcher or parser alone caused them. final_confusions.csv lists the largest per-window label confusions.

External depth MAE is 21.8205 ft over 18 windows with matched depth boundaries. One shared calibration, ext_kgs_EmmaWard1_p002, has mean matched-boundary depth error 384.4000 ft. Its cached scale is 3.66429 ft/pixel, while the supplied truth’s interval depth increments are approximately 0.13333 ft/pixel. This is a calibration failure inherited from the shared depth stage, not a runtime failure. It is retained in the final result. The matched-boundary-only metric changes its contributing boundaries with each strategy, so baseline depth MAEs are not a constant calibration comparison.

There were zero runtime failures. Six external windows and one in-domain window have zero composite score. Peak RSS across preprocessing and inference was 1.2202 GiB. Maximum strategy time was 4.221 s on external test and 1.810 s in-domain; thread count was capped at three.

Confidence has limited resolution. Grouped dev selection chose logistic C=0.3 with 0.25 shrinkage toward 0.5; Brier loss is 0.2473 versus 0.2500 for fixed 0.5. These are selection estimates. Confidence is provided per interval in sidecars and does not alter the six-column submission schema.

All 13 submission CSVs pass the SPEC.md fallback checks: exact column order, integer full-image pixel rows, sorted non-overlapping bottom-exclusive intervals, finite monotone depths rounded to 0.1. They cover only the 13 fixed evaluation windows. The frozen code/model hashes still match FINAL_LOCK.json.

All strategies by source: per_source.csv. All strategies by in-domain family: per_family.csv, mapped using analysis/label_shares_per_log.csv. Complete comparisons: REPORT.md and comparison.json.
