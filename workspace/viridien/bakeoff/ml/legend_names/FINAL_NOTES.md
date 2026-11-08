The external improvement transfers to held-out test wells: S16 row accuracy is
60.69%, versus S15's 17.11%. On dev, window-macro accuracy is 81.31% versus
50.34%; well-macro accuracy is 72.35% versus 43.48%. Boundary F1 and cluster ARI
are unchanged from S15 on all three splits, consistent with changing names only.

In-domain interval accuracy is essentially unchanged (53.43% versus 53.42%).
Legend truth is sparse: one non-eval page has 18 annotated entries, and one eval
page has 17. The non-eval page improves in correct usable names from 16/18 to
17/18. The eval page regresses from 10/17 to 9/17; exact text accuracy before
the usable-rock filter falls from 17/17 to 15/17. These denominators include
technical legend entries as supplied by the human annotations, so usable-name
rates are not conditional on a rock-only truth subset.

The two frozen eval naming errors are `Sandy Clay/Shale` -> `Sand Clay/Shale`
and `LOT/FIT` -> `LOTFIT`. The latter also evades the existing technical-symbol
exclusion rule and is incorrectly allowed as a lithology. These are reported
as observed; no parser, threshold, vocabulary, model, or settings were changed
after final evaluation began.

Name evaluation uses one-to-one IoU >= 0.5 associations and all annotated
entries as its denominator. Names on continuation pages are evaluated once per
log, not once per interval window. Strategy averages are over the 59 dev,
20 test, and 13 in-domain eval windows. The shared empty-truth convention is
retained: the one in-domain window with no filled truth rows receives perfect
row/boundary/ARI scores. Baseline final metrics are reused from the matching
lane; its strategy and checkpoint hashes still match the current versions.

`audit.json` confirms unchanged frozen dependency hashes, the expected output
counts, ordered non-overlapping predictions within each window, zero runtime
errors, and maximum observed RSS of 1.41 GiB. `tests.log` records six passing
parser regression tests. Inference and data access used CPU only with at most
three compute threads per numerical engine; all generated files are in this lane.
