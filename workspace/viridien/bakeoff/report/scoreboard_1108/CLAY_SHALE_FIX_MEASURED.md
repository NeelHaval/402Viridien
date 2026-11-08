# Log06: identity fix measurement and source audit

The owner requests full label credit for Clay, Shale, Clay/Shale and Clay-Shale in every direction. The grading helper and regression checks are implemented. Raw labels and contacts stay intact. This is a local policy, not a verified official challenge rule. S23 remains a separate optional matching experiment.

## Current approved grade on cached predictions

| Metric | S21 baseline | S23 distinct captions |
|---|---:|---:|
| Row accuracy | 0.768565 | 0.768565 |
| Boundary F1 | 0.625000 | 0.588235 |
| Product | 0.480353 | 0.452097 |

No inference was rerun for these values. The same-window strict products are 0.083250 and 0.092555. Boundary F1 and depth error do not change with the label policy. The approved grade removes the apparent S23 label gain in this window. It does not remove the Siltstone-versus-Sandstone error. See [the raw prediction view](log06_dino_equivalent_detail.html).

## Historical strict-label measurement

This addendum uses the saved gold-window records for case `6b663182fbfb4f87`, rows 4404–11195. It is one window, not a general result. Both strategies use the same strict scorer. The full multi-window evaluation is separate.

| Metric | S21, visual twins | S23, distinct captions | Change |
|---|---:|---:|---:|
| Row accuracy | 0.133200 | 0.157343 | +0.024143 |
| Boundary F1 | 0.625000 | 0.588235 | −0.036765 |
| Score | 0.083250 | 0.092555 | +0.009304 |
| Cluster ARI | 0.760660 | 0.702580 | −0.058080 |

Under the historical strict-label scorer, row accuracy rose by 2.414 percentage points; boundary F1 and ARI fell. S23 separates the candidates: of 4,004 gold Shale pixels, 3,671 are predicted Clay and 145 are predicted Shale. Under the owner's clarified policy, both Clay and Shale predictions qualify for full class credit against Shale. The other gold Shale pixels are assigned to SIllty Sandstone (20) or Sand and Sandstone (168). The saved S23 prediction contains 3,732 Clay pixels and 145 Shale pixels overall. These counts are pixel rows, not geological samples; they do not themselves constitute a recalculated score.

The first S21 and S23 recorded times were 190.31 and 134.94 seconds respectively. Treat these as descriptive measurements: cold-start conditions and system contention must be matched before attributing a speed difference to the identity change.

## Native swatch and source audit

The original legend visibly labels two separate swatches Clay and Shale. Their cached native RGB arrays each match the corresponding inner crop of the original log image and its cached legend PNG exactly. There is no observed cache pixel mismatch or swapped caption in these two entries. The annotated [source crop](../../out/scoreboard_1108/log06_clay_shale_source.png) shows both original captions and patterns; the rectangles mark the parser's swatch boxes.

The 104 × 148 swatches are not identical, but 2.222% of aligned pixels differ. Their bidirectional v4 mask NCC is 0.8216, above the 0.75 twin threshold. Their SHA-256 values over contiguous native RGB bytes are `62e246331098f74d95315809828af61deede1c1eac6ea8bcd02209fe954df440` for Clay and `936f61010d56aa7bf64ea1a94d6d78e0168bf8741b9206544d6b236df2ce936b` for Shale. The [audit data](../../out/scoreboard_1108/log06_clay_shale_swatches.json) includes coordinates and the matching checks.

As a narrow CPU-only diagnostic, the v4 hand distance preferred Clay at column row 4500 (Clay 2.6744, Shale 2.8121) and at row 5250 (Clay 3.2130, Shale 3.3283). These are two 48-row crops, not an aggregate explanation of S23 predictions. They do not measure the DINO similarity contribution and do not establish that the source patterns are intrinsically impossible to distinguish. The small GPU smoke on a tiled native Shale swatch returned Shale; that synthetic check is not evidence of performance on the log column.

Reproduce the source audit from `bakeoff/` with `PYTHONPATH` set to that directory and `python -X utf8 out/scoreboard_1108/inspect_log06_swatches.py`. Run `python -X utf8 out/scoreboard_1108/log06_label_overlap.py` to recompute the saved prediction and gold overlap counts. Both scripts read inputs only and perform no GPU inference.
