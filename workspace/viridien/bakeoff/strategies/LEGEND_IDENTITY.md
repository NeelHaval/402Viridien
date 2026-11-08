# Grade the approved Clay/Shale family consistently

The owner defines **Clay, Shale, Clay/Shale and Clay-Shale as equivalent for grading**. Give full label credit for every pairing of these names, in either direction. Case and spaces around the slash or hyphen do not matter. Clay-Shale is an accepted spelling of the compound name. This is the owner's local grading rule. It is not a verified official challenge rule.

Use `strategies.grading_policy.score_equivalent(pred_rows, truth_rows, window)`. It copies both sets of rows, maps only these names to one grading class, then calls the unchanged shared harness scorer. It preserves contacts, intervals, cluster IDs, depths and raw output labels. Save the strict score too, with an explicit policy name for each series.

```python
from strategies.grading_policy import score_equivalent

accepted_metrics = score_equivalent(predictions, truth, window)
```

A Clay/Shale label can be a valid answer even when the legend prints Clay and Shale separately. A genuine single compound caption also remains valid. The optional S23 strategy keeps printed entries distinct during matching; that is a separate experiment and is not required to receive equivalent grading credit.

## Prevent the regression

From `bakeoff/`, run:

```text
python -X utf8 -m pytest strategies/test_grading_policy.py strategies/test_legend_identity.py -q
```

The tests require all 16 approved name pairings to receive full credit. They check case and spacing, unchanged inputs, and preserved bed contacts. They also reject extra credit for Oil shale, Claystone, Shale grey, Unknown and unrelated rocks. Qualified captions such as Clay-S, Clay-M and Clay-SACS are not in this family. A joined Clay-M / Clay-S / Clay-SACS label does not get credit against Clay-S. Do not use a `startswith('clay')` shortcut. Add another equivalence only when the owner approves its meaning.

For new pipeline outputs, `legend_identity.validate_prediction_labels(rows, captions)` accepts exact printed captions and this approved family. It rejects other invented visual-group names. Similar pictures alone do not authorize merging Mudstone, Wackstone or other distinct classes.

## Next pipeline PR

Use one grading policy at every evaluation entry point, on both predictions and truth. Save `strict_harness` and `clay_shale_equivalent_v1` metrics with those names. Re-score saved CSVs; a grading rule change does not need another inference run. Check that overlays use the same declared policy as their error colors. Keep raw labels readable beside the accepted score.

Include the test output and same-window score parts in the PR. Preserve the interval CSV format, depth behavior and real contacts. The first strict log06 result was low partly because it rejected Clay/Shale against Shale; do not describe that as a current grading error under the approved rule. See [the handoff](../report/scoreboard_1108/CLAY_SHALE_FIX.md) and [the measured note](../report/scoreboard_1108/CLAY_SHALE_FIX_MEASURED.md).
