# Lithology strategy rules

For new or changed grading, legend matching, class grouping, or output labels, read [LEGEND_IDENTITY.md](LEGEND_IDENTITY.md) before you edit code.

For any new score report or score summary, also read [SCORING_CONTRACT_V2.md](SCORING_CONTRACT_V2.md). Apply metric availability before you calculate means. Use `grading_availability_v2.truth_structure` and `metric_view`; do not use a raw legacy score as an available grade. No filled truth rows means unavailable accuracy and product, not zero or one. Keep failed predictions at zero when valid truth is available. Show the denominator and exclusion reasons for each metric. Keep source-A silver diagnostics separate from qualified development results. Never use an unreviewed vote tie as consensus truth.

Grade Clay, Shale, Clay/Shale and Clay-Shale as one class in both directions. Use `grading_policy.score_equivalent` on predictions and truth; keep strict scores as a reference. This is the owner's local policy, not a verified official challenge rule. Preserve raw labels and contacts. Store any other visual similarity as separate evidence. Validate new output labels with `legend_identity.validate_prediction_labels`, which permits the approved family.

Run the identity regression checks in `LEGEND_IDENTITY.md` after the last edit. Report the real result and any remaining classification error.

S20, S21 and their snapshots are historical comparison baselines. Keep their recorded behavior intact. Make a separate candidate for an inference change and use the unchanged shared scorer.
