# Read results and avoid overfitting

The scorer version is saved with every run. It uses interval arithmetic, not full-height image arrays. Do not compare a saved legacy notebook/bake-off score directly with this scorer: the notebook allowed many predictions to match one boundary, and the older bake-off sampled rows for ARI.

Version 0.2 uses `interval-v2`. It checks shared-boundary depths, accepts 0.01-unit quantization differences in the imported vector truth, averages agreeing endpoint readings, and exposes depth coverage. Re-run both sides before comparing against older scorer hashes/versions.

| Metric | Meaning |
|---|---|
| `row_acc` | Correct lithology over truth-filled original pixel rows; case/whitespace normalization plus the organizer's explicit Sand/Sandstone alias |
| `row_acc_weighted` | Correct labelled pixels divided by all labelled pixels across cases |
| `macro_f1` | Per-class F1 averaged across classes, to expose minority-class failures |
| `boundary_f1` | One-to-one matching of internal boundaries within saved pixel tolerance; outer window endpoints excluded |
| `cluster_ari` | Exact interval contingency adjusted Rand index, invariant to cluster numbering |
| `depth_mae_ft` | Absolute depth error on matched boundaries, only with comparable feet and datum metadata |
| `depth_coverage` | Matched depth pairs divided by eligible true internal boundaries; report beside conditional MAE |
| `score` | Per-case row accuracy × boundary F1; mean across labelled cases |
| `coverage`, `unknown_rate` | Predicted coverage and fraction assigned Unknown; read with accuracy |
| `seconds`, `peak_rss_mb` | Wall time including worker startup and sampled process-tree RSS |
| Completion/errors | Every requested case remains visible, including timeout/crash/invalid output |

Both boundary sets empty produces F1=1, which means no missed internal boundaries. Empty predictions can still have that value on a one-bed window, so always read row accuracy, score, predicted interval count and coverage. Unlabelled cases have null supervised metrics. Errors on labelled cases contribute zero quality rather than disappearing from the denominator. Depths absent from either side never become zero error.

Run directories are unique and are not reused. `per_case.jsonl` includes status and errors. `metrics.json` reports aggregates by source, well group, split and tags. `provenance.json` records inputs and package versions. `summary.md` is the readable result and contains links to per-case predictions and overlays. Treat these as experimental artifacts and retain them alongside any model-promotion decision.

`compare` requires matching data fingerprints, selected cases, split, input mode, scorer hash and version. It reports right-minus-left deltas per case and overall. A different dataset version, truth edit, input assistance or scorer needs a fresh baseline run. Model settings/checkpoints are intentionally allowed to differ because that is the experiment.

Depth-error improvements are suppressed when the two runs score different true boundary sets. Omitting difficult depth predictions can lower a conditional MAE, so it must never be interpreted without pair identities and coverage. Input depths must not decrease; very thin beds can legitimately have equal endpoints after 0.1-ft rounding.

Compare well-level results, not just a grand mean: two human-labelled pages and a few external wells are a small sample. Inspect the worst cases, minority classes and error counts. Long single-rock intervals can dominate pixel accuracy, while equal per-window means can over-weight wells with many windows. Repeated tuning turns a test set into development evidence. Preserve an actual unseen group/verification source and run it sparingly through its custodian.

The shared `external-diagnostic` set was previously tuned by s13–s18. Synthetic data and clean outside vector logs each cover only part of the challenge distribution. A higher score on them alone is not evidence of improved unseen Viridien scan accuracy. Comparisons include paired group-bootstrap 95% intervals (seed 0, 2,000 resamples); with fewer than two labelled groups these are null. Intervals over only a few wells remain uncertain and do not establish broad generalisation.
