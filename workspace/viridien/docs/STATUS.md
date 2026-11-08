# Status and limits

This publication combines existing work from 8 November 2026. It adds setup,
data separation, integrity checks, and use guides. It does not claim a new model
accuracy result or certify the synthetic dataset.

The [catalog](../data/catalog.json) fixes the published counts and checksums.
Local labelling and generation can continue after this snapshot. New changes on
the original host do not update these files automatically.

## Available work

- Current pattern labeller, shared saves, full document editor, reviewer files,
  fixed crop split, normalized export, and quality holds.
- Tile learner, generator lab, column assembler, atlas, progress board,
  historical synthetic v5, and critiques.
- Baseline and learned strategy experiments through S22, benchmark API,
  outside-data converters, and saved model artifacts.
- Two separately published outside subsets, human interval labels, organiser
  example truth, and separate AI labels.
- Council reports and expanded outside-data source searches.

## Material limits

The [council](../bakeoff/notes/council/merged_codex.md) found excessive reliance
on AI crop agreement for model selection. It confirmed twin-pattern merge
errors, dark fills treated as blank, grid lines named as rock, unsupported bed
merges, damaged legend parsing, and stale caches. Some local fixes can be present;
the complete pipeline still needs a fresh real-interval evaluation.

Native-scale DINOv2 matching showed a useful limited result. Full integration,
fixed holdout comparison, and family-level validation were not complete in the
source thread. A proposed fix must not be described as shipped.

The [generation reviews](../reports/generation/) found rails, seams, missing rock
types, pattern-box heights used as thickness, unlabelled areas treated as blank,
and an exact-pixel quarantine bypass in v1–v5. Synthetic outputs remain
experimental. The source contains fix work; the whole dataset is not certified.

Outside strip logs test robustness and new legends. They do not represent most
Gulf layouts. The search reports include BSEE, state records, research wells,
and vendor examples. A source lead is not a downloaded, verified test set.

## Checks

[CHECKS.md](CHECKS.md) records commands and final output for this publication.
Source and data checks establish file integrity and selected test behaviour.
They do not establish model accuracy. Browser checks and full training are
marked not run when they were not run here.
