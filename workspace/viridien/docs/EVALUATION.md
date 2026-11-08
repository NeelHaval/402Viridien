# Evaluate without mixing data roles

Report row accuracy, boundary F1, cluster ARI, elapsed time, and failed windows.
The project's combined development score is row accuracy × boundary F1. The
organiser slides did not specify that exact formula or boundary tolerance.

| Suite or data | What it establishes |
| --- | --- |
| `ext_shared` | A quick run and development check on published outside data |
| `viridien_gold` | Performance on the organiser's single example window |
| `viridien_eval` | Mixed-evidence development results; some truth is AI consensus |
| `human_labels/` | Human interval labels, with their documented QA and limits |
| Real crop `test.jsonl` | Crop classification; not a complete boundary test |
| `ext_shared_verify` | Separate published check, previously used in our CV |
| Synthetic outputs | Regression and development behaviour, not real accuracy |

From `viridien/bakeoff/`:

```bash
python -m harness.bench --strategy s11 --suite ext_shared --workers 1
python -m harness.bench --strategy s11 --suite ext_shared_verify --workers 1
```

Each run writes a new directory. Keep its configuration and metrics. Use
`--no-cache` for fresh predictions. Original-log work needs `LITHOLOGY_DATASET`.
Learned strategies need optional packages and artifacts. See
[BENCH.md](../bakeoff/harness/BENCH.md) for the complete API.

Before comparing a candidate, freeze the suite, scorer, boundary tolerance,
label policy, and baseline. Keep pages of a well together. Choose thresholds
on development wells. Score test wells after selection. Count crashes and empty
predictions. Keep retrospective and prospective test claims separate.

## Score your CSVs

Run from the repository root:

```bash
python viridien/external_shared/score_csvs.py viridien/external_shared PRED_DIR
python viridien/external_shared_verify/score_csvs.py viridien/external_shared_verify PRED_DIR
python viridien/external_shared/score_csvs.py viridien/human_labels PRED_DIR
```

Use complete-image rows. Preserve depth units; some outside logs use metres.
Each subset's `eval_set.json` defines windows and `truth/` supplies labels.
Read its README for ambiguous names, unlabelled gaps, and single-class windows.

## Read older results

Crop classification, legend matching, AI agreement, boundary detection, and
complete-interval scoring are different tasks. Their percentages cannot be
exchanged. A high crop score can coexist with poor boundary accuracy. A synthetic
score cannot establish real generalisation. An LLM review is a noisy check.

The council found some settings and confidence models evaluated on their fitting
samples. Keep those reports for traceability, but read [the limits](STATUS.md)
before citing a figure. No new model accuracy is claimed by this consolidation.
