# SHARED-VERIFY-SUBSET

> **SHARED-VERIFY-SUBSET — 2026-11-08.** Use this subset to check how models generalise.

Only well **ForsythB2-17** (Kansas Geological Survey, KGS) is included: logs ext_kgs_ForsythB2-17_p015 through ext_kgs_ForsythB2-17_p022. Eight pages give eight windows, about 9% of the 89 outside windows.

This subset is disjoint from our internal test, our sealed verify, and the earlier shared dev subset.

**Prior use:** It came from our dev split. Our strategies s13–s18 used it in grouped cross-validation (CV) and tuning. From now on, we exclude it from training and tuning. Other teams' models have never seen it. It is not fresh held-out evidence for our tuned strategies. This tag does not mean it is the sealed verify split.

## Files

The images, bbox_annotation, truth and preview folders each have eight files. TIFF images and JSON boxes use the challenge format. Truth CSVs use full-image rows, with the bottom row excluded. eval_set.json defines the eight windows. manifest.csv gives source URLs and extraction notes. score_csvs.py is copied from the earlier shared subset.

SHARED_TAG.json gives SHA-256 for every other file. Text uses LF line endings; binary bytes are unchanged. The tag file does not hash itself.

## Run s11

From viridien/bakeoff:

```bash
LITHOLOGY_DATASET=../external_shared_verify python -m harness.run --strategy s11 --set eval --items ../external_shared_verify/eval_set.json --gt-dir ../external_shared_verify/truth --out-tag external_shared_verify
```

Use plain python. Isolated mode can cause a pyarrow DLL error. The --set option selects a mode; --items supplies the window file.

S11 mean over eight windows on 2026-11-08: row_acc **0.470948**, boundary_f1 **0.473666**, cluster_ari **0.479147**. All eight windows were scored; no crashes.

## Score another model

Write prediction CSVs with top_y,bottom_y,lithology and optional cluster_id,top_depth,bottom_depth. Use full-image rows, with the bottom row excluded. From viridien/bakeoff:

```bash
python ../external_shared_verify/score_csvs.py ../external_shared_verify PRED_DIR
```

The scorer tries PRED_DIR/<id>__y<y0>_<y1>.csv, then PRED_DIR/<id>.csv. Missing predictions count as empty predictions. It reports row accuracy, boundary F1, cluster ARI and depth error per window and source.

## Limits and source credit

These born-digital strip logs test boundary placement and matching to unseen legends. They do not cover noisy scans or all challenge layouts. Legend abbreviations can limit label reading. Ambiguous labels have no truth row and are not scored. Read each page's manifest notes.

KGS publishes these public records. Source URLs are in manifest.csv. We share the files inside this private team repo for research and testing. Credit KGS when you show a page.
