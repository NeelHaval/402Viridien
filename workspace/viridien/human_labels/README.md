# Human-labelled challenge pages (Neel)

> **Tag: HUMAN-LABELS-NEEL, built 2026-11-08 from a 01:39 UTC snapshot.** Hand labels by Neel Haval (NeelHaval) for two Viridien challenge pages, checked by independent AI reviewers against the scans. Use them to develop and score methods on real challenge layouts. `HUMAN_TAG.json` lists every file with its SHA-256.

2 pages, 646 beds, 10 scoring windows (6000 px each, as in `external_shared`).

| Page | Beds | Labelled depth | Rocks | Depth calibration |
|---|---|---|---|---|
| `60811405120000_1inch tvd mudlog` | 357 | 14,851.9–21,514.8 ft TVD (whole log) | Clay/Shale 166, Sandstone 159, Siltstone 32 | 2 anchors (px 1399 = 14,800 ft, px 2000 = 15,100 ft); last bed ends 1.8 ft from the printed final TVD 21,513 ft |
| `60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400` | 289 | 6,876.3–23,104.0 ft (about 70% of the log; Neel is still labelling the rest) | Shale 136, Siltstone 73, Sandstone 54, Limestone 21, Halite 5 | 231 anchors, one per printed 100 ft tick, detected from the tick lines (3.000 px/ft, max residual 1.0 ft) |

Shenzi has one very thick salt bed: Halite from about 10,326 to 19,312 ft (about 9,000 ft), with one thin shale bed inside it and a thin gap at 19,022 ft where the "Gas" show was removed.

## Layout

Same as `viridien/external_shared` (PR 49):

- `truth/<id>.csv`: `top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth`, one row per bed, full-image rows. Gaps have no row and are not scored.
- `bbox_annotation/<id>.json`: the challenge boxes for the page (copied unchanged).
- `labels/<id>.json`: the full labelling-tool file (legend boxes, depth anchors, beds), after the QA fixes below.
- `eval_set.json`: scoring windows (`log`, `y0`, `y1`, `tier: "human"`), only inside the labelled interval.
- `manifest.csv`: per page counts, depth range, px/ft, calibration and the fixes applied.
- `preview/<id>.png`: page thumbnail with the labelled column beside it.
- `qa/`: reviewer findings, adjudication verdicts and the exact fixes applied.

The page images are the challenge's own TIFFs, so they are **not** copied here. Point `LITHOLOGY_DATASET` at the challenge dataset.

## Run it

Bake-off harness (from `viridien/bakeoff`):

```bash
LITHOLOGY_DATASET=../dataset python -m harness.run --strategy s11_fft_legend \
  --items ../human_labels/eval_set.json --gt-dir ../human_labels/truth --out-tag human_labels
```

**Baseline (checked 2026-11-08):** `s11_fft_legend` runs on all 10 windows. On 1inch it gets row accuracy 0.864 and 0.558 (boundary F1 0.154 and 0.121). On Shenzi it returns **no rows at all** (empty CSV in about 5 ms per window), so every Shenzi window scores row accuracy 0. That is a failure of the strategy on this layout (20-entry legend, 3 px/ft), not of the truth. Note that on the three windows that are all Halite, an empty prediction still gets boundary F1 and ARI 1.0, so read row accuracy there.

Any other solution: write one CSV per window and score it with `python ../external_shared/score_csvs.py ../human_labels PRED_DIR`.

## How it was checked

1. **Two independent reviews of every bed.** Both columns were cut into 105 chunks (600 source px each, 2x zoom: scan beside the same scan with Neel's boundaries and labels). Two separate sets of GPT-6.1 Terra reviewers, with different chunk groupings, checked every bed against the page legend for wrong rock, boundary and non-rock entries. Set A flagged 22 beds and set B flagged 20; 11 were flagged by both.
2. **Two independent adjudications of every flag.** Each of the 29 flagged beds got a 4x close-up with a source-row ruler and was judged separately by two Terra reviewers at high effort. They answered with exact edits.
3. **Only agreed edits were applied** (same edit from both adjudicators, rows within 1 px):
   - 1inch: inserted a missing Clay/Shale band at px 10794–10800; relabelled bed 282 Siltstone → Sandstone and bed 352 Sandstone → Clay/Shale (dominant rock across the column width).
   - Shenzi: bed 25 Siltstone → Shale; removed the Oil and Gas "beds" (show symbols, not rocks) and one 1 px unassigned bed overlapping its neighbour; inserted missing thin beds (Siltstone px 18793–18808, Shale px 31875–31887 inside the halite).
   - Shenzi legend names cleaned: "Halite = Shale = Clay" → Halite (an OCR merge of three neighbouring names), and stray `|` removed from Chalk, Granite and Test.
4. **Rejected flags.** 19 flags were judged correct as labelled by both adjudicators, including all 9 "boundary too high" claims from one reviewer (boundaries are within about 1 px of the colour change).
5. **Depth checks.** Two Terra readers independently read 10 printed depth labels on Shenzi; all match the tick lattice within 2 ft. The 1inch anchors reproduce the printed final depth.

Every page passes the labelling tool's validator with 0 errors.

## Known caveats

- **Side-by-side intervals (1inch).** In places the 1inch log draws two rocks side by side across the column. Each bed takes the dominant rock; where both cover about equal width (beds 237, 240, 334, 345, 350) the reviewers kept Neel's choice, so those beds are a coin toss between Clay/Shale and Sandstone or Siltstone.
- **Small gaps.** A few 1–3 px gaps between beds remain; they have no truth row and are not scored.
- **Width warnings (1inch).** 14 beds are boxed a little narrower than the column. This does not affect the truth rows.
- **AI review is not ground truth.** The review catches clear mislabels and missing beds; it does not re-label the page.
- **Shenzi is partial.** Windows cover only the labelled interval. Rebuild with `tools/build_dataset.py` (applies `qa/applied_fixes.json`) when Neel finishes.
- Not included: `60811404100000_gc743_dc141…` (2 beds, legend unnamed).
