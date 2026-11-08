# Outside test logs (shared dev subset)

> **Tag: SHARED-DEV-SUBSET, shared 2026-11-08 by silmoon04.** These logs are now visible to the whole team, so they are **not held-out evidence**. Use them to develop, debug and smoke-test. The outside **test** split (3 wells) and **verification** split (3 wells, encrypted) are kept private for unbiased checks; ask silmoon04 to run them. `SHARED_TAG.json` lists every file with its SHA-256 so we can always tell this subset apart.

12 log pages from 3 wells, 17 scoring windows, about 19% of our 89 outside windows. Each page is a born-digital strip log PDF from a public government archive, rendered to TIFF. The rock truth comes from the PDF's vector drawing, so it is **exact**, not hand-labelled.

| Well | Source | Pages | Windows | Notes |
|---|---|---|---|---|
| 386825 | Kansas Geological Survey | 1 | 2 | 4 rock types |
| Barlow3-14 | Idaho Oil and Gas Conservation Commission | 2 | 6 | many thin beds; colour legend, grey column |
| Flat Bay 2 | Newfoundland and Labrador final well report | 9 | 9 | depths in metres; often 1 to 3 beds per page |

The format is the same as the Viridien challenge:

- `images/<id>.tif`
- `bbox_annotation/<id>.json`, with `bounding_boxes` for `depth_column`, `interpreted_lithology_column` and `lithology_legend`, each `[x0,y0,x1,y1]` with x1/y1 exclusive.
- `truth/<id>.csv`, with columns `top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth`.
- `eval_set.json`: the scoring windows (`log`, `y0`, `y1`).
- `manifest.csv`: source URL, pixels per foot, legend-parse coverage and notes for each page.
- `preview/`: one picture per page.

Any solution that reads the challenge's images and boxes runs on this folder unchanged.

## Run it

Bake-off harness (from `viridien/bakeoff`):

```bash
LITHOLOGY_DATASET=../external_shared python -m harness.run --strategy s11_fft_legend \
  --items ../external_shared/eval_set.json --gt-dir ../external_shared/truth --out-tag external_shared
```

For any other solution, write one CSV per window (`top_y,bottom_y,lithology` at least, full-image rows), then score it with the same scorer:

```bash
cd viridien/bakeoff
python ../external_shared/score_csvs.py ../external_shared PRED_DIR
```

`score_csvs.py` looks for `PRED_DIR/<id>__y<y0>_<y1>.csv` and then `PRED_DIR/<id>.csv`. A missing prediction scores 0. It prints row accuracy, boundary F1, cluster ARI and depth error per window and per source.

## Known caveats

- **Legend labels are abbreviations** (`Lmst`, `Sltst`, `Clyst dkgy`). The harness legend parser reads about half of them, so row accuracy partly measures label reading.
- **Truth gaps**: intervals whose legend name was ambiguous were left out of the truth (no row). Blank rows are not scored, as in the challenge scorer.
- **Units**: Flat Bay 2 depth labels are metres, while the truth `*_depth` columns follow the page labels.
- **Layout coverage**: every page is a strip log with a "Rock Types"-style legend. That is closest to our Texas family and unlike the Gulf of Mexico 3-panel logs. See `GENERALISATION.md`.

## Licence and attribution

These are public records published by the agencies above; the source URL of every page is in `manifest.csv`. They are shared only inside this private team repo for research and testing. Credit the source agency if you show a page.
