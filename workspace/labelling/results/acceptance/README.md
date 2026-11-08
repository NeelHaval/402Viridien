# Browser acceptance exercise — 2026-11-07

Passed in a temporary, nonpersistent WebKit browser against an isolated server.
These annotations exercise interaction and export; they are **not reviewed geological ground truth**.
Real `labels/` files were not replaced. The page's two depth numbers and three
legend swatches were visually inspected; bed placements are deliberately test geometry.

- Full-page tile prep: 1.9 s streaming, 3.2 s total, ~107 MB reported peak on this page.
- Calibrated y=1770 → 6100 ft and y=21598 → 16000 ft.
- Boxed Clay-shale, Limestone and Fine sandstone legend swatches.
- Drew five full-width exercise layers and one matching-depth 80/20 two-slice layer.
- Loaded unassigned proposals into gaps, then undid the entire insertion in one step.
- Toggled uncertain and undid it. Reloaded the saved page and checked composition.
- Validation: 0 errors, 0 warnings; done tick displayed.
- Exported JSON, CSV and composition sidecar. Changed one of six equal-height layers
  in a copied prediction and evaluated it. Accuracy is 83.33%, as expected; boundary
  F1 is 100% at both tolerances. Composition L1 is 0.333.

Artifacts:

- [Label JSON](labels/60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev.json)
- [Challenge CSV](labels_csv/60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev.csv)
- [Composition JSON](labels_csv/60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev.composition.json)
- [Viewer screenshot](viewer.png)
- [Eval output](eval-output.txt)
- [Metric summary](scores/one_changed/summary.json)

## Eval output

```text
model one_changed  (thin-ft 0.0; BF1 in %, pages = scored-with-pred/total)
block  agg     pages  acc%  mF1%  BF1@2  BF1@5  err ft  ARI   pur%   dMAE  dMax  s/page  comp L1
test   mean    1/1    83.3  45.4  100.0  100.0  0.00    0.00  100.0  0.02  0.05  -       0.33
test   pooled  1/1    83.3  45.4  100.0  100.0  0.00    0.00  100.0  0.02  0.05  -       0.33
wrote results/acceptance/scores/one_changed/per_page.csv, summary.json, confusion.csv
```
