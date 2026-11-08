# Unlabelled log demo

Open [the gallery](index.html). It has 27 contact sheets with source pixels and
label overlays. The run made 64 intervals on 15 pages. Two other pages had no
usable parsed legend. This is a partial demo of the 651 input pages. The other
634 pages have no reviewed boxes and were not run. See [the full list](inventory.csv).

The input folder has no semantic box files. I made boxes for 17 pages in this
lane. These include all eight new round-four Oregon pages. I also made image
previews for 26 pages across all nine sources. The boxes are in [boxes.json](boxes.json).
They came from image pixels. They contain no bed labels. Companion legends
come from the same source log.

S04 was selected from five fixed strategies with no inherited fitted weights.
The selection uses saved outside dev results. Each record was filtered by the
current `eval_set_dev.json` before its score was used. This gives 51 windows
from five wells. ForsythB2-17 is excluded. Each well has equal weight in the
selection score. The score for each window is row accuracy times boundary F1.

| Strategy | Mean score by well | Mean score by window |
| --- | ---: | ---: |
| S04 | 0.17227 | 0.17748 |
| S01 | 0.10550 | 0.13949 |
| S11 | 0.10413 | 0.15678 |
| S02 | 0.05557 | 0.07962 |
| S03 | 0.01099 | 0.00849 |

See [selection.json](selection.json). These are dev selection results. They are
not new test results. No new cross-validation was done. The strategies have
fixed settings, so no model fit was needed for this comparison. S04 fits local
visual clusters during inference. It does not use truth or a saved trained model.
This comparison does not establish the best strategy among all possible models.

The inherited learned lanes report training and selection on the old six-well
dev set. That set included ForsythB2-17. Thus, their weights and their selection
results have historical exposure to the current shared verify subset. Those
weights, parser settings, and confidence models were not used here. This demo
uses the shared original legend parser. The full history of its fixed rules
was not independently checked.

| Source | Pages with demo boxes | Pages with predictions | Result |
| --- | ---: | ---: | --- |
| Oregon DOGAMI | 10 | 10 | Many parsed names are damaged. |
| DHI Wister | 1 | 1 | Two broad labels; mixtures are lost. |
| Precision Well Logging | 2 | 2 | Parsed names are short fragments. |
| Stratagraph | 1 | 1 | Only one parsed name is usable by the API. |
| Mississippi | 2 | 0 | No usable parsed legend. |
| Utah FORGE | 1 | 1 | Readable legend; thin beds need review. |

All confidence values are **unverified**. S04 has no calibrated confidence.
No probability was invented. The [confidence files](confidence/) contain a
null confidence value for each interval. They also contain a raw descriptor
check: the nearest swatch name, its relative distance gap, and agreement with
the output name. This check uses unscaled descriptors. It is separate from the
S04 cluster decision. It is not a measure of accuracy and does not change labels.

The following failures need attention:

- On CC23-04-65, S04 emits `S4Ad` for the whole box. The source has many visible
  changes. The legend parser returned damaged names. This output needs review.
- On CoosCounty1-7, the legend parser accepts text fragments such as `LoGY 1`
  and `10` as names. The overlay preserves these names so the failure is visible.
- On Precision log-01, names such as `le` and `st` are fragments. The legend
  has text to the left of the swatches. The shared parser does not read it well.
- On the Stratagraph example, the parser supplies only `ONITE`. One name cannot
  describe the visible colour changes. This is a parser failure case.
- On Wister page 2, the output has two broad intervals. The source shows several
  rocks side by side. A single name per row loses these mixtures.
- On Utah FORGE, S04 makes short intervals in the mostly empty area above the
  first filled beds. Grid lines can be treated as rock texture.
- Both Mississippi pages are blocked. The small legend has no usable parsed
  entry. No prediction CSV was written for either page.
- The COSO sample is sideways. The NWT and Singles samples include headers or
  mostly empty tracks. These samples have previews but no demo inference.
- Long scans can drift sideways. A fixed column box can include a border or
  part of another track. These boxes need a further check before production use.

The gallery includes [parsed legend pictures](legends/). No names were repaired
by hand. There is no exact truth for these pages. No accuracy, boundary F1,
cluster ARI, or depth error is claimed. Depth fields are empty because this demo
does not calibrate depth.

Each of the 15 supported column boxes was processed in one strategy call.
The 51 gallery views are crops of those page results. A view contains at most
4,000 source rows. View edges do not add bed boundaries to the CSV files.
Cluster IDs belong to one source page. The PNG overlays use a white background,
flat colours, and readable text. The local HTML page scales to phone width.

The cold-cache run took 18.25 seconds and used at most 0.604 GiB of RAM. The
final run reused the local OCR cache. It took 9.92 seconds and used at most
0.247 GiB of RAM. The final strategy calls took less than one second per page.
OMP, MKL, OpenBLAS, and NumExpr were limited to three threads. OpenCV used one
thread. The OCR engine uses two threads per pool. No torch model was loaded.
See [resources.txt](resources.txt), [run.log](run.log), and [summary.json](summary.json).

No final benchmark was run. No held-out truth file was opened. No sealed well
was searched for or reconstructed. During the initial audit, I opened old
reports that also contain historical final-set scores. A broad read of an old
dev selection file also displayed scores for the former dev well. Those scores
were not used for selection. Only the filtered 51-window dev scores above were
used. No training, threshold change, or model fit used any final-set data.

All new code, caches, previews, predictions, and reports are in this lane's
folder. No source data, shared strategy, or team service was changed. No file
was downloaded. No child agent, commit, or push was used.

From the repository root, use the project Python environment:

```text
python -I bakeoff/notes/unlabelled_demo/demo.py
python -I -B -m pytest -q -p no:cacheprovider bakeoff/notes/unlabelled_demo/test_artifacts.py --basetemp=bakeoff/notes/unlabelled_demo/pytest_tmp
```

All six checks passed. The checks cover all input page statuses, dev split size, CSV row order and
bounds, confidence records, blocked pages, gallery links, and view coverage.
The saved results are in [results.json](results.json). Source and dev split
hashes are saved with the results.
