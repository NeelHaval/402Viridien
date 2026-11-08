# Vision silver-label pilot

The pilot has 10 intervals on two non-eval logs. It labels 1,779 pixel rows.
The source crops, review notes, CSV files, and checks are in this folder.
Open [the image review page](index.html) to see the labels and source images.

## Reference limit

There is no eligible silver reference. The two silver folders have 26 CSV files.
All 26 file names match the 13 eval log names. The check used file names only.
No reference CSV contents were read. See [reference_audit.json](reference_audit.json).

Agreement, row accuracy, boundary F1, and error rates are **not available**.
The pilot has no measured calibration. Do not treat a review grade as a probability.
No independent second review was made.

## Labels

| Log | Window, pixels | Intervals | Filled rows | Blank rows |
|---|---:|---:|---:|---:|
| Callaway 3-33 mudlog to TD.PDF | 5096 to 6896 | 1 | 1573 | 227 |
| Mudlog_MD_FREEMAN 3 24 ST | 1833 to 2193 | 9 | 206 | 154 |
| Total | 2160 rows | 10 | 1779 | 381 |

All bottom rows are excluded. Blank rows have no CSV interval.
The Callaway pattern matches the printed SILTY SHALE swatch.
The Freeman labels are SAND, CLAY-MUDSTONE, and SILTSTONE.
There are 1,751 rows with a high review grade and 28 rows with a medium grade.

The thin red and grey Freeman bands have a medium grade. Their height is only
seven pixels. The red band can be CLAY. The grey bands can be SHALE.
The yellow bands match SAND. SANDSTONE remains a possible alternative.
The Callaway alternative is SHALE GRAY. Its marks are less dense than the observed marks.

The estimated boundary error is ±2 pixels. This estimate comes from image review.
It has no reference check. The last interval in each window ends at the crop edge.
That end is not a new rock boundary. Depth columns are empty because no depth
calibration was made. Thin white seams can affect the boundary estimate.

## Files for use

- `labels/visual/`: the rock names read from the printed legends by direct image review.
  These CSV files have the six harness columns. Some names are missing from the shared parser.
- `labels/harness/`: names that match `harness.legend`, version 13, or `Unknown`.
  The files keep the visual cluster IDs. They keep adjacent intervals at drawn boundaries.
- `labels/training/`: five SAND interiors, with 158 pixel rows in total.
  Each source interval has a high grade and an exact parser name match.
  Two pixels are removed from each end. Medium-grade bands and `Unknown` rows are omitted.
  The Callaway training CSV has a header only.
- [training_items.json](training_items.json): the five allowed training ranges.
  Paths in this file start from this folder. The labels remain uncalibrated.
- [review.json](review.json): label decisions, alternatives, grades, and source views.
- [sources.json](sources.json): source image names, boxes, and SHA-256 hashes.
- [summary.json](summary.json): row counts and the missing agreement result.
- [artifact_hashes.json](artifact_hashes.json): hashes of the labels and image evidence.

The shared parser joins several Callaway names into one entry. It does not return
SILTY SHALE as a separate name. It also omits CLAY-MUDSTONE and SILTSTONE on Freeman.
Thus, 1,601 reviewed rows become `Unknown` in the harness export. The visual export
keeps their names. No false match to a joined parser name was made.
No shared parser code or cache was changed.

## Method and provenance

Codex used `view_image` to inspect saved legend and track crops. The review was
made in this session on 2026-11-08. The first image review came before the parser run.
The large view makes source pixels larger. It does not add image detail.
The Freeman source views extend to row 3633. Only rows 1833 to 2193 have pilot labels.
No class decision came from OCR or a pixel classifier.

RapidOCR was used after the visual review to check the shared parser names.
Its cache is under `parser_cache/` in this folder. It did not assign rock classes.
See [parser_provenance.json](parser_provenance.json).

No inherited lithology weights were used. No model was trained, tuned, or selected.
Codex and RapidOCR pretraining data were not audited. Their prior data overlap is unknown.
No claim of a new independent accuracy test is made.

No eval log image or eval truth was opened. Only the eval ID list was read to block
those logs. No outside data was read. The shared-verify well and the sealed wells
were not searched for or read. No benchmark was run. The final benchmark is for
the orchestrator. These two windows do not show performance on other wells.

## Repeat the exports and checks

Run from the repository root. Set `LITHOLOGY_DATASET` to the dataset folder.
Use the project Python interpreter. On Linux, set the thread limits before the commands:

```sh
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export PYTHONDONTWRITEBYTECODE=1
python bakeoff/ml/label_pilot/prepare.py
python bakeoff/ml/label_pilot/legend_names.py
python bakeoff/ml/label_pilot/build.py
python -m pytest bakeoff/ml/label_pilot/test_pilot.py -q -o cache_dir=bakeoff/ml/label_pilot/.pytest_cache
```

The Python code uses paths from the repository or `LITHOLOGY_DATASET`.
It has no fixed VM or Windows path. On Windows, set the same environment variables
with the shell in use. All generated files stay in this lane folder.
The scripts repeat the crops and exports. They do not repeat the Codex image review.
The review decisions are fixed in `review.json`.

Seven checks passed. They check the eval block, unknown-log block, window coverage,
overlap rejection, evidence hashes, parser names, training masks, and the missing
reference result. The measured crop run used 374,960 KiB peak RAM, about 0.36 GiB.
The parser run with a fresh cache used 603,084 KiB, about 0.58 GiB. This was the
largest measured peak. The separate resource logs give the measured runs.
See [validation.json](validation.json) for the check result. Thread limits are at
most three. Torch was not used.
