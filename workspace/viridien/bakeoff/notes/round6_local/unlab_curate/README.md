# Unlabelled TEST lane handoff

The local curation and metric code are complete. Heavy work now belongs on
the boat VM. Do not restart local inference or image processing.

## Frozen data

The dataset is `external_unlabelled_test/` under `viridien_work/`.
It contains images, challenge-format region boxes, 130 fixed windows,
a source manifest, TEST rules and a flat gallery at `index.html`.
Never train or tune on this data, its boxes or its results.
The dataset's existing `README.md` was checked and left untouched after
the owner pinned the files.

| Layout guess | Pages |
|---|---:|
| Gulf 3-panel | 60 |
| Texas Rock Types | 13 |
| MC765 colour grid | 25 |
| One-off | 5 |
| Geoservices grid | 8 |
| D-582 1-bit grid | 12 |
| MC761 mineral/lithology | 7 |
| Total | 130 |

There are 36 well-name groups and 61 source URLs. The source audit found
no missing URL, licence note or visual review. Reuse terms remain
unverified. Family names describe similar styles. They do not confirm
challenge-family membership. The MC761 pages are a mineral-volume stress
subset. There is no rock truth.

The eval-set SHA256 is:
`55a2961965557fe334524d40b8e99459933e3bcef6852bbc840ebbe15d909365`.

The pinned set already includes 43 round-6 pages: 27 Mizzen, four Norway
and 12 partial-bulk Oregon pages. See `round6_already_in_frozen.csv`.
Do not change the frozen set. The parent will build any supplementary set
separately on the VM and must check overlap with these 43 identifiers.

## Completed checks

- Scored 1,007 candidate pages against 38 non-eval challenge exemplars.
- Checked all 130 selected annotations on 33 contact sheets.
- Checked enlarged depth crops for the new monochrome pages.
- The freeze check found zero exact duplicate images and zero banned wells.
- Added label-free metrics, 18 unit tests and the `unlab_test` suite.
- Backed up `bench.py` to `bench.py.bak_r6` before the change.
- Metric unit command: `python -X utf8 -m pytest harness/test_unlabelled_metrics.py -q`.
  Last summary: `18 passed in 11.82s`.
- Combined metric and bench command: `python -X utf8 -m pytest harness/test_unlabelled_metrics.py harness/test_bench.py -q`.
  Last summary: `1 failed, 30 passed in 19.26s`. The existing PNG/TIFF reader test failed.

## Remaining work

The parent is stopping the local run and resuming it on the boat VM.
The last local journal observed here reached 125/520 clean cases.
The VM will use the pinned runner, models and dataset. It will finish the
clean run, make the fixed medium scan copy and finish the scan run.
The parent will return outputs to `runs/clean/` and `runs/medium/`.
The final per-family strategy comparison is not yet available.

Do not change `run_suite.py`, `make_degraded.py`, the metric module,
`bench.py`, the dataset or model files while the remote job is pinned.
No further local round-6 fold is due.

When both complete result sets return, run this from
`viridien_work/bakeoff/`:

```text
python -I -X utf8 notes/round6_local/unlab_curate/report_results.py .. notes/round6_local/unlab_curate
```

It requires 520 records in each run, the prediction CSV files, metric
reports, matching model fingerprints and unchanged-file records. It writes
`REPORT.md`, `result_summary.json` and `comparison.html` here. It reads saved
tables only. The report command has not yet been run.

## Main records

- `CURATION_METHOD.md`: selection, box checks, source limits and freeze record.
- `METRICS.md`: definitions, missing-data rules and run handoff.
- `page_scores.csv`: all page scores.
- `selection_decisions.json`: selection and rejection reasons.
- `visual_review.json`: approved identifiers and sheet hashes.
- `set_validation.json`: freeze-check output.
- `provenance_check.json`: URL, licence-note and review check.
- `freeze_phase1/`: frozen manifest, windows and review records.
- `initial_attempt/`: first-attempt log and partial journal.
- `preprocessing_cache_check.json`: matching failure status and messages.
- `metrics_checks.log`, `tests.log`: actual test output.

No commit or push was made.
