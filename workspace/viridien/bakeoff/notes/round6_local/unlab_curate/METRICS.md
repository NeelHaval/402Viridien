# TEST diagnostics without rock truth

Never train or tune on this set or its results. These checks do not measure rock-type accuracy.
They can show failed runs, weak image reading, unused column rows, and unstable output.

The harness keeps its normal accuracy fields empty. It reads no truth for `unlab_test`.
The added files are `unlabelled_metrics.csv`, `unlabelled_by_family.csv`,
`strategy_agreement.csv`, `strategy_agreement_by_family.csv`, and `unlabelled_report.json`.

| Field | Meaning and limit |
|---|---|
| execution_failure | The harness could not complete the run. The error remains in results.json. |
| failure_rate | One for a failed run or a run with no named, assigned rows. Mean includes every page. |
| assigned_share | Union of rows with a named rock assignment, divided by window height. Unknown and blank rows do not count. Overlaps do not count twice. A wrong named rock can still give full coverage. |
| depth_monotonic | Fitted depth increases as image row increases. |
| depth_ft_per_px | The calibration slope, after the harness unit conversion. A missing or wrong source unit can still cause error. |
| depth_plausible | Slope is between 0.005 and 5 feet per pixel. This is a broad fixed check, not a fitted limit. |
| depth_fit_ok | Positive and plausible slope, at least three anchors, and a residual of at most 16 pixels. Two-anchor fits are retained but do not pass this check. Missing fits do not pass. |
| predicted_depth_monotonic | Depth in the model's output does not decrease. This also checks the adjusted depth used by s19. |
| legend_entries_found / named | Rock entries found and entries with a name. Generic cluster numbers and Unknown do not count as names. |
| legend_parser | Identifies the parser measured. s11, s17 and s22 use the shared harness parser. The run script records s19's actual v3 entries and adjusted depth, with the same fixed model settings. If preprocessing fails, only the available shared-parser evidence is retained. |
| output_names | Distinct named rock types in the output. It does not prove the names are correct. |
| short_bed_rate | Share of output intervals shorter than eight pixels, after clipping to the window. A real thin bed can also meet this condition. |
| beds_per_1000px | Number of output intervals per 1,000 image rows. Use it to compare fragmentation. There is no known true bed count. |
| oversegmentation_rate | Share of touching output intervals that repeat the same name across their boundary. This is a redundant-boundary proxy. It cannot detect every false bed. |
| ari | Adjusted Rand agreement between s19 and s17 or s22. One means the same grouping, even if class numbers differ. It uses only rows assigned by both runs. Common blank rows never improve it. One constant class can give one, so class counts and support are also reported. |
| scan_ari | The same agreement check between a clean page and its degraded copy, for the same strategy. The fixed medium scanify preset adds blur, noise, fade, colour change and JPEG loss. Pixel geometry is unchanged. No TEST result selects this preset. |

The scan check does not apply shear or resolution scaling. Thus it measures photometric stress,
not all scan damage. A failed side or fewer than two common assigned rows gives no agreement
score. The report also gives the common assigned share, eligible-pair count and failure rate.

Family means weight pages equally. Most pages from one well are related evidence. The family
tables give the page count and well count. Missing measurements stay missing; observed-page
counts show their support. The named layout groups are analogues, not verified family membership.

The run script uses owned cache paths and two workers with one compute thread each.
The RapidOCR model and image settings stay fixed; only its thread pools are bounded.
It caches deterministic depth-fit ValueErrors under the benchmark's exact image, annotation
and inference-file identity. This avoids the same failed OCR fit for every strategy.
The first-attempt journal verifies matching failure status and messages after the change.
It reads the installed model weights and fixed settings. It does not fit a model,
select a threshold, or use rock truth.

## Run status and report handoff

The local run is incomplete. The last observed local journal reached
125 of 520 clean cases. This is not a final comparison of the strategies.
Several historic pages failed in the shared depth fit before any strategy
could predict. The recorded error was
`ValueError: Depth calibration extrapolates above zero at column top`.
Other pages completed prediction. Keep execution failures and empty named
output as separate fields in the final report.

The owner moved the remaining heavy work to the boat VM. The parent will
copy back `runs/clean/` and `runs/medium/`. Each completed run must contain
520 result records for this 130-page release. Both runs must have
`results.json`, `unlabelled_report.json`, `model_fingerprints.json`,
`model_files_unchanged.json` and their prediction CSV files.

After both complete runs return, run this from `viridien_work/bakeoff/`:

```text
python -I -X utf8 notes/round6_local/unlab_curate/report_results.py .. notes/round6_local/unlab_curate
```

This light report step reads saved tables only. It does not run a model or
process images. It checks the result counts, absent truth scores, matching
inference-file hashes and the unchanged-file records. It writes `REPORT.md`,
`result_summary.json` and `comparison.html` in this lane. The old 09:00
source-fold gate was removed after the owner's instruction. No `--draft`
flag or late-source file is required.

The last metric unit check was:
`python -X utf8 -m pytest harness/test_unlabelled_metrics.py -q`.
Its last summary line was `18 passed in 11.82s`.
The combined metric and bench check ended with
`1 failed, 30 passed in 19.26s`. The failed existing test was
`test_notebook_callable_with_spawn_and_cached_second_run`.
It raised `tifffile.TiffFileError: not a TIFF file b'\x89PNG'`.
The frozen dataset uses TIFF files. The image reader is outside this lane's
write scope. The final result report has not been run because both complete
VM result sets are still required.
