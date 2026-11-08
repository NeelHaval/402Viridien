# Wave 3 final results

Report time: 2026-11-08T03:30:20.190180+00:00. The choice was fixed at 2026-11-08T03:21:01.493115+00:00.

**Chosen strategy: S19.** The final scores did not change this choice. No model, setting, threshold, or code was changed after the freeze.

**Resource limit failed.** The final process tree peaked at 5.423 GiB RSS. This exceeds the 4 GiB limit. The two-window dev smoke check peaked at 2.301 GiB and did not reveal this full-run peak. The final scores are retained as run. There was no held-out rerun. See `ml/final/resource_check.json`.

## Selection evidence

Selection used the current 51 dev windows from five wells and synthetic columns from non-eval in-domain source logs. Each dev fold excluded the whole query well from both model fits. Training and held synthetic source logs were separate. No real non-eval interval labels are present in this mirror. The synthetic check is a limited substitute, not real-log interval CV.

The corrected, fixed S19 decoder gives dev score 0.510669 by well and synthetic score 0.515755 by family. The selection value is 0.511941. It gives dev weight 0.75 and synthetic weight 0.25. S21 reports 0.310140 for the same weighting, with a different synthetic set. These are selection estimates. They are not nested CV or an independent outer-well test.

S11 has dev score 0.1041 by well. S22 has dev score 0.1839 by well; its synthetic check measures boundary proposals only. S20 has clean parser CV, but its deployed wrapper loads old S18 weights. It cannot be selected as a clean full system. Different synthetic checks limit a direct rank across lanes. S19 has the strongest permitted full-pipeline selection evidence among the clean deployed candidates.

The S19 check used the lane settings without a new grid. The bug fixes did not change its CV scores. See `ml/final/selection_recheck.json` and `ml/final/FREEZE.json` for exact values and settings.

A separate smoke check used two permitted dev windows and S19-S22. All eight predictions passed. Four workers used one compute thread each. Peak total process-tree RSS was 2.301 GiB. This checked execution and memory only; it did not change the choice or any setting. See `ml/final/dev_smoke.log`.

## Final table

Each mean gives every window equal weight. This differs from the well means used for selection. Score is row accuracy times boundary F1 for each window. Quality failures count as zero. Depth MAE is in feet and uses matched boundaries with finite depth values only. A dash means no value. The machine table gives the count for each metric. ARI can be negative. The final dev results use full-dev model fits; they are training-set checks, not CV.

S13, S14, S15, S16, S17, S18 and S20 have historical Forsyth contamination. Their rows are comparisons only. S19, S21 and S22 use clean new weights. S11 has no learned checkpoint. Do not treat shared-verify scores for contaminated models as held-out evidence.

| Suite | Strategy | Windows | Failures | Row accuracy | Boundary F1 | ARI | Depth MAE ft | Score | Seconds | Mean RSS GiB | Max RSS GiB | API calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ext_dev | s11 | 51 | 0 | 0.438561 | 0.333060 | 0.685827 | 0.758822 | 0.156784 | 0.115093 | 0.544453 | 0.605759 | 0.000000 |
| ext_shared_verify | s11 | 8 | 0 | 0.440514 | 0.471692 | 0.468326 | 0.670646 | 0.261609 | 0.068754 | 0.529281 | 0.559010 | 0.000000 |
| ext_test | s11 | 20 | 0 | 0.102084 | 0.216408 | 0.538804 | 0.626885 | 0.058662 | 0.162833 | 0.534033 | 0.600723 | 0.000000 |
| viridien_eval | s11 | 13 | 0 | 0.428177 | 0.498363 | 0.558934 | 0.271429 | 0.207237 | 0.249156 | 0.690728 | 1.294937 | 0.000000 |
| ext_dev | s13 | 51 | 0 | 0.457324 | 0.524696 | 0.821962 | 0.306114 | 0.231473 | 0.094497 | 0.480887 | 0.487804 | 0.000000 |
| ext_shared_verify | s13 | 8 | 0 | 0.515817 | 0.609416 | 0.551722 | 0.360705 | 0.289240 | 0.061013 | 0.484714 | 0.487343 | 0.000000 |
| ext_test | s13 | 20 | 0 | 0.155672 | 0.324212 | 0.430262 | 0.487394 | 0.071227 | 0.153989 | 0.482415 | 0.489346 | 0.000000 |
| viridien_eval | s13 | 13 | 0 | 0.454137 | 0.508379 | 0.608391 | 0.325000 | 0.245895 | 0.139492 | 0.637673 | 1.244328 | 0.000000 |
| ext_dev | s14 | 51 | 0 | 0.439938 | 0.340820 | 0.679919 | 0.778254 | 0.161431 | 0.239663 | 0.481637 | 0.563530 | 0.000000 |
| ext_shared_verify | s14 | 8 | 0 | 0.628587 | 0.449543 | 0.487456 | 0.576195 | 0.337847 | 0.145159 | 0.483759 | 0.485313 | 0.000000 |
| ext_test | s14 | 20 | 0 | 0.226451 | 0.247973 | 0.480288 | 0.636880 | 0.088270 | 0.340279 | 0.483117 | 0.489483 | 0.000000 |
| viridien_eval | s14 | 13 | 0 | 0.486983 | 0.374740 | 0.473956 | 0.290909 | 0.161240 | 0.406422 | 0.632294 | 1.246170 | 0.000000 |
| ext_dev | s15 | 51 | 0 | 0.484372 | 0.520774 | 0.873241 | 0.287054 | 0.252204 | 0.376548 | 0.481069 | 0.488785 | 0.000000 |
| ext_shared_verify | s15 | 8 | 0 | 0.672782 | 0.623994 | 0.655156 | 0.358616 | 0.423192 | 0.192835 | 0.484721 | 0.487385 | 0.000000 |
| ext_test | s15 | 20 | 0 | 0.171101 | 0.301344 | 0.531533 | 0.475643 | 0.072752 | 0.663778 | 0.483067 | 0.489098 | 0.000000 |
| viridien_eval | s15 | 13 | 0 | 0.522115 | 0.503005 | 0.591260 | 0.000000 | 0.342898 | 0.413441 | 0.629393 | 1.243206 | 0.000000 |
| ext_dev | s16 | 51 | 0 | 0.891198 | 0.520774 | 0.873241 | 0.287054 | 0.461658 | 0.393760 | 0.481674 | 0.488960 | 0.000000 |
| ext_shared_verify | s16 | 8 | 0 | 0.913598 | 0.623994 | 0.655156 | 0.358616 | 0.564172 | 0.187439 | 0.485914 | 0.488319 | 0.000000 |
| ext_test | s16 | 20 | 0 | 0.606899 | 0.301344 | 0.531533 | 0.475643 | 0.225426 | 0.703130 | 0.483518 | 0.490940 | 0.000000 |
| viridien_eval | s16 | 13 | 0 | 0.522243 | 0.503005 | 0.591260 | 0.000000 | 0.343793 | 0.470572 | 0.646137 | 1.249119 | 0.000000 |
| ext_dev | s17 | 51 | 0 | 0.488869 | 0.529473 | 0.852348 | 0.260439 | 0.246435 | 0.407294 | 0.481325 | 0.487953 | 0.000000 |
| ext_shared_verify | s17 | 8 | 0 | 0.661482 | 0.645572 | 0.636063 | 0.430871 | 0.431059 | 0.187665 | 0.485687 | 0.487400 | 0.000000 |
| ext_test | s17 | 20 | 0 | 0.184544 | 0.268273 | 0.525444 | 0.391027 | 0.058983 | 0.670641 | 0.483017 | 0.489544 | 0.000000 |
| viridien_eval | s17 | 13 | 0 | 0.598475 | 0.589530 | 0.661617 | 0.311111 | 0.377920 | 0.373629 | 0.639013 | 1.244980 | 0.000000 |
| ext_dev | s18 | 51 | 0 | 0.490145 | 0.562916 | 0.864796 | 0.362804 | 0.287787 | 0.527607 | 0.481003 | 0.488495 | 0.000000 |
| ext_shared_verify | s18 | 8 | 0 | 0.652644 | 0.686391 | 0.774943 | 0.353408 | 0.418842 | 0.273008 | 0.484431 | 0.488537 | 0.000000 |
| ext_test | s18 | 20 | 0 | 0.245823 | 0.409927 | 0.602055 | 0.498429 | 0.111626 | 0.939775 | 0.483083 | 0.489704 | 0.000000 |
| viridien_eval | s18 | 13 | 0 | 0.516765 | 0.618111 | 0.602494 | 0.266667 | 0.303059 | 0.752157 | 0.635534 | 1.247845 | 0.000000 |
| ext_dev | s19 | 51 | 0 | 0.932454 | 0.580897 | 0.889715 | 0.335179 | 0.544187 | 0.801015 | 0.541880 | 0.669975 | 0.000000 |
| ext_shared_verify | s19 | 8 | 0 | 0.918973 | 0.842432 | 0.691640 | 0.440201 | 0.774895 | 0.369836 | 0.508263 | 0.671715 | 0.000000 |
| ext_test | s19 | 20 | 0 | 0.750781 | 0.389570 | 0.610992 | 0.486250 | 0.300972 | 1.445819 | 0.526311 | 0.680908 | 0.000000 |
| viridien_eval | s19 | 13 | 0 | 0.533168 | 0.641596 | 0.649675 | 0.333333 | 0.398659 | 0.828924 | 0.631146 | 1.243214 | 0.000000 |
| ext_dev | s20 | 51 | 0 | 0.897206 | 0.562916 | 0.864796 | 0.362804 | 0.508234 | 0.586623 | 0.481582 | 0.489471 | 0.000000 |
| ext_shared_verify | s20 | 8 | 0 | 0.894068 | 0.686391 | 0.774943 | 0.353408 | 0.601715 | 0.272794 | 0.484433 | 0.487411 | 0.000000 |
| ext_test | s20 | 20 | 0 | 0.693314 | 0.409927 | 0.602055 | 0.498429 | 0.290080 | 0.987764 | 0.483645 | 0.488960 | 0.000000 |
| viridien_eval | s20 | 13 | 0 | 0.518521 | 0.618111 | 0.602494 | 0.266667 | 0.304283 | 0.790259 | 0.646287 | 1.249107 | 0.000000 |
| ext_dev | s21 | 51 | 0 | 0.483712 | 0.553178 | 0.864629 | 0.250052 | 0.248392 | 0.431796 | 0.481235 | 0.487946 | 0.000000 |
| ext_shared_verify | s21 | 8 | 0 | 0.652222 | 0.720678 | 0.629147 | 0.427383 | 0.497131 | 0.237753 | 0.485363 | 0.487400 | 0.000000 |
| ext_test | s21 | 20 | 0 | 0.353708 | 0.299272 | 0.643474 | 0.374804 | 0.088061 | 0.659401 | 0.482660 | 0.489536 | 0.000000 |
| viridien_eval | s21 | 13 | 0 | 0.576134 | 0.545066 | 0.597335 | 0.423077 | 0.296256 | 0.314524 | 0.640307 | 1.245010 | 0.000000 |
| ext_dev | s22 | 51 | 0 | 0.462725 | 0.574122 | 0.843243 | 0.285216 | 0.252280 | 0.123128 | 0.480159 | 0.488689 | 0.000000 |
| ext_shared_verify | s22 | 8 | 0 | 0.387298 | 0.816577 | 0.666444 | 0.384228 | 0.323126 | 0.065782 | 0.484060 | 0.487213 | 0.000000 |
| ext_test | s22 | 20 | 0 | 0.114944 | 0.353953 | 0.504313 | 0.446344 | 0.062836 | 0.196284 | 0.482543 | 0.489086 | 0.000000 |
| viridien_eval | s22 | 13 | 0 | 0.482855 | 0.692642 | 0.667460 | 0.335714 | 0.315822 | 0.124080 | 0.634364 | 1.247532 | 0.000000 |

The JSON file also contains well-bootstrap 95% confidence limits for all reported means. It records all per-window rows, failures, source hashes, truth hashes and prediction cache keys. A single shared-verify well gives no estimate of between-well variation.

## Execution and failures

The one final invocation wrote `out/bench/20261108T032117_db6f0e94`. It used four workers and one compute thread per worker. There are 1012 result rows, 0 prediction or preprocessing failures, and 0 render failures. Cached predictions: 0. Peak total process-tree RSS: 5.423 GiB. All 11 requested strategies were present. No strategy ID was omitted.

The source fingerprint includes harness, strategy and ML code, selected settings and deployed model files. Prediction, strategy and preprocessing caches include this fingerprint. Image and annotation hashes identify each source. Depth caches also identify exact depth-track and unit-context pixels. Old prediction keys cannot match this run. Raw parser OCR caches use exact image pixels.

Final dependency SHA-256: `5e14eccbdb1de22f7b9de4917a3c91cb08d853ec25f99af91922217a28bd841c`.

No prediction or preprocessing failed.

## Review findings and fixes

Both review files were read before any final evaluation. The detailed resolution is in `notes/wave3_reviews/final_resolution.md`.

- The checkpoint audit confirms old Forsyth use in S13-S18 and deployed S20. These assets were kept as labelled historical comparisons. New S19, S21 and S22 checkpoint metadata name only the five approved wells. The audit checked current dev identities and synthetic source separation. S19 fold hashes match its fit records. S19 starts its CNNs from random weights and its matchers from new fits. It does not load the old confidence weights.
- Old boundary and matcher caches have unsafe positional identities and remain outside this lane ownership. They were not used for the choice or for new training. The historical S18 prepare and select commands now refuse to resume finalized experiments or reuse unversioned interval caches. A full clean legacy rebuild was not safe within the confirmed-fix scope.
- Benchmark keys now include ML code, settings and weights. Strategy and preprocessing caches use the same digest. Paired comparisons reject source or truth hash differences.
- Depth unit detection now reads “Depth in feet” before unrelated metre text. Equal-support depth fits prefer lower residual before longer span. Pixel content identifies the depth cache. No fixed depth offset or per-log rule was added.
- The runner and scorer reject non-finite and conflicting boundary depths. Legacy quality failures now count as zero in all main quality means. Figure errors do not change valid prediction scores. Truth-read errors cannot trigger a second unguarded read. Both runners sort labeler paths for a fixed consensus tie rule.
- Joint decoding splits at known blank gaps before the duration check. Rule restoration checks each consecutive cut and applies the strong-probability policy separately. Restored rows use the probability at their actual top. The S19 allowed CV check was repeated after these fixes.
- The old S18 “nested” diagnostic is not independent outer-well validation. It was not used for this choice. Earlier final runs exist for inherited designs; the present refits cannot remove that history. The local Emil retrain used six reserved eval images for pseudo labels. Default Drive model provenance is incomplete. S12 was not requested and was not run.

## Tests

Commands ran from bakeoff with the project Python environment and CPU thread limits. The logs retain the full output.

```text
python -m pytest harness/test_bench.py harness/test_depth.py harness/test_run.py harness/test_score.py harness/test_wave3_fixes.py ml/integrate/test_integrate.py -q -o cache_dir=ml/final/.pytest_cache
78 passed, 1 subtests passed in 8.64s

python -m pytest ml/combine/test_combine.py ml/legend_v4/test_legend_v4.py ml/synth_v2/test_synth_v2.py ml/boundary_v2/test_boundary_v2.py -q -o cache_dir=ml/final/.pytest_cache
1 failed, 49 passed in 3.88s

python -m pytest harness/test_bench.py harness/test_depth.py harness/test_run.py harness/test_score.py harness/test_wave3_fixes.py ml/integrate/test_integrate.py ml/combine/test_combine.py ml/legend_v4/test_legend_v4.py ml/synth_v2/test_synth_v2.py ml/boundary_v2/test_boundary_v2.py -q -k 'not test_equal_support_depth_fit_uses_residual_before_span' -o cache_dir=ml/final/.pytest_cache
129 passed, 1 deselected, 1 subtests passed in 7.65s
```

The initial lane failure is an obsolete assertion in another lane: it expects the shared depth fitter to keep the old wrong slope greater than 3 ft per pixel. The fixed fitter now gives about 0.13323. That test file was not changed. The final test command omits that one obsolete assertion. The owned replacement `test_depth_equal_support_prefers_residual_before_span` checks the correct result.

## Data found and request list

The benchmark has 13 in-domain eval logs, 51 external dev windows from five wells, 20 external test windows from three wells, and eight Forsyth shared-verify windows from one well. The three sealed wells were not opened, searched for, or rebuilt. The split declaration was read only as metadata. No sealed suite was run.

Round three has 32 page logs and 32 exact-truth windows with 250 intervals from one Mirror Lake well. Keep these pages in one well group. Round three also supplied 643 unlabelled pages. Round four added eight Oregon scan pages and no exact-truth pages. These sets were not used for model choice or the final table. Current manifest record counts are in `ml/final/audit.json`; images were not scored.

BSEE pilot request list: `notes/bsee_list/pilot_request.csv`. It lists 60 files from 20 API well groups, about 1.49 GB. The full list is `notes/bsee_list/request_documents.csv`, with 959 files from 126 groups, about 16.04 GB. The owner instructions are in `notes/bsee_list/REPORT.md`. No request was sent.

## Files

- `ml/final/final_table.json` and `final_table.csv`: complete suite means, metric counts and failures.
- `ml/final/final_windows.json` and `final_windows.csv`: every benchmark window and source identity.
- `ml/final/FREEZE.json`: the choice, selection evidence and code/model fingerprints before final scoring.
- `ml/final/audit.json`: checkpoint and data audit.
- `out/bench/20261108T032117_db6f0e94/report.html`: the flat benchmark report with source, predictions, truth and legends.

No services were changed. No child agent was started. No commit or push was made.
