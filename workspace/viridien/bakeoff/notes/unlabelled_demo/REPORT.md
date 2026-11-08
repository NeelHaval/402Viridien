# S19 demo on unlabelled pages

S19 is the frozen choice in `bakeoff/ml/final/FREEZE.json`. This run used that choice. No new selection was run.

The input set has 651 pages. All 17 reviewed pages were checked. S19 made 2879 intervals on 17 pages. There were 0 errors and 0 blocked pages. The other 634 pages have no reviewed boxes. They were skipped.

The [S04 baseline report](baseline_s04/REPORT.md) and [S04 summary](baseline_s04/summary.json) are kept with the baseline code and files. S04 made 64 intervals on 15 pages. It could not use the legends on two Mississippi pages. S19 made intervals on both pages. This result does not prove that their labels are correct.

| Page | Result | Legend names | Intervals | Labelled rows |
| --- | --- | ---: | ---: | ---: |
| ul_r4_bw_hatch_oregon_dogami_CC23-04-65_mud_p001 | predicted | 9 | 326 | 100.0% |
| ul_r4_bw_hatch_oregon_dogami_CC33-03-65_mud_p001 | predicted | 10 | 283 | 93.9% |
| ul_r4_bw_hatch_oregon_dogami_CC43-34-65-RD_mud-r_p001 | predicted | 8 | 177 | 100.0% |
| ul_r4_bw_hatch_oregon_dogami_CoosCounty1-7_mud_p001 | predicted | 6 | 393 | 99.9% |
| ul_r4_bw_hatch_oregon_dogami_Hammerberg1-RD_mud-R_p001 | predicted | 8 | 312 | 97.7% |
| ul_r4_bw_hatch_oregon_dogami_Keys1_mud1_p001 | predicted | 8 | 557 | 96.6% |
| ul_r4_bw_hatch_oregon_dogami_OM12C-03-65_mud_p001 | predicted | 8 | 143 | 90.9% |
| ul_r4_bw_hatch_oregon_dogami_CC12-19-65_mud_p001 | predicted | 13 | 131 | 98.1% |
| ul_r3_oregon_dogami_Adams24-34-75_mud_p001 | predicted | 10 | 267 | 93.7% |
| ul_r3_oregon_dogami_Adams31-34-75-RD_mud_p001 | predicted | 16 | 127 | 96.9% |
| ul_r3_dhi_wister_wister_p002 | predicted | 19 | 12 | 100.0% |
| ul_r3_precision_wl_log-01_p001 | predicted | 11 | 5 | 100.0% |
| ul_r3_precision_wl_log-01_p002 | predicted | 11 | 20 | 100.0% |
| ul_r3_stratagraph_Measured_Depth_Log_Example_p002 | predicted | 1 | 1 | 100.0% |
| ul_r3_mississippi_mdl123045000360000a-001_p001 | predicted | 3 | 6 | 100.0% |
| ul_r3_mississippi_mdl123045000360000a-002_p001 | predicted | 3 | 1 | 100.0% |
| ul_r3_utah_forge_mud_log_p001 | predicted | 15 | 118 | 100.0% |

All labels need review. There is no interval truth for these pages. No accuracy or depth error is claimed. Depth fields are empty.

S19 returns raw match cost, match margin, and boundary probability. The [confidence files](confidence/) keep these native scores for each interval. Boundary probability is a boundary score. It is not a probability that the rock name is correct. The current S19 code returns no interval confidence. Its confidence value is null. All confidence records are marked unverified. If S19 returns native confidence, the demo will keep it. No S04 descriptor distance is used as S19 confidence. No S18 confidence model is used.

The larger interval count needs review. Many short intervals can be false bed boundaries. OCR can give wrong or incomplete names. The stock v3 parser can clean printed names. No name was changed by hand. A single label cannot show side-by-side rock mixtures. Empty tracks, grid lines, and scan drift can affect results. The reviewed boxes are the same boxes as in the S04 demo.

The [offline gallery](index.html) has 53 column views and 27 contact sheets. There are 53 white PNG overlays, 17 legend images, 17 prediction CSV files, and 17 confidence JSON files. Each supported column was processed in one call. The pictures are crops of that result. View edges do not add interval boundaries.

The demo calls the current S19 run function with a page, a full column window, and CPU context. S19 reads its own v3 legends. A pixel adapter supplies the reviewed companion legend when needed. The same model files and fixed decoder settings are used. Their hashes are in [summary.json](summary.json). Only the resource wrapper is removed to set one thread. The S19 models are loaded before cache paths are moved into this demo folder. RapidOCR uses the same settings with one thread per pool. Torch and numeric pools use one thread.

The last run took 31.28 seconds after imports. Peak RAM was 0.458 GiB. The limit was 4 GB. The run used one CPU thread for compute. See [run.log](run.log) and [resources.txt](resources.txt).

Only external_unlabelled image pixels and its manifest were used as input data. No training, model change, threshold change, reserved truth read, benchmark run, or service change was made. All new files and caches are in this demo folder. No child agent, commit, or push was used.

From the repository root, use a Python environment with the project packages. The code and local links work on Windows. No machine font path is needed.

```text
python -I -B bakeoff/notes/unlabelled_demo/demo.py --strategy s19
python -I -B -m pytest -q -p no:cacheprovider bakeoff/notes/unlabelled_demo/test_artifacts.py --basetemp=bakeoff/notes/unlabelled_demo/pytest_tmp
```

The artifact checks cover the frozen choice, baseline evidence, page counts, prediction format, row bounds, native scores, image files, view coverage, and local gallery links. See [validation.log](validation.log) for the last check result.

Last summary: 17 predicted pages; 2879 intervals; 0 errors; 0 blocked pages; 634 skipped pages.
