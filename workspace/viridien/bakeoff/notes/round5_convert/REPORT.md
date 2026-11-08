# Round 5 conversion report

Date: 8 November 2026. Lane: r5_convert.

**Result: 13 distinct source logs, 17 distinct source body pages, and 17 review windows.**
Five logs are real mud or geologist logs. One log is a fictional example.
Seven logs are LWD mineral-volume plots from seven named holes at three sites.
The LWD plots are not seven independent site groups. All layouts have reviewed similarity 2 of 3.
There is no score-3 layout. This score is a visual feature check against the source family description.
It is not an image score against challenge data.

## Sources and reviewed layouts

| Source log | Full source pages | Active pages / windows | Score | Checked layout | Source URL |
| --- | ---: | ---: | ---: | --- | --- |
| OCG-S-2490 | 1 | 1 / 1 | 2 | Historic Gulf mud log; depth, four-symbol key, patterns, gas and drilling tracks. Scan skew remains. | [Original source](https://pubs.usgs.gov/ds/760/Downloads/OS/OFFSHORE/OCG-S-2490_MUDLOG_200_17500.tif) |
| OCG-S-2523 | 1 | 1 / 1 | 2 | Historic Gulf geologist log; ten-symbol key, patterns, depth, sample text and ROP. Header is partly sideways. | [Original source](https://pubs.usgs.gov/ds/760/Downloads/OS/OFFSHORE/OCG-S-2523_GEOLOGIST_1240_17388.tif) |
| JIP2-AC21-A | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/AC21-A/field_log_plots/AC21-A_Ecoview_Solutions.pdf) |
| JIP2-AC21-B | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/AC21-B/field_log_plots/AC21-B_Ecoview_Solutions.pdf) |
| JIP2-GC955-H | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/GC955-H/field_log_plots/GC955-H_Ecoview_Solutions.pdf) |
| JIP2-GC955-I | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/GC955-I/field_log_plots/GC955-I_Ecoview_Solutions.pdf) |
| JIP2-GC955-Q | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/GC955-Q/field_log_plots/GC955-Q_Ecoview_Solutions.pdf) |
| JIP2-WR313-G | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/WR313-G/field_log_plots/WR313-G_Ecoview_Solution.pdf) |
| JIP2-WR313-H | 1 | 1 / 1 | 2 | Gulf LWD plot; mineral-volume column, mineral key, depth and curves. It is not a cuttings mud log. | [Original source](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/WR313-H/field_log_plots/WR313-H_Ecoview_Solutions.pdf) |
| MPC-10-4 | 15 | 2 / 2 | 2 | Onshore Mississippi; cuttings percentages, depth, key, gas and rock text. Scale is 5 inches per 100 ft. | [Original source](https://edx.netl.doe.gov/storage/f/edx/2019/06/2019-06-10T17:53:14.903Z/d751e382-c7ff-48ca-8791-207d32dc91e1/5440-mpc-10-4-mudlog-5in.pdf) |
| MPC-34-1 | 4 | 2 / 2 | 2 | Onshore Mississippi; cuttings percentages, depth, key, gas and rock text. Scale is 1 inch per 100 ft. | [Original source](https://edx.netl.doe.gov/storage/f/edx/2019/06/2019-06-13T13:44:21.570Z/6e190c69-df50-42c6-89b9-eff2f2597719/5750-advanced-resources-1inch-td.pdf) |
| MRCSP-FEGENCO-1 | 55 | 2 / 2 | 2 | Ohio; cuttings patterns, depth, key, gas and rock text. Displayed page is landscape. | [Original source](https://edx.netl.doe.gov/storage/f/edx/2018/07/2018-07-30T20:20:31.315Z/7b1937a0-85f2-4b1c-9de0-af062588d680/mud-log-fegenco-1.pdf) |
| FICTIONAL-BIG-OIL-GUSHER-1 | 4 | 2 / 2 | 2 | Fictional Texas example; cuttings, depth, key, curves, photos and rock text. It is not a real Gulf well. | [Original source](https://mud-log.com/wp-content/uploads/2026/07/Selman_Example_Mudlog_B.pdf) |

All source URLs above returned downloaded bytes in the completed input lanes.
This lane used those local bytes. It made no new network request.
The source manifest records each SHA-256 hash, byte count, page count, and source decision.

## Full volume and converted volume

The six input lanes provide 26 downloaded score-2 records with 100 source pages.
Some records are individual slices of the same well. The count is not 26 distinct logs.
Eleven of the 100 pages are very long TIFF or PDF log pages.

| Decision | Source records | Full source pages | Active body pages / windows |
| --- | ---: | ---: | ---: |
| Accepted | 13 | 87 | 17 / 17 |
| Existing Mississippi page copies | 10 | 10 | 0 / 0 |
| Repeated OCG-S-2523 well interval | 1 | 1 | 0 / 0 |
| No confirmed legend | 2 | 2 | 0 / 0 |
| Total | 26 | 100 | 17 / 17 |

Eighteen complete source page images are retained. This includes the Selman page-1 header.
The other 69 accepted source pages are retained in original PDFs. They were not fully rendered or boxed.
EDX contributes 74 original pages. Only its first two pages per log were converted.
The Selman file has four pages. Pages 2 and 3 give active windows. Page 1 supplies the header and key.
Page 4 is a short body end with too few useful depth anchors. It adds no window.

The USGS catalogue has 40 Gulf files from 15 wells. Thirteen have mud-log or geologist-log type names.
Those 13 catalogue rows cover seven wells. Only four source TIFFs were downloaded and checked by the input lane.
Two distinct Gulf source logs were converted here. No catalogue count was used as converted image volume.
The BSEE inventory has 15,923 Gulf mud-log TIFF records and 3,319 ZIP records.
Those are metadata counts. Useful image volume and anonymous package access remain unknown.
The literature lane has no verified score-2 or score-3 candidate. Its context reports add no target pages.
The names lane supplies the three EDX sources. It also supplies provenance and exclusion metadata.

## Boxes, source images, and truth

Each active page has three reviewed boxes. The total is 51 boxes.
Each box gives its source page, source coordinates, image coordinates, source hash, and image hash.
See `external_unlabelled/round5/reviewed_boxes.csv` and its `bbox_annotation/` folder.
All bounds came from actual source pages. Review sheets and native region crops retain the visual evidence.
Five continuation images copy a header and key from page 1. Their explicit maps passed pixel checks.
Each complete selected source page is retained before a header copy or top crop.
The two Gulf images retain full source TIFFs. The PDF files also remain complete.

The OCG-S-2490 scan is skewed. Its axis-aligned column box can contain small border or depth-label parts.
This is a geometry limit. It must not be used as exact interval truth.
FEGENCO source boxes use its displayed landscape page. The 90-degree PDF rotation was checked.
There are 220 depth-label anchors. Their positions are label evidence, not a calibrated depth map.

**No rock truth was made.** Sixteen selected PDF pages were checked for vector content.
No exact interval-to-rock semantic map was validated. No interval truth CSV was saved.
Cuttings percentages and LWD mineral fractions do not establish one rock class for each interval.
The EcoView keys are mineral templates. They do not prove which minerals occur.
The dataset records this distinction in its column types.

## Duplicate and provenance checks

All ten Mississippi source pages match existing source URLs and RGB pixels.
They add source-access evidence, but no new log, page, or window.
The OCG-S-2523 mud and geologist logs cover the same well interval.
The clearer geologist log was kept. The mud log adds provenance only.
OCG-S-2486 and the WellSight screenshot stay out because no source legend was confirmed.

The check compared source URLs and hashes across all lane triage files.
It also compared existing unlabelled and evaluation source metadata.
The 51 challenge names and all 24 known API10 roots were exclusion inputs.
Accepted data has zero known evaluation source-URL, name, or API-root matches.
Challenge-related BSEE records remain provenance metadata only. No related log package was used.
The common BSEE source hypothesis remains unproved by exact file identity.

No original challenge-image hash catalogue was available. Unknown renamed or cropped copies remain a check limit.
No challenge image, evaluation image, evaluation truth, or sealed source was opened.
The required well exclusions were kept. No excluded well was sought or reconstructed.
AC21-A and AC21-B share a printed API. Their different body plots and named holes were retained.
The shared site groups and the unusual GC955-I printed API are explicit. No ID was repaired.

## Commands and actual results

Run these commands from the workspace root. The available Python is the local virtual-environment Python.
All source parsing and rendering commands use `-I`. Rendering used one CPU thread.
The build and data check also fixed CPU affinity to one CPU.

```bash
/home/user/lith/.venv/bin/python -I external_untrusted/round5/convert/inspect_sources.py > external_untrusted/round5/convert/inspection.log 2>&1
/home/user/lith/.venv/bin/python -I external_untrusted/round5/convert/build_dataset.py > external_untrusted/round5/convert/build.log 2>&1
/home/user/lith/.venv/bin/python -I external_untrusted/round5/convert/validate_dataset.py > external_untrusted/round5/convert/validation.log 2>&1
/home/user/lith/.venv/bin/python -I external_untrusted/round5/convert/write_docs.py > external_untrusted/round5/convert/docs.log 2>&1
/home/user/lith/.venv/bin/python -I external_untrusted/round5/convert/check_release.py > external_untrusted/round5/convert/release_check.log 2>&1
```

1. Input inspection: six complete triage files read; all 26 eligible file hashes and sizes passed.
   It saved a hash snapshot of 1,324 pre-existing unlabelled files and bounded source previews.
2. Build: 13 original source hashes passed; 17 active TIFF/box pairs and 17 windows were written.
   It retained 18 complete source page images and 87 original source pages.
3. Data check: **PASS**. All 17 IDs match images, boxes, manifest rows, and item windows.
   All 51 boxes are within bounds. All 220 anchor labels are within their review windows.
   All 51 harness RGB crops equal Pillow crops. All 22 source-page pixel maps match.
   The 17 active RGB hashes are distinct. All 13 original hashes and 18 page-image hashes pass.
   All 17 review sheets and 51 region-crop hashes pass. All 1,324 old files remain unchanged.
   The thumbnail check covered 651 existing thumbnails. No exact normalized thumbnail match was found.
   Thumbnail distances are review aids. They are not proof against unknown renamed copies.
   There are zero known evaluation metadata matches in active data. There are zero truth files.
4. Report check: source and box tables were made from the passed records. Exact source URLs are above.
5. Release check: see `external_unlabelled/round5/release_check.json` for file, link, hash, report-count, and ownership checks.

The checks used the harness I/O contract only. No strategy, OCR, training, scoring, or benchmark ran.
No dataset or registry outside the three owned folders was changed.
No data was added to test or verify. No downloaded code was run.
No sign-in, email, child agent, service, or commit was used.
No new download was needed, so no host rate limit was exercised.
Public access does not establish all third-party artwork rights. Keep the source terms and credits.
