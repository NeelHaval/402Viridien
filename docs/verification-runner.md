# Run the final pipeline on new data

Use Python 3.12. Run each command from the clone root. Keep the clone writable:
OCR writes a legend cache in the model folder. The four S19 model files are in
`pipeline/s19/models/`.

## Install and check

No system OpenCV libraries or build tools are needed. On a bare Ubuntu image,
install `python3 python3-venv git` first.

```sh
git clone --depth 1 https://github.com/NeelHaval/402Viridien.git
cd 402Viridien
python -m venv .venv
.venv/bin/python -m pip install -r requirements-final.txt
.venv/bin/python -m pipeline.final --selftest
.venv/bin/python -m pipeline.final --images /path/to/images --boxes /path/to/bbox_annotation --out outputs/verification --workers 2
```

On Ubuntu, use `python3` for the venv command if `python` is absent.
`requirements-final.txt` includes `requirements-s19.txt`; use either entry point
for the same pinned runtime. The vendored RapidOCR wheel changes dependency
metadata only. Its source, OCR model files and license match upstream 1.4.4.
Both EasyOCR and RapidOCR use `opencv-python-headless==4.11.0.86`.

On Windows PowerShell, use these commands after the clone:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-final.txt
.venv/Scripts/python.exe -m pipeline.final --selftest
.venv/Scripts/python.exe -m pipeline.final --images 'C:/data/images' --boxes 'C:/data/bbox_annotation' --out outputs/verification --workers 2
```

The selftest has a 55 second outer limit. Its small page tests imports, model
loading and exports. It does not prove that OCR weights for new legends are
available offline.

## Prepare for offline use

While the network is available, run:

```sh
.venv/bin/python scripts/prepare_final_ocr.py
```

On Windows, use `.venv/Scripts/python.exe`. Keep the same user home for the offline
run. EasyOCR stores `english_g2.pth` under `~/.EasyOCR/model/`. For Docker, keep this
home in a volume and mount it in the offline container. Installing Python
packages does not download these weights. Without the weights, a new page can
fail with `URLError`. The manifest records this failure and the batch continues.

## Input folders

Pass the organiser folders as supplied. TIFF, PNG, JPG and JPEG extensions are
case insensitive. Image and JSON searches include nested folders. Match JSON
files by image stem; use matching relative paths if stems repeat. JSON can hold
the box keys directly or under `bounding_boxes`. The required boxes are
`lithology_legend` and `interpreted_lithology_column`. A `depth_column` allows
depth OCR. Paths can contain spaces and Unicode characters.

For a root with `images/` and `bbox_annotation/`, this command is equivalent:

```sh
.venv/bin/python -m pipeline.final --data /path/to/dataset --out outputs/verification
```

The training folders in the clone are `data/dataset/` and
`data/test/external_shared/`. The 2026-11-08 organiser source has 52 images and
51 matching JSON files. The extra image has no annotation and must produce a
failed page, not stop the batch.

The legacy command `python -m pipeline.run` uses the same requirements. Use
`pipeline.final` for the bounded batch, manifest, nested inputs and resume checks.

## Output and repeat runs

Each output page folder contains notebook and slide CSVs under `csv/`, a raw CSV,
a scaled column preview, `document.sidecar.json` and `worker.log`. Original pixel
coordinates stay in the CSV; the preview can use a smaller vertical scale.
`official_manifest.json` records every page, its status, failure type, input
hashes, output hashes and depth status.

Exit 0 means all pages have status `ok`. Exit 1 means at least one page failed.
Exit 2 means an invalid command or missing input folder. Always check
`depth_complete`: status `ok` can have blank depths when OCR or unit checks fail.
Missing depths stay blank. Unknown printed units are not treated as verified.

The output folder can exist and contain other files. A repeat run reuses pages
whose successful receipt has the same image hash, JSON hash and units option.
It retries failed pages. Use `--force` to replace page outputs. Use a new output
folder after changing the model, code or runtime: resume does not validate these
versions or repair modified output CSVs.

Two workers run by default. Each page has a 600 second limit; use `--timeout 30`
for a short failure check. A timeout has `failure_type: timeout`. Other pages
continue. More workers can use more RAM and can make each page slower.

Supported TIFF strips avoid full-page decoding. When a TIFF needs the Pillow
fallback, the default budget shares half of available RAM across workers, with
a maximum of 4,096 MB per page. Linux container memory limits are included in
this calculation. RGB fallback storage is estimated at four bytes per pixel.
Crops have a 32 million pixel limit. Multi-frame TIFF input uses frame zero.

For trusted large scans on a machine with enough RAM, set an explicit budget
and use fewer workers. The team VM test uses three workers and 3,072 MB per page:

```sh
.venv/bin/python -m pipeline.final --data /path/to/dataset --out outputs/verification --workers 3 --raster-budget-mb 3072 --timeout 600
```

The budget bounds fallback raster storage. Models, OCR, crop arrays and the
Python process also use RAM. Inputs above the chosen budget have a typed
`decode_limit` failure. Lower the worker count or select a larger budget if the
machine has room for those pages.

## Compare with the released CSVs

Run the three released hashed source pages with `--allow-assumed-units`, then
compare. This flag belongs to the runner, not the comparison script:

```sh
.venv/bin/python -m pipeline.final --images /path/to/official-images --boxes /path/to/official-json --out outputs/reproduced --allow-assumed-units
.venv/bin/python scripts/compare_final.py --actual outputs/reproduced --official outputs/official --report outputs/diff.json
```

Assumed units keep the `ft-assumed` status. Use the default units check on new
validation data. The comparison checks CSV headers, row counts and every cell.
A missing CSV or an empty official CSV produces a failed report.
## Verification results: 2026-11-08

These checks used fresh public clones and fresh virtual environments. The VM
has 16 cores and 62 GiB RAM. Clone time is excluded from install times. Linux
times measure pip installation; Windows includes venv creation and cached wheel
installation. The final packages contain only headless OpenCV.

| Environment | Install time | Initial failure | Final selftest |
|---|---:|---|---|
| Docker `python:3.12-slim`, no apt packages or build tools | 41 s | GUI OpenCV required libGL | PASS |
| Docker Ubuntu 24.04, system Python 3.12 | 44 s | GUI OpenCV required libGL | PASS |
| Docker `python:3.12` | 41 s | GUI OpenCV required libGL | PASS |
| VM host, Python 3.12.3 | 41 s | No install failure | PASS |
| Windows, Python 3.12.7 | 186.23 s | No install failure | PASS |

Ubuntu received Python and venv packages only. No libGL package was added in
the final checks. `pip check` reports `No broken requirements found.` in all five
environments. `pip freeze` shows `opencv-python-headless==4.11.0.86` and no
`opencv-python` package. The slim selftest was repeated after the PELT change.

| Scenario | Before the fixes | After the fixes |
|---|---|---|
| 10 organiser pages with random new stems, upper-case extensions, wrapped and direct box JSON | 9 OK, 1 decoder limit | 10 OK |
| 51 paired pages, workers 16 | Earlier run: 38 OK, 10 decoder limits, 3 timeouts. Test-lane run after first budget fix: 40 OK, 8 decoder limits, 3 timeouts | All 13 earlier failures rerun: 13 OK. Final fresh public clone + fresh venv: 51 OK in one 16-worker batch, with 3,072 MB per-page raster budget |
| 51 pages, workers 1, timeout 30 s | 28 OK, 8 old decoder limits, 15 typed timeouts; finished in 837.87 s | Full inference with one worker was not repeated after the PELT fix |
| 4 GiB memory cap, 2 CPUs, 2 workers, 10 renamed pages | Missing libGL; then 9 OK and 1 old decoder limit | 10 OK with headless packages |
| Offline container, no weights, 10 renamed pages | 9 typed URLError failures and 1 old decoder limit | With weights prepared: 10 OK under `--network none` |
| Mixed corrupt TIFF, zero bytes, missing JSON, corrupt JSON, unmatched JSON, nested folders, Unicode/space paths, PNG/JPG | Old runner had no nested search and no typed page reason | 3 valid pages OK; 4 typed failures; unmatched JSON listed; batch finished |
| Existing non-empty output directory | Rejected | Run completed; unrelated file retained |
| Repeat same output | Rejected | Initial check: 45 CSV hashes unchanged. Final 51-page repeat with workers 1: exit 0 in 0.777 s; all 255 CSV hashes unchanged |
| Timeout 30 s, 51 pages, workers 16 | Untyped timeout reason | 38 typed timeouts, 10 old decoder limits, 3 OK; all 51 jobs finished |
| Official reproduction with assumed units | Baseline retained | All 12 CSVs match exactly after the PELT fix |
| Legacy `pipeline.run` on bundled PNG | TIFF decoder exception | PNG run and exports pass |

The long-page bottleneck was PELT's repeated variance calculation. Stack samples
showed `CostL2.error` scanning long slices. Cumulative sums now give the same L2
objective with constant-time interval cost. Tests compare float32 and float64
costs and boundaries. The official comparison still matches all 12 CSVs.

Focused check command:

```sh
python -m pytest tests/test_final.py tests/test_compare_final.py tests/test_legacy_runner.py tests/test_pelt_cost.py -q
```

Last summaries: Linux `21 passed in 5.94s`; Windows `21 passed in 17.11s`.
The runtime install does not include pytest; install pytest for this developer
check. The judge selftest does not need pytest.

The [JSON evidence](verification-runner-results.json) records each page status,
row count, depth status, input hashes, runner hash and output source. A successful
page can still have blank depths.

### Final 51-page result

The table uses the final fresh-clone batch, not a merge of prior batches. All 51
pages have status `ok`; 28 have calibrated depths and 23 have blank depths.
`unavailable` depths remain blank; they are not a failed export.

| Page stem | Status | Rows | Depth status | Seconds |
|---|---|---:|---|---:|
| 0000000009_d-582_0000020435-0001_log01_fe_1in_md | ok | 1866 | unavailable | 79.463 |
| 0000000009_d-582_0000020435-0001_log02_fe_5in_md | ok | 153 | unavailable | 29.528 |
| 0000000009_d-582_0000020435-0001_log03_chromatograph_ratio | ok | 145 | unavailable | 29.708 |
| 0000000009_d-582_0000020435-0001_log04_fe_5in_md | ok | 152 | unavailable | 29.519 |
| 0000000009_d-582_0000020435-0001_log05_chromatograph_ratio | ok | 155 | unavailable | 31.256 |
| 0000000009_d-582_0000020435-0001_log06_fe_5in_md | ok | 152 | unavailable | 28.841 |
| 0000000009_d-582_0000020435-0001_log07_chromatograph_ratio | ok | 169 | unavailable | 30.125 |
| 0000000009_d-582_0000020435-0001_log08_fe_5in_md | ok | 91 | unavailable | 26.651 |
| 0000000009_d-582_0000020435-0001_log09_chromatograph_ratio | ok | 104 | unavailable | 25.287 |
| 60807402860000_gb506#2_md_mudlog | ok | 137 | calibrated | 34.691 |
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | ok | 593 | unavailable | 76.865 |
| 60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev | ok | 81 | calibrated | 43.492 |
| 60811405120000_1inch tvd mudlog | ok | 188 | calibrated | 44.446 |
| 60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400 | ok | 416 | unavailable | 80.042 |
| 60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2 | ok | 5 | calibrated | 24.02 |
| 60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft | ok | 131 | calibrated | 118.523 |
| 60811405800300_mdn-1_st01bp01_5 inch tvdmudlog_4560ft to 20251ft | ok | 26 | calibrated | 111.176 |
| 60811405800500_608114058005_g16786_001_03_00_tvd5_mud | ok | 143 | calibrated | 99.883 |
| 60811406050000_608114060500_g31765 001 st00 bp00__master log_5640-28740_5 in_md | ok | 112 | unavailable | 184.147 |
| 60811406050100_ardennes_ocs-g31765 001 st00 bp01__master log_5640-36552_5 in_md | ok | 524 | unavailable | 184.077 |
| 60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872 | ok | 254 | calibrated | 50.69 |
| 60811407570000_bp_gc743_15607_dc802_st00bp00_mudlog_5md_6900-17643 | ok | 281 | calibrated | 80.307 |
| 608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD | ok | 83 | calibrated | 41.082 |
| 60812400480000_ocs-g 26419 001 st00bp00_masterlog_7860-27400_5in_md_2 | ok | 958 | calibrated | 102.727 |
| 60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721 | ok | 97 | calibrated | 37.36 |
| 60812401020000_xom_ocs-g_20351_ju103_st00bp00_mudlog_10200-30900ft_5in_tvd | ok | 545 | unavailable | 201.363 |
| 60817410140000_mc822#10 1-1200 mud log | ok | 256 | unavailable | 46.224 |
| 60817410140100_mc822#10bp01 1-240 tvd mud log | ok | 806 | unavailable | 109.718 |
| 60817411400100_mc761_1_st1_mudlog_1md_17133_22886 | ok | 77 | calibrated | 19.464 |
| 60817411400100_mc761_1_st1_mudlog_1tvd_17129_20968 | ok | 52 | unavailable | 18.793 |
| 60817411400200_mc761_1_st1_bp01_mudlog_1md_17133_25775 | ok | 88 | unavailable | 29.699 |
| 60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md | ok | 295 | calibrated | 103.519 |
| 60817411700000_mc776-5_5in_md_combo_mud_log_20415_mms | ok | 273 | calibrated | 96.711 |
| 60817412840000_mc_776_6_ml_mudlog_5757_23721_5in_md_final | ok | 377 | calibrated | 88.797 |
| 60818400600100_608184006001_g08037_002_01_00_5 inch mudlog_13452ft-19912ft | ok | 177 | calibrated | 34.556 |
| 60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378 | ok | 227 | unavailable | 32.124 |
| Callaway 3-33 mudlog to TD.PDF | ok | 597 | calibrated | 55.882 |
| Critter Creek 201-1807H_Mud_PDF_Vertical.pdf | ok | 173 | calibrated | 36.24 |
| Critter Creek 202-1807H_Mud_Vertical_PDF.pdf | ok | 340 | calibrated | 39.593 |
| MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft | ok | 103 | calibrated | 28.675 |
| MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft | ok | 134 | unavailable | 67.455 |
| MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft | ok | 118 | unavailable | 57.382 |
| MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft | ok | 48 | unavailable | 25.368 |
| Mudlog-Trail Mtn1Palisades | ok | 85 | calibrated | 57.577 |
| Mudlog_MD_FREEMAN 3 24 ST | ok | 168 | calibrated | 41.585 |
| Rio_Lavaca_1748_Unit_1_241_MUD | ok | 1191 | calibrated | 82.527 |
| Rio_Nueces_3744_Unit_1_212_MUD_5IN | ok | 249 | calibrated | 48.484 |
| Rio_Nueces_3744_Unit_1_222_5IN_MUD | ok | 458 | calibrated | 49.579 |
| Rio_San_Gabriel_2829_Unit_2_272_MUD | ok | 236 | calibrated | 37.629 |
| Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft | ok | 31 | calibrated | 16.615 |
| Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft | ok | 244 | unavailable | 55.178 |

The final VM output is
`/srv/viridien/submission/20261108/swarm/validation-runner/results/final51-w16/`.
Its manifest has `complete: true`. The extra unpaired organiser image is outside
this 51-page paired set; it remains a typed `missing_annotation` failure when
all 52 images are supplied.
