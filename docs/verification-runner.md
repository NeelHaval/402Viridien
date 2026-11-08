# Running on new verification data

Use Python 3.12 and install the S19 dependencies once:

```sh
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-final.txt
python -m pipeline.final --images <images_dir> --boxes <bbox_json_dir> --out <new_output_dir> --workers 2
```

Give the command the organiser folders as supplied. It accepts TIFF, PNG and JPG
images and matches each JSON file by image stem. Use an empty output directory.
Large trusted images are accepted. Each page has a 600 second timeout; use
`--timeout <seconds>` to change it. A failed page does not stop other pages. The
command returns 1 if any page fails. Check `depth_complete` in the manifest too:
a successful page can have blank depths when OCR or unit checks fail.

Allow about 10–30 seconds for a short page and 1–5 minutes for a long page on the
team VM. Two pages run at once by default. The first run can download the EasyOCR
recognition weights. Install and run a check before the verification deadline.

Each `<new_output_dir>/<image_stem>/` contains:

- `csv/<image_stem>.official.raw.notebook.csv` with the official notebook header.
- `csv/<image_stem>.official.raw.slide.csv` with the `base_depth` alias.
- `<image_stem>.official.raw.csv` and `<image_stem>.official.full-column.png`.
- Output sidecars, `document.sidecar.json`, `worker.log` and a private OCR cache.

`official_manifest.json` lists all pages, input and output hashes, source and model
hashes, dimensions, full column bounds, calibration and unit status. The command
prints a final table with page, row count, depth status and elapsed seconds.
Depth values use the detected printed units. There is no unit conversion. An
unknown unit or a failed fit gives blank depth fields and a reason. No zero value
is inserted to replace a missing depth.

The released official bundle has provisional depths with assumed feet units.
For a comparison with that bundle only, add `--allow-assumed-units`. This option
keeps the `ft-assumed` status and does not mark the depths as verified. Use the
default command for new verification data.

On the team VM, the environment and wrapper are installed at `/srv/viridien/verify/`:

```sh
/srv/viridien/verify/run_final.sh <images_dir> <bbox_json_dir> <new_output_dir>
```

Set `FINAL_WORKERS` or `FINAL_TIMEOUT` to change the wrapper defaults. To check CSV
values against the released bundle:

```sh
python scripts/compare_final.py --actual <output_dir> --official outputs/official --report diff.json
python -m pytest tests/test_final.py -q
```
