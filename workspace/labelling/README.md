# Lithology labelling tool

Local annotation and evaluation for the HackTheBurgh well-log challenge. The viewer
shows tiled full-page scans, calibrates depths from printed numbers, and labels
beds against each page's own legend. Each bed has one rock at 100%.

## Run locally

Use Python 3.12 or newer. From the repo root, run these commands in order.
Stop if any command fails; a pip upgrade notice alone is harmless.

```sh
cd labelling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prep_labelling.py --workers 2
python scripts/label_server.py
```

First preparation may take several minutes. It creates the local tile cache and
never overwrites existing labels, assignments or dataset files.

The shared dataset lives in `../viridien/dataset/` (`images/` and
`bbox_annotation/`). On a fresh full clone, prep automatically restores missing
inputs from the legacy dataset snapshot already present in this branch's Git
history. This does not add the dataset back to tracked files. A shallow clone
needs `git fetch --unshallow` first. Alternatively, extract the organiser dataset
there, or set `LITHOLOGY_DATASET` to your existing dataset directory before prep
and server. Tile caches, datasets and virtual environments stay ignored.
`dataset_synthetic/` is not needed for labelling.

Choose your initial using the **Labeller** menu, or open your direct link:

- E: <http://127.0.0.1:8765/?assignee=E>
- B: <http://127.0.0.1:8765/?assignee=B>
- N: <http://127.0.0.1:8765/?assignee=N>

The tool remembers your initial and opens a page belonging to you. Changing the
initial switches to that person's pages after saving the current page.
Use **Next page** to save and open the next page in the selected labeller group.
You can switch pages at any time; existing annotations stay saved. The validation
tick checks annotation validity and does not certify that a whole page is labelled.

For automatic legend OCR, install Tesseract once (`brew install tesseract` on
macOS or `sudo apt-get install tesseract-ocr` on Ubuntu). Python packages do not
include that executable. Without Tesseract, swatch detection still works and you
can enter names manually in the review dialog.

If you see `ModuleNotFoundError: No module named 'cv2'`, OpenCV was not installed
in the Python environment running the tool. With `.venv` activated, run:

```sh
python -m pip install -r requirements.txt
python -c "import sys, cv2; print(sys.executable, cv2.__version__)"
```

Use `python -m pip` if your shell's `pip` points at another interpreter. Both prep
and server now check dependencies and explain how to fix them before starting.
On Linux, install `libtiff6` if the shared libtiff is unavailable. On macOS the
streaming reader uses libtiff bundled with Pillow.

## Labelling together

See [the assignment list](LABELLING_ASSIGNMENTS.md). Hadrian is already being
labelled and is reserved; do not replace its JSON with an empty copy.

Each teammate runs their own local copy. Open `/?assignee=E`, `/?assignee=B`, or
`/?assignee=N` to start on the assigned pages. Return only the assigned
`labels/<stem>.json` files, preserving everybody else's files. The server binds
localhost; this is not a publicly hosted web app.

1. Choose the assigned page and calibrate at least two printed depths (MD when both
   MD and TVD are present). Feet are stored; metre inputs are converted.
2. Detect legend swatches and review the names, or box them manually. Keep colour
   and pattern distinctions exactly as printed.
3. Draw beds or load unassigned boundary proposals and assign a legend entry.
4. Check validation. A tick means no errors. Wait for **Saved** before closing.

Useful controls: Lithology only / Show full page, +/− zoom, pan, the minimap,
edge scrolling while dragging, U for uncertain, and Ctrl/Cmd+Z for undo.

## Validate, export and evaluate

```sh
python scripts/validate_labels.py
python scripts/export_gt_csv.py
python eval.py preds/my-model --gt-dir labels_csv
```

Export produces challenge CSV plus a sidecar containing each single-rock layer.
Gaps are ignored in evaluation. See [the detailed spec](.plans/lithology-labelling-tool-spec.md)
for data contracts and scoring rules.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --input-type=module --check < labelling/app.js
```

Native macOS browser regression checks are documented in [tests/browser](tests/browser/README.md).
Prep never overwrites labels. Save history and migration backups remain local under
`labels/.history/` and `labels/.backup/`.
