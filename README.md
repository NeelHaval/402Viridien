# Strata

Strata reads scanned well logs and turns the rock column into layers you can use: rock type, top and bottom depth, and where each layer sits on the page.

Built by **402_Payment_Required** for Viridien's *Decoding the Subsurface* challenge at HackTheBurgh 2026.

![Strata map with a well profile open](docs/showcase/app-well-profile.png)

## The problem

A well log is a long scanned strip. The rocks found at each depth are drawn as black-and-white patterns, with a legend on the side that says which pattern is which rock. That information is useful for drilling, storage and planning decisions, but it is locked inside an image.

Strata takes the scan and produces a CSV in the challenge format:

```csv
top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id
```

## How the pipeline works

1. **Slice the rock column into thin horizontal strips.** For each strip we measure ink intensity, colour, the number of connected marks, and the spacing between them. These features were picked to tell the patterns apart.
2. **Find the boundaries.** When those features change sharply, a new layer starts. Change-point detection splits the column into segments.
3. **Match each segment to the legend.** We compute the same features for each legend swatch, then give each segment the rock type of its closest swatch.
4. **Read the depth scale and export.** OCR reads the depth ticks, rows are mapped to depths, and the layers are written as CSV plus an annotated image.

![Strip features down a scan. Red lines are detected boundaries, grey dashed lines are hand labels](docs/showcase/pipeline-strip-features.png)

*Strip features on one log. Red lines are the boundaries we detected and grey dashed lines are hand labels. Boundary F1 is 0.857 within ±20 px on this log.*

![Legend swatches matched to rock labels](docs/showcase/02-fixture-legend.png)

### What we tried that didn't work

- **Clustering strips without the legend.** Grouping similar strips split a single rock into stripes, and rock-type accuracy was too low. Matching against the legend worked better.
- **One merged Clay/Shale class.** It hid a real distinction. We now keep the page's own labels and only treat the two as equivalent when scoring.
- **A large vision model (DINOv2) in the main path.** It helped on some logs, but its first call was far too slow for the time budget.

## How we tested it

Labelling by hand was slow, so we built our own test data in stages:

1. A small hand-labelled set.
2. AI-assisted labelling, checked by people.
3. **Pattern Labeller**, our own tool for marking at least five examples of every legend pattern on every page (82 pattern groups across 51 documents).
4. **Synthetic logs** generated from those labelled patterns, plus public strip logs (the closest public match was KGS) so the test set looks more like real data.

All of this lives in a shared evaluation setup with CI. Every strategy is scored on the same cases using row accuracy × boundary F1. Failures are drawn as pictures, so you can see *where* a version went wrong. A new version only replaced the current best after it scored higher and passed review.

![Where a version went wrong: hand labels next to two strategies, errors marked in red](docs/showcase/eval-failure-overlay.png)

## Results

**Released scans.** Here is the source column next to our predicted layers on the three released pages. These pages have no answer key, so this is a visual check, not an accuracy number.

![Released pages: source column beside predicted layers](docs/showcase/01-released-pages-s19.png)

**Generated scan.** On a generated four-layer log, the three predicted boundaries landed within 4 px of where they were drawn.

![Generated four-layer scan and its predicted layers](docs/showcase/03-fixture-source-prediction.png)

## Official challenge outputs

The required outputs for the three released test documents are in [`outputs/official/`](outputs/official/): one CSV per document in the challenge header (`top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`, file `*.official.raw.notebook.csv`), the slide variant with `base_depth` (`*.official.raw.slide.csv`), and the annotated full rock column (`*.official.full-column.png`) showing boundaries, depths, cluster IDs and lithology labels. All were produced automatically with no manual correction. Depth units were read by OCR and are marked as assumed feet, not verified.

## The Strata app

Upload a scan, see the well on a map, and click it to see its layers next to the depth scale. You can also select an area to compare the wells inside it.

| Map | Upload |
|---|---|
| ![Map](docs/showcase/app-map.png) | ![Upload dialog](docs/showcase/app-upload.png) |

The scores on the well panel (drilling ease, cost, CO₂ storage) are an early idea of where this could go. They are not validated numbers.

## Run it

You need Python 3.12, Node/npm and Make.

```sh
git clone https://github.com/NeelHaval/402Viridien.git
cd 402Viridien

# quick check on a generated example (no model download)
python scripts/local_test.py

# the app
cd demo
make setup
make run      # then open http://localhost:8000
```

To write a challenge CSV from saved predictions, see [docs/output-contract.md](docs/output-contract.md).

### The pipeline notebook

[`notebooks/pipeline_final.ipynb`](notebooks/pipeline_final.ipynb) walks through the whole pipeline step by step: load a page, segment the rock column, read the legend, classify each segment, then plot and score the result. The last sections run it on a list of pages and evaluate a batch against ground-truth label files.

### Run our model on the challenge data
Use Python 3.12 and create a virtual environment: `python -m venv .venv`.
Activate it, then run `pip install -r requirements-s19.txt`.
The four S19 model files are included in `pipeline/s19/models/`; OCR weights download on first use.
Run `python -m pipeline.run --images <provided-folder>/images --boxes <provided-folder>/bbox_annotation --out outputs/s19`.
Use the organiser folders; TIFF, PNG and JPG are accepted. CSV depths stay blank when calibration fails.
See [example output](examples/s19_output/) and [official outputs](outputs/official/).

## Repo map

| Path | What's there |
|---|---|
| `demo/` | The Strata app: FastAPI backend (upload, OCR, pipeline) and React + MapLibre frontend |
| `src/viridian402/` | Evaluation CLI, metrics, reports and model adapters |
| `scripts/challenge_export.py` | Checks saved rows and writes both challenge CSV layouts |
| `tests/`, `stress_testing/` | Test suite and stress cases |
| `docs/` | Output format, experiment guide, and how the figures were made |

## Limits

- Very thin layers, faded scans and unreadable legends can give wrong or missing layers.
- If a scan's depth scale can't be read, its layers stay in pixels. We leave the depths blank rather than guess.

See [docs/output-contract.md](docs/output-contract.md) for the CSV format and its checks.

## Credits

Team 402_Payment_Required. We used AI coding tools (OpenAI Codex/GPT models and Claude) for implementation, testing and review, as the event rules require us to disclose. We built on open-source libraries including NumPy, SciPy, OpenCV, PyTorch, RapidOCR, FastAPI, React and MapLibre.


## Running on new verification data

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
