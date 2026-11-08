# Set up and run the workspace

Use Python 3.11 or 3.12. The core example runs on a CPU. Start in a fresh clone:

```bash
git clone --filter=blob:none --sparse https://github.com/NeelHaval/viridian402.git
cd viridian402
git sparse-checkout set workspace
cd workspace
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` in Linux or macOS. Then install the core packages:

```bash
python -m pip install -r viridien/requirements.txt
python viridien/tools/workspace.py doctor
python viridien/tools/workspace.py verify
```

The doctor reports whether original scans and lab seed data are installed. Both
are optional for the shared example below. Some OCR packages download weights
on first use. The core example needs no paid model service.

## Run the shared example

These images, region boxes, and answer CSVs are already in Git:

```bash
cd viridien/bakeoff
python -m harness.bench --strategy s11 --suite ext_shared --workers 1
```

The command prints a new run directory under `out/bench/`. It writes prediction
CSVs, metrics, a per-window table, and an HTML report. Open the report from that
directory. This is a stress test on outside data.

To score your own prediction CSVs, run from the repository root:

```bash
python viridien/external_shared/score_csvs.py viridien/external_shared PRED_DIR
```

Name a prediction `<window-id>.csv`, with the ID from `external_shared/eval_set.json`.
The scorer also accepts `<log>__y<top>_<bottom>.csv`. Use complete-image row
coordinates. A missing prediction counts as an empty prediction.

## Use the original scans

Extract the organiser archive. Find the directory with both `images/` and
`bbox_annotation/`. Set its absolute path before starting a tool.

Windows PowerShell:

```powershell
$env:LITHOLOGY_DATASET = 'D:/data/Viridien_HTB_challenge/hack27/project/dataset'
```

Linux or macOS:

```bash
export LITHOLOGY_DATASET=/absolute/path/to/project/dataset
```

Then, from `viridien/bakeoff/`:

```bash
python -m harness.bench --strategy s11 --suite viridien_gold --workers 1
```

This scores the organiser's example window. Other `viridien_eval` windows can
use AI consensus labels. Keep those results separate from human truth.

## Get crop images and optional outputs

Run these commands from the repository root. Use your own GitHub account:

```bash
gh auth login
python viridien/tools/workspace.py download real
python viridien/tools/workspace.py verify --require-assets
```

The download checks the ZIP checksum and refuses to replace a different local
file. You can also download the ZIP in the browser, then install it:

```bash
python viridien/tools/workspace.py download real --archive /path/to/real-labels.zip
```

The [data catalog](data/catalog.json) lists five separate bundles. Read the
[data guide](data/README.md) before choosing training or test records.

- [Review human labels](docs/LABELLING.md).
- [Inspect or generate fake patterns](docs/GENERATION.md).
- [Compare models](docs/EVALUATION.md).
- [Read the known limits](docs/STATUS.md).
- [Trace the four work streams](docs/WORK_RECORD.md).

For learned strategies, install `viridien/requirements-models.txt` after the core
packages, then run `python viridien/tools/workspace.py download models`. Some
experiments also need rebuilt caches or OCR weights. Read their source and reports.

Use the current [contribution guide](../../CONTRIBUTING.md) for branches and
pull requests. Historical tool ports are in [services](../docs/services.md).
