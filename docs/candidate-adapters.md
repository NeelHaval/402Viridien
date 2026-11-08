# Candidate adapters

## Notebook ink rule

Use `viridian402.notebook_ink:predict` with `configs/notebook-ink.json`.
The adapter uses the existing plugin interface and bounded RGB reader.
SciPy is required. It is in the existing `legacy` extra:

```sh
python -m pip install -c requirements.lock -c requirements-legacy.lock -e ".[legacy,test]"
```

The source is `notebooks/main_pipeline.ipynb`, cells 10, 12, 20 and 22.
The adapter copies the seven stripe features, z-score, ink rule, Lab/run-length
descriptor, mixed-bed split rule, and adjacent-name merge rule. It does not
execute the notebook. It does not train, load weights, read labels, use notebook
prediction caches, or call a service. The CNN is not part of this candidate.
No claim is made for the notebook's published CNN scores.

The adapter reads only the image and public boxes in `request`. It clips output
to the intersection of the requested window and lithology box. Output uses the
original image coordinates. IDs start from zero within the page request.
Repeated visual patterns use the same ID. With supplied swatches, IDs follow
swatch order and do not copy external legend IDs. Labels come from supplied
public swatches. Without swatches, labels are `Unknown`. White gaps are omitted.
Depth endpoints are absent, even if a supplied calibration is available. No
unit or datum is invented. The existing scorer reports missing depth.

For example, the focused test has red, blue and red beds at original rows
20–120, 120–220 and 220–320. The supplied-swatch result keeps these coordinates
and returns cluster IDs 0, 1 and 0.

The notebook names output coordinates `y_top`/`y_bottom` in tables and
`top_px`/`bottom_px` in saved records. Its composition IDs refer to legend entries.
The adapter returns the actual harness schema: `top_y`, `bottom_y`, `lithology`
and integer `cluster_id`. It does not pass notebook records to the evaluator.

## S19 inspection

The read-only source strategy `bakeoff/strategies/s19_combine.py` imports
`ml.combine.runtime.run`. The runtime loads a selected config, boundary model,
pair matcher, optional second model pair, v3 legend parser and OCR assets.
Its final `run` adds the window offset to local `top_y`/`bottom_y` and optionally
adds calibrated depths. The runtime also writes and reuses OCR caches in its
source tree. A direct import would not meet the read-only source rule.

S19 is not included in this change. It needs a separate asset and training
history review, checksum-pinned files, a private cache location and a bounded
reader adapter. No S19 model files or OCR caches were copied or uploaded.
The notebook rule is the smaller candidate that can run now without training.
The notebook states that its naming choices were tuned on two exposed pages.
Its inherited independence is unknown. These results are not held-out evidence.

## Adapter-only smoke check (before integration)

The existing `demo-000` fixture is synthetic. Its image is 256 × 600 pixels.
The scored window has 420 rows. The check used a 20-second worker timeout,
512 MiB sampled RSS limit and no previews. It did not generate new data.

The unassisted candidate completed one case with no errors. It had full
coverage, boundary F1 0.285714, cluster ARI 0.441237, unknown rate 1.0,
row accuracy 0, macro F1 0 and no depth pairs. Worker time was 0.930761 seconds;
peak sampled RSS was 85.339844 MiB. The run is
`runs/notebook-smoke/20261108T051538Z-5911e1997a`.

The separate supplied-legend diagnostic completed one case with no errors.
Row accuracy was 0.873810, macro F1 0.888849, boundary F1 0.6,
cluster ARI 0.641923 and score 0.524286. Coverage was 1.0 and depth was absent.
Worker time was 1.165694 seconds; peak sampled RSS was 84.972656 MiB.
The run is `runs/notebook-assisted/20261108T051631Z-76d06af8f1`.
This mode has extra public assistance. Do not compare it to the full-image run.

A fresh full-image baseline run is
`runs/notebook-paired/20261108T051628Z-372d1d6da5`.
The existing compare command accepted its dataset, scorer, mode and case
fingerprints against the unassisted candidate. Dataset SHA-256:
`158248bcfb8e3ce8eb0fd6105fca56241167d3e69ff78c0254efb85eb329a934`.
The saved comparison is `runs/notebook-comparison.json`.
This is a one-case diagnostic; it does not establish an overall model winner.
No older baseline was assumed to match. For another dataset, scorer, mode or
case selection, make a fresh paired run.

## Full-scan limits

Real full-scan inference: **not run**. Full-scan readiness is not established.
Only the small existing demo and targeted contract fixtures were measured.

The default limits are 12 million cropped pixels, 512 million decode bytes,
10,000 feature stripes and 256 component centres per stripe for pair distances.
At stripe height 8, the stripe limit permits at most 80,000 rows. The pixel
limit can be reached first. The source notebook permits 2,000 component centres;
this config lowers that cap to bound pair-distance memory. Dense noisy stripes
can therefore have different feature values. The source-parity test covers
ordinary stripes below the cap.

Tests confirm that over-budget windows raise an error. There is no automatic
rescale, tile fallback, alternate model or partial-success output. The existing
reader can decode TIFF strips when supported. Its Pillow fallback can decode
the full image and is subject to its decode guard. RSS is sampled and can miss
short peaks. Do not infer tall-scan memory use from the small demo measurement.

The source rule uses eight-pixel stripes and robust change peaks. Small beds
can be missed. Pixels below mean RGB 128 count as ink, which can hide colour
boundaries between dark beds. There is no automatic legend OCR. A full-image
production candidate still needs naming, depth and real-scan resource evidence.

## Reproduce with the existing CLI

Run from the repository root with the package installed, or set `PYTHONPATH=src`.
No new CLI, scorer, exporter or baseline service is needed.

```sh
python -X utf8 -m pytest -q tests/test_notebook_ink_adapter.py tests/test_models.py
python -X utf8 -m viridian402 evaluate --manifest examples/demo/manifest.json --model viridian402.notebook_ink:predict --config configs/notebook-ink.json --case-id demo-000 --timeout 20 --max-worker-rss-mb 512 --no-previews --output runs/notebook-smoke
python -X utf8 -m viridian402 evaluate --manifest examples/demo/manifest.json --model builtin:baseline --config configs/baseline.json --case-id demo-000 --timeout 20 --max-worker-rss-mb 512 --no-previews --output runs/notebook-paired
python -X utf8 -m viridian402 evaluate --manifest examples/demo/manifest.json --model viridian402.notebook_ink:predict --config configs/notebook-ink.json --case-id demo-000 --mode supplied_legend --timeout 20 --max-worker-rss-mb 512 --no-previews --output runs/notebook-assisted
python -X utf8 -m viridian402 compare runs/notebook-paired/20261108T051628Z-372d1d6da5 runs/notebook-smoke/20261108T051538Z-5911e1997a --output runs/notebook-comparison.json
```

Use the new run IDs printed by `evaluate` when repeating the comparison.
The focused command above reported `12 passed`. It covers the adapter and the
existing baseline model tests. `git diff --check` reported no whitespace errors.

Private real-data fetch, real full-scan evaluation and the whole test suite were
not run for the adapter-only commit. The integration PR records its own checks.

## Candidate selection and final unseen test

Select candidates only from paired runs with identical dataset, scorer, mode
and case fingerprints. Use the existing compare command. Keep the split and
metric version identical too. A supplied-legend run cannot select a winner
against an unassisted run. Keep failures and resource limits in the comparison.

The five reviewed windows are exposed training/development evidence. They are
not fresh verification. The local real-window bundle is unlabelled. Its parent
train/dev assignment conflicts, and parent catalog validation has not run.
Keep it local. Resolve the parent assignment and validate the combined catalog
before any release review. Do not change splits or accepted labels to pass a
check. See [window exchange](review-window-exchange.md).

A final unseen test needs a named custodian and isolated truth. The custodian
must keep its images, labels and results out of candidate tuning. Freeze the
candidate, config, data and scorer fingerprints before that test. No final
unseen test was run in this integration. These are experiment review rules;
they do not state official hackathon rules or scoring.

Keep render failures in the visual report. Missing required images prevent a
promotion claim. Complete images alone do not establish a quality gain. See
[visual evidence](visual-evidence.md).
