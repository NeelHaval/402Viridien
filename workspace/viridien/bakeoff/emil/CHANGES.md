# Live notebook port

Source: **emil/drive/HTB2026.ipynb, 56 cells**; the obsolete notebook is not used.
`port_manifest.json` hashes the notebook and current Drive evaluators.
`notebook.diff` records source substitutions. AST extraction retains functions,
classes, decorators, constants and imports. Colab installs, Drive mount, examples
and automatic loops are omitted. All notebook function names are preserved.

## Changes

* ROOT defaults to bakeoff/emil; DATASET, LABEL_DIR, MODEL_PATH and RUN_DIR are
  configurable. DATASET honours LITHOLOGY_DATASET. The default model is
  **emil/drive/boundary_net.pt**, never the local retrain. Changing model resets
  the in-memory network. No input paths are written.
* `load_legend` retains the notebook's default `auto` GT fallback. Fair runs
  explicitly select **detect** (swatches + OCR); **labels** is oracle legend.
* The live prediction schema and `save_record` validation are preserved. All
  three evaluators score the **same final records**, as in the live notebook.
  Section-2 raw boundaries are saved separately for diagnosis.
* `build_train_pages` wraps cell 21 without training on import. It respects
  splits.json, excludes test JSON filenames and 0000000009*, and requires a
  column >=5000 rows. Both complete labelled pages are excluded. The reconstructed
  current selection is in validation.json. Saved notebook output says 39;
  historical checkpoint training-page metadata is unavailable.
* PIL crops use the shared strip decoder with unchanged geometry. `load_column`
  closes its image/decoder. No whole tall page is converted to RGB.
* Empty detected legends have explicit DataFrame columns. Blank-only matching
  gets infinite runner-up margin instead of IndexError. It produces no named
  lithology rows. Other computational functions are AST-checked against source.
* The live notebook calls **score_run but never defines it**. A local adapter
  runs the unchanged evaluators and returns per-page metrics, allowing
  `run_experiment` to finish. Its `naming` argument remains metadata only, as in
  the notebook. Algorithm, thresholds and prediction logic are not changed.
* Isolated-mode sklearn hits a broken Conda Arrow DLL. bootstrap.py loads only
  the installed working user Arrow wheel, without enabling general user-site
  imports. External readers use `python -I`; external_untrusted is never executed.

## Measurements

Main results use Emil's Drive checkpoint; every record includes its SHA256.
The completed earlier 2000-step retrain is a **secondary** check. It used NumPy
sampling seed 0 and torch seed 0; the notebook sets no torch initialization seed.
Its original selection and loss log remain in training.json/log.

Complete in-domain pages: Hadrian (680 layers), 1inch tvd mudlog (356 layers).
gc743 has two labelled layers and is reported separately, excluded from complete
aggregates. Its unlabelled rows are not reliable negative/background truth.

The prior conversion of 76 external pages / 89 windows is reused. Validation
checks exact truth intervals, legend IDs/names, converter coordinate transforms
and the matching dataset view. No external test data or converter is edited.
Per-window records avoid counting unsampled gaps as background. Full-page
in-domain inference uses full-column normalization/context; harness/outside
inference normalizes within each requested window.

s12 caches swatch detection/OCR by legend-pixel SHA256, relocating boxes by origin.
The harness wrapper maps OCR names to shared OCR labels at string similarity
>=0.72; cluster_id remains Emil's detected ID. Direct evaluator runs preserve raw
OCR names. True names do not enter fair prediction or mapping. oracle_emil is an
explicit control reading true legend boxes/names only. Gold log06 has no supplied
true legend, so oracle evaluation there is unavailable.

Direct runs bypass shared depth OCR equally for all methods. Direct s11 CSVs are
reused from the prior completed evaluation of the unchanged strategy; CSV-read
time is not inference time. Shared harness results separately retain depth
preprocessing failures, comparing with the existing s11 baseline.

S11's cluster IDs are independent pattern-cluster IDs, including Unknown, and
cannot be copied into composition.legend_id. Its comparison adapter uses the
evaluators' supported layer.lithology field and retains cluster_id separately.
Its detected physical legend is unchanged; records still pass save_record.

Wall seconds, process peak RSS (including children), CUDA peak reserved bytes and
tracemalloc peak are recorded. Traced full-page runs include tracing overhead;
normal shared-harness inference does not enable tracing. Cached OCR is warm.

Evaluators are copied **byte for byte from emil/drive/eval_cells**. Only CONFIG
is changed in memory, with all other lines checked for identity.
`IMAGE_COORDS=True` is a CONFIG change for full-image prediction boxes.
Segmentation PixAcc measures coverage, not boundaries. Classification PixAcc
includes background; harness row accuracy excludes GT blanks. ARI measures
partitions independently of names and can be 1 for an entirely wrong constant
prediction on a single-class window. These conventions remain unchanged.

## Re-run (workspace root)

```
python -I bakeoff/emil/port_notebook.py
python -I bakeoff/emil/validate.py
python -I -u bakeoff/emil/evaluate.py --domain viridien
python -I -u bakeoff/emil/run_all.py
python -I bakeoff/emil/live_report.py
python -m pytest bakeoff/harness bakeoff/external -q
```

prepare_external.py is retained for provenance, not rerun after passing validation.
Main outputs are in results_live; results/ retains the earlier retrain comparison.

## Linux VM continuation (2026-11-08)

* Dataset resolution is portable (`paths.py`): honour `LITHOLOGY_DATASET`, read
  `dataset_paths.json`, then try paths relative to the workspace. The VM config
  points at `/home/user/lith/dataset`. Archived Windows prediction-source paths
  are mapped to this mirror; the live notebook/evaluator source remains intact.
  `run_all.py` uses `sys.executable`, `pathlib`, and `subprocess`, with no shell
  commands. Python entry points that read external input retain `-I`.
* The only available outside cohort is **dev 59 windows / 54 pages**, **test 20
  windows / 16 pages**, **combined 79 windows / 70 pages** (9 wells). The earlier
  conversion was filtered to those available pages and validated again; it was
  not regenerated. `split_results.py` now evaluates only dev, test and dev_test.
  Removed-well rows and stale artifacts were retired by allowlist/filename,
  without opening their individual artifacts. No verify inference or aggregate
  is produced. `external_tests` and `kgs_extract.py` were not edited.
* Reused the live notebook, original Drive checkpoint SHA256
  `45273902c94e642e0bc5edac8d51295f9e7106372b6b2f2bb137a54f9be40079`,
  both complete in-domain runs, 79 main detect outputs, 74 external s11 direct
  outputs, and original shared s12-detect/s11 preprocessing outcomes. The same
  computational definitions are AST-checked against the live notebook; the
  s11/s12 strategy source was not changed.
* A first CPU replay differed from 12 archived oracle predictions, so **all 79
  labels/oracle windows were rerun on CPU**. Five empty s11 CSVs proved to be
  partial outputs of shared depth failures; reran those five with the unchanged
  strategy and fresh shared RapidOCR legend parsing. Direct scores now include
  successful lithology inference for those windows. Shared comparisons retain
  their original **3 dev / 2 test** depth-preprocessing failures.
* The shared oracle comparison applies recorded original preprocessing outcomes
  to replayed oracle predictions from the identical `predict()` used by
  `oracle_emil.run`. This is explicitly labelled a reconstruction of shared
  pixel scores; full depth OCR was not rerun. It has actual per-window CSVs and
  dev/test/combined aggregates. Original GPU/laptop timings remain provenance,
  separate from the new CPU measurements.
* `benchmark_vm.py` measured all 162 main s12 calls (two modes on 2 complete
  in-domain pages plus 79 outside windows) without tracemalloc, using six CPU
  torch threads and warm OCR caches. **162/162 final predictions match exactly**.
  `vm_resources.json` and report/page_resources.csv contain timings and process
  peak RSS. GPU allocation is zero. Outside per-page times sum sampled windows,
  and are not advertised as full original-page runtime.
* Reused the earlier completed 2000-step retrained checkpoint and loss history;
  reran detect/labels on both complete pages. It used 29 pages; the current live
  training rule selects 31. Both complete labelled pages are excluded from both
  selections. This remains a secondary check, not the main checkpoint.
* EasyOCR models are not installed on the VM and network DNS prevents downloads.
  OCR files now live in a writable workspace cache. `read_swatch_labels` can
  reuse archived raw OCR only when **both legend-pixel SHA256 and every detected
  box agree**. No GT legend supplies detect names. `port_notebook.py` reproduces
  these environment/cache adaptations. The real `run_experiment` smoke test
  passes with exact reference PixAcc. gc743 oracle is reported separately;
  its uncached detect attempt fails for missing EasyOCR weights/download DNS,
  and is reported as unavailable rather than a quality score.
* The unchanged team segmentation/classification evaluators reject empty final
  predictions. External detect rejects 59/59 dev and 13/20 test (72/79 total).
  Their accepted-only PixAcc and rejection counts are explicit; every harness
  metric includes every window. Undefined legend precision is N/A; it is
  conditional on pages with detections. Legend recall includes zero detections.
* Rebuilt the standalone white HTML report, JSON, Markdown, per-page resource
  CSV, split/source charts, diagnostic panels and six detected/true-legend
  overlays in `report/`. Direct results and shared-preprocessing results are
  separate. `audit_results.py` rescored all saved CSVs, verified the cohorts,
  checked evaluator identities, and validated checkpoint/replay provenance.
* Validation: data acceptance passes. Harness tests: **9 passed**. Full requested
  suite: **36 passed, 3 failed, 8 errors**; the original dev fixture
  `external_untrusted/kgs/386825.pdf` is absent, and an existing metric-PDF test
  exceeds the unchanged converter's 160-million-pixel render guard. The sample
  was not reconstructed and the protected converter/tests were not modified.
  See `acceptance.json`, `tests_live.log`, and `tests_harness_vm.log`.

### VM commands (workspace root)

```text
source ~/lith/.venv/bin/activate
export LITHOLOGY_DATASET=~/lith/dataset
export MPLCONFIGDIR=/tmp/emil_mpl
python -I bakeoff/emil/port_notebook.py
python -I bakeoff/emil/validate.py
python -I -u bakeoff/emil/evaluate.py --domain external --modes labels --refresh
python -I -u bakeoff/emil/evaluate.py --domain external --modes s11 --refresh --recompute-s11 --keys ext_utah_4304150011_p155 ext_kgs_ForsythB2-17_p020 ext_nl_flat-bay-3_p054 ext_nl_flat-bay-3_p055 ext_nl_storm_p071
python -I -u bakeoff/emil/evaluate.py --domain viridien --modes detect labels --tag retrained_live --model bakeoff/emil/boundary_net.pt
python -I bakeoff/emil/benchmark_vm.py
python -I bakeoff/emil/smoke_live.py
python -m pytest bakeoff/harness bakeoff/external -q
python -m pytest bakeoff/harness -q
python -I bakeoff/emil/audit_results.py
python -I bakeoff/emil/live_report.py
```

`python -I -u bakeoff/emil/run_all.py` resumes valid saved results and recreates
all current aggregates and the report on either platform. Windows users set the
same environment variables using their shell; Python scripts contain no new
Windows-specific paths or shell-specific calls. Main outputs: `results_live/`;
secondary comparison: `results_retrained_live/`; explicit reuse manifest:
`vm_resume.json`. Historical laptop logs are preserved as provenance, not rerun
commands. Final active metadata uses mapped paths.

Final report verification: 7 HTML tables, 66 local links with no broken targets,
21 valid PNGs (4 comparison charts, 11 diagnostic panels, 6 legend overlays).
`report/artifact_check.json` records this check. Diagnostic selections are
explicitly non-random, while all tabulated means use the complete allowed cohort.
