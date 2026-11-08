# Analysis: research pipeline -> served app

Written before any code change. Sections: A pipeline, B demo audit, C target structure and API changes.

## A. Research pipeline (`notebooks/main_pipeline.ipynb`, 52 cells)

**Entry point.** None: a notebook driven by globals (`STEM`, `ROOT`, `GT_DIR`). The production path is Sections 1-4 plus Section 6. Section 5 (evaluators, plots) needs ground-truth labels and is not part of serving.

**Stages, in order** (all coordinates are pixel rows of the full scan unless noted):
1. *Load column* (`load_column`, `crop_rgb`): scan + bbox JSON -> RGB array of `interpreted_lithology_column` and its `y_offset`. Needs the bbox JSON.
2. *Stripe features* (`trim_borders`, `stripe_features`, `column_features`): RGB array -> `(n_stripes, 7)` float32, z-scored. Stripes are `H=8` px; features are ink density, component count, mean component distance, mean size, background RGB.
3. *Boundary rule* (`boundary_score`, `ink_boundaries`, `to_table`): features -> peaks (`WINDOW=3`, `MIN_GAP=2`, `INK_K=2.0`, median + k*MAD) -> beds table `y_top, y_bottom`. Not learned.
4. *Legend* (`detect_swatches`, `read_swatch_labels`, `load_legend`): scan + `lithology_legend` bbox -> table `legend_id, lithology, box`. Swatches by morphology; names by OCR of the text line right of each swatch.
5. *Naming* (`prototypes`, `split_mixed`, `name_segments`, `merge_named`): beds + legend -> named layers `y_top, y_bottom, legend_id, lithology`. Colour+texture descriptor distance to each swatch; splits mixed segments (16 px windows, run >= 3); blank = `legend_id -1`; neighbours with the same id merge.
6. *Depth + CSV* (`calibrate_depth`, `fit_depth_line`, `write_submission_csv`): OCR of the depth column, RANSAC line `depth = a*y + b`, then layers -> CSV `top_y, bottom_y, cluster_id, lithology, top_depth, base_depth`. Unnamed layers become cluster `Unknown`.

**Models.** There is no trained network in the served path. Boundaries are a fixed rule (`boundary_net_f7_h8.pt` exists in `data/predictions/` but is unused). The only learned model is the OCR engine: EasyOCR in the notebook (`torch`, weights downloaded to `predictions/ocr_models`). The demo's `tools/ocr.py` already wraps RapidOCR (ONNX, bundled weights) with a Tesseract fallback. **Decision:** the served pipeline uses the `tools` OCR engine for legend names and depth, so `torch`/`easyocr` are not needed. The "model" that `model.py` loads and warms up is that engine.

**Global state.** `PAGES` cache, `read_swatch_labels.reader`, `_reader`, `torch.manual_seed(0)`, `torch.set_num_threads`, `Image.MAX_IMAGE_PIXELS=None`, monkey-patched `page_paths` in Section 5. **Paths assumed:** `data/dataset/images|bbox_annotation`, `data/test` (labels), `data/predictions`; OCR weights under `predictions/ocr_models`. **Hard-coded:** `STEM`, all thresholds (H, WINDOW, RUN_BINS, W_COLOR/W_INK/W_TEXTURE, STRIP=1800, INLIER_PX=12, MIN_INLIERS=5). **Non-determinism:** only RANSAC (`default_rng(0)`, fixed) and a 2000-component subsample (`default_rng(0)`); OCR is deterministic on CPU. The stages are deterministic.

**Failure points.** Missing bbox JSON (columns and legend need it); empty column box; no swatches found (all layers become `Unknown`); OCR engine missing (names `legend_N`, depth missing); fewer than 5 consistent depth labels (no calibration, empty depths); `legend` OCR only used for names, so a wrong swatch box silently mis-names beds. The labelled-legend shortcut (`load_legend(..., "labels")`) reads `data/test` ground truth and must never be used when serving.

**Gap between research and the demo tools.** `tools.preprocess` (layer 1) already does what stages 1 and 6 do without a bbox (column detection, depth calibration, header location). Stage 1 therefore takes its crop and `y_offset` from `Layer1Result`, and stage 6 takes the calibration from it. Layer 1 does not read the `lithology_legend` box from the bbox JSON, so I add one optional field, `legend_box`, to `Layer1Result` (additive; no behaviour change). Without a bbox there is no legend: layers are named `Unknown` and a warning `legend_unavailable` is returned. Nothing is invented.

**Timing.** Not measured yet; filled in section D after the first real run.

## B. Demo audit

Dummy / random / seeded / hard-coded data, all removal targets:
- `pipeline.py`: `time.sleep(4)`, `np.random` beds, lithologies, lat/lon, name. -> deleted, replaced by the real pipeline.
- `seed.py`: ten invented wells with fixed coordinates, patched dataframes, `np.random.seed`. -> deleted.
- `main.py`: `seed_wells()` in lifespan; provisional upload marker `lat=26.5, lon=-90`; `BackgroundTasks` (no queue); JSON-file registry in `storage.py`. -> replaced.
- `scoring.py`: `water_depth_m()` is a made-up function of lat/lon, and `RULES['water_lat'|'water_lon'|'water_range'|'water_weights']`; `nearest` defaults to 50 km when there are no neighbours (a made-up number); scores hard-code lithology names `Halite`, `Shale grey`, `Sandstone`... that real OCR names will not match.
- `column.py`: `COLORS[...]` raises `KeyError` for any lithology outside eight names.
- `frontend/src/App.tsx`: optimistic upload marker `{name: "Uploading well", lat: 26.5, lon: -90}`. `Map.tsx` centre/zoom are viewport defaults, not data (kept). `types.ts` has no fixtures.
- Fine, not data: `tools/preprocess.py` `ransac_depth(seed=0)` and `scripts/bench_tools.py` (synthetic benchmark scan, a dev script; kept, renamed hits explained in the final report), tests that build tiny synthetic images in code.
- `adapter.py` calls `from pipeline import run_pipeline`; its job moves to `services/ingest.py`, so it is deleted.
- Existing tests: `test_tools`, `test_validation`, `test_scoring` (direction checks only).

## C. Target structure and API changes

Structure as in the brief (`app/{main,config}.py`, `api/`, `pipeline/{stages,model,types}`, `services/`, `store/`, `tools/`, `validation/`, `scripts/`, `data/`, `tests/`). Stages under `pipeline/stages/`: `load_column`, `features`, `boundaries`, `legend`, `naming`, `depth` (six, same order as the notebook; `depth` also writes the six-column frame). Imports of `tools`/`validation` become `app.tools`/`app.validation`.

Decisions that go beyond the brief (small, each needed by it):
- DB: extra columns `file_hash` (dedupe by hash), `warnings_json`, `timings_json` (for `warnings`, per-stage timings). Well id = first 16 hex of the file hash, so re-uploading the same scan returns the existing well (failed ones are re-queued).
- `scores_json` stores `{values, logged_ft, flags}`; `logged_ft` is the aggregation weight, `flags` feed `data_quality`.
- Scoring works on `validation.config.coarse_class` (mud, evaporite, sand_silt) rather than eight literal names; `Halite/Anhydrite/Shale grey/Sandstone/Siltstone` map as before, `Marl` now counts as mud.
- Scoring: no bathymetry -> cost uses depth + distance terms renormalised, flag `water_depth_unavailable`; no neighbours -> distance term dropped; no location -> `cost` and `price` are `null`.
- Column colours come from the coarse class when the name is unknown.
- API paths stay mounted at `/` and `/api` (the Vite proxy strips `/api`). Frontend `Well` gains `located`, `validation_confidence`, `status` adds `queued`; `Report` gains `data_quality`.
- Missing data I cannot supply: `bathymetry_gulf.csv`, `macrostrat_grid.csv`, `boem_wells.csv` (only `transitions.json` is in the repo; the fetch script needs network and a manual GEBCO download).
