# Stratum demo

FastAPI backend (`backend/app`) serving the lithology pipeline from `notebooks/main_pipeline.ipynb`, plus a React/MapLibre frontend. All data is live: wells come from the SQLite database, which is filled by uploads or by `make ingest`. There is no seed or sample data. An empty database shows an empty map.

## Setup (from `demo/`)
1. `make setup` (needs Python 3.12; 3.15 has no scipy wheels. Use `make setup PYTHON=/path/to/python3.12` if `python3.12` is not on your PATH).
2. `make run` builds the frontend and serves app and API together on http://localhost:8000.

For frontend development, run `make run-backend` and `make run-frontend` (http://localhost:5173, proxies `/api` to the backend).

Maps and fonts need internet. Run one backend worker: the job queue is in-process.

## Model weights and reference data
- The only learned model is the OCR engine (RapidOCR, ONNX). Its weights ship inside the `rapidocr-onnxruntime` package, so nothing to download; keep it at the pinned `==1.2.3` (the tools use its internal API). Tesseract is the fallback. If no engine loads, the app still starts, `/health` reports `model_loaded: false` with the reason, and uploads return 503.
- Reference data goes in `backend/data/reference/`: `transitions.json` (included), `bathymetry_gulf.csv`, `macrostrat_grid.csv`, `boem_wells.csv` (build with `backend/scripts/fetch_reference_data.py`, needs network; GEBCO bathymetry is a manual download). Without `bathymetry_gulf.csv` the cost score drops its water-depth term and reports `water_depth_unavailable`.
- Runtime data (`app.db`, `scans/`, `csv/`) lives in `backend/data/` and is git-ignored. `APP_DATA_DIR` moves it.

## Ingest the dataset
`make ingest DATASET=../data/dataset` (folder with `images/` and `bbox_annotation/`). Uses the same ingest service as `/upload`, skips scans already ingested (file hash), prints a summary. Takes roughly 30 s per scan. After changing scoring rules: `backend/.venv/bin/python backend/scripts/rescore.py`.

## Tests
`make test` runs pytest, the frontend typecheck and the build. `pytest -m model` runs only the real-OCR test (skipped when OCR or the dataset is missing). Tests stub the OCR model and layer 1 only in `backend/tests/conftest.py`.

For reproducible adversarial tests and offline browser probes, see
[the isolated stress guide](../stress_testing/README.md#test-the-current-backend-and-frontend).
Its separate Python 3.12 environment is pinned in
`stress_testing/requirements-demo.lock`. CI now checks the demo in addition to
the original evaluation package.

Duplicate uploads are idempotent only when their bbox input also matches;
changed bboxes return `409 input_conflict` and preserve the first job.
The queue defaults to 32 jobs including the running job (`APP_MAX_PENDING_JOBS`);
when full, uploads return `503 queue_busy` for retry later. These limits do not
provide a hard memory or whole-job deadline for native image/OCR processing.

## API (also under `/api`; errors are `{"error": {"code", "message"}}`)
- `GET /health` model, database and reference-data status
- `GET /wells[?status=]` wells for the map; `GET /wells/{id}` detail, column, validation; `GET /wells/{id}/csv` corrected CSV
- `POST /upload` multipart `file` + optional `bbox_json`; returns `{id, status}` at once (400/413/415/503 on rejection)
- `POST /report` `{polygon: [[lon, lat], ...]}` returns `{wells, scores, column, data_quality}`
