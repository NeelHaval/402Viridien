# Stratum demo
One-time setup from the repository root:
1. `python3 -m venv demo/backend/.venv && demo/backend/.venv/bin/pip install -r demo/backend/requirements.txt`
2. `npm --prefix demo/frontend ci`
Run from the repository root: `bash demo/run.sh`
Open http://localhost:8000. FastAPI serves both the built frontend and `/api` routes.
The launcher rebuilds the frontend before starting; restart it after frontend edits.
Tests: `cd demo/backend && .venv/bin/python -m pytest`; first startup seeds wells in about 40 seconds.
Cache lives in `backend/storage/`; run one backend worker. Maps/fonts require internet.
Upload markers start at the Gulf centre until processing supplies their location; area weights use `total_depth_ft`.
