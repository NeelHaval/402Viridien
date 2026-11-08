-include .env
export

PY := backend/.venv/bin/python

PYTHON ?= python3.12

.PHONY: setup run test run-backend run-frontend ingest

# one-time install of backend and frontend dependencies
setup:
	$(PYTHON) -m venv backend/.venv
	backend/.venv/bin/pip install -r backend/requirements.txt
	npm --prefix frontend ci

# build the frontend, then serve app and API together on http://localhost:8000
run:
	npm --prefix frontend run build
	cd backend && .venv/bin/uvicorn app.main:app --port 8000

test:
	cd backend && .venv/bin/python -m pytest
	npm --prefix frontend run typecheck
	npm --prefix frontend run build

run-backend:
	cd backend && .venv/bin/uvicorn app.main:app --port 8000

run-frontend:
	npm --prefix frontend run dev

ingest:
	@test -n "$(DATASET)" || (echo "usage: make ingest DATASET=path/to/dataset" && exit 2)
	cd backend && .venv/bin/python scripts/ingest_dataset.py "$(abspath $(DATASET))"
