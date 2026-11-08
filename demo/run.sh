#!/usr/bin/env bash
set -euo pipefail
demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$demo_dir/frontend"
npm run build
cd "$demo_dir/backend"
exec .venv/bin/uvicorn main:app --reload
