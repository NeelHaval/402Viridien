#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${FINAL_PYTHON:-python}
if [ "$#" -eq 2 ]; then
    exec "$PYTHON" -m pipeline.final --data "$1" --out "$2" --workers "${FINAL_WORKERS:-2}" --timeout "${FINAL_TIMEOUT:-600}"
elif [ "$#" -eq 3 ]; then
    exec "$PYTHON" -m pipeline.final --images "$1" --boxes "$2" --out "$3" --workers "${FINAL_WORKERS:-2}" --timeout "${FINAL_TIMEOUT:-600}"
fi
echo 'Usage: run_final.sh <data_root> <out> OR <images> <boxes> <out>' >&2
exit 2
