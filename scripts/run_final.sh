#!/bin/sh
set -eu
if [ "$#" -ne 3 ]; then
    echo 'Usage: run_final.sh <images_dir> <boxes_dir> <out_dir>' >&2
    exit 2
fi
cd /srv/viridien/verify/repo
exec /srv/viridien/verify/venv/bin/python -m pipeline.final \
    --images "$1" --boxes "$2" --out "$3" --workers "${FINAL_WORKERS:-2}" \
    --timeout "${FINAL_TIMEOUT:-600}"
