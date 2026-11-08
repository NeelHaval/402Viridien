#!/bin/bash
set -eu
cd /home/user/viridian402-jobs/v4-cpu-20261108T063000Z-f785e5/benedek-newtest-1010
echo queued > queue.state
date -u +%FT%TZ > queued.utc
exec 9>/home/user/viridian402-jobs/v4-cpu-worker.lock
if ! flock -w 60 9; then echo queue_timeout > queue.state; exit 75; fi
echo running > queue.state
date -u +%FT%TZ > started.utc
CORES=$(python3 -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0))[:2])))')
ulimit -v 10485760
set +e
timeout -k 10s 300s taskset -c "$CORES" /home/user/lith/.venv/bin/python -u paired_worker.py > worker.log 2>&1
STATUS=$?
set -e
echo "$STATUS" > worker.exit
echo finished > queue.state
date -u +%FT%TZ > finished.utc
exit "$STATUS"
