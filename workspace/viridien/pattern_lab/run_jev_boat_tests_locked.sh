#!/bin/bash
set -u
umask 077
cd "$(dirname "$0")"
printf 'queued\n' > test-checks.state
(
  flock -w 1800 9 || exit 75
  printf 'started\n' > test-checks.state
  timeout --signal=TERM --kill-after=10 300s taskset -c 0,1 env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$PWD/home" TMPDIR="$PWD/tmp" "$PWD/venv/bin/python" -m pip install --disable-pip-version-check pytest==8.4.1 pluggy==1.6.0 iniconfig==2.3.0 pygments==2.19.2 packaging==26.3
  rc=$?
  if [ "$rc" = 0 ]; then
    timeout --signal=TERM --kill-after=10 300s env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$PWD/home" TMPDIR="$PWD/tmp" "$PWD/venv/bin/python" -X utf8 "$PWD/run_tests.py"
    rc=$?
  fi
  printf '%s\n' "$rc" > test-checks.exit
  if [ "$rc" = 0 ]; then printf 'completed\n'; else printf 'failed\n'; fi > test-checks.state
  exit "$rc"
) 9>/home/user/viridian402-jobs/v4-cpu-worker.lock
