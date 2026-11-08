#!/bin/bash
set -u
umask 077
cd "$(dirname "$0")"
stage="$1"
case_id="${2:-checks}"
printf 'queued\n' > "$stage-$case_id.state"
(
  flock -w 1800 9 || exit 75
  printf 'started\n' > "$stage-$case_id.state"
  if [ "$stage" = report ]; then script=make_report.py; args="$case_id";
  elif [ "$stage" = compare ]; then script=compare.py; args="";
  elif [ "$stage" = test ]; then script=run_tests.py; args="";
  elif [ "$stage" = export ] || [ "$stage" = evaluate ]; then script=worker_v2.py; args="$stage $case_id";
  else exit 64; fi
  timeout --signal=TERM --kill-after=10 900s env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$PWD/home" TMPDIR="$PWD/tmp" "$PWD/venv/bin/python" -X utf8 "$PWD/$script" $args
  rc=$?
  printf '%s\n' "$rc" > "$stage-$case_id.exit"
  if [ "$rc" = 0 ]; then printf 'completed\n'; else printf 'failed\n'; fi > "$stage-$case_id.state"
  exit "$rc"
) 9>/home/user/viridian402-jobs/v4-cpu-worker.lock
