"""Run bounded metric, report and held-launch regression checks."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BAKE = HERE.parents[2]
env = dict(os.environ, PYTHONPATH=str(BAKE), OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
tests = ['strategies/test_grading_policy.py', 'strategies/test_grading_availability_v2.py',
         'report/scoreboard_1108/availability_v2',
         'report/scoreboard_1108/jev_comparison/test_registry_private.py',
         'report/scoreboard_1108/jev_comparison/test_continuation.py']
command = [sys.executable, '-X', 'utf8', '-B', '-m', 'pytest', '-q', *tests,
           '--tb=short', '-o', 'cache_dir=out/availability_v2_pytest_cache']
started = datetime.now(timezone.utc).isoformat()
with (HERE/'checks.log').open('w', encoding='utf-8') as log:
    try:
        result = subprocess.run(command, cwd=BAKE, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=180, check=False)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
        log.write('\nChecks reached the 180-second limit.\n')
receipt = dict(started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
               command=command, cwd=str(BAKE), exit_code=code, timeout_seconds=180,
               inference_or_API_started=False)
(HERE/'checks_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print('Availability checks exit', code, '; see checks.log')
raise SystemExit(code)
