"""Run the relevant strategy, grading, report and shared harness checks once."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--without-harness', action='store_true',
                    help='Keep the earlier full harness result and check the later local changes')
args = parser.parse_args()
tests = ['harness', 'strategies/test_scoreboard_v4.py',
         'strategies/test_s21_v4dino.py', 'strategies/test_s21_fusion.py',
         'strategies/test_s23_identity.py', 'strategies/test_grading_policy.py',
         'strategies/test_legend_identity.py', 'strategies/test_challenge_export.py']
tests += [str(path.relative_to(BAKEOFF)) for path in sorted(HERE.glob('test_*.py'))]
if args.without_harness:
    tests.remove('harness')
prefix = 'focused_final_checks' if args.without_harness else 'final_checks'
command = [sys.executable, '-X', 'utf8', '-B', '-m', 'pytest', *tests, '-q', '--tb=short',
           '-o', 'cache_dir=out/scoreboard_pytest_cache']
env = dict(os.environ, PYTHONPATH=str(BAKEOFF), OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
start = datetime.now(timezone.utc).isoformat()
with (HERE / (prefix + '.log')).open('w', encoding='utf-8') as stream:
    try:
        result = subprocess.run(command, cwd=BAKEOFF, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=300, check=False)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
        stream.write('\nCheck process exceeded the 300-second limit.\n')
receipt = dict(started_utc=start, ended_utc=datetime.now(timezone.utc).isoformat(),
               command=command, cwd=str(BAKEOFF), exit_code=code,
               log=prefix + '.log', timeout_seconds=300)
(HERE / (prefix + '_exit.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print('Checks exited', code, '; see', prefix + '.log')
raise SystemExit(code)
