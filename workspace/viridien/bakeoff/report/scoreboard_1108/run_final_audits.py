"""Run the two final saved-result audits in a finite process."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
env = dict(os.environ, PYTHONPATH=str(BAKEOFF), OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
results = []
for name in ('audit_saved_predictions.py', 'audit_run_integrity.py'):
    command = [sys.executable, '-X', 'utf8', '-B', str(HERE / name)]
    started = datetime.now(timezone.utc).isoformat()
    with (HERE / (name.removesuffix('.py') + '.log')).open('w', encoding='utf-8') as log:
        try:
            process = subprocess.run(command, cwd=BAKEOFF, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=300, check=False)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write('\nAudit reached the 300-second limit.\n')
    results.append(dict(command=command, started_utc=started,
                        ended_utc=datetime.now(timezone.utc).isoformat(), exit_code=code))
    (HERE / 'final_audit_commands.json').write_text(
        json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(name, 'exit', code, flush=True)
    if code:
        raise SystemExit(code)
