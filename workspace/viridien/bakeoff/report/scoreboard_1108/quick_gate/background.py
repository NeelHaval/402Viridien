import json
from pathlib import Path
import subprocess
import sys
import time
folder = Path(__file__).resolve().parent
started = time.perf_counter()
command = [sys.executable, '-X', 'utf8', '-B', str(folder / 'run.py')]
with (folder / 'run.log').open('w', encoding='utf-8') as stream:
    try:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
receipt = {'command': command, 'timeout_seconds': 180, 'exit_code': code, 'wall_seconds': time.perf_counter() - started}
(folder / 'command_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
