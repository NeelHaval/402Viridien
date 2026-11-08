"""Check the documented legacy runner on a PNG page."""
from pathlib import Path
import subprocess
import sys

def test_run_accepts_png(tmp_path):
    sample = Path('tests/assets/final-sample')
    result = subprocess.run([sys.executable, '-m', 'pipeline.run', '--images', str(sample/'images'), '--boxes', str(sample/'bbox_annotation'), '--out', str(tmp_path)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path/'sample.csv').is_file()
    assert (tmp_path/'sample.png').is_file()
