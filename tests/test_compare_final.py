"""Check reports for incomplete official exports."""
import json
from pathlib import Path
import subprocess
import sys

def test_empty_official_csv_writes_failure_report(tmp_path):
    official = tmp_path / 'official'
    actual = tmp_path / 'actual'
    official.mkdir()
    actual.mkdir()
    name = 'sample.notebook.csv'
    (official / name).write_text('', encoding='utf-8')
    (actual / name).write_text('', encoding='utf-8')
    report = tmp_path / 'nested ü path' / 'report.json'
    result = subprocess.run([sys.executable, 'scripts/compare_final.py', '--actual', str(actual), '--official', str(official), '--report', str(report)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    data = json.loads(report.read_text(encoding='utf-8'))
    assert data['match'] is False
    assert data['files'][0]['error'] == 'Official CSV has no header'
