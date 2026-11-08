"""Check fallback reads and review-round selection without changing fleet data."""
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('board', HERE / 'board.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
b = m.Board()
original = b.lab
b.jobs_at = time.time()
with patch.object(m, 'urlopen', side_effect=OSError('Lab restarting')), patch.object(m.time, 'sleep', side_effect=InterruptedError):
    try:
        b.poll()
    except InterruptedError:
        pass
assert b.lab is original and not b.lab_ok
print('PASS: last good lab snapshot retained on outage')
s = json.loads((HERE / 'checked-state.json').read_text(encoding='utf-8'))
e = next(e for e in s['entries'] if e['real'])
q = parse_qs(urlparse(e['real'][0]).query)
box = [int(v) for v in q['box'][0].split(',')]
with patch.object(m, 'urlopen', side_effect=OSError('Lab restarting')):
    blob = b.crop(q['doc'][0], box)
assert blob.startswith(b'\x89PNG\r\n\x1a\n')
print('PASS: real source crop served locally while lab is unavailable')

did = '16c9064b18b77b61'
fixture = {
    m.DATA / 'candidates' / (did + '.json'): dict(doc_id=did, name='Test', entries=[dict(legend_id=1, label='Shale')]),
    m.PROPOSALS / did / 'legend-1.json': dict(doc_id=did, legend_id=1, status='made', round=2),
    m.PROPOSALS / did / 'legend-review-r1.json': {'1': dict(verdict='confident', note='Old approval')},
    m.PROPOSALS / did / 'legend-review-r2.json': {'1': dict(verdict='reject', note='Current round rejection')},
}
b.lab = {}
b.jobs = {}
with patch.object(m, 'read_dict', side_effect=lambda p: fixture.get(p, {})):
    result = b.build()
assert len(result['entries']) == 1
assert result['entries'][0]['state'] == 'Rejected'
assert result['entries'][0]['note'] == 'Current round rejection'
print('PASS: proposal uses its own review round; old approval does not leak forward')
print('PASS: resilience checks made no writes to fleet or lab data')
