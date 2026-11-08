"""Check saved failure exports and reports. No run file is changed.

Usage: python -I check_consistency.py LANE A_failure_audit.json [clean|medium]
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

lane, audit_path = Path(sys.argv[1]), Path(sys.argv[2])
run_name = sys.argv[3] if len(sys.argv) > 3 else 'clean'
run = lane / 'runs' / run_name
spec = importlib.util.spec_from_file_location('denominators', Path(__file__).with_name('denominators.py'))
den = importlib.util.module_from_spec(spec)
spec.loader.exec_module(den)
checks, info = [], []


def check(name, ok, detail=''):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def key(row):
    return row['strategy'], row['log'], int(float(row['y0'])), int(float(row['y1']))


# Independent flag interpretation; do not call the export's flag helper.
def failed(row):
    return row.get('failure') is True or str(row.get('failure')).casefold() == 'true'


before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
raw = read(run / 'results.json')['results']
exp = read(run / 'derived/corrected_records.json')
cases = exp['records']
raw_by = {key(r): r for r in raw}
case_by = {key(c): c for c in cases}
check('unique cases and exact raw/export join', len(raw_by) == len(raw) == len(case_by) == len(cases) and set(raw_by) == set(case_by))
check('export raw hash matches', exp['provenance']['results_json_sha256'] == before[run / 'results.json'])
check('export report hash matches', exp['provenance']['unlabelled_report_json_sha256'] == before[run / 'unlabelled_report.json'])
check('flags agree with independent raw recomputation', set(raw_by) == set(case_by) and all(
      c['execution_failure'] == failed(raw_by[k]) for k, c in case_by.items()))
info.append(dict(name='failure count', detail=sum(failed(r) for r in raw)))
info.append(dict(name='categories', detail={cat: sum(c['error_category'] == cat for c in cases)
                                          for cat in sorted({c['error_category'] for c in cases})}))
# Independent aggregation from raw flags plus saved metric values.
metrics = {key(m): m for m in read(run / 'unlabelled_report.json')['metrics']}
check('report case join', set(metrics) == set(raw_by))
rows = {}
for strategy in sorted({r['strategy'] for r in raw}):
    group = [r for r in raw if r['strategy'] == strategy]
    ms = [metrics[key(r)] for r in group]
    rows[strategy] = dict(attempted=len(group), execution_failures=sum(failed(r) for r in group),
        completed=sum(not failed(r) for r in group),
        completed_empty=sum(not failed(r) and bool(m.get('empty_prediction')) for r, m in zip(group, ms)),
        scorable_non_empty=sum(not failed(r) and not bool(m.get('empty_prediction')) for r, m in zip(group, ms)),
        depth_available=sum(m.get('depth_ft_per_px') is not None for m in ms))
if run_name == 'clean':
    check('independent clean audit present', audit_path.is_file(), str(audit_path))
    if audit_path.is_file():
        audit = read(audit_path)
        check('independent audit raw hash', audit['results_sha256'] == before[run / 'results.json'])
        for strategy, v in audit['report_denominators'].items():
            want = dict(attempted=v['attempted'], execution_failures=v['execution_failures'],
                        completed=v['completed_without_exception'], scorable_non_empty=v['scorable_non_empty'],
                        depth_available=v['depth_available']['depth_ft_per_px'])
            check('independent clean audit ' + strategy, all(rows.get(strategy, {}).get(k) == n for k, n in want.items()))
        check('independent failure total', sum(failed(r) for r in raw) == audit['failures_total'])
summary = read(lane / 'failure_audit.json')
stored = {r['strategy']: r for x in summary['runs'] if x['run'] == run_name for r in x['rows'] if r['family'] == 'All'}
for strategy, row in rows.items():
    check('failure_audit counts ' + strategy, all(stored.get(strategy, {}).get(k) == n for k, n in row.items()))
rs = lane / 'result_summary.json'
if rs.is_file():
    for t in read(rs)['totals']:
        if t['family'] != 'All': continue
        d = t.get('denominators', {}).get(run_name, {})
        check('result_summary counts ' + t['strategy'], all(d.get(k) == v for k, v in rows[t['strategy']].items()))
    html = lane / 'comparison.html'
    if html.is_file(): check('comparison contains denominators', '"denominators"' in html.read_text(encoding='utf-8'))
after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
check('all run files unchanged', before == after)
ok = all(c['ok'] for c in checks)
for item in info: print('INFO', item['name'], item['detail'])
for c in checks: print('PASS' if c['ok'] else 'FAIL', c['check'], c['detail'])
print('SUMMARY', 'PASS' if ok else 'FAIL', f"{sum(c['ok'] for c in checks)}/{len(checks)} checks")
target = lane / f'consistency_{run_name}.json'
den.backup(target)
target.write_text(json.dumps(dict(ok=ok, checks=checks, info=info), indent=1), encoding='utf-8')
sys.exit(0 if ok else 1)
