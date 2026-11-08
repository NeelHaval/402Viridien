"""Corrected per-case export for an unlabelled TEST run. Raw files are read, never changed.

Usage: python -I export_corrected.py <run dir>   (writes <run dir>/derived/)

Why: bench embeds a legend-only label-free measurement before depth calibration. When
calibration then raises, the record's `failure` is True but its embedded
`unlabelled_metrics.execution_failure` stays False. `unlabelled_report.json` already
recomputes the flag from `failure`; this export makes that canonical flag, the error
category and metric availability explicit per case, so every consumer reads one source.
No inference and no metric formula change.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

RULE = ('execution_failure = raw results.json record.failure (canonical, as recomputed by '
        'unlabelled_metrics.write_report); embedded unlabelled_metrics.execution_failure is ignored')
DEPTH = re.compile(r'depth|anchor|calibration', re.I)
FIELDS = ('strategy', 'log', 'family', 'source', 'well', 'y0', 'y1', 'execution_failure', 'error_category',
          'error', 'raw_embedded_execution_failure', 'embedded_flag_corrected', 'empty_prediction',
          'failure_rate', 'scorable', 'depth_available', 'depth_fit_ok', 'legend_available', 'prediction_csv')


def truthy(v):
    return v is True or str(v).lower() == 'true'


def error_category(rec):
    if not truthy(rec.get('failure')):
        return 'none'
    err = str(rec.get('error') or '')
    etype = err.split(':', 1)[0]
    if 'Timeout' in etype or 'timed out' in err.lower():
        return 'timeout'
    if etype == 'MemoryError' or 'BrokenProcessPool' in err or 'Killed' in err:
        return 'crash_or_resource'
    if etype == 'ValueError' and DEPTH.search(err):
        return 'depth_calibration_unavailable'
    if etype == 'FileNotFoundError':
        return 'missing_input_or_output'
    return 'other_exception'


def key(r):
    return (r['strategy'], r['log'], int(float(r['y0'])), int(float(r['y1'])))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(run):
    run = Path(run)
    raw = json.loads((run / 'results.json').read_text(encoding='utf-8'))['results']
    report = json.loads((run / 'unlabelled_report.json').read_text(encoding='utf-8'))['metrics']
    metrics = {key(m): m for m in report}
    records = {}
    for r in raw:
        k = key(r)
        assert k not in records, f'duplicate case {k}'
        m = metrics.get(k)
        assert m is not None, f'case missing from unlabelled_report.json: {k}'
        canonical = truthy(r.get('failure'))
        assert bool(m.get('execution_failure')) == canonical, f'report flag disagrees with record: {k}'
        emb = r.get('unlabelled_metrics') if isinstance(r.get('unlabelled_metrics'), dict) else {}
        raw_emb = emb.get('execution_failure') if emb else None
        records[k] = dict(
            strategy=r['strategy'], log=r['log'], family=r.get('family'), source=r.get('source'), well=r.get('well'),
            y0=k[2], y1=k[3], execution_failure=canonical, error_category=error_category(r),
            error=r.get('error') if canonical else None, raw_embedded_execution_failure=raw_emb,
            embedded_flag_corrected=(raw_emb is not None and bool(raw_emb) != canonical),
            empty_prediction=bool(m.get('empty_prediction')), failure_rate=m.get('failure_rate'),
            scorable=not canonical and not bool(m.get('empty_prediction')),
            depth_available=m.get('depth_ft_per_px') is not None, depth_fit_ok=m.get('depth_fit_ok'),
            legend_available=m.get('legend_entries_found') is not None,
            prediction_csv=r.get('csv'))
    assert set(records) == set(metrics), 'report has cases not in results.json'
    rows = [records[k] for k in sorted(records, key=str)]
    out = run / 'derived'
    out.mkdir(exist_ok=True)
    provenance = dict(rule=RULE, run=run.name, results_json_sha256=sha(run / 'results.json'),
                      unlabelled_report_json_sha256=sha(run / 'unlabelled_report.json'),
                      script_sha256=sha(__file__), cases=len(rows),
                      execution_failures=sum(r['execution_failure'] for r in rows),
                      embedded_flags_corrected=sum(r['embedded_flag_corrected'] for r in rows),
                      categories={c: sum(r['error_category'] == c for r in rows)
                                  for c in sorted({r['error_category'] for r in rows})})
    (out / 'corrected_records.json').write_text(json.dumps(dict(provenance=provenance, records=rows), indent=1),
                                                encoding='utf-8')
    with (out / 'corrected_records.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return provenance, rows


if __name__ == '__main__':
    prov, _ = export(sys.argv[1])
    print(json.dumps(prov, indent=1))
