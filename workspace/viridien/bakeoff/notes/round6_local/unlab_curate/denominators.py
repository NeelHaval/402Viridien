"""Failure and denominator appendix for the unlabelled TEST runs (no inference, no new formula).

Usage: python -I denominators.py <lane dir>
For clean and medium runs it reads the existing corrected per-case exports.
Raw runs and their derived files are never changed.
Writes FAILURE_AUDIT.md and failure_audit.json in the lane. REPORT.md averages skip missing
values; this file states, for every average, how many pages it is taken over.
"""
import collections
import importlib.util
import json
import math
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location('export_corrected', Path(__file__).with_name('export_corrected.py'))
export_corrected = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_corrected)

AVERAGED = ('assigned_share', 'depth_fit_ok', 'legend_entries_found', 'legend_entries_named',
            'legend_named_share', 'short_bed_rate', 'beds_per_1000px', 'oversegmentation_rate')


def present(v):  # same rule as report_results.mean(): finite numbers (bools count) only
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def summarise(run):
    stored = json.loads((run / 'derived/corrected_records.json').read_text(encoding='utf-8'))
    provenance, cases = stored['provenance'], stored['records']
    assert provenance['results_json_sha256'] == export_corrected.sha(run / 'results.json')
    assert provenance['unlabelled_report_json_sha256'] == export_corrected.sha(run / 'unlabelled_report.json')
    metrics = json.loads((run / 'unlabelled_report.json').read_text(encoding='utf-8'))['metrics']
    by_key = {export_corrected.key(m): m for m in metrics}
    errors = collections.Counter((c['strategy'], c['error']) for c in cases if c['execution_failure'])
    rows = []
    for family in ['All'] + sorted({c['family'] for c in cases}):
        for strategy in sorted({c['strategy'] for c in cases}):
            g = [c for c in cases if c['strategy'] == strategy and (family == 'All' or c['family'] == family)]
            m = [by_key[(c['strategy'], c['log'], c['y0'], c['y1'])] for c in g]
            failed = [c for c in g if c['execution_failure']]
            rows.append(dict(family=family, strategy=strategy, attempted=len(g), execution_failures=len(failed),
                             failure_classes=dict(collections.Counter(c['error_category'] for c in failed)),
                             completed=len(g) - len(failed),
                             completed_empty=sum(not c['execution_failure'] and c['empty_prediction'] for c in g),
                             scorable_non_empty=sum(c['scorable'] for c in g),
                             depth_available=sum(c['depth_available'] for c in g),
                             legend_available=sum(c['legend_available'] for c in g),
                             failure_rate_denominator=len(g),
                             averaged_metric_denominators={**{k: len(m) if k == 'depth_fit_ok' else sum(present(x.get(k)) for x in m) for k in AVERAGED},
                                 'depth_fit_ok_available': sum(x.get('depth_ft_per_px') is not None for x in m)}))
    return dict(run=run.name, records=len(cases), export=provenance,
                errors={f'{s}|{e}': n for (s, e), n in errors.most_common()}, rows=rows)


def backup(path):
    target = path.with_name(path.name + '.bak_review')
    if path.exists() and not target.exists(): target.write_bytes(path.read_bytes())


def main(lane):
    runs = [lane / 'runs' / n for n in ('clean', 'medium') if (lane / 'runs' / n / 'results.json').exists()]
    out = dict(note='Every attempted page stays in failure_rate. Other averages skip missing values; '
                    'their page counts are in averaged_metric_denominators. Numeric depth values are unavailable when calibration fails. '
                    'Depth check % (all) counts a missing fit as a failed check; Depth check % (available) uses only pages with a fit. '
                    'Other means omit missing values. Failure flags and categories '
                    'come from each run\'s derived/corrected_records.json.',
               runs=[summarise(r) for r in runs])
    backup(lane / 'failure_audit.json')
    backup(lane / 'FAILURE_AUDIT.md')
    (lane / 'failure_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    md = ['# Failures and denominators (read with REPORT.md)', '', out['note'], '']
    for run in out['runs']:
        md += [f"## {run['run']} run: {run['records']} cases "
               f"(export sha {run['export']['results_json_sha256'][:12]}..., "
               f"{run['export']['embedded_flags_corrected']} embedded flags corrected)", '',
               '| Family | Strategy | Attempted | Exec. failures (class) | Completed | Empty | Scorable | Depth available | Averages taken over (assigned / short-bed / overseg.) |',
               '|---|---|---|---|---|---|---|---|---|']
        for r in run['rows']:
            d = r['averaged_metric_denominators']
            cls = ', '.join(f'{k} {v}' for k, v in r['failure_classes'].items()) or '-'
            md.append(f"| {r['family']} | {r['strategy']} | {r['attempted']} | {r['execution_failures']} ({cls}) | "
                      f"{r['completed']} | {r['completed_empty']} | {r['scorable_non_empty']} | {r['depth_available']} "
                      f"(else unavailable) | {d['assigned_share']} / {d['short_bed_rate']} / {d['oversegmentation_rate']} |")
        md += ['', 'Recorded errors:', ''] + [f'- {n} x {k}' for k, n in run['errors'].items()] + ['']
    (lane / 'FAILURE_AUDIT.md').write_text('\n'.join(md), encoding='utf-8')
    return out


if __name__ == '__main__':
    lane = Path(sys.argv[1])
    print('wrote', lane / 'FAILURE_AUDIT.md', [r['run'] for r in main(lane)['runs']])
