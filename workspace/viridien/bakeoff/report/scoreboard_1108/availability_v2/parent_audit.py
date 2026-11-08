"""Check derived availability against preserved records without new inference."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import itertools
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
BAKE = BASE.parents[1]
METRICS = ('score', 'row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft')
LITHOLOGY = {'score', 'row_acc', 'cluster_ari'}
INVALID = {'6fb4e860a31a7c9b', 'dc6d9c7cccfe7e79'}
SILVER = 'unreviewed-two-source-A-tie-break'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def same(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a-b) < 1e-11

def overall(rows):
    order = {'human_qa': 0, 'human_gold': 1, 'viridien_eval': 2, 'ext_dev': 3}
    docs = defaultdict(list)
    for row in rows:
        if row['suite'] in order:
            docs[(row['strategy'], row['image_sha256'])].append(row)
    selected = []
    for values in docs.values():
        best = min(order[r['suite']] for r in values)
        chosen = [r for r in values if order[r['suite']] == best]
        name = min(r['log'] for r in chosen)
        selected += [dict(r, log=name) for r in chosen]
    return selected

def main():
    target = HERE / 'derived_scores.json'
    data = read(target)
    baseline = read(BASE / 'scores.json')
    receipt = read(BASE / 'baseline_v1_preservation_receipt.json')
    saved = {(r['case_id'], r['strategy']): r for r in baseline['records']}
    rows = data['records']
    errors = []
    checks = defaultdict(int)

    def check(condition, message):
        checks['assertions'] += 1
        if not condition:
            errors.append(message)

    preserved = {'scores_json_sha256': BASE/'scores.json',
                 'raw_results_json_sha256': BASE/'raw_results.json',
                 'shared_scorer_sha256': BAKE/'harness/score.py',
                 'preserved_html_sha256': BASE/'index_baseline_v1.html'}
    for key, path in preserved.items():
        check(sha(path) == receipt[key], 'Preserved baseline changed: '+key)
    for path, expected in data['pins'].items():
        check(sha(path) == expected, 'Pinned source/metadata changed: '+path)
    for path, expected in data['prediction_and_truth_sha256'].items():
        check(sha(path) == expected, 'Cached CSV bytes changed: '+path)
        checks['csv_hashes'] += 1
    keys = [(r['case_id'], r['strategy']) for r in rows]
    check(len(keys) == 272 and len(set(keys)) == 272 and set(keys) == set(saved),
          'Derived case matrix differs from the preserved baseline')
    check(data['selection_promoted'] is False and data['generalization_verified'] is False,
          'Derived result must not promote a champion or certify generalization')
    for row in rows:
        old = saved[(row['case_id'], row['strategy'])]
        for field in ('log', 'suite', 'family', 'window', 'truth_sha256', 'image_sha256', 'annotation_sha256'):
            check(row[field] == old[field], 'Case identity changed: '+field)
        st = row['structural_supervision']
        for view, series_set in row['views'].items():
            old_view = old if view == 'primary' else old['pixel_only']
            for policy in ('strict', 'owner_equivalent'):
                raw = {k: old_view.get(k) for k in METRICS} if policy == 'strict' else old_view['accepted_metrics']
                for metric in METRICS:
                    check(same(row['raw_metrics'][view][policy][metric], raw.get(metric)),
                          'Raw metric changed: '+row['case_id']+'/'+metric)
                for series in ('reference_diagnostic', 'qualified_development'):
                    for metric in METRICS:
                        eligible = not st['missing_truth'] and st['filled_rows'] > 0
                        if metric in LITHOLOGY and row['case_id'] in INVALID:
                            eligible = False
                        if series == 'qualified_development':
                            if row['truth_status'] == 'human-unqa' or row['truth_status'] == SILVER and metric in LITHOLOGY:
                                eligible = False
                        if metric == 'depth_mae_ft' and not st['finite_depth_boundaries']:
                            eligible = False
                        value = raw.get(metric)
                        if eligible and row['prediction_failed'][view] and metric != 'depth_mae_ft':
                            value = 0.0
                        expected = value if eligible else None
                        cell = series_set[series][policy][metric]
                        check(cell['eligible'] == eligible and cell['available'] == (expected is not None)
                              and same(cell['value'], expected),
                              'Availability/value mismatch: '+row['case_id']+'/'+series+'/'+metric)
                        if expected is None:
                            check(bool(cell['reasons']), 'Unavailable cell has no reason')
                        checks['metric_cells'] += 1
    cohorts = {s: [r for r in rows if r['suite'] == s] for s in {r['suite'] for r in rows}}
    cohorts['overall_unique_log'] = overall(rows)
    cohorts['qa_gold_only'] = overall([r for r in rows if r['truth_status'] in ('QA-human', 'gold-development')])
    for summary in data['summaries']:
        group = [r for r in cohorts[summary['suite']] if r['strategy'] == summary['strategy']
                 and (summary['family'] == 'all' or r['family'] == summary['family']) and summary['view'] in r['views']]
        for metric, actual in summary['metrics'].items():
            cells = [(r, r['views'][summary['view']][summary['series']][summary['policy']][metric]) for r in group]
            values = defaultdict(list)
            for row, cell in cells:
                if cell['available']:
                    values[row['log']].append(cell['value'])
            expected = float(np.mean([np.mean(v) for v in values.values()])) if values else None
            check(same(expected, actual['mean']), 'Mean mismatch: '+summary['suite']+'/'+metric)
            counts = actual['denominator']
            expected_counts = dict(total_windows=len(cells), available_windows=sum(c['available'] for _, c in cells),
                eligible_windows=sum(c['eligible'] for _, c in cells), total_logs=len({r['log'] for r in group}),
                available_logs=len(values), eligible_logs=len({r['log'] for r,c in cells if c['eligible']}))
            for unit in ('windows', 'logs'):
                expected_counts['excluded_'+unit] = expected_counts['total_'+unit]-expected_counts['available_'+unit]
            for name, count in expected_counts.items():
                check(counts[name] == count, 'Denominator mismatch: '+summary['suite']+'/'+metric+'/'+name)
            checks['summary_metrics'] += 1
    for pair in data['paired']:
        if pair['family'] != 'all' or pair['suite'] not in ('overall_unique_log', 'viridien_eval', 'qa_gold_only'):
            continue
        group = cohorts[pair['suite']]
        left = {r['case_id']: r for r in group if r['strategy'] == pair['a'] and pair['view'] in r['views']}
        right = {r['case_id']: r for r in group if r['strategy'] == pair['b'] and pair['view'] in r['views']}
        by_log = defaultdict(list)
        available = []
        for cid in sorted(left.keys() & right.keys()):
            a, b = left[cid], right[cid]
            x = a['views'][pair['view']][pair['series']][pair['policy']]['score']
            y = b['views'][pair['view']][pair['series']][pair['policy']]['score']
            if x['available'] and y['available']:
                by_log[a['log']].append(x['value']-y['value'])
                available.append(cid)
        actual = pair['metrics']['score']
        check(actual['paired_case_ids'] == available and actual['paired_logs'] == sorted(by_log), 'Paired availability mismatch')
        vals = np.asarray([np.mean(by_log[k]) for k in sorted(by_log)], dtype=float)
        check(actual['ci']['n_logs'] == len(vals), 'Paired log count mismatch')
        if len(vals):
            means = vals[np.random.default_rng(1108).integers(0,len(vals),(10000,len(vals)))].mean(axis=1)
            check(same(float(vals.mean()),actual['ci']['mean']) and
                  same(float(np.quantile(means,.025)),actual['ci']['low']) and
                  same(float(np.quantile(means,.975)),actual['ci']['high']), 'Paired bootstrap mismatch')
        else:
            check(actual['ci']['mean'] is None, 'Empty pair has a mean')
        checks['paired_product_cis'] += 1
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), derived_sha256=sha(target),
                  audit_source_sha256=sha(__file__), records=len(rows), checks=dict(checks), errors=errors,
                  inference_started=False, original_scores_unchanged=True, selection_promoted=False)
    (HERE/'parent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PARENT_AVAILABILITY_AUDIT: {len(rows)} records; {checks["metric_cells"]} metric cells; {checks["summary_metrics"]} summary metrics; {checks["paired_product_cis"]} paired product CIs; {len(errors)} errors')
    for error in errors[:12]:
        print(error)
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
