"""Replay receipt-backed cached metrics with shared availability masks; no inference."""
import copy
import hashlib
import html
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
BAKEOFF = REPORT.parent.parent
COMMON = REPORT / 'availability_v2/derived_scores.json'
OLD = HERE.parent / 'scores.json'
RECEIPT = BAKEOFF.parent / 'pattern_lab/data/jev-e2e-20261108/comparison.json'
HELPER = BAKEOFF / 'strategies/grading_availability_v2.py'
COMMON_SHA = '6502e4df226becf194b95c5f2d49dd75075f1a5c897dae17c059504b73102db8'
HELPER_SHA = '80c35cfde8c12c5a4ee6c44de2639ee3c78796a57a6218ea125f28e86d5b2175'
JEV = 's30_jev_sourcecards'
POLICY = {'strict': 'strict_harness', 'owner_equivalent': 'clay_shale_equivalent_v1'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, expected=None):
    data = path.read_bytes()
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Hash mismatch: ' + str(path))
    return json.loads(data)


def build():
    common = load(COMMON, COMMON_SHA)
    old = load(OLD)
    receipt = load(RECEIPT, old['comparison_sha256'])
    if digest(HELPER) != HELPER_SHA or common['pins'].get(str(HELPER)) != HELPER_SHA:
        raise ValueError('Availability helper hash mismatch')
    sys.path.insert(0, str(BAKEOFF))
    spec = importlib.util.spec_from_file_location('grading_availability_v2', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    ids = old['cases']
    candidates = common['candidates'] + [JEV]
    if old['status'] != 'checkpoint' or old['jev_saved_cases'] != 30 or old['matched_cases'] != 29 or len(set(ids)) != 29:
        raise ValueError('Expected the held 30 saved / 29 paired checkpoint')
    if '4b9fc8eda216939d' in ids:
        raise ValueError('The held case must remain unpromoted')
    cached = {(r['case_id'], r['strategy']): r for r in old['records']}
    receipted = {(r['case_id'], r['strategy']): r for r in receipt['records']}
    baseline = {(r['case_id'], r['strategy']): r for r in common['records']}
    if len(cached) != 29 * 5 or set(cached) != {(i, s) for i in ids for s in candidates} or set(cached) != set(receipted):
        raise ValueError('Cached receipt cases or methods differ')
    for key, record in cached.items():
        for field, value in receipted[key].items():
            if record.get(field) != value:
                raise ValueError('Receipt content mismatch: ' + key[0] + ' ' + field)
    rows = []
    for ident in ids:
        reference = baseline[(ident, candidates[0])]
        for method in candidates:
            prior = cached[(ident, method)]
            for field in ('case_id', 'log', 'suite', 'family', 'window', 'truth_sha256', 'image_sha256', 'annotation_sha256'):
                if prior[field] != reference[field]:
                    raise ValueError('Case identity mismatch: ' + ident + ' ' + field)
            if method != JEV:
                row = copy.deepcopy(baseline[(ident, method)])
                for field in ('truth_status', 'legend_vocabulary_invalid', 'structural_supervision'):
                    if row[field] != reference[field]:
                        raise ValueError('Baseline supervision mismatch: ' + ident + ' ' + field)
                if any(row['raw_metrics']['primary'][p] != prior[old_p] for p, old_p in POLICY.items()):
                    raise ValueError('Baseline metrics differ from receipt: ' + ident)
                if row['prediction_references']['primary']['sha256'] != prior['prediction_sha256']:
                    raise ValueError('Baseline prediction hash mismatch: ' + ident)
                row['views'] = {'primary': row['views']['primary']}
                row['raw_metrics'] = {'primary': row['raw_metrics']['primary']}
                row['prediction_failed'] = {'primary': row['prediction_failed']['primary']}
                row['prediction_references'] = {'primary': row['prediction_references']['primary']}
            else:
                row = copy.deepcopy(reference)
                row['strategy'] = JEV
                row['raw_metrics'] = {'primary': {p: prior[old_p] for p, old_p in POLICY.items()}}
                row['prediction_failed'] = {'primary': prior['error'] is not None}
                row['prediction_references'] = {'primary': {'sha256': prior['prediction_sha256'], 'source': 'receipt-backed cached comparison'}}
                row['views'] = {'primary': {}}
                for series in helper.SERIES:
                    row['views']['primary'][series] = {p: helper.metric_view(row['raw_metrics']['primary'][p], row['structural_supervision'], series=series, truth_status=row['truth_status'], legend_invalid=row['legend_vocabulary_invalid'], prediction_failed=row['prediction_failed']['primary']) for p in POLICY}
            for series in helper.SERIES:
                for policy in POLICY:
                    recalculated = helper.metric_view(row['raw_metrics']['primary'][policy], row['structural_supervision'], series=series, truth_status=row['truth_status'], legend_invalid=row['legend_vocabulary_invalid'], prediction_failed=row['prediction_failed']['primary'])
                    if recalculated != row['views']['primary'][series][policy]:
                        raise ValueError('Availability mask mismatch: ' + ident + ' ' + method)
            rows.append(row)
    for ident in ids:
        same = [r for r in rows if r['case_id'] == ident]
        for series in helper.SERIES:
            for policy in POLICY:
                for metric in helper.METRICS:
                    eligible = {r['views']['primary'][series][policy][metric]['eligible'] for r in same}
                    if len(eligible) != 1:
                        raise ValueError('Common eligibility mismatch: ' + ident + ' ' + metric)
    summaries, paired = helper.aggregate(rows, candidates)
    output = {'version': 'availability_v2', 'status': 'checkpoint', 'complete': False,
              'selection_promoted': False, 'generalization_verified': False,
              'saved_cases': 30, 'paired_cases': 29, 'expected_cases': 68,
              'unpromoted_case': '4b9fc8eda216939d', 'cases': ids, 'candidates': candidates,
              'series_definitions': common['series_definitions'], 'method': common['method'],
              'warnings': ['Local metrics are not official challenge grades. Qualified development is not unseen-well evidence.',
                           'Silver is unreviewed source-A tie-break data, not consensus or independent truth. Empty supervision, invalid legend vocabulary, and absent depth truth are masked by the shared helper.',
                           'Manager ASTRA-01 identified 12 Labeller TRAIN pages overlapping seven protected wells, including seven exact evaluation pages. DJS raw and gallery review was exposed; this is not blind.',
                           'Registry binding: 15 excluded, two unknown, 51 DEV-eligible. All 68 API launches remain held; no protected or unknown label generation is allowed.',
                           'Jev uses a machine-legend fallback while the v4-family native legend cache is missing on some DEV logs; this compares pipelines with different inputs.',
                           'Hadrian is unreviewed; Shenzi can retain machine boundary anchors. No model or champion is automatically promoted.'],
              'pins': {'common_scores_sha256': COMMON_SHA, 'helper_sha256': HELPER_SHA,
                       'receipt_checkpoint_sha256': digest(OLD), 'receipt_sha256': old['comparison_sha256'],
                       'scoreboard_manifest_digest': old['scoreboard_manifest_digest'], 'builder_sha256': digest(Path(__file__))},
              'records': rows, 'summaries': summaries, 'paired': paired}
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'scores.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    esc = html.escape
    warnings = ''.join('<li>' + esc(w) + '</li>' for w in output['warnings'])
    table = ''
    for s in summaries:
        if s['suite'] != 'overall_unique_log' or s['family'] != 'all' or s['view'] != 'primary':
            continue
        v = s['metrics']['score']; d = v['denominator']
        value = 'unavailable' if v['mean'] is None else f"{v['mean']:.4f}"
        table += '<tr>' + ''.join('<td>' + esc(str(x)) + '</td>' for x in (s['series'], s['policy'], s['strategy'], value, d['available_windows'], d['available_logs'])) + '</tr>'
    page = '<!doctype html><html><head><meta charset="utf-8"><title>Jev availability checkpoint</title><style>body{font:15px system-ui;max-width:1100px;margin:auto;padding:20px}table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ccc;text-align:left}.warning{background:#fff1d0;padding:12px}</style></head><body><h1>Jev availability-v2 checkpoint</h1><p class="warning">29 receipt-backed pairs of 68; 30 cases saved. The extra held case is not promoted. No champion is selected.</p><ul>' + warnings + '</ul><h2>Primary view: score by unique physical log</h2><p>Available-window and log counts show the denominator for each score. Full metric cells, reasons, matched case IDs, and paired log confidence intervals are in scores.json.</p><table><tr><th>Series</th><th>Policy</th><th>Method</th><th>Mean score</th><th>Windows</th><th>Logs</th></tr>' + table + '</table></body></html>'
    (HERE / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({'cases': len(ids), 'records': len(rows), 'summaries': len(summaries), 'paired': len(paired), 'status': output['status']}))


if __name__ == '__main__':
    build()
