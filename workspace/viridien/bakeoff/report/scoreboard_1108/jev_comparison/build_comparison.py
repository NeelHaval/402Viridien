"""Verify and report the receipt-backed, exact-pair Jev comparison.

This reads saved predictions and scores only. It makes no model calls and
never supplies truth to inference. A checkpoint is not a final comparison.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BAKE = REPORT.parents[1]
JEV = BAKE.parent / 'pattern_lab' / 'data' / 'jev-e2e-20261108'
METHODS = ('s19_combine', 's20_v4', 's21_v4dino', 's23_v4dino_identity', 's30_jev_sourcecards')
POLICIES = ('strict_harness', 'clay_shale_equivalent_v1')
FIELDS = ('score', 'row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft')
IDENTITY = ('case_id', 'suite', 'log', 'window', 'image_sha256', 'annotation_sha256', 'truth_sha256')
EXPECTED_DIGEST = '1da7b757c9846b7cf1b7d79a1f01519195d3011a6c4041ac02c05f6798795301'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            digest.update(data)
    return digest.hexdigest()


def equal(a, b):
    return a is None and b is None or (a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9)


def bootstrap(bylog):
    """Paired log bootstrap on within-log mean deltas; deterministic seed."""
    values = np.array([np.mean(bylog[name]) for name in sorted(bylog)], dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(1108)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(delta=float(values.mean()), low=float(np.quantile(draws, .025)),
                high=float(np.quantile(draws, .975)), logs=len(values), samples=10000,
                note='Single-log interval has no useful precision' if len(values) == 1 else None)


def build(require_complete):
    manifest = load(REPORT / 'manifest_1108.json')
    canonical = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if canonical != EXPECTED_DIGEST:
        raise ValueError('Current scoreboard manifest differs from the specified freeze')
    reference = load(REPORT / 'scores.json')
    audit = load(REPORT / 'audit_saved_predictions.json')
    integrity = load(REPORT / 'result_audit.json')
    if not reference.get('complete') or reference['manifest_digest'] != canonical or audit['errors'] or integrity['errors']:
        raise ValueError('Baseline scoreboard is incomplete or not audited')
    scorer = sha(BAKE / 'harness' / 'score.py')
    grading = sha(BAKE / 'strategies' / 'grading_policy.py')
    if scorer != reference['grading_policy']['shared_scorer_sha256'] or grading != reference['grading_policy']['source_sha256']:
        raise ValueError('Shared scorer or grading policy differs from audited baseline')
    experiment_file = JEV / 'manifest.json'
    experiment = load(experiment_file)
    experiment_hash = sha(experiment_file)
    if experiment_hash != load(JEV / 'lock.json')['manifest_sha256']:
        raise ValueError('Jev manifest lock mismatch')
    if experiment['policy']['model'] != 'jev-1.14.0':
        raise ValueError('Unexpected Jev model')
    cases = {c['case_id']: c for c in manifest['cases']}
    if len(cases) != 68 or len(experiment['cases']) != 68:
        raise ValueError('Expected exactly 68 frozen cases')
    for case in experiment['cases']:
        current = cases.get(case['case_id'])
        if current is None or any(case.get(k) != current.get(k) for k in IDENTITY):
            raise ValueError('Experiment/current case identity mismatch: ' + case['case_id'])
    changed = []
    for source, frozen in experiment['source_hashes'].items():
        actual = sha(source)
        if actual != frozen:
            changed.append(dict(path=source, frozen_sha256=frozen, current_sha256=actual))
            if Path(source).name != 'grading_policy.py' or actual != grading:
                raise ValueError('Unexpected frozen Jev source drift: ' + source)
    receipt = load(JEV / 'comparison-receipt.json')
    comparison_path = JEV / 'comparison.json'
    if sha(comparison_path) != receipt['files']['comparison.json']:
        raise ValueError('Comparison receipt is stale or incomplete; retry after active write')
    comp = load(comparison_path)
    if (comp['metric_sha256'] != scorer or comp['grading_sha256'] != grading or
            comp['input_sha256'] != receipt['input_sha256'] or
            comp['source_fingerprint'] != receipt['source_fingerprint']):
        raise ValueError('Boat scoring receipt or source fingerprints differ')
    own = {r['case_id']: r for r in load(JEV / 'results.json')['records']}
    trace_provenance = {}
    baseline = {(r['case_id'], r['strategy']): r for r in reference['records']}
    if len(baseline) != 272 or len(reference['records']) != 272:
        raise ValueError('Baseline records are not the unique 68-by-4 set')
    pair = {}
    for r in comp['records']:
        key = (r['case_id'], r['strategy'])
        if key in pair or r['case_id'] not in cases or r['strategy'] not in METHODS:
            raise ValueError('Duplicate or unexpected paired record')
        c = cases[r['case_id']]
        if any(r.get(k) != c.get(k) for k in IDENTITY):
            raise ValueError('Paired record identity mismatch')
        original = own.get(r['case_id']) if r['strategy'] == METHODS[-1] else baseline.get(key)
        if original is None or any(original.get(k) != c.get(k) for k in IDENTITY):
            raise ValueError('Original record identity mismatch')
        csv_path = Path(original['prediction_csv'])
        recorded_sha = original.get('prediction_sha256') if r['strategy'] == METHODS[-1] else audit['prediction_csv_sha256'].get(str(csv_path))
        if recorded_sha is None or sha(csv_path) != recorded_sha or r['prediction_sha256'] != recorded_sha:
            raise ValueError('Prediction is missing or differs from its audited hash: ' + str(key))
        if r['strategy'] == METHODS[-1]:
            allowed_manifests = set(experiment.get('accepted_prior_manifest_hashes', [])) | {experiment_hash}
            if original['manifest_sha256'] not in allowed_manifests:
                raise ValueError('Unknown experiment manifest migration')
            diag_path = Path(original['diagnostics'])
            diag = load(diag_path)
            source_labels = {int(e['id']): e['label'] for e in diag['legend']}
            if len(source_labels) != len(diag['legend']):
                raise ValueError('Duplicate source legend IDs in Jev diagnostics')
            for chunk in diag['trace']:
                answer = chunk.get('jev')
                if answer is not None and source_labels.get(int(answer['legend_id'])) != answer['label']:
                    raise ValueError('Jev returned a label outside the source legend')
            packet_hashes = [call['packet_sha256'] for call in diag.get('api', {}).get('calls', [])
                             if call.get('packet_sha256')]
            trace_provenance[r['case_id']] = dict(diagnostics_sha256=sha(diag_path),
                                                  source_legend_entries=len(source_labels),
                                                  chunks=len(diag['trace']),
                                                  request_packet_sha256=packet_hashes,
                                                  hash_note='Packet hash binds request body, model and query pixel hashes; cached requests may appear separately.')
            for field in FIELDS:
                if not equal(r['strict_harness'].get(field), original.get(field)):
                    raise ValueError('Boat and saved Jev strict score mismatch: ' + str(key) + ' ' + field)
                if original.get('clay_shale_equivalent_v1') is not None and not equal(
                        r['clay_shale_equivalent_v1'].get(field), original['clay_shale_equivalent_v1'].get(field)):
                    raise ValueError('Boat and saved Jev equivalent score mismatch: ' + str(key) + ' ' + field)
        else:
            for field in FIELDS:
                if not equal(r['strict_harness'].get(field), original.get(field)) or not equal(r['clay_shale_equivalent_v1'].get(field), original['accepted_metrics'].get(field)):
                    raise ValueError('Boat and audited baseline score mismatch: ' + str(key) + ' ' + field)
        pair[key] = r
    common = [ident for ident in cases if all((ident, method) in pair for method in METHODS)]
    if len(common) != comp['common_cases'] or len(common) != comp['jev_cases']:
        raise ValueError('Five-way comparison coverage does not reconcile')
    if require_complete and (len(common) != 68 or len(own) != 68):
        raise ValueError(f'Final gate not met: {len(common)} paired cases and {len(own)} saved Jev cases')
    summaries = []
    for suite in sorted({cases[i]['suite'] for i in common}):
        ids = [i for i in common if cases[i]['suite'] == suite]
        for policy in POLICIES:
            for method in METHODS:
                perlog = defaultdict(list)
                deltas = defaultdict(list)
                for ident in ids:
                    log = cases[ident]['log']
                    perlog[log].append(pair[(ident, method)][policy]['score'])
                    if method != METHODS[-1]:
                        deltas[log].append(pair[(ident, METHODS[-1])][policy]['score'] - pair[(ident, method)][policy]['score'])
                means = [np.mean(v) for v in perlog.values()]
                summaries.append(dict(suite=suite, policy=policy, strategy=method, cases=len(ids),
                                      logs=len(perlog), score=float(np.mean(means)),
                                      paired_jev_minus_strategy=bootstrap(deltas)))
    coverage = Counter()
    for ident in common:
        entry = baseline[(ident, 's20_v4')]
        state = entry.get('strategy_diagnostics', {}).get('native_cache_present')
        coverage[(cases[ident]['suite'], 'present' if state is True else 'missing' if state is False else 'unknown')] += 1
    coverage_rows = [dict(suite=k[0], native_cache=k[1], cases=v) for k, v in sorted(coverage.items())]
    # Use only already-returned and receipt-verified annotated error images.
    worst = sorted(common, key=lambda ident: pair[(ident, METHODS[-1])]['clay_shale_equivalent_v1']['score'])[:5]
    pictures = []
    for ident in worst:
        root = JEV / 'boat-results' / 'exports' / ident
        receipt_path, img = root / 'report-receipt.json', root / 'error-zoom.png'
        artifact = None
        if receipt_path.is_file() and img.is_file():
            values = load(receipt_path)['artifacts']
            if values.get('error-zoom.png') != sha(img):
                raise ValueError('Annotated image receipt mismatch: ' + ident)
            artifact = str(img.resolve())
        pictures.append(dict(case_id=ident, suite=cases[ident]['suite'], log=cases[ident]['log'],
                             score=pair[(ident, METHODS[-1])]['clay_shale_equivalent_v1']['score'],
                             image=artifact))
    if require_complete and any(not p['image'] for p in pictures):
        raise ValueError('Final worst-case annotated images are not all available')
    final_ready = len(common) == len(own) == 68 and all(p['image'] for p in pictures)
    output = dict(status='complete' if final_ready else 'checkpoint',
                  created_utc=datetime.now(timezone.utc).isoformat(), expected_cases=68,
                  jev_saved_cases=len(own), matched_cases=len(common), cases=common,
                  scoreboard_manifest_digest=canonical, baseline_scores_sha256=sha(REPORT / 'scores.json'),
                  baseline_audit_sha256=sha(REPORT / 'audit_saved_predictions.json'),
                  jev_manifest_sha256=experiment_hash, jev_prior_manifest_hashes=experiment.get('accepted_prior_manifest_hashes', []),
                  jev_migration=experiment.get('migration'), source_changes=changed,
                  model=experiment['policy']['model'], inference_policy=experiment['policy'],
                  frozen_source_sha256=experiment['source_hashes'],
                  boat_source_manifest_sha256=sha(JEV / 'boat-packet' / 'source-manifest.json'),
                  control_hashes={k: v for k, v in load(JEV / 'boat-queue.json').items()
                                  if k in ('source_manifest_sha256', 'api_control_sha256', 'local_control_sha256')},
                  jev_trace_provenance=trace_provenance,
                  scorer_sha256=scorer, grading_sha256=grading, comparison_sha256=sha(comparison_path),
                  boat_source_fingerprint=comp['source_fingerprint'], boat_input_sha256=comp['input_sha256'],
                  metric='Per-window row accuracy times boundary F1, averaged by log; local proxy, not official challenge grade',
                  bootstrap='10000 paired log resamples, seed 1108; per-log mean delta',
                  warnings=['Only exact matched cases enter all five-way comparisons.',
                            'Manager review ASTRA-01 identified 12 Labeller TRAIN pages overlapping seven protected evaluation wells from 13 evaluation logs, including seven exact evaluation pages. The registry and exclusions are pending; these results are not blind.',
                            'DJS raw and gallery review was exposed and was not blind. No new training, tuning, label generation or missing API label calls may proceed until the registry check and the separate security and ownership releases.',
                            'Owner equivalence is local and is not an official challenge rule.',
                            'Hadrian is unreviewed; Shenzi may contain machine boundary anchors.',
                            'DEV only; sealed test and verify data excluded.',
                            'v4-family native legend cache is missing on some DEV logs; Jev uses machine-legend fallback. Treat this as a coverage difference, not Jev quality.',
                            'The Jev fallback is a different pipeline input, so this is not an isolated matcher comparison.',
                            'Only automatic Jev predictions are scored; weak or assisted outputs are not human-approved labels.',
                            'Jev is outside the two prior Astra review scopes; no new review was run.',
                            'The older 65/68 result was crop classification, not this automatic end-to-end test.'],
                  coverage=coverage_rows, summaries=summaries, worst=pictures,
                  records=[dict(pair[(ident, method)], runtime_and_api={
                      field: (own[ident] if method == METHODS[-1] else baseline[(ident, method)]).get(field)
                      for field in ('seconds', 'preprocess_seconds', 'end_to_end_seconds', 'api_calls',
                                    'api_cached_calls', 'partial_api_failures', 'chunks', 'review_chunks')})
                      for ident in common for method in METHODS])
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'scores.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    esc = html.escape
    rows = ''.join('<tr>' + ''.join('<td>' + esc(str(v)) + '</td>' for v in
                   (s['suite'], s['policy'], s['strategy'], s['cases'], s['logs'], f"{s['score']:.4f}",
                    '—' if s['paired_jev_minus_strategy'] is None else
                    f"{s['paired_jev_minus_strategy']['delta']:+.4f} [{s['paired_jev_minus_strategy']['low']:+.4f}, {s['paired_jev_minus_strategy']['high']:+.4f}]")) + '</tr>' for s in summaries)
    imgs = ''
    for picture in pictures:
        imgs += '<h3>' + esc(picture['suite'] + ' · ' + picture['log']) + '</h3><p>Case ' + esc(picture['case_id']) + ', equivalent score ' + f"{picture['score']:.4f}" + '.</p>'
        if picture['image']:
            data = base64.b64encode(Path(picture['image']).read_bytes()).decode('ascii')
            imgs += '<img alt="Annotated source and prediction error" src="data:image/png;base64,' + data + '">'
        else:
            imgs += '<p>Annotated error image not available in this checkpoint.</p>'
    page = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Jev matched comparison</title><style>body{font:15px system-ui;color:#222;max-width:1200px;margin:auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #ccc;text-align:left;padding:6px}img{max-width:100%;height:auto}p{line-height:1.45}.warning{background:#fff0d0;padding:12px}</style></head><body><h1>Jev end-to-end matched comparison</h1><p class="warning">' + esc(output['status'].upper()) + ': ' + str(len(common)) + '/68 exact five-way pairs. ' + ('The final comparison is incomplete.' if output['status'] != 'complete' else 'All planned cases are paired.') + '</p><p class="warning">Exposure limit: manager review ASTRA-01 identified 12 Labeller TRAIN pages overlapping seven protected evaluation wells from 13 evaluation logs, including seven exact evaluation pages. Registry and exclusions are pending. DJS raw and gallery review was exposed. This evaluation is not blind.</p><p>These scores are a local proxy, not an official challenge grade. Jev begins with automatic v4 bed detection, not truth beds. The old 65/68 crop result is a different test. Assisted labels are not included. Jev is outside the two prior Astra reviews.</p><p>DEV native-legend gaps affect baseline coverage; Jev uses a machine-legend fallback. This is a pipeline comparison with different legend inputs, not an isolated matcher comparison. Hadrian is unreviewed; Shenzi can retain machine boundary anchors. Results do not establish held-out performance.</p><div style="overflow-x:auto"><table><thead><tr><th>Suite</th><th>Policy</th><th>Method</th><th>Cases</th><th>Logs</th><th>Score</th><th>Jev minus method, paired log 95% CI</th></tr></thead><tbody>' + rows + '</tbody></table></div><h2>Lowest Jev scores: annotated source and error</h2>' + imgs + '<p>Hashes, runtime fields and migration details are in scores.json.</p></body></html>'
    (HERE / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(status=output['status'], jev_saved=len(own), matched=len(common), expected=68,
                          native_coverage=coverage_rows, annotated_worst=sum(bool(p['image']) for p in pictures))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    build(args.require_complete)
