"""Audit one saved Jev case and its four baselines without API or inference."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BAKE = HERE.parents[2]
REPORT = HERE.parent
JEV = BAKE.parent / 'pattern_lab' / 'data' / 'jev-e2e-20261108'
sys.path.insert(0, str(BAKE))
from harness import io, score  # noqa: E402
from strategies.grading_policy import score_equivalent  # noqa: E402
from build_comparison import sha, equal, IDENTITY, FIELDS, METHODS, load  # noqa: E402


def replay(ident):
    manifest = load(REPORT / 'manifest_1108.json')
    case = next(c for c in manifest['cases'] if c['case_id'] == ident)
    jev = next(r for r in load(JEV / 'results.json')['records'] if r['case_id'] == ident)
    baseline = [r for r in load(REPORT / 'scores.json')['records'] if r['case_id'] == ident]
    audit = load(REPORT / 'audit_saved_predictions.json')
    if len(baseline) != 4 or audit['errors']:
        raise ValueError('Missing or unaudited baseline')
    root = JEV / 'boat-results' / 'exports' / ident
    export = load(root / 'export.json')
    receipt = load(root / 'receipt.json')
    evaluation = load(root / 'evaluation.json')
    eval_receipt = load(root / 'evaluation-receipt.json')
    decision = load(JEV / 'boat-decisions' / (ident + '.json'))
    if receipt['export_sha256'] != sha(root / 'export.json') or decision['export_sha256'] != receipt['export_sha256']:
        raise ValueError('Export/decision receipt mismatch')
    for name, digest in receipt['artifacts'].items():
        if sha(root / name) != digest:
            raise ValueError('Export artifact hash mismatch: ' + name)
    if eval_receipt['evaluation_sha256'] != sha(root / 'evaluation.json') or eval_receipt['prediction_sha256'] != sha(root / 'prediction.csv'):
        raise ValueError('Evaluation receipt mismatch')
    if any(export.get(k) != case.get(k) for k in ('window', 'image_sha256', 'annotation_sha256')):
        raise ValueError('Export/source identity mismatch')
    if evaluation['truth_sha256'] != case['truth_sha256'] or sha(case['truth_csv']) != case['truth_sha256']:
        raise ValueError('Saved truth hash mismatch')
    if set(decision['by_pixel']) != {q['pixel_sha256'] for q in export['queries']} or any(v is None for v in decision['by_pixel'].values()):
        raise ValueError('Decisions do not cover the frozen queries')
    report_receipt = load(root / 'report-receipt.json')
    for name, digest in report_receipt['artifacts'].items():
        if sha(root / name) != digest:
            raise ValueError('Report artifact hash mismatch: ' + name)
    truth = io.load_rows(case['truth_csv'])
    rows = []
    for r in baseline + [jev]:
        if any(r.get(k) != case.get(k) for k in IDENTITY):
            raise ValueError('Case identity mismatch: ' + r['strategy'])
        path = Path(r['prediction_csv'])
        expected = r.get('prediction_sha256') if r is jev else audit['prediction_csv_sha256'].get(str(path))
        if not expected or sha(path) != expected:
            raise ValueError('Prediction hash mismatch: ' + r['strategy'])
        pred = io.load_rows(path)
        strict = score.score(pred, truth, tuple(case['window']), min_layer_px=0)
        equivalent = score_equivalent(pred, truth, tuple(case['window']), min_layer_px=0)
        if r.get('error'):
            strict = dict(score=0., row_acc=0., boundary_f1=0., cluster_ari=0., depth_mae_ft=None)
            equivalent = dict(strict)
        old_equivalent = evaluation['clay_shale_equivalent_v1'] if r is jev else r['accepted_metrics']
        if any(not equal(strict.get(k), r.get(k)) or not equal(equivalent.get(k), old_equivalent.get(k)) for k in FIELDS):
            raise ValueError('Saved score differs from replay: ' + r['strategy'])
        rows.append(dict(case_id=ident, strategy=r['strategy'], strict_harness=strict,
                         clay_shale_equivalent_v1=equivalent, prediction_sha256=expected))
    result = dict(status='cached_replay_pass', case_id=ident, image_sha256=case['image_sha256'],
                  annotation_sha256=case['annotation_sha256'], truth_sha256=case['truth_sha256'],
                  export_sha256=receipt['export_sha256'], decision_sha256=sha(JEV / 'boat-decisions' / (ident + '.json')),
                  evaluation_sha256=eval_receipt['evaluation_sha256'], api_calls_made=0,
                  source_receipt_note='Existing evaluation receipt does not itself bind decision file hash.', records=rows)
    (HERE / (ident + '.cached_replay.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('PASS:', ident, 'five saved predictions replayed under both policies; no API calls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case_id')
    replay(parser.parse_args().case_id)
