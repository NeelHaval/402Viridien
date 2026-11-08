"""Independently re-score the saved CSV files with the unchanged harness scorer."""
from pathlib import Path
from collections import defaultdict
import json
import hashlib
import math
import sys

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))
from harness import io, score
from strategies import grading_policy
R = Path(__file__).resolve().parent
FIELDS = ('score', 'row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft')


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def main():
    data = json.loads((R/'scores.json').read_text(encoding='utf-8'))
    manifest = json.loads((R/'manifest_1108.json').read_text(encoding='utf-8'))
    rows = data['records']
    cases = {c['case_id']: c for c in manifest['cases']}
    original_candidates = {'s19_combine', 's20_v4', 's21_v4dino'}
    candidates = manifest.get('candidates', sorted(original_candidates))
    allowed_candidates = original_candidates | {'s23_v4dino_identity'}
    expected = {(c['case_id'], s) for c in manifest['cases'] for s in candidates}
    keys = [(r['case_id'], r['strategy']) for r in rows]
    errors = []
    if not original_candidates.issubset(candidates) or not set(candidates).issubset(allowed_candidates):
        errors.append('Unexpected candidate list in the manifest')
    if len(candidates) != len(set(candidates)):
        errors.append('Duplicate candidates in the manifest')
    if len(keys) != len(set(keys)) or set(keys) != expected:
        errors.append('Coverage differs from the frozen manifest')
    if not data.get('complete'):
        errors.append('The score file is incomplete')
    declared_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                               separators=(',', ':')).encode('utf-8')).hexdigest()
    if data.get('manifest_digest') != declared_digest:
        errors.append('Top-level manifest digest mismatch')
    policy = data.get('grading_policy', {})
    policy_source = Path(grading_policy.__file__).resolve()
    policy_sha = hashlib.sha256(policy_source.read_bytes()).hexdigest()
    scorer_source = Path(score.__file__).resolve()
    required_policy = dict(id=grading_policy.POLICY_ID, source_path=str(policy_source),
                           source_sha256=policy_sha,
                           equivalent_labels=list(grading_policy.EQUIVALENT_LABELS),
                           canonical_label='Clay/Shale', shared_scorer_path=str(scorer_source),
                           shared_scorer_sha256=hashlib.sha256(scorer_source.read_bytes()).hexdigest(),
                           owner_configured=True, official_challenge_rule_verified=False)
    for key, value in required_policy.items():
        if policy.get(key) != value:
            errors.append(f'Grading policy metadata mismatch {key}')
    successful = diagnostics = accepted_successful = accepted_diagnostics = 0
    checked_inputs = {}
    prediction_hashes = {}

    def pin_prediction(path):
        path = Path(path).resolve()
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = prediction_hashes.setdefault(str(path), value)
        if previous != value:
            errors.append(f'Prediction changed during replay {path}')
    identity_fields = ('log', 'suite', 'window', 'image', 'annotation', 'truth_csv',
                       'image_sha256', 'annotation_sha256', 'truth_sha256', 'truth_sources')
    for rec in rows:
        if rec.get('manifest_digest') != declared_digest:
            errors.append(f"Record manifest digest mismatch {rec['case_id']}")
        pins = manifest.get('freeze', {}).get('dino', {})
        for scope, diag in [('primary', rec.get('strategy_diagnostics', {})),
                            ('pixel', rec.get('pixel_only', {}).get('strategy_diagnostics', {}))]:
            model = diag.get('model')
            if (diag.get('dino_chunk_vectors', 0) or 0) > 0 and not model:
                errors.append(f"Missing {scope} DINO model evidence {rec['case_id']}")
            if model:
                for key in ('weights_sha256', 'config_sha256'):
                    if not pins.get(key) or model.get(key) != pins[key]:
                        errors.append(f"Wrong {scope} model pin {rec['case_id']} {key}")
        case = cases.get(rec['case_id'])
        if case is None:
            errors.append(f"Unknown case {rec['case_id']}")
            continue
        for field in identity_fields:
            if rec.get(field) != case.get(field):
                errors.append(f"Record/manifest mismatch {rec['case_id']} {field}")
        inputs = [(case['image'], case['image_sha256']),
                  (case['annotation'], case['annotation_sha256']),
                  (case['truth_csv'], case['truth_sha256'])]
        inputs += [(s['path'], s['sha256']) for s in case['truth_sources']]
        for path, digest in inputs:
            if (path, digest) in checked_inputs:
                continue
            h = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    h.update(block)
            checked_inputs[(path, digest)] = h.hexdigest()
            if h.hexdigest() != digest:
                errors.append(f'Input hash mismatch {path}')
        truth = io.load_rows(Path(rec['truth_csv']))
        pin_prediction(rec['prediction_csv'])
        pred = io.load_rows(Path(rec['prediction_csv']))
        window = tuple(rec['window'])
        if truth is None or pred is None:
            errors.append(f"Missing CSV {rec['case_id']} {rec['strategy']}")
            continue
        previous = window[0]
        for p in pred:
            if not window[0] <= p['top_y'] < p['bottom_y'] <= window[1] or p['top_y'] < previous:
                errors.append(f"Invalid interval {rec['case_id']} {rec['strategy']}")
            previous = p['bottom_y']
        if rec.get('error'):
            if any(not same(rec.get(f), 0) for f in FIELDS[:-1]):
                errors.append(f"Nonzero failed record {rec['case_id']}")
            for f in FIELDS:
                if f not in rec.get('accepted_metrics', {}) or not same(rec['accepted_metrics'][f], rec.get(f)):
                    errors.append(f"Failed accepted metric mismatch {rec['case_id']} {f}")
        else:
            actual = score.score(pred, truth, window, min_layer_px=0)
            successful += 1
            for f in FIELDS:
                if not same(actual[f], rec.get(f)):
                    errors.append(f"Metric mismatch {rec['case_id']} {rec['strategy']} {f}")
            accepted = grading_policy.score_equivalent(pred, truth, window, min_layer_px=0)
            accepted_successful += 1
            for f in FIELDS:
                if f not in rec.get('accepted_metrics', {}) or not same(accepted[f], rec['accepted_metrics'][f]):
                    errors.append(f"Accepted metric mismatch {rec['case_id']} {rec['strategy']} {f}")
            for f in ('boundary_f1', 'depth_mae_ft'):
                if not same(actual[f], accepted[f]):
                    errors.append(f"Label policy changed contacts or depth {rec['case_id']} {f}")
        if rec.get('pixel_only'):
            diag = rec['pixel_only']
            pin_prediction(diag['prediction_csv'])
            pixel_pred = io.load_rows(Path(diag['prediction_csv']))
            actual = score.score(pixel_pred, truth, window, min_layer_px=0)
            diagnostics += 1
            for f in FIELDS:
                if not same(actual[f], diag.get(f)):
                    errors.append(f"Pixel metric mismatch {rec['case_id']} {rec['strategy']} {f}")
            accepted = grading_policy.score_equivalent(pixel_pred, truth, window, min_layer_px=0)
            accepted_diagnostics += 1
            for f in FIELDS:
                if f not in diag.get('accepted_metrics', {}) or not same(accepted[f], diag['accepted_metrics'][f]):
                    errors.append(f"Accepted pixel metric mismatch {rec['case_id']} {rec['strategy']} {f}")
            for f in ('boundary_f1', 'depth_mae_ft'):
                if not same(actual[f], accepted[f]):
                    errors.append(f"Label policy changed pixel contacts or depth {rec['case_id']} {f}")
    summaries = [(item, False) for item in data['summaries']]
    if not data.get('accepted_summaries'):
        errors.append('Missing accepted summaries')
    summaries += [(item, True) for item in data.get('accepted_summaries', [])]
    for item, accepted_series in summaries:
        if item['suite'] == 'overall_unique_log':
            continue
        selected = [r for r in rows if r['suite'] == item['suite'] and r['strategy'] == item['strategy']
                    and (item['family'] == 'all' or r['family'] == item['family'])]
        grouped = defaultdict(list)
        for rec in selected:
            grouped[rec['log']].append(rec)
        for f in FIELDS:
            means = []
            for group in grouped.values():
                metric_rows = [r.get('accepted_metrics', {}) if accepted_series else r for r in group]
                values = [r[f] for r in metric_rows if r.get(f) is not None]
                if values:
                    means.append(sum(values)/len(values))
            actual = sum(means)/len(means) if means else None
            if not same(actual, item['metrics'].get(f)):
                errors.append(f"Mean mismatch accepted={accepted_series} {item['suite']} {item['strategy']} {item['family']} {f}")
    source_checks = 0
    for path, digest in manifest['source_hashes'].items():
        p = B/path
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        source_checks += 1
        if h != digest:
            errors.append(f'Executed source hash mismatch {path}')
    manifest_digest = hashlib.sha256((R/'manifest_1108.json').read_bytes()).hexdigest()
    for path in tuple(prediction_hashes):
        pin_prediction(path)
    record_digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    result = dict(records=len(rows), expected=len(expected), rescored=successful,
                  manifest_sha256=manifest_digest, records_sha256=record_digest,
                  input_hash_checks=len(checked_inputs), source_hash_checks=source_checks,
                  prediction_csv_sha256=prediction_hashes,
                  prediction_csv_hash_checks=len(prediction_hashes),
                  pixel_diagnostics_rescored=diagnostics,
                  grading_policy_id=grading_policy.POLICY_ID,
                  grading_policy_source_sha256=policy_sha,
                  accepted_rescored=accepted_successful,
                  accepted_pixel_diagnostics=accepted_diagnostics, errors=errors)
    result['summary'] = f"{len(rows)}/{len(expected)} records; {successful} predictions and {diagnostics} pixel diagnostics re-scored; {len(errors)} errors"
    (R/'audit_saved_predictions.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['summary'])
    for error in errors[:30]:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
