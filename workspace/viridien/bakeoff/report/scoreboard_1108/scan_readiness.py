"""Preview existing prediction coverage without inference or truth reads."""
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPORT = Path(__file__).resolve().parent
BAKEOFF = REPORT.parents[1]
PREDICTIONS = (BAKEOFF / 'out' / 'scoreboard_1108').resolve()
APPROVED_PREDICTION_ROOTS = (PREDICTIONS, (REPORT / 'remote').resolve())
sys.path.insert(0, str(BAKEOFF))
from strategies.challenge_export import read_rows  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def annotation(case):
    path = Path(case['annotation'])
    data = path.read_bytes()
    actual = sha(data)
    if actual != case['annotation_sha256']:
        raise ValueError('annotation hash differs from frozen manifest')
    parsed = json.loads(data)
    box = parsed['bounding_boxes']['interpreted_lithology_column']
    if list(box) != case['column_box']:
        raise ValueError('annotation lithology box differs from frozen manifest')
    return box, parsed.get('height')



def validated_prediction_path(path, approved_roots=None):
    """Resolve the saved CSV and reject traversal outside the two approved roots."""
    resolved=Path(path).resolve()
    roots=APPROVED_PREDICTION_ROOTS if approved_roots is None else tuple(Path(root).resolve() for root in approved_roots)
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError('Prediction CSV is outside the approved local and remote output roots')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    source_records=REPORT/('scores.json' if args.final else 'raw_results.json')
    manifest_bytes = (REPORT / 'manifest_1108.json').read_bytes()
    record_bytes = source_records.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = json.loads(record_bytes)['records']
    cases = {c['case_id']: c for c in manifest['cases']}
    candidates = manifest['candidates']
    if len(cases) != len(manifest['cases']):
        raise ValueError('Duplicate case IDs in manifest')
    allowed = {'viridien_eval', 'human_qa', 'human_gold', 'hadrian_unqa', 'ext_dev'}
    cache, details, seen = {}, [], set()
    for record in records:
        case = cases[record['case_id']]
        if case['suite'] not in allowed or record['suite'] != case['suite']:
            raise ValueError('Record is outside the frozen allowed suites')
        strategy = record['strategy']
        key = (case['case_id'], strategy)
        if strategy not in candidates or key in seen:
            raise ValueError('Unexpected or duplicate candidate/case record')
        seen.add(key)
        item = {'case_id': case['case_id'], 'suite': case['suite'], 'strategy': strategy,
                'log': case['log'], 'window': case['window'],
                'physical_document': case['image_sha256'],
                'calibration': record.get('depth_calibration'),
                'native_cache_present': record.get('strategy_diagnostics', {}).get('native_cache_present'),
                'record_error': record.get('error') or record.get('failure')}
        try:
            cache_key = (case['annotation'], case['annotation_sha256'])
            if cache_key not in cache:
                cache[cache_key] = annotation(case)
            box, height = cache[cache_key]
            if list(box) != case['column_box']:
                raise ValueError('case box differs from cached annotation')
            y0, y1 = map(int, case['window'])
            if not (box[1] <= y0 < y1 <= box[3]):
                raise ValueError('window exceeds the annotated lithology box')
            item.update(lithology_box=box, image_height=height,
                        extent='full_lithology_box' if [y0, y1] == [box[1], box[3]] else 'partial_lithology_box')
            pred = validated_prediction_path(record['prediction_csv'])
            item['prediction_sha256'] = sha(pred.read_bytes())
            rows = read_rows(pred, image_height=height)
            if any(int(r['top_y']) < y0 or int(r['bottom_y']) > y1 for r in rows):
                raise ValueError('prediction interval lies outside recorded window')
            item['rows'] = len(rows)
            item['validation'] = 'valid_nonempty' if rows else 'empty_unverified'
            item['depth_finite_ordered'] = all(float(r['top_depth']) <= float(r['bottom_depth']) for r in rows) if rows else None
            item['depth_ordered_between_intervals'] = all(float(a['bottom_depth']) <= float(b['top_depth']) for a, b in zip(rows, rows[1:])) if rows else None
            item['depth_confidence'] = 'unverified: finite values and calibration metadata do not establish depth accuracy'
        except (KeyError, OSError, ValueError, TypeError) as exc:
            item['validation'] = 'invalid_or_unavailable'
            item['validation_error'] = f'{type(exc).__name__}: {exc}'
            item.setdefault('extent', 'unknown')
        details.append(item)
    stats = {}
    for strategy in candidates:
        group = [d for d in details if d['strategy'] == strategy]
        docs = defaultdict(list)
        for d in group:
            docs[d['physical_document']].append(d)
        full = {doc for doc, rows in docs.items() if any(d['extent'] == 'full_lithology_box' for d in rows)}
        valid_full = {doc for doc, rows in docs.items() if any(d['extent'] == 'full_lithology_box' and d['validation'] == 'valid_nonempty' and not d['record_error'] for d in rows)}
        stats[strategy] = {'records': len(group), 'unique_documents': len(docs),
                           'documents_with_full_lithology_box_record': len(full),
                           'documents_with_valid_nonempty_full_box_csv': len(valid_full),
                           'documents_only_partial_or_unknown': len(set(docs) - full),
                           'record_extents': dict(Counter(d['extent'] for d in group)),
                           'record_validation': dict(Counter(d['validation'] for d in group)),
                           'depth_not_ordered_records': sum(d.get('depth_finite_ordered') is False for d in group),
                           'depth_not_ordered_between_interval_records': sum(d.get('depth_ordered_between_intervals') is False for d in group),
                           'missing_native_cache_records': sum(d['native_cache_present'] is False for d in group),
                           'native_cache_unknown_records': sum(d['native_cache_present'] is None for d in group)}
    manifest_docs = {c['image_sha256'] for c in cases.values()}
    observed_docs = {d['physical_document'] for d in details}
    snapshot = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': sha(manifest_bytes), 'record_snapshot_sha256': sha(record_bytes),
        'source_record_file': str(source_records.resolve()),
        'manifest_cases': len(cases), 'candidates': candidates,
        'expected_case_candidate_pairs': len(cases) * len(candidates),
        'observed_record_pairs': len(seen), 'manifest_unique_physical_documents': len(manifest_docs),
        'observed_unique_physical_documents': len(observed_docs),
        'record_coverage_complete': len(seen)==len(cases)*len(candidates),
        'submission_readiness_verified': False,
        'complete': False,
        'incomplete_reason': (('Final record coverage diagnostic: ' if args.final else 'Preview only: ') +
                              ('some case/candidate pairs are missing; ' if len(seen) < len(cases) * len(candidates) else '') +
                              'full-document correctness, depth accuracy and visual output are not established.'),
        'counts_are': 'per strategy; physical documents deduplicated by frozen image SHA-256, not summed across suites',
        'limitations': ['Empty outputs can be valid blank columns and require review.',
                        'Successful CSV validation and full box extent do not establish prediction accuracy or submission readiness.',
                        'Depth ordering and calibration metadata are not confidence or accuracy estimates.',
                        'Human QA Shenzi may use machine boundary anchors; see evaluation_protocol.json.',
                        'DEV missing native legends, when present, are reported separately from QA truth limitations.'],
        'by_strategy': stats, 'records': details,
    }
    output = REPORT / 'readiness_preview.json'
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: snapshot[k] for k in ('manifest_cases', 'expected_case_candidate_pairs',
          'observed_record_pairs', 'manifest_unique_physical_documents',
          'observed_unique_physical_documents', 'by_strategy')}, indent=2))


if __name__ == '__main__':
    main()
