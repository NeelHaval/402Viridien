"""Compare final notebook/slide CSV values with an official bundle."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', required=True, type=Path)
    parser.add_argument('--official', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    results = []
    for expected in sorted(args.official.rglob('*.csv')):
        if not expected.name.endswith(('.notebook.csv', '.slide.csv')):
            continue
        target = args.actual / expected.relative_to(args.official)
        matches = [target] if target.is_file() else []
        record = dict(file=str(expected.relative_to(args.official)))
        if len(matches) != 1:
            record['error'] = 'Missing or duplicate output'
        else:
            with expected.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                wanted = list(reader)
            if not fields:
                record['error'] = 'Official CSV has no header'
                results.append(record)
                continue
            with matches[0].open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                record['header_match'] = reader.fieldnames == fields
                actual = list(reader)
            record.update(expected_rows=len(wanted), actual_rows=len(actual),
                          differences={field: sum(a.get(field) != b.get(field) for a, b in zip(actual, wanted))
                                       for field in fields})
            record['match'] = record['header_match'] and len(actual) == len(wanted) and not any(record['differences'].values())
        results.append(record)
    report = dict(match=bool(results) and all(r.get('match', False) for r in results), files=results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['match'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
