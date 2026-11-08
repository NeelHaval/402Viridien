"""Audit a generated checkpoint against pinned metadata and shared aggregation."""
import importlib.util
import json
import sys
from pathlib import Path

import build_checkpoint as builder


def main():
    root = builder.HERE
    out = builder.load(root / 'scores.json')
    common = builder.load(builder.COMMON, builder.COMMON_SHA)
    old = builder.load(builder.OLD)
    assert out['pins']['builder_sha256'] == builder.digest(root / 'build_checkpoint.py')
    assert out['pins']['receipt_checkpoint_sha256'] == builder.digest(builder.OLD)
    assert out['pins']['common_scores_sha256'] == builder.COMMON_SHA
    assert out['pins']['helper_sha256'] == builder.HELPER_SHA == builder.digest(builder.HELPER)
    assert out['status'] == 'checkpoint' and not out['complete'] and not out['selection_promoted']
    assert out['cases'] == old['cases'] and (out['saved_cases'], out['paired_cases'], out['expected_cases']) == (30, 29, 68)
    assert out['unpromoted_case'] not in out['cases'] and len(out['records']) == 145
    by_id = {(r['case_id'], r['strategy']): r for r in out['records']}
    assert len(by_id) == 145
    for row in out['records']:
        for series in ('reference_diagnostic', 'qualified_development'):
            for policy in ('strict', 'owner_equivalent'):
                cells = row['views']['primary'][series][policy]
                if not row['structural_supervision']['filled_rows']:
                    assert not any(cell['available'] for cell in cells.values())
                if row['legend_vocabulary_invalid']:
                    assert all(not cells[m]['available'] for m in ('score', 'row_acc', 'cluster_ari'))
                if not row['structural_supervision']['finite_depth_boundaries']:
                    assert not cells['depth_mae_ft']['available']
                if row['truth_status'] == 'unreviewed-two-source-A-tie-break' and series == 'qualified_development':
                    assert all(not cells[m]['available'] for m in ('score', 'row_acc', 'cluster_ari'))
    sys.path.insert(0, str(builder.BAKEOFF))
    spec = importlib.util.spec_from_file_location('grading_availability_v2', builder.HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    summaries, paired = helper.aggregate(out['records'], out['candidates'])
    assert out['summaries'] == summaries and out['paired'] == paired
    assert common['series_definitions'] == out['series_definitions']
    print('PASS: 29 pairs, 145 records, common masks and all summary/paired denominators recomputed; no truth or image reads')


if __name__ == '__main__':
    main()
