"""Diagnostics without rock truth. High agreement does not establish accuracy.

Limits are fixed engineering checks, not values fitted on this TEST set.
Missing evidence stays missing. Failed and empty runs stay in denominators.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import adjusted_rand_score

SHORT_BED_PX = 8
MIN_FT_PER_PX = .005
MAX_FT_PER_PX = 5.
MAX_RESIDUAL_PX = 16.


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def named(label):
    text = str(label or '').strip()
    return bool(re.search(r'[a-zA-Z]', text)) and not bool(re.fullmatch(
        r'(?:unknown|unassigned|unlabelled|unlabeled|none|null|nan|n/?a|blank|'
        r'(?:cluster|class|entry|lithology|rock|pattern|swatch)[\s_#-]*\d*)', text, re.I))


def _segments(rows, window):
    y0, y1 = map(int, window)
    if y1 <= y0:
        raise ValueError('Window must have positive height')
    for row in rows or []:
        top, bottom = _number(row.get('top_y')), _number(row.get('bottom_y'))
        if top is None or bottom is None:
            continue
        a, b = max(y0, math.ceil(top)), min(y1, math.ceil(bottom))
        if b > a:
            yield a, b, row


def raster(rows, window):
    """Return class ids and a separate assigned mask; gaps are never a class."""
    y0, y1 = map(int, window)
    if y1 <= y0:
        raise ValueError('Window must have positive height')
    values = np.full(y1-y0, -1, dtype=np.int32)
    identities = {}
    for a, b, row in _segments(rows, window):
        if not named(row.get('lithology')):
            continue
        # A partition is invariant to class names. Prefer explicit cluster ids.
        cluster = row.get('cluster_id')
        identity = ('cluster', str(cluster)) if cluster is not None and str(cluster) not in ('', '-1', 'nan') else (
            'name', re.sub(r'\s+', ' ', str(row['lithology']).strip().casefold()))
        code = identities.setdefault(identity, len(identities))
        values[a-y0:b-y0] = code
    return values, values >= 0


def agreement(rows_a, rows_b, window, *, failure_a=False, failure_b=False):
    """ARI on common assigned rows, plus support. Never score common blanks."""
    a, assigned_a = raster(rows_a, window)
    b, assigned_b = raster(rows_b, window)
    common = assigned_a & assigned_b
    support = int(common.sum())
    return dict(ari=(float(adjusted_rand_score(a[common], b[common]))
                     if support >= 2 and not failure_a and not failure_b else None),
                common_assigned_share=float(common.mean()), common_assigned_px=support,
                clusters_a=int(len(np.unique(a[common]))) if support else 0,
                clusters_b=int(len(np.unique(b[common]))) if support else 0,
                assignment_mask_agreement=float(np.mean(assigned_a == assigned_b)),
                eligible=bool(support >= 2 and not failure_a and not failure_b))


def depth_sanity(calibration=None, rows=None):
    a = _number(_value(calibration, 'a'))
    residual = _number(_value(calibration, 'rmse_px'))
    anchors = _number(_value(calibration, 'n_points'))
    monotonic = a > 0 if a is not None else None
    plausible = MIN_FT_PER_PX < a < MAX_FT_PER_PX if a is not None else None
    fit_ok = bool(monotonic and plausible and residual is not None and
                  0 <= residual <= MAX_RESIDUAL_PX and anchors is not None and anchors >= 3)
    points = []
    for r in rows or []:
        for edge in ('top', 'bottom'):
            y, d = _number(r.get(edge+'_y')), _number(r.get(edge+'_depth'))
            if y is not None and d is not None:
                points.append((y, d))
    points.sort(key=lambda p:p[0])
    prediction_monotonic = all(d2 >= d1 for (_, d1), (_, d2) in zip(points, points[1:])) if len(points) >= 2 else None
    return dict(depth_monotonic=monotonic, depth_ft_per_px=a, depth_plausible=plausible,
                depth_residual_px=residual, depth_anchors=anchors, depth_fit_ok=fit_ok,
                depth_method=_value(calibration, 'method'),
                predicted_depth_monotonic=prediction_monotonic)


def measure(rows, window, *, calibration=None, legend_entries=None, failure=False,
            short_bed_px=SHORT_BED_PX):
    """Measure output structure. Segment density/redundancy are only proxies."""
    if short_bed_px <= 0:
        raise ValueError('short_bed_px must be positive')
    y0, y1 = map(int, window)
    values, assigned = raster(rows, window)
    segments = sorted(_segments(rows, window), key=lambda r: (r[0], r[1]))
    lengths = [b-a for a,b,r in segments]
    short = sum(length < short_bed_px for length in lengths)
    redundant = adjacent = overlaps = 0
    for (a,b,r), (c,d,s) in zip(segments, segments[1:]):
        overlaps += int(c < b)
        if b == c:
            adjacent += 1
            redundant += int(str(r.get('lithology','')).strip().casefold() ==
                             str(s.get('lithology','')).strip().casefold())
    entries = None if legend_entries is None else list(legend_entries)
    entry_names = None if entries is None else sum(named(_value(e, 'label')) for e in entries
                                                   if _value(e, 'is_lithology', True))
    found = None if entries is None else sum(bool(_value(e, 'is_lithology', True)) for e in entries)
    result = dict(execution_failure=bool(failure), empty_prediction=not bool(assigned.any()),
                  failure_rate=float(bool(failure) or not bool(assigned.any())),
                  assigned_share=float(assigned.mean()), interval_count=len(segments),
                  output_names=len({str(r.get('lithology','')).strip().casefold()
                                    for _,_,r in segments if named(r.get('lithology'))}),
                  legend_entries_found=found, legend_entries_named=entry_names,
                  legend_named_share=entry_names/found if found else None,
                  short_bed_rate=short/len(lengths) if lengths else None,
                  short_bed_px=short_bed_px, beds_per_1000px=len(segments)*1000/(y1-y0),
                  oversegmentation_rate=redundant/adjacent if adjacent else None,
                  redundant_boundaries=redundant, overlapping_intervals=overlaps)
    result.update(depth_sanity(calibration, rows))
    return result


def _read_rows(root, record):
    path = record.get('csv')
    if not path:
        return []
    with (root/path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_report(frame, out_dir, *, clean_dir=None):
    """Write diagnostics for bench records. Optional clean run gives stability."""
    import pandas as pd
    root = Path(out_dir)
    records = frame.to_dict('records') if hasattr(frame, 'to_dict') else list(frame)
    clean = {}
    if clean_dir:
        clean_dir = Path(clean_dir)
        clean_records = json.loads((clean_dir/'results.json').read_text(encoding='utf-8'))['results']
        clean = {(r['strategy'], r['log'], r['y0'], r['y1']): r for r in clean_records}
    by_window = {}
    metrics = []
    for r in records:
        if r.get('scored'):
            continue
        a,b=_number(r.get('y0')),_number(r.get('y1'))
        valid_window=a is not None and b is not None and b>a
        win = (int(a),int(b)) if valid_window else None
        rows = _read_rows(root,r)
        diag = r.get('unlabelled_metrics')
        if not valid_window:
            diag=dict(execution_failure=True,empty_prediction=True,failure_rate=1.,
                      assigned_share=None,interval_count=0,output_names=0,
                      legend_entries_found=None,legend_entries_named=None,legend_named_share=None,
                      short_bed_rate=None,beds_per_1000px=None,oversegmentation_rate=None,
                      **depth_sanity())
        elif not isinstance(diag,dict):
            diag = measure(rows,win,failure=r.get('failure',False))
        else:
            diag = dict(diag)
            diag['execution_failure'] = bool(r.get('failure',False))
            diag['failure_rate'] = float(diag['execution_failure'] or diag['empty_prediction'])
        m = {k:r.get(k) for k in ('strategy','suite','log','source','well','family','y0','y1')}
        m.update(diag)
        m['legend_measurement'] = diag.get('legend_parser','shared harness parser; parser failed or used as fallback')
        reference = clean.get((r['strategy'],r['log'],r['y0'],r['y1'])) if valid_window else None
        if reference is not None:
            stable = agreement(_read_rows(clean_dir,reference),rows,win,
                               failure_a=reference.get('failure',False),failure_b=r.get('failure',False))
            m.update({'scan_'+k:v for k,v in stable.items()})
        metrics.append(m)
        if valid_window:
            by_window.setdefault((r['suite'],r['log'],*win),[]).append((r,rows))
    pairs=[]
    for (suite,log,y0,y1),group in by_window.items():
        baseline=next(((r,rows) for r,rows in group if re.match(r'^s19(?:_|$)',r['strategy'])),None)
        if baseline is None:continue
        for r,rows in group:
            if not re.match(r'^s(?:17|22)(?:_|$)',r['strategy']):continue
            ref,ref_rows=baseline
            pair=dict(suite=suite,log=log,family=r['family'],strategy_a=ref['strategy'],strategy_b=r['strategy'],y0=y0,y1=y1)
            pair.update(agreement(ref_rows,rows,(y0,y1),failure_a=ref.get('failure',False),failure_b=r.get('failure',False)))
            pairs.append(pair)
    detail=pd.DataFrame(metrics)
    detail.to_csv(root/'unlabelled_metrics.csv',index=False)
    summaries=[]
    if not detail.empty:
        for (strategy,family),group in detail.groupby(['strategy','family'],dropna=False):
            summary=dict(strategy=strategy,family=family,pages=len(group),wells=group.well.nunique())
            summary['legend_observed_pages']=int(group.legend_entries_found.notna().sum())
            summary['depth_observed_pages']=int(group.depth_ft_per_px.notna().sum())
            for key in group.select_dtypes(include=['number','bool']).columns:
                if key not in ('y0','y1'):summary[key]=float(group[key].mean()) if group[key].notna().any() else None
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(root/'unlabelled_by_family.csv',index=False)
    pd.DataFrame(pairs).to_csv(root/'strategy_agreement.csv',index=False)
    pair_summaries=[]
    if pairs:
        for keys,group in pd.DataFrame(pairs).groupby(['family','strategy_a','strategy_b']):
            pair_summaries.append(dict(zip(('family','strategy_a','strategy_b'),keys))|
                dict(pages=len(group),eligible=int(group.eligible.sum()),ari=float(group.ari.mean()) if group.ari.notna().any() else None,
                     common_assigned_share=float(group.common_assigned_share.mean())))
    pd.DataFrame(pair_summaries).to_csv(root/'strategy_agreement_by_family.csv',index=False)
    payload=dict(metrics=metrics,by_family=summaries,agreement=pairs,agreement_by_family=pair_summaries,
                 limits=dict(short_bed_px=SHORT_BED_PX,ft_per_px=[MIN_FT_PER_PX,MAX_FT_PER_PX],residual_px=MAX_RESIDUAL_PX),
                 note='TEST diagnostics only. Agreement and smoothness do not establish rock accuracy. '
                      'Oversegmentation is the share of adjacent equal-name boundaries; bed density is also reported. '
                      'Failure rate includes execution failure or no assigned rows. Missing fits are not sane fits.')
    (root/'unlabelled_report.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--clean-dir',type=Path)
    args=parser.parse_args()
    records=json.loads((args.run_dir/'results.json').read_text(encoding='utf-8'))['results']
    result=write_report(records,args.run_dir,clean_dir=args.clean_dir)
    print(f"Written {len(result['metrics'])} diagnostics and {len(result['agreement'])} strategy pairs")


if __name__=='__main__':main()
