"""Report saved timing regimes and native-cache evidence without changing records."""
from __future__ import annotations
from collections import defaultdict
import html,math,statistics


def diagnostics(record):
    return record.get('strategy_diagnostics') or record.get('diagnostics') or {}


def runtime_regime(record):
    scope=str(record.get('runtime_scope') or '')
    remote=record.get('remote') is True or any(x in scope.casefold() for x in ('remote','boat','vm cpu','predecoded','decoded lossless'))
    return ('remote · '+(scope or 'CPU crop inference; scope missing')) if remote else ('local · '+(scope or 'source-page run()'))


def runtime_summary(records):
    groups=defaultdict(list)
    for r in records:groups[(r['suite'],r['strategy'],runtime_regime(r))].append(r)
    result=[]
    for (suite,strategy,regime),rows in sorted(groups.items()):
        times=[float(r['seconds']) for r in rows if isinstance(r.get('seconds'),(int,float)) and math.isfinite(r['seconds'])]
        result.append(dict(suite=suite,strategy=strategy,regime=regime,cases=len(rows),timed=len(times),missing=len(rows)-len(times),mean_seconds=statistics.mean(times) if times else None,median_seconds=statistics.median(times) if times else None))
    return result


def native_legend_status(record):
    if record['strategy'] not in ('s20_v4','s21_v4dino','s23_v4dino_identity'):return 'not_applicable','none'
    for source,diag in [('primary',diagnostics(record)),('pixel_only',diagnostics(record.get('pixel_only',{})))]:
        if diag.get('native_cache_present') is False or diag.get('native_entries')==0:return 'missing_native_legend',source
        if diag.get('native_entries',0)>0:
            return ('empty_usable_legend' if diag.get('legend_entries')==0 else 'available'),source
    return 'unknown','none'


def native_legend_summary(records):
    groups=defaultdict(list)
    for r in records:groups[(r['suite'],r['strategy'])].append(r)
    result=[]
    for (suite,strategy),rows in sorted(groups.items()):
        counts={k:0 for k in ('missing_native_legend','empty_usable_legend','available','unknown','not_applicable')};missing_logs=set();pixel_evidence=0
        for r in rows:
            status,source=native_legend_status(r);counts[status]+=1
            if status=='missing_native_legend':missing_logs.add(r['log'])
            pixel_evidence+=source=='pixel_only'
        result.append(dict(suite=suite,strategy=strategy,cases=len(rows),logs=len(set(r['log'] for r in rows)),missing_native_logs=len(missing_logs),pixel_only_evidence_cases=pixel_evidence,**counts))
    return result


def runtime_table(data,label=lambda x:x):
    esc=lambda x:html.escape(str(x));fmt=lambda x:'unknown' if x is None else f'{x:.2f}'
    out=['<div class="table-wrap scroll"><table><thead><tr><th>Suite / strategy / timing scope</th><th>Cases</th><th>Timed</th><th>Missing</th><th>Mean s</th><th>Median s</th></tr></thead><tbody>']
    for r in data:out+=['<tr><td>'+esc(r['suite']+' / '+label(r['strategy']))+'<br>'+esc(r['regime'])+'</td>'+''.join('<td>'+str(r[k])+'</td>' for k in ('cases','timed','missing'))+'<td>'+fmt(r['mean_seconds'])+'</td><td>'+fmt(r['median_seconds'])+'</td></tr>']
    return ''.join(out)+'</tbody></table></div>'


def native_table(data,label=lambda x:x):
    esc=lambda x:html.escape(str(x))
    out=['<div class="table-wrap scroll"><table><thead><tr><th>Suite / strategy</th><th>Cases</th><th>Missing native</th><th>Missing logs</th><th>Empty usable</th><th>Unknown</th><th>Not applicable</th></tr></thead><tbody>']
    for r in data:out+=['<tr><td>'+esc(r['suite']+' / '+label(r['strategy']))+'</td>'+''.join('<td>'+str(r[k])+'</td>' for k in ('cases','missing_native_legend','missing_native_logs','empty_usable_legend','unknown','not_applicable'))+'</tr>']
    return ''.join(out)+'</tbody></table></div>'
