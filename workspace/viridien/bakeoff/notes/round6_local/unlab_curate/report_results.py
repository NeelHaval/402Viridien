"""Summarise completed TEST runs without reading rock truth."""
import argparse
from collections import Counter
import csv
import json
import math
import importlib.util
from pathlib import Path
from statistics import fmean

# Failure flags, error categories and availability come from one corrected export per run
# (derived/corrected_records.json, built by export_corrected.py via denominators.py).
_spec=importlib.util.spec_from_file_location('denominators',Path(__file__).with_name('denominators.py'))
denominators=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(denominators)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(rows,key):
    values=[float(r[key]) for r in rows if isinstance(r.get(key),(int,float)) and math.isfinite(float(r[key]))]
    return fmean(values) if values else None


def num(x,scale=1,digits=3):
    return 'missing' if x is None else f'{x*scale:.{digits}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('lane',type=Path)
    p.add_argument('--draft',action='store_true')
    p.add_argument('--combined-test-log',type=Path)
    a=p.parse_args()
    dataset=a.root/'external_unlabelled_test';clean_dir=a.lane/'runs/clean';scan_dir=a.lane/'runs/medium'
    clean=read(clean_dir/'unlabelled_report.json');scan=read(scan_dir/'unlabelled_report.json')
    cr=read(clean_dir/'results.json')['results'];sr=read(scan_dir/'results.json')['results']
    items=read(dataset/'eval_set.json');contract=read(dataset/'curation.json')
    assert contract['status']=='FROZEN TEST'
    assert len(cr)==len(sr)==len(items)*4
    assert all(not r['scored'] and r.get('score') is None for r in cr+sr)
    assert read(clean_dir/'model_fingerprints.json')==read(scan_dir/'model_fingerprints.json')
    assert all(read(d/'model_files_unchanged.json')['unchanged'] for d in (clean_dir,scan_dir))
    late_path=a.lane/'late_source_check.json'
    # Owner moved inference to the boat VM and cancelled the late fold.
    # A late-source note is optional. Completed clean and scan runs are required.
    with (dataset/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
    strategies=sorted({r['strategy'] for r in cr});families=list(contract['per_family'])
    unit_summary=(a.lane/'metrics_checks.log').read_text(encoding='utf-8').strip().splitlines()[-1]
    combined_summary='not re-run'
    if a.combined_test_log:
        combined_summary=a.combined_test_log.read_text(encoding='utf-8').strip().splitlines()[-1]
    audit=denominators.main(a.lane)
    assert [r['run'] for r in audit['runs']]==['clean','medium'],'corrected export needs both runs'
    counts_by={(run['run'],r['family'],r['strategy']):r for run in audit['runs'] for r in run['rows']}
    category_of={(run_name,c['strategy'],c['log'],c['y0'],c['y1']):c['error_category']
                 for run_name,d in (('clean',clean_dir),('medium',scan_dir))
                 for c in json.loads((d/'derived/corrected_records.json').read_text(encoding='utf-8'))['records']}
    totals=[]
    for family in ['All']+families:
        for strategy in strategies:
            c=[r for r in clean['metrics'] if r['strategy']==strategy and (family=='All' or r['family']==family)]
            s=[r for r in scan['metrics'] if r['strategy']==strategy and (family=='All' or r['family']==family)]
            totals.append(dict(family=family,strategy=strategy,pages=len(c),
                legend_observed=sum(r.get('legend_entries_found') is not None for r in c),
                depth_observed=sum(r.get('depth_ft_per_px') is not None for r in c),
                depth_fit_ok_available=(sum(bool(r.get('depth_fit_ok')) for r in c if r.get('depth_ft_per_px') is not None) /
                    sum(r.get('depth_ft_per_px') is not None for r in c)) if any(r.get('depth_ft_per_px') is not None for r in c) else None,
                legend_parser_counts=dict(Counter(r.get('legend_measurement','missing') for r in c)),
                **{key:((sum(bool(r.get(key)) for r in c)/len(c)) if c else None) if key=='depth_fit_ok' else mean(c,key) for key in ('execution_failure','failure_rate','assigned_share','depth_fit_ok',
                    'legend_entries_found','legend_entries_named','legend_named_share','short_bed_rate',
                    'beds_per_1000px','oversegmentation_rate')},
                scan_failure_rate=mean(s,'failure_rate'),scan_ari=mean(s,'scan_ari'),
                scan_support=mean(s,'scan_common_assigned_share'),
                scan_eligible=sum(bool(r.get('scan_eligible')) for r in s),
                scan_multiclass=sum(bool(r.get('scan_eligible')) and r.get('scan_clusters_a',0)>1
                                    and r.get('scan_clusters_b',0)>1 for r in s),
                scan_multiclass_ari=mean([r for r in s if r.get('scan_clusters_a',0)>1
                                         and r.get('scan_clusters_b',0)>1],'scan_ari'),
                denominators={run:{k:v for k,v in counts_by.get((run,family,strategy),{}).items()
                                   if k not in ('family','strategy')} for run in ('clean','medium')}))
            for run,group in (('clean',c),('medium',s)):
                d=totals[-1]['denominators'][run]
                assert d.get('attempted',0)==len(group),(run,family,strategy,'attempted')
                assert d.get('execution_failures',0)==sum(bool(r.get('execution_failure')) for r in group),(run,family,strategy)
    lines=['# Unlabelled TEST results','',
           'This set has no rock truth. These checks cannot establish rock-type accuracy. '
           'Never train or tune on the set, predictions or diagnostics.','',
           f"The frozen set has {len(items)} pages and windows, {len({r['well'] for r in items})} well-name groups, "
           f"and {len({r['source_url'] for r in manifest})} source URLs. "
           'Pages from one well are related evidence. Well names are grouping proxies.','',
           '| Layout guess | Pages | Well-name groups |','|---|---:|---:|']
    for family in families:
        rows=[r for r in manifest if r['family']==family]
        lines.append(f"| {family} | {len(rows)} | {len({r['well'] for r in rows})} |")
    lines+=['','No outside source is a confirmed score-3 match. Gulf pages include historic onshore logs, '
            'an offshore Canadian log and vendor samples. Much of MC765 is geothermal colour-grid material. '
            'Geoservices includes four Norwegian pages with its printed key and four DHI analogues. '
            'The seven MC761 mineral-volume pages are a separate stress subset.','',
            '## Clean output and scan stability','',
            'Failure includes execution errors and empty named output. Coverage counts named, assigned rows. '
            'Scan agreement uses only rows assigned on both copies. ARI 1 can also mean one constant class. '
            'The multiclass column excludes pairs with only one class on either side. '
            'Numeric depth values are unavailable when calibration fails. Depth check % (all) counts a missing fit as a failed check; '
            'Depth check % (available) uses only pages with a fit. Other means omit missing values. '
            'Legend means use observed pages only; the count in parentheses gives that support. '
            'Available shared-parser evidence remains when preprocessing fails before the strategy runs.','',
            '| Strategy | Failure % | Coverage % | Depth check % (all) | Depth check % (available, pages) | Named / found keys (pages) | Short beds % | Beds / 1,000 px | Redundant boundaries % | Scan ARI | Eligible scan pairs | Multiclass scan ARI (pairs) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in [r for r in totals if r['family']=='All']:
        lines.append(f"| {r['strategy']} | {num(r['failure_rate'],100,1)} | {num(r['assigned_share'],100,1)} | "
            f"{num(r['depth_fit_ok'],100,1)} | {num(r['depth_fit_ok_available'],100,1)} ({r['depth_observed']}) | {num(r['legend_entries_named'],digits=1)} / {num(r['legend_entries_found'],digits=1)} ({r['legend_observed']}) | "
            f"{num(r['short_bed_rate'],100,1)} | {num(r['beds_per_1000px'],digits=1)} | {num(r['oversegmentation_rate'],100,1)} | "
            f"{num(r['scan_ari'])} | {r['scan_eligible']}/{r['pages']} | {num(r['scan_multiclass_ari'])} ({r['scan_multiclass']}) |")
    lines+=['','## Denominators (clean / medium scan)','',
            'Every attempted page stays in Failure %. Other means skip missing values. This table gives their support. '
            'Numeric depth values are unavailable when calibration fails. Depth check % (all) counts a missing fit as a failed check; '
            'Depth check % (available) uses only pages with a fit. Other means omit missing values. These are TEST diagnostics, '
            'not an accuracy result. Per-family detail: `FAILURE_AUDIT.md`. Per-case flags, error category and '
            'availability: `runs/*/derived/corrected_records.csv`.','',
            '| Strategy | Attempted | Execution failures (category) | Completed | Empty | Scorable | Depth available |',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in [r for r in totals if r['family']=='All']:
        d=r['denominators']
        cell=lambda k:f"{d['clean'].get(k,0)} / {d['medium'].get(k,0)}"
        cats=' / '.join(', '.join(f'{k} {v}' for k,v in d[run].get('failure_classes',{}).items()) or 'none'
                        for run in ('clean','medium'))
        lines.append(f"| {r['strategy']} | {cell('attempted')} | {cell('execution_failures')} ({cats}) | "
                     f"{cell('completed')} | {cell('completed_empty')} | {cell('scorable_non_empty')} | {cell('depth_available')} |")
    lines+=['','## Per-family checks','',
            '| Layout guess | Strategy | Failure % | Coverage % | Depth check % (all) | Depth check % (available, pages) | Named / found keys (pages) | Short beds % | Beds / 1,000 px | Redundant boundaries % | Scan ARI (eligible pairs) |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in [r for r in totals if r['family']!='All']:
        lines.append(f"| {r['family']} | {r['strategy']} | {num(r['failure_rate'],100,1)} | {num(r['assigned_share'],100,1)} | "
            f"{num(r['depth_fit_ok'],100,1)} | {num(r['depth_fit_ok_available'],100,1)} ({r['depth_observed']}) | {num(r['legend_entries_named'],digits=1)} / {num(r['legend_entries_found'],digits=1)} ({r['legend_observed']}) | "
            f"{num(r['short_bed_rate'],100,1)} | {num(r['beds_per_1000px'],digits=1)} | {num(r['oversegmentation_rate'],100,1)} | "
            f"{num(r['scan_ari'])} ({r['scan_eligible']}/{r['pages']}) |")
    lines+=['','## Agreement with s19','',
            'ARI compares partitions and ignores class names. It is not a rock-name accuracy score. '
            'Missing or failed pairs are excluded from its mean; the eligible count gives its support.','',
            '| Layout guess | Pair | Mean ARI | Eligible / pages | Common assigned % | Multiclass ARI (pairs) |',
            '|---|---|---:|---:|---:|---:|']
    for r in clean['agreement_by_family']:
        group=[q for q in clean['agreement'] if q['family']==r['family'] and q['strategy_b']==r['strategy_b']
               and q['eligible'] and q['clusters_a']>1 and q['clusters_b']>1]
        lines.append(f"| {r['family']} | {r['strategy_a']} / {r['strategy_b']} | {num(r['ari'])} | "
            f"{r['eligible']}/{r['pages']} | {num(r['common_assigned_share'],100,1)} | {num(mean(group,'ari'))} ({len(group)}) |")
    lines+=['','## Execution problems','',
            '| Copy | Strategy | Error | Pages |','|---|---|---|---:|']
    errors=[]
    lines[-2:]=['| Copy | Strategy | Category | Error | Pages |','|---|---|---|---|---:|']
    for name,run_name,rows in [('clean','clean',cr),('medium scan','medium',sr)]:
        counts=Counter((r['strategy'],category_of[(run_name,r['strategy'],r['log'],int(float(r['y0'])),int(float(r['y1'])))],
                        r.get('error','unknown')) for r in rows
                       if category_of[(run_name,r['strategy'],r['log'],int(float(r['y0'])),int(float(r['y1'])))]!='none')
        assert sum(counts.values())==sum(bool(r['failure']) for r in rows),name
        for (strategy,category,error),count in counts.most_common():
            lines.append(f"| {name} | {strategy} | {category} | {str(error).replace('|',' / ').replace(chr(10),' ')[:400]} | {count} |")
            errors.append(dict(copy=name,strategy=strategy,category=category,error=error,pages=count))
    if not errors:lines.append('| Both | All four | none | No execution error | 0 |')
    lines+=['','## Limits and checks','',
        '- The depth check requires a positive, plausible slope, at least three anchors and at most 16 px residual. '
        'Missing fits and two-anchor fits do not pass.',
        '- The shared parser is measured for s11, s17 and s22. The runner captures s19’s actual v3 legend and adjusted depth. '
        'If shared preprocessing fails first, only its available evidence remains.',
        '- Short beds are under eight pixels. Redundant same-name boundaries are an over-segmentation proxy. '
        'Neither measure gives a true bed error rate.',
        '- The fixed medium scanify preset adds blur, noise, fade, colour change and JPEG loss. '
        'It does not change geometry, apply shear or rescale the page.',
        '- Family means weight pages equally. Coverage, agreement and a readable key can all be high for a wrong rock assignment.',
        '- All image regions were reviewed before inference. The source-page, crop, rotation and key placement are recorded in each annotation.',
        '- Source reuse rights remain limited by the manifest licence notes. Public access alone is not a reuse licence.',
        '- Clean and scan inference file hashes match. The files were unchanged through both runs. No model or threshold was fitted here.',
        f'- Unit check: `python -X utf8 -m pytest harness/test_unlabelled_metrics.py -q` — `{unit_summary}`.',
        f'- Combined check: `{combined_summary}`.','',
        '## Commands and files','',
        'The harness uses plain Python. Downloaded image processing uses `-I` with explicit paths. '
        'The runs use two workers with one compute thread each and fixed model settings. Deterministic depth-fit ValueErrors '
        'are cached for the exact image and annotation. All new caches stay in this lane.','',
        'From `bakeoff/`, the actual runner is:','',
        '```powershell',
        'python -X utf8 notes/round6_local/unlab_curate/run_suite.py ../external_unlabelled_test notes/round6_local/unlab_curate/runs/clean --workers 2',
        'python -I -X utf8 notes/round6_local/unlab_curate/make_degraded.py .. ../external_unlabelled_test',
        'python -X utf8 notes/round6_local/unlab_curate/run_suite.py ../external_unlabelled_test/degraded notes/round6_local/unlab_curate/runs/medium --workers 2 --clean-dir notes/round6_local/unlab_curate/runs/clean',
        '```','',
        'The registered suite is `unlab_test`, with `truth_mode=none`. The runner sets `LITHOLOGY_DATASET` '
        'and the three `BENCH_UNLAB_TEST_*` path variables, then calls `harness.bench.evaluate`.',
        '',
        '- Dataset: `external_unlabelled_test/`; flat gallery: `index.html`.',
        '- Page scores and rejection reasons: `page_scores.csv`, `selection_decisions.json`.',
        '- Box checks: `visual_review.json`, `set_validation.json`, `details/depth_zoom_*.jpg`.',
        '- Method and metric limits: `CURATION_METHOD.md`, `METRICS.md`.',
        '- Raw outputs: `runs/clean/`, `runs/medium/`.',
        '- Per-page and family tables: `unlabelled_metrics.csv`, `unlabelled_by_family.csv` in each run.',
        '- Strategy comparisons: `strategy_agreement.csv`, `strategy_agreement_by_family.csv`.',
        '- Full machine-readable diagnostics: `unlabelled_report.json` in each run.',
        '- Bench backup: `bakeoff/harness/bench.py.bak_r6`. No commit or push was made.']
    if late_path.exists():
        late=read(late_path);lines+=['','## Late source check','',json.dumps(late,ensure_ascii=False,indent=2)]
    payload=dict(pages=len(items),families=contract['per_family'],totals=totals,errors=errors,
                 agreement_by_family=clean['agreement_by_family'])
    for name in ('result_summary.json','REPORT.md','comparison.html'): denominators.backup(a.lane/name)
    (a.lane/'result_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    (a.lane/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    template=(a.lane/'comparison_template.html').read_text(encoding='utf-8')
    template=template.replace('Depth checks passed (%)','Depth checks passed, all (%)').replace('<option value="scan_ari">','<option value="depth_fit_ok_available">Depth checks passed, available (%)</option><option value="scan_ari">')
    template=template.replace('`${r.pages} pages in the denominator`',"`${key==='depth_fit_ok_available'?r.depth_observed:r.pages} pages in the denominator`")
    template=template.replace("key==='depth_fit_ok'?","key.startsWith('depth_fit_ok')?")
    plot=dict(pages=len(items),families=contract['per_family'],totals=[
        {k:r[k] for k in ('family','strategy','pages','assigned_share','failure_rate','depth_fit_ok',
                         'depth_fit_ok_available','depth_observed','scan_ari','scan_eligible','scan_multiclass','denominators')} for r in totals])
    plot_json=json.dumps(plot,allow_nan=False).replace('<','\\u003c')
    (a.lane/'comparison.html').write_text(template.replace('__RESULT_DATA__',plot_json),encoding='utf-8')
    print('Report written:',len(items),'pages;',len(cr)+len(sr),'runs; execution errors',sum(e['pages'] for e in errors),flush=True)


if __name__=='__main__':main()
