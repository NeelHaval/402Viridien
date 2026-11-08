"""Standalone split-aware report for the live notebook, CPU replay and original harness."""
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from emil import build_report as b
from emil import pipeline as p
from emil.split_results import selected, membership
from harness import io
import numpy as np

OUT = HERE/'report'
MODES = ['detect', 'labels', 's11']
SPLITS = ['dev', 'test', 'dev_test']


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def optional(path): return read(path) if path.exists() else None


def metric_row(result):
    h, e = result['harness'], result['emil']; leg=e['legend']['aggregate']
    seg=e['segmentation']['aggregate']; cla=e['classification']['aggregate']
    return dict(row_acc=h['row_acc'], boundary_f1=h['boundary_f1'], cluster_ari=h['cluster_ari'],
        failures=h['crashes'], n_windows=h['n_windows'],
        emil_coverage=seg['pixel_accuracy_macro'] if seg else None,
        emil_pixacc=cla['per_file_mean']['pixel_accuracy'] if cla else None,
        emil_segmentation_rejected=len(e['segmentation'].get('failed',{})),
        emil_classification_rejected=len(e['classification'].get('failed',{})),
        legend_precision=leg['bbox_precision_macro'], legend_recall=leg['bbox_recall_macro'],
        legend_text=leg['text_accuracy_macro'])


def runtime_stats(rows):
    return dict(n_calls=len(rows),mean_seconds=float(np.mean([r['seconds'] for r in rows])),
        median_seconds=float(np.median([r['seconds'] for r in rows])),
        p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),
        max_rss_gb=max(r['peak_rss_gb'] for r in rows),max_gpu_gb=max(r['peak_gpu_gb'] for r in rows),
        traced=False,device='cpu',ocr_cache='warm',scope='Inference only; external timings cover sampled windows')


def legend_stats(rr, metrics):
    unique={r['log']:r for r in rr if r['mode']=='detect'}
    rows=[metrics['emil']['legend']['per_file'][r['key']+'.json'] for r in unique.values()]
    def defined_mean(key):
        values=[x[key] for x in rows if x[key] is not None and np.isfinite(x[key])]
        return float(np.mean(values)) if values else None
    return dict(n_pages=len(unique),zero_detected=sum(r.get('n_detected',0)==0 for r in unique.values()),
        nonempty_rate=sum(r.get('n_detected',0)>0 for r in unique.values())/len(unique),
        mean_swatches=float(np.mean([r.get('n_detected',0) for r in unique.values()])),
        precision=defined_mean('bbox_precision'),
        precision_defined_pages=int(sum(x['bbox_precision'] is not None and np.isfinite(x['bbox_precision']) for x in rows)),
        recall=defined_mean('bbox_recall'),
        text_accuracy=defined_mean('text_accuracy'),
        aggregation='Unique physical pages, independent of window count')


def table_md(table):
    rows=['| Set | Method | Row acc | Boundary F1 | ARI | Failures | Emil coverage | Emil PixAcc |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for domain, modes in table.items():
        for mode,t in modes.items():
            fmt=lambda value: f'{value:.4f}' if value is not None else 'N/A'
            rows.append(f"| {domain} | {mode} | {t['row_acc']:.4f} | {t['boundary_f1']:.4f} | {t['cluster_ari']:.4f} | {t['failures']}/{t['n_windows']} | {fmt(t['emil_coverage'])} | {fmt(t['emil_pixacc'])} |")
    return '\n'.join(rows)


def main():
    OUT.mkdir(exist_ok=True)
    # Reporting must never overwrite dataset configuration.
    allowed={w['log'] for w in read(p.WORK/'external_tests/eval_set.json')}
    with (p.WORK/'external_tests/manifest.csv').open(encoding='utf-8',newline='') as f:
        manifest={r['id']:r for r in csv.DictReader(f) if r['id'] in allowed}
    records={d:read(HERE/'results_live'/d/'records.json') for d in ['viridien','external']}
    records['external']=selected(records['external'],'dev_test')
    metrics={'viridien':{m:read(HERE/'results_live/viridien'/m/'metrics.json') for m in MODES}}
    for split in SPLITS:
        metrics[split]={m:read(HERE/'results_live/external'/split/m/'metrics.json') for m in MODES}
    table={d:{m:metric_row(x) for m,x in mm.items()} for d,mm in metrics.items()}
    groups={'Viridien (2 pages)':{m:b.mean([r for r in records['viridien'] if r['mode']==m]) for m in MODES}}
    for split in SPLITS:
        groups[split.replace('_','+')]= {m:b.mean([r for r in selected(records['external'],split) if r['mode']==m]) for m in MODES}
    b.plot_sources(groups)
    sources=sorted({r['source'] for r in manifest.values()})
    source_groups={}
    for split in SPLITS:
        rr=selected(records['external'],split)
        present=sorted({manifest[r['log']]['source'] for r in rr})
        source_groups[split]={source:{m:b.mean([r for r in rr if r['mode']==m and manifest[r['log']]['source']==source]) for m in MODES} for source in present}
        b.plot_sources(source_groups[split],f'per_source_{split}.png',f'External {split.replace("_","+")}: macro means over sampled windows\nFair detect, oracle labels and s11; same scorer, depth OCR bypassed')
    # Diagnostic selection: two distinct pages per source, both labelled Viridien pages.
    chosen=[('viridien',r) for r in records['viridien'] if r['mode']=='detect']
    oracle={r['key']:r for r in records['external'] if r['mode']=='labels'}
    for source in sources:
        candidates=[r for r in records['external'] if r['mode']=='detect' and manifest[r['log']]['source']==source]
        first=max(candidates,key=lambda r:oracle[r['key']]['cluster_ari']-oracle[r['key']]['row_acc'])
        others=[r for r in candidates if manifest[r['log']]['well']!=manifest[first['log']]['well']]
        others=others or [r for r in candidates if r['log']!=first['log']]
        second=(max(others,key=lambda r:oracle[r['key']]['cluster_ari']-oracle[r['key']]['row_acc'])
                if source=='nl' else min(others,key=lambda r:oracle[r['key']]['boundary_f1']))
        chosen.extend([('external',first),('external',second)])
    panels=[]; overlays=[]; seen=set(); lookup=membership()
    for domain,r in chosen:
        labels=read((p.WORK/'emil/labels' if domain=='viridien' else HERE/'external_labels')/(r['log']+'.json'))
        panel=b.panel(domain,r,labels); panel['split']=lookup.get(r['log'],'viridien');panels.append(panel)
        source=manifest[r['log']]['source'] if domain=='external' else r['log']
        if source not in seen:
            overlays.append(b.overlay(domain,r,labels)); seen.add(source)
    shared={s:read(io.ROOT/'out'/strategy/'external/metrics.json') for s,strategy in
        [('detect','s12_emil'),('labels','oracle_emil'),('s11','s11_fft_legend')]}
    shared_table={split:{m:shared[m]['splits'][split] for m in MODES} for split in SPLITS}
    gold=read(io.ROOT/'out/s12_emil/metrics.json')
    with (io.ROOT/'out/_rerun/summary.csv').open(encoding='utf-8',newline='') as f:
        gold_s11=next(r for r in csv.DictReader(f) if r['s']=='s11_fft_legend' and r['gt']=='gold')
    failed={r['log'] for r in shared['detect']['logs'] if 'error' in r}
    example=next(r for r in records['external'] if r['mode']=='detect' and r['log'] in failed)
    panel=b.panel('external',example,read(HERE/'external_labels'/(example['log']+'.json')),shared=True)
    panel['split']=lookup[example['log']];panels.append(panel)
    resources=read(HERE/'vm_resources.json')
    runtime={'viridien':{m:runtime_stats([r for r in resources if r['domain']=='viridien' and r['mode']==m]) for m in ['detect','labels']}}
    for split in SPLITS:
        rs=selected([r for r in resources if r['domain']=='external'],split)
        runtime[split]={m:runtime_stats([r for r in rs if r['mode']==m]) for m in ['detect','labels']}
    page_resources=[]
    for domain in ['viridien',*SPLITS]:
        rs=[r for r in resources if r['domain']=='viridien'] if domain=='viridien' else selected([r for r in resources if r['domain']=='external'],domain)
        for mode in ['detect','labels']:
            for name in sorted({r['log'] for r in rs}):
                rr=[r for r in rs if r['log']==name and r['mode']==mode]
                page_resources.append(dict(set=domain,log=name,mode=mode,n_windows=len(rr),
                    seconds=sum(r['seconds'] for r in rr),peak_rss_gb=max(r['peak_rss_gb'] for r in rr),
                    peak_gpu_gb=0.0,scope='Full column' if domain=='viridien' else 'Sum of evaluated windows; not entire original page'))
    with (OUT/'page_resources.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(page_resources[0]));writer.writeheader();writer.writerows(page_resources)
    runtime_pages={}
    for domain in runtime:
        runtime_pages[domain]={m:dict(mean_seconds=float(np.mean([r['seconds'] for r in page_resources if r['set']==domain and r['mode']==m])),
            n_pages=sum(r['set']==domain and r['mode']==m for r in page_resources)) for m in ['detect','labels']}
    legend={'viridien':legend_stats(records['viridien'],metrics['viridien']['detect'])}
    for split in SPLITS:legend[split]=legend_stats(selected(records['external'],split),metrics[split]['detect'])
    legend_sources={}
    for split in SPLITS:
        legend_sources[split]={}
        for source in source_groups[split]:
            rr=[r for r in selected(records['external'],split) if manifest[r['log']]['source']==source]
            stats=legend_stats(rr,metrics[split]['detect']); dims=[]
            for name in {r['log'] for r in rr}:
                dims.extend((e['box'][2]-e['box'][0],e['box'][3]-e['box'][1]) for e in read(HERE/'external_labels'/(name+'.json'))['legend'])
            stats.update(true_swatches=len(dims),size_and_aspect_eligible=sum(25<=w<=200 and 15<=h<=120 and .7<=w/h<=3.5 for w,h in dims))
            legend_sources[split][source]=stats
    reproduction={f.stem:read(f) for f in (HERE/'results_live/viridien/reproduction').glob('*.json')}
    for name,data in reproduction.items():
        data['modes']={r['mode']:dict(r,emil_classification=metrics['viridien'][r['mode']]['emil']['classification']['per_file'][name+'.json'],
            emil_legend=metrics['viridien'][r['mode']]['emil']['legend']['per_file'][name+'.json']) for r in records['viridien'] if r['log']==name}
    failures=[
      dict(finding='Most external legends yield no detected swatches, so detect mode cannot name the column.',example='ext_utah_4304150011_p153'),
      dict(finding='Two detected boxes on EmmaWard1 do not recover its true legend; OCR/naming still gives zero row accuracy.',example='ext_kgs_EmmaWard1_p002'),
      dict(finding='Oracle names do not solve texture confusion: flat-bay-2 has nearly correct partitions with wrong lithology names.',example='ext_nl_flat-bay-2_p078'),
      dict(finding='Bed clustering misses thin-layer boundaries. Hadrian has 680 labelled layers but only 47 Section-2 segments.',example='60808400140000_hadrian 2_masterlog_2in_10000-21962'),
      dict(finding='Shared depth calibration fails before lithology inference on five allowed windows; direct scoring exposes the separate inference result.',example=example['log'])]
    report=dict(model=dict(path=str(p.DEFAULT_MODEL.relative_to(p.WORK)),sha256=hashlib.sha256(p.DEFAULT_MODEL.read_bytes()).hexdigest()),
        source=read(HERE/'port_manifest.json'),table=table,direct_metrics=metrics,source_groups=source_groups,
        shared_table=shared_table,harness=shared,gold=dict(s12=gold,s11=gold_s11),reproduction=reproduction,
        legend_detection=legend,legend_by_source=legend_sources,runtime=runtime,runtime_per_page=runtime_pages,
        page_resources=page_resources,panels=panels,overlays=overlays,failure_modes=failures,
        panel_selection='Two diagnostic windows per source, distinct wells where available, plus both complete in-domain pages and a shared-preprocessing failure; not a random sample.',
        incomplete={m:optional(HERE/'results_live/incomplete'/m/'metrics.json') for m in ['detect','labels']},
        incomplete_records=read(HERE/'results_live/incomplete/records.json'),
        retrained_secondary={m:read(HERE/'results_retrained_live/viridien'/m/'metrics.json') for m in ['detect','labels']},
        training_secondary=read(HERE/'training.json'),validation=read(HERE/'validation.json'),
        acceptance=optional(HERE/'acceptance.json'),smoke=optional(HERE/'smoke_live.json'),
        reuse=optional(HERE/'vm_resume.json'),n_external_pages=len(manifest),n_external_wells=len({(r['source'],r['well']) for r in manifest.values()}),
        leakage=dict(section3='Default auto reads the GT legend for both 0.858 and true-segment 0.989 scores',
            section4='Saved live Section 4 follows explicit detect mode; 85.57% reproduced in both detect and labels',
            oneinch='Excluded by live TRAIN_PAGES rule and splits.test; historical main-checkpoint training manifest unavailable'),
        environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','scikit-learn','easyocr','Pillow']}),
        limitations=['Only two complete in-domain pages; external windows are correlated within wells.',
          'Removed verify wells were excluded throughout. No verify metrics are generated.',
          'Direct scores bypass shared depth OCR equally; shared tables retain recorded preprocessing failures.',
          'Oracle shared results combine CPU-replayed predictions with original preprocessing outcomes, not a fresh end-to-end depth-OCR run.',
          's11 predictions and detect shared-harness outcomes are reused. CSV-loading time is not s11 inference runtime.',
          'CPU replay uses exact cached OCR. EasyOCR weights are absent and DNS blocks downloading them; uncached gc743 detection is unavailable.',
          'Coverage PixAcc is not boundary accuracy; classification PixAcc includes background. ARI ignores class-name correctness.',
          'An empty prediction can have ARI=1 on a single-class window; boundary F1 is also 1 when both boundary sets are empty.',
          'Unchanged team segmentation/classification evaluators reject empty predictions. Their PixAcc means condition on accepted files; rejection counts are explicit. Harness means include all windows.',
          'Runtime uses CPU, six torch threads, warm OCR and no tracemalloc; original laptop timings are retained separately in source records.',
          'Both labelled pages are excluded by reconstructed training rules. Historical Drive checkpoint training metadata is unavailable.',
          'The existing retrain used 29 pages; current live selection differs as recorded in validation.json. It is a secondary check.'])
    b.dump(OUT/'report.json',report);b.dump(OUT/'failure_modes_live.json',failures)
    write_summary(report)
    summary=(OUT/'summary.md').read_text(encoding='utf-8')
    # A self-contained narrative and numeric tables are visible in HTML, not only linked.
    try:
        import markdown
        content=markdown.markdown(summary,extensions=['tables','fenced_code'])
    except ImportError:
        content=render_markdown(summary)
    images=['per_source.png',*[f'per_source_{s}.png' for s in SPLITS],*overlays,*[r['file'] for r in panels]]
    cards=''.join(f'<figure><a href="{html.escape(f)}"><img loading="lazy" src="{html.escape(f)}"></a><figcaption>{html.escape(f)}</figcaption></figure>' for f in images)
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Emil evaluation: dev and test</title><style>body{font:15px/1.5 Arial,sans-serif;color:#222;background:#fff;margin:28px auto;max-width:1300px;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}img{max-width:100%;height:auto}figure{margin:24px 0}a{color:#176494}</style><main>'+content+cards+'</main></html>',encoding='utf-8')
    print(json.dumps(dict(table=table,shared=shared_table,legend=legend,runtime=runtime,runtime_per_page=runtime_pages,panels=len(panels),overlays=len(overlays)),indent=2))


def render_markdown(source):
    """Small escaped fallback so the offline report still has real HTML tables."""
    def inline(line):
        line=html.escape(line)
        line=re.sub(r'`([^`]+)`',r'<code>\1</code>',line)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',line)
    lines=source.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].startswith('|'):
                block.append(lines[i]);i+=1
            out.append('<table>')
            for n,row in enumerate(block):
                if re.fullmatch(r'[|:\-\s]+',row):continue
                tag='th' if n==0 else 'td'
                out.append('<tr>'+''.join(f'<{tag}>{inline(c.strip())}</{tag}>' for c in row.strip('|').split('|'))+'</tr>')
            out.append('</table>');continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
        elif line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):
                out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        elif line=='{':
            block=[]
            while i<len(lines) and lines[i].strip():block.append(lines[i]);i+=1
            out.append('<pre>'+html.escape('\n'.join(block))+'</pre>');continue
        else:out.append('<p>'+inline(line)+'</p>')
        i+=1
    return '\n'.join(out)


def write_summary(r):
    lines=['# Emil live notebook evaluation: dev and test','',
      'Main checkpoint: `'+r['model']['path']+'`; SHA256 `'+r['model']['sha256']+'`.','',
      'Hadrian reproduces Emil’s 95 beds → 47 segments, row accuracy 0.858, 85.57% classification PixAcc in both detect and labels, 0.989 naming accuracy with true segments, and 94.44% detected-legend text accuracy. Section 3’s default auto loads the GT legend. Live Section 4 selects detect explicitly.','',
      '1inch tvd mudlog is in splits.test and excluded by the current TRAIN_PAGES rule. The historical main checkpoint has no training-page manifest. gc743 has only two labelled layers and is excluded from complete aggregates.','',
      '## Same-window direct results','',
      'Macro means over 2 complete in-domain columns, external dev (59 windows), test (20), and dev+test (79). All strategies bypass depth OCR. detect uses page pixels and cached OCR; labels uses true legend boxes/names.','',table_md(r['table']),'',
      'The unchanged team segmentation/classification evaluators reject empty predictions: external detect rejects 59/59 dev, 13/20 test, 72/79 combined. PixAcc is N/A with no accepted files; otherwise its mean uses accepted files only. Every harness metric still includes every window. Exact rejection counts and messages are retained in report.json.','',
      '## Shared harness including preprocessing failures','',
      's12-detect and s11 reuse completed original harness outcomes. s12-labels uses CPU-replayed identical predict() with those recorded preprocessing outcomes. Five available windows fail depth calibration; failure rows remain in all means.','',
      '| Split | Method | Row acc | Boundary F1 | ARI | Failures |','|---|---|---:|---:|---:|---:|']
    for split,mm in r['shared_table'].items():
        for m,t in mm.items(): lines.append(f"| {split} | {m} | {t['row_acc']:.4f} | {t['boundary_f1']:.4f} | {t['cluster_ari']:.4f} | {t['crashes']}/{t['n_windows']} |")
    g=r['gold']['s12']['mean'];s=r['gold']['s11']
    lines+=['',f"Gold log06, s12-detect: row {g['row_acc']:.4f}, F1 {g['boundary_f1']:.4f}, ARI {g['cluster_ari']:.4f}. Existing s11: row {float(s['row']):.4f}, F1 {float(s['bf1']):.4f}, ARI {float(s['ari']):.4f}. Only one gold page is scored; other eval pages are timing-only. No supplied true legend is available for log06 oracle scoring.",'',
      '## Complete-page reproduction','', '| Page | Method | Beds | Segments | Row acc | Boundary F1 | ARI | Emil PixAcc |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,d in r['reproduction'].items():
        for m,x in d['modes'].items():
            lines.append(f"| {name} | {m} | {x.get('n_beds','—')} | {x.get('n_segments','—')} | {x['row_acc']:.4f} | {x['boundary_f1']:.4f} | {x['cluster_ari']:.4f} | {x['emil_classification']['pixel_accuracy']:.4f} |")
        lines+=['',f"True-segment naming ({name}): `{d['true_segments']}`.",'']
    lines+=['## Legend detection','', '| Set | Pages with any detections | Box precision | Box recall | Text accuracy |','|---|---:|---:|---:|---:|']
    for d,t in r['legend_detection'].items():
        precision=f"{t['precision']:.4f}" if t['precision'] is not None else 'N/A'
        lines.append(f"| {d} | {t['n_pages']-t['zero_detected']}/{t['n_pages']} | {precision} | {t['recall']:.4f} | {t['text_accuracy']:.4f} |")
    lines+=['','Unique physical pages are counted once; precision conditions on pages with detections (undefined otherwise). Recall and text include zero detections. Window-weighted legend scores and per-source dev/test/combined breakdowns are in report.json. Swatch-size filter eligibility is diagnostic only.','',
      '## CPU runtime and memory','', '| Set | Method | Seconds / window | Seconds / page | Peak process GiB | GPU GiB |','|---|---|---:|---:|---:|---:|']
    for d,mm in r['runtime'].items():
        for m,t in mm.items():lines.append(f"| {d} | {m} | {t['mean_seconds']:.4f} | {r['runtime_per_page'][d][m]['mean_seconds']:.4f} | {t['max_rss_gb']:.3f} | 0 |")
    lines+=['','External page time is the sum of its sampled evaluation windows, not full original-page inference. CPU, six torch threads, warm content-addressed OCR, no tracemalloc. Times exclude shared depth/legend parsing and rendering. [Per-page timings](page_resources.csv).','',
      '## Secondary retrained model','',table_md({'retrained / 2 pages':{m:metric_row(x) for m,x in r['retrained_secondary'].items()}}),'',
      'Reused the completed 2000-step retrain (29 pages, NumPy/torch seed 0), checkpoint `'+r['training_secondary']['checkpoint_sha256']+'`; reran both modes on both complete pages. Training loss and selection remain in ../training.log and ../training.json. The current live selection differs; validation.json records the differences.','',
      '## Incomplete gc743','',
      'Only two layers (342 labelled rows), one with unknown/null legend ID and one numeric type. Oracle scoring is separate; uncached detect cannot run because EasyOCR weights are absent and downloads fail DNS. This is an environment failure, not a measured zero-quality prediction.','']
    for m,x in r['incomplete'].items():
        if x:lines.append(f"{m}: `{metric_row(x)}`")
    lines+=['','## Five observed failure modes','',*['- '+f['finding']+' Example: `'+f['example']+'`.' for f in r['failure_modes']],
      '', '## Reuse and verification','',json.dumps(r['reuse'],indent=2),'',
      'Evaluator files are byte-identical to the live Drive scripts; only CONFIG changes in memory. Data acceptance passed; the full existing test suite is not green. Harness: 9 passed. Full suite: 36 passed, 3 failed, 8 errors (missing original dev PDF and unchanged converter render-size guard). Details: ../acceptance.json, ../tests_live.log and ../tests_harness_vm.log.','',
      '## Limits','',*['- '+x for x in r['limitations']],'',
      '## Files','', '- [Full report and figures](index.html)', '- [All metrics and provenance](report.json)', '- [Overall comparison](per_source.png)',
      *['- [Per-source '+s.replace('_','+')+'](per_source_'+s+'.png)' for s in SPLITS],
      *['- ['+x['split']+' '+x['key']+']('+quote(x['file'])+')' for x in r['panels']],
      *['- [Legend overlay]('+quote(x)+')' for x in r['overlays']]]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
