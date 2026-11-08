"""Build a portable static dashboard: python -m harness.report."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import io

def build(out_root=None,report_dir=None):
    out_root=Path(out_root or io.ROOT/'out');report_dir=Path(report_dir or io.ROOT/'report')
    report_dir.mkdir(parents=True,exist_ok=True)
    datasets=[]
    paths=sorted(out_root.glob('*/metrics.json'))
    for path in paths:
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            data['_path']=path;datasets.append(data)
        except (ValueError,OSError):continue
    def esc(value):return html.escape(str(value),quote=True)
    def rel(path):return quote(Path(os.path.relpath(path,report_dir)).as_posix(),safe='/')
    def label(data):return str(data.get('strategy',data['_path'].parent.name))+(' (synthetic)' if data.get('set')=='synthetic' else '')
    def mean(data):return data.get('mean',data.get('summary',{}))
    datasets.sort(key=lambda d:mean(d).get('score') if mean(d).get('score') is not None else -1,reverse=True)
    fields=[('Strategy',None),('Score','score'),('row_acc','row_acc'),('boundary_f1','boundary_f1'),('cluster_ari','cluster_ari'),('depth_mae_ft','depth_mae_ft'),('s/window','seconds'),('Peak RAM (GB)','peak_rss_gb'),('Peak GPU (GB)','peak_gpu_gb'),('API calls','api_calls'),('Crashes','crashes')]
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lithology bake-off</title>',
           '<style>body{font:15px Arial,sans-serif;background:white;color:#222;margin:28px}h1,h2,h3{font-weight:500}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}.gallery{display:flex;gap:18px;overflow-x:auto;align-items:flex-start}figure{margin:0;flex:0 0 540px}figure img{width:100%;height:auto;border:1px solid #ddd}figcaption{margin:8px 0}a{color:#245a96}.error{color:#b22}section{margin:36px 0}</style>',
           '<h1>Lithology bake-off</h1><p>Scores average windows with available truth. Timing measures the strategy call. Shared OCR preparation is excluded. Memory values average per-window process peaks; GPU memory measures the PyTorch allocator. Synthetic metrics are saved separately under each strategy.</p>',
           '<table><thead><tr>'+''.join('<th>'+esc(k)+'</th>' for k,_ in fields)+'</tr></thead><tbody>']
    for data in datasets:
        values=mean(data);cells=[]
        for _,key in fields:
            value=label(data) if key is None else values.get(key)
            if value is None:rendered='—'
            elif isinstance(value,(int,float)):rendered=f'{value:.4g}'
            else:rendered=esc(value)
            cells.append('<td>'+rendered+'</td>')
        parts.append('<tr>'+''.join(cells)+'</tr>')
    parts.append('</tbody></table>')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for data in datasets:
        m=mean(data);seconds=m.get('seconds');value=m.get('score')
        if seconds is None or value is None:continue
        seconds=max(float(seconds),1e-7)
        ax.scatter(seconds,value,s=45)
        ax.annotate(label(data),(seconds,value),xytext=(5,6),textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.set_xlabel('Seconds per window (log scale)');ax.set_ylabel('Score');ax.set_ylim(-.03,1.08)
    ax.grid(True,color='#e5e5e5');fig.savefig(report_dir/'score_vs_seconds.png',dpi=150);plt.close(fig)
    parts.append('<section><h2>Accuracy and time</h2><img src="score_vs_seconds.png" width="900" alt="Score versus seconds per window"></section>')
    galleries=defaultdict(list)
    for data in datasets:
        logs=data.get('logs',data.get('per_log',[]))
        if isinstance(logs,dict):logs=[dict(value,log=key,key=key) for key,value in logs.items()]
        for record in logs:
            key=record.get('key',record.get('log','unknown'))
            galleries[key].append((data,record))
    for key,items in sorted(galleries.items()):
        parts.append('<section><h2>'+esc(key)+'</h2><div class="gallery">')
        for data,record in items:
            png=data['_path'].parent/record.get('png',key+'.png')
            parts.append('<figure><figcaption>'+esc(label(data)))
            if record.get('score') is not None:parts.append(' · score '+f'{record["score"]:.4f}')
            if not record.get('scored',True):parts.append(' · no GT')
            parts.append('</figcaption>')
            if png.exists():parts.append(f'<a href="{esc(rel(png))}"><img loading="lazy" src="{esc(rel(png))}" alt="{esc(label(data)+": "+key)}"></a>')
            if record.get('error'):parts.append('<p class="error">'+esc(record['error'])+'</p>')
            parts.append('</figure>')
        parts.append('</div></section>')
    parts.append('</html>')
    (report_dir/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    return report_dir/'index.html'

if __name__=='__main__':print(build())
