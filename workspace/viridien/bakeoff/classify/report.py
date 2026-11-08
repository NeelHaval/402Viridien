"""Self-contained mobile report plus exportable figures."""
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from classify.data import HERE, OUT, MANIFEST, font

NAMES=['colour','fft_hatch','dinov2','siglip','cnn_synth','jev_choice','jev_gallery','terra','hybrid']


def esc(v):return html.escape(str(v),quote=True)


def pct(v):return '—' if pd.isna(v) else f'{v*100:.1f}%'


def number(v):return '—' if pd.isna(v) else f'{v:.3f}'


def confidence(row):
    return pct(row.get('top1_ci_low'))+'–'+pct(row.get('top1_ci_high'))


def embed(image):
    image=Image.open(image) if isinstance(image,(str,Path)) else image
    b=io.BytesIO();image.convert('RGB').save(b,format='WEBP',quality=78,method=4)
    return 'data:image/webp;base64,'+base64.b64encode(b.getvalue()).decode('ascii')


def heatmap(summary):
    sources=[s for s in ['SAMPLE2','TRUE','CONSENSUS'] if s in summary.source.values]
    height=sum(max(1,summary[summary.source==s].family.nunique())*.8+1.1 for s in sources)
    fig,axes=plt.subplots(len(sources),1,figsize=(13,max(3,height)),squeeze=False,
                          gridspec_kw={'height_ratios':[max(1,summary[summary.source==s].family.nunique())+1 for s in sources]},layout='constrained')
    for ax,source in zip(axes[:,0],sources):
        sub=summary[summary.source==source]
        families=list(dict.fromkeys(sub.family))
        values=np.full((len(families),len(NAMES)),np.nan)
        ns=np.zeros_like(values,dtype=int)
        intervals={}
        for _,r in sub.iterrows():
            y=families.index(r.family);x=NAMES.index(r.classifier)
            values[y,x]=r.top1;ns[y,x]=r.n_patches
            intervals[y,x]=confidence(r)
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#ededed')
        ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap=cmap,aspect='auto')
        ax.set_xticks(range(len(NAMES)),NAMES,rotation=30,ha='right',fontsize=9)
        ax.set_yticks(range(len(families)),families,fontsize=9)
        ax.set_title(source+' · top-1 / 95% Wilson CI / eligible n',loc='left',fontsize=12)
        for y in range(len(families)):
            for x in range(len(NAMES)):
                val=values[y,x]
                text='NA\nn='+str(ns[y,x]) if np.isnan(val) else f'{val:.0%}\n{intervals[y,x]}\nn={ns[y,x]}'
                ax.text(x,y,text,ha='center',va='center',fontsize=7,color='white' if val>.65 else '#222')
    path=OUT/'heatmap.png';fig.savefig(path,dpi=150);plt.close(fig)
    return path


def selected_examples(case,df):
    ids=[p['patch_id'] for p in case['patches']]
    # Give paired wins/losses priority, then cover distinct truth labels.
    scores={pid:0 for pid in ids}
    for pid,g in df.groupby('patch_id'):
        if pid not in scores:continue
        by={r.classifier:r for _,r in g.iterrows()}
        j=by.get('jev_choice');local=by.get('fft_hatch')
        if j is not None and local is not None and j.predicted and local.predicted:
            scores[pid]=3 if j.correct!=local.correct else 1
    chosen=[];labels=set()
    for p in sorted(case['patches'],key=lambda p:(-scores[p['patch_id']],p['patch_y0'])):
        if p['lithology'] not in labels:
            chosen.append(p);labels.add(p['lithology'])
            if len(chosen)==6:break
    for pid in sorted(ids,key=lambda p:(-scores[p],ids.index(p))):
        if len(chosen)>=6:break
        if pid not in [p['patch_id'] for p in chosen]:
            chosen.append(next(p for p in case['patches'] if p['patch_id']==pid))
    return chosen


def wrap(text,width=55):
    import textwrap
    return '\n'.join(textwrap.wrap(str(text),width=width) or [''])


def examples(case,df):
    chosen=selected_examples(case,df)
    canvas=Image.new('RGB',(1120,max(1,(len(chosen)+1)//2)*505+45),'white')
    d=ImageDraw.Draw(canvas)
    d.text((12,10),case['info']['source']+' / '+case['info']['family'],fill='black',font=font(20))
    cards=[]
    for i,p in enumerate(chosen):
        x,y=(i%2)*560,(i//2)*505+45
        d.rectangle((x+5,y+5,x+550,y+495),outline='#ddd')
        d.multiline_text((x+15,y+12),wrap(f"Truth: {p['lithology']} | rows {p['patch_y0']}:{p['patch_y1']}",52),font=font(16),fill='black')
        im=Image.fromarray(p['rgb']);im.thumbnail((520,155),Image.Resampling.NEAREST)
        canvas.paste(im,(x+15,y+65))
        g=df[df.patch_id==p['patch_id']]
        guesses=[]
        for j,name in enumerate(NAMES):
            row=g[g.classifier==name]
            row=row.iloc[0] if len(row) else None
            guess=row.predicted if row is not None and row.predicted else 'unavailable'
            correct=bool(row.correct) if row is not None else False
            color='#b62222' if guess!='unavailable' and not correct else '#222'
            d.text((x+15,y+230+j*28),f'{name}: {guess}'[:65],font=font(14),fill=color)
            guesses.append(f'<li class="{"wrong" if color=="#b62222" else ""}"><b>{esc(name)}</b>: {esc(guess)}</li>')
        cards.append('<article><p><b>Truth: '+esc(p['lithology'])+'</b><br>Rows '+str(p['patch_y0'])+'–'+str(p['patch_y1'])+
                     '</p><img class="patch" src="'+embed(Image.fromarray(p['rgb']))+'" alt="Lithology patch"><ul>'+''.join(guesses)+'</ul></article>')
    path=OUT/('examples_'+case['info']['key']+'.png');canvas.save(path)
    return path,''.join(cards)


def findings(df,summary,policy):
    statements=[]
    local=policy.get('local','fft_hatch')
    for (source,family),g in df.groupby(['source','family'],sort=False):
        base=g[g.classifier==local].set_index('patch_id')
        for name in ['jev_choice','jev_gallery']:
            j=g[g.classifier==name].set_index('patch_id')
            common=[pid for pid in base.index.intersection(j.index) if base.loc[pid].predicted and j.loc[pid].predicted]
            if not common:continue
            wins=[pid for pid in common if j.loc[pid].correct and not base.loc[pid].correct]
            losses=[pid for pid in common if base.loc[pid].correct and not j.loc[pid].correct]
            text=f'{source} / {family}: {name} fixes {len(wins)} and breaks {len(losses)} of {len(common)} paired patches versus {local}.'
            for word,ids in [('Fix',wins),('Loss',losses)]:
                if ids:
                    a=base.loc[ids[0]];b=j.loc[ids[0]]
                    text+=f' {word} at rows {a.patch_y0}–{a.patch_y1}: truth {a.truth}; local {a.predicted}; JEV {b.predicted}.'
            statements.append(text)
    if not statements:
        statements.append('No paired successful JEV/local predictions are available; a claim about JEV winning or losing would be unsupported.')
    return statements


def build(df,summary,cases,metadata):
    heat=heatmap(summary)
    policy=metadata['hybrid']
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Legend-family classification benchmark</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#202428;background:white;margin:0 auto;padding:18px;max-width:1100px}h1{font-size:27px;line-height:1.2}h2{font-size:23px;margin-top:36px}h3{font-size:19px}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:9px 6px;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:18px}article{border:1px solid #ddd;padding:12px;overflow-wrap:anywhere}article ul{padding-left:20px;font-size:14px}.wrong{color:#b62222}.patch{image-rendering:auto;max-height:170px;width:auto}small,.muted{color:#59616b}code{overflow-wrap:anywhere}details{border-top:1px solid #ddd;margin:14px 0}summary{padding:10px 0;cursor:pointer}a{color:#185c99}</style></head><body>',
        '<h1>Given this log’s legend, which rock is this patch?</h1>',
        '<p><b>SAMPLE2</b> leads this benchmark: fixed-seed sampling targets 60 intervals per family, covers as many logs as possible, caps each log at 15, and balances available tiers and interval-height strata. It excludes log06, intervals under 16 rows and intervals within 30 rows of either column edge. MC761 has only three logs, so its maximum is 45. Classifiers receive the complete native-scale interval.</p>',
        '<p>Reference readers independently label sheets of up to eight patches from one log, with 40 rows of context and that log’s original legend. Only identical, non-unknown Sol (high) and Terra (medium) answers are retained. Terra classification uses a fixed 25% per-family subsample (rounded down). Agreement favours easier patches and can retain shared mistakes; Terra is not independent of these reference labels.</p>',
        '<p><b>TRUE</b> retains the organisers’ 17 log06 patches. <b>CONSENSUS</b> retains the original window-based Sol/Terra agreements. These older sources use central crops up to 160 rows, a 12-row minimum and a 40-patch window cap. Each source is scored separately.</p>',
        '<p>Hardware: '+esc(metadata['gpu'] or metadata['device'])+'. All scores are per interval patch, not per pixel row or depth. Small samples and repeated patterns within a single log limit generalisation.</p>',
        '<p><b>Completion:</b> '+esc(json.dumps(metadata['state'],ensure_ascii=False))+'</p>',
        '<h2>SAMPLE2 reference coverage and agreement</h2>']
    plan=metadata.get('sample2_plan',{}).get('families',[])
    agreement={r['family']:r for r in metadata.get('sample2_agreement',{}).get('families',[])}
    if plan:
        parts.append('<p>Reference pass: '+esc(metadata.get('sample2_status',{}).get('status','not run'))+'. Planned counts are before agreement filtering.</p>')
        sampling=metadata['sample2_plan']
        if sampling.get('reader_call_shortfall',0):
            parts.append('<p>The plan needs '+str(sampling['reader_calls'])+' reader calls and exceeds the remaining shared quota by '+str(sampling['reader_call_shortfall'])+'. The call cap will leave partial reference coverage.</p>')
        parts.append('<div class="scroll"><table><tr><th>Family</th><th>Planned n</th><th>Logs</th><th>Paired n</th><th>Agreed n</th><th>Agreement</th><th>Agreed logs</th></tr>')
        for item in plan:
            stat=agreement.get(item['family'],{})
            parts.append('<tr><td>'+esc(item['family'])+f'</td><td>{item["n"]}</td><td>{item["logs_covered"]}</td><td>{stat.get("n_paired",0)}</td><td>{stat.get("n_agreed",0)}</td><td>'+pct(stat.get('agreement_rate'))+f'</td><td>{stat.get("logs_covered",0)}</td></tr>')
        parts.append('</table></div><p>Agreement is agreed / completed reader pairs; pending calls are excluded from that denominator.</p>')
    else:
        parts.append('<p>SAMPLE2 has not been prepared or labelled.</p>')
    parts.append('<h2>Which classifier per family?</h2><p>The highest top-1 wins only when its 95% Wilson lower bound exceeds the runner-up’s top-1; otherwise tie. Terra uses fewer patches, so compare its n and logs. Patch intervals do not account for within-log correlation, and this selection rule is descriptive.</p>')
    parts.append('<div class="scroll"><table><tr><th>Family</th><th>Choice</th><th>Best observed</th><th>n</th><th>Logs</th><th>Top-1</th><th>95% CI</th><th>Runner-up</th></tr>')
    for item in metadata.get('family_choices',[]):
        parts.append('<tr><td>'+esc(item['family'])+'</td><td>'+esc(item['classifier'])+'</td><td>'+esc(item['best'] or '—')+
            f'</td><td>{item["n"]}</td><td>{item["logs_covered"]}</td><td>'+pct(item['top1'])+'</td><td>'+pct(item['ci_low'])+'–'+pct(item['ci_high'])+'</td><td>'+esc(item['runner_up'] or '—')+'</td></tr>')
    parts.extend(['</table></div>','<h2>Family × classifier</h2><p>Top-1 includes a 95% Wilson interval; n is the eligible patch count. Grey / NA means unavailable.</p>',
        '<img src="'+embed(heat)+'" alt="Top-1, confidence intervals and n; SAMPLE2 first">'])
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    for source in ['SAMPLE2','TRUE','CONSENSUS']:
        parts.append('<h2>'+source+'</h2>')
        sub=summary[summary.source==source]
        if sub.empty:parts.append('<p>No eligible ground truth was available for this source.</p>')
        for family,g in sub.groupby('family',sort=False):
            parts.append('<h3>'+esc(family)+'</h3><div class="scroll"><table><tr><th>Classifier</th><th>Top-1</th><th>95% CI</th><th>Top-3</th><th>Macro-F1</th><th>n</th><th>Logs</th><th>Returned</th><th>s/patch</th><th>Calls</th></tr>')
            for _,r in g.iterrows():
                parts.append('<tr><td>'+esc(r.classifier)+'</td><td>'+pct(r.top1)+'</td><td>'+confidence(r)+'</td><td>'+pct(r.top3)+'</td><td>'+pct(r.macro_f1)+
                             f'</td><td>{r.n_patches}</td><td>{r.get("logs_covered",0)}</td><td>{r.n_scored}</td><td>'+number(r.sec_per_patch)+'</td><td>'+number(r.total_calls)+'</td></tr>')
            parts.append('</table></div>')
        if source=='CONSENSUS':
            present={c['info']['key'] for c in cases if c['info']['source']==source}
            absent=[v['family'] for k,v in manifest.items() if k not in present]
            if absent:parts.append('<p>Families without usable consensus in this run: '+esc('; '.join(absent))+'.</p>')
    parts.append('<h2>JEV compared with local methods</h2>')
    for text in findings(df,summary,policy):parts.append('<p>'+esc(text)+'</p>')
    parts.extend(['<p>Interpretation: nearest-colour methods discard hatch structure, so monochrome patterns can tie. FFT compares directional frequency and ink density; embeddings compare learned visual similarity. Fine-grained adjacent legend colours, scan colour shifts, narrow/full-width mixtures and OCR label or swatch errors can hurt all methods. The paired examples show errors; these mechanisms are hypotheses rather than causal measurements.</p>',
        '<h2>Confidence-gated hybrid</h2><p>'+esc(json.dumps(policy,ensure_ascii=False))+'</p>',
        '<div class="scroll"><table><tr><th>Source / family</th><th>Top-1</th><th>95% CI</th><th>Top-3</th><th>n</th><th>Sent to JEV</th><th>JEV used</th></tr>'])
    for _,r in summary[summary.classifier=='hybrid'].iterrows():
        parts.append('<tr><td>'+esc(r.source+' / '+r.family)+'</td><td>'+pct(r.top1)+'</td><td>'+confidence(r)+'</td><td>'+pct(r.top3)+f'</td><td>{r.n_patches}</td><td>'+pct(r.share_sent_jev)+'</td><td>'+pct(r.share_jev_used)+'</td></tr>')
    parts.extend(['</table></div><p>The best local classifier and a single top-two probability margin threshold are selected on log06 TRUE only: maximise top-1, break ties by fewer JEV calls. JEV choice is fixed in advance. The policy is frozen for CONSENSUS. Hybrid scores use cached standalone outputs and incur no extra benchmark requests; latency and calls estimate a sequential deployment. Missing JEV responses retain the local answer. Threshold transfer between families is unvalidated.</p>',
        '<h2>Calls, latency and memory</h2><p>'+esc(metadata['monetary_cost'])+'</p><p>Actual API requests: '+esc(json.dumps(metadata['actual_calls']))+'. Reported token usage: '+esc(json.dumps(metadata['usage']))+'. Local classifiers have no paid API calls; electricity/hardware costs are not measured.</p>',
        '<p>'+esc(metadata['timing'])+'</p><p>Unavailable API timings are time spent on failed attempts and skipped requests, not measured model inference latency. CLI invocations count toward the Terra cap even when startup fails; successful remote requests and actual charges cannot be inferred from those invocation counts.</p><p>'+esc(metadata['memory'])+'</p>',
        '<div class="scroll"><table><tr><th>Classifier</th><th>s/patch</th><th>Calls/patch</th><th>Peak RSS GB</th><th>GPU GB</th></tr>'])
    for name in NAMES:
        g=df[df.classifier==name]
        if g.empty:continue
        parts.append(f'<tr><td>{name}</td><td>{g.seconds.mean():.3f}</td><td>{g.calls.mean():.3f}</td><td>{g.peak_rss_gb.max():.2f}</td><td>{g.peak_gpu_gb.max():.2f}</td></tr>')
    parts.append('</table></div><h2>Examples</h2><p>Up to six distinct patches per window. Red guesses are wrong; unavailable methods are marked. Expand to see readable patch cards.</p>')
    for case in cases:
        _,cards=examples(case,df[df.key==case['info']['key']])
        parts.append('<details><summary>'+esc(case['info']['source']+' / '+case['info']['family'])+'</summary><div class="grid">'+cards+'</div></details>')
    available={c['info']['key'] for c in cases if c['info']['source']=='CONSENSUS'}
    for key,item in manifest.items():
        if key in available:continue
        placeholder=Image.new('RGB',(1000,160),'white');draw=ImageDraw.Draw(placeholder)
        draw.text((20,20),'CONSENSUS / '+item['family'],font=font(20),fill='black')
        draw.text((20,65),'n=0: no agreed consensus intervals were supplied.',font=font(19),fill='#555')
        draw.text((20,105),'Six labelled examples cannot be produced without ground truth.',font=font(19),fill='#555')
        placeholder.save(OUT/('examples_'+key+'.png'))
    parts.extend(['<h2>What this measures</h2>',
        '<p>Classifiers receive interval crops with boundaries already supplied. This does not measure segmentation, boundary detection or depth calibration. No real-image training labels are used by the local models: the CNN is trained only on synthetic augmentations of this log’s own swatches, with S7’s 20-second training limit. DINOv2 uses cosine similarity; SigLIP uses S6’s equal visual/text score blend. Colour uses S1’s dominant colour in Lab; FFT uses S11’s hatch and colour descriptor over tiles spanning the patch.</p>',
        '<p>JEV choice uses original legend rock-entry strips beside one query patch. JEV gallery uses labelled cropped swatches beside the same patch. Each makes one choice question and ranks the returned probabilities, with no network retries. When the parser finds no rocks, VLMs use vocab.json and the original legend crop; swatch-based local methods are unavailable. Terra uses gpt-6.1-terra at low effort, eight numbered patches per image, at most 60 invocations in total. Every API response and failure is cached; repeated transport failures stop further calls.</p>',
        '<p>'+esc(metadata['metrics'])+'</p>',
        '<p>TRUE is only one monochrome family, with repeated hatch motifs and 17 correlated intervals. The organisers’ “Sandstone” is matched to the legend’s “Sand and Sandstone” via the harness scorer alias. CONSENSUS is machine agreement, not an independent geologist’s ground truth; agreement filtering favours easier rows and can preserve shared model mistakes. The vocabulary comes from parsed legend entries, whose OCR can be wrong. Labels outside the legend vocabulary and short intervals are excluded and logged. Stratification changes natural class prevalence. Top-3 can be forgiving in small vocabularies; macro-F1 averages classes observed in truth. The hybrid’s TRUE score is fitted and optimistic. No result establishes accuracy on unseen wells.</p>',
        '<h2>Input audit</h2>'])
    for case in cases:
        inf=case['info']
        parts.append('<details><summary>'+esc(inf['source']+' / '+inf['family'])+f' — {inf["n_selected"]} selected / {inf["n_accepted"]} eligible; {len(inf["excluded"])} excluded</summary><p>Vocabulary: '+esc(', '.join(inf['vocabulary']))+'</p><p>Log: <code>'+esc(inf['log'])+'</code></p><p>Exclusions: '+esc(json.dumps(inf['excluded'],ensure_ascii=False))+'</p><img src="'+embed(case['legend'])+'" alt="Original legend crop"></details>')
    if metadata.get('consensus_job',{}).get('error'):
        parts.append('<h2>Execution limits</h2><p>Upstream consensus job: '+esc(metadata['consensus_job']['error'])+'</p>')
    if metadata.get('edge_check'):
        parts.append('<p>Headless Edge render check: '+esc(json.dumps(metadata['edge_check'],ensure_ascii=False))+'</p>')
    parts.append('<p class="muted">Generated '+esc(metadata['created'])+'. Data: out/results.csv; aggregates: out/summary.csv; raw requests: out/api_cache/; audit: out/calls.jsonl and out/run.json.</p></body></html>')
    path=HERE/'report.html';path.write_text('\n'.join(parts),encoding='utf-8')
    assert path.stat().st_size<15_000_000, 'Report exceeds 15 MB'
