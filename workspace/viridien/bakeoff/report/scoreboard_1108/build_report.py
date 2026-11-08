"""Build the offline 1108 score report. No inference or protected-split access."""
from __future__ import annotations
import argparse, copy, csv, hashlib, html, itertools, json, math, re, statistics, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
from report_inputs import validate_records, paired_case_rows, build_dev_map, validate_audit, digest, validate_prediction_bytes
from report_integrity import require_integrity_audit
from report_policy_gate import require_policy
from report_diagnostics import diagnostics, runtime_summary, runtime_table, native_legend_summary, native_table, native_legend_status
DEV_GROUPS={}
NAMES = {'s19_combine':'s19', 's20_v4':'v4', 's21_v4dino':'v4 + DINOv2 (baseline)', 's23_v4dino_identity':'DINOv2 · distinct labels'}
COLORS = ['#1674b8','#087c70','#b65417','#824b87']
METRICS = ['score','row_acc','boundary_f1','cluster_ari','seconds','depth_mae_ft','preprocess_seconds','end_to_end_seconds','task_wall_seconds']

def avg(values):
    clean=[float(x) for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    return statistics.mean(clean) if clean else None

def esc(x): return html.escape(str(x))
def num(x,d=3): return '—' if x is None else f'{x:.{d}f}'
def strategy(r): return r.get('strategy',r.get('strategy_id','unknown'))
def label(s): return NAMES.get(s,s)
def field(r,m): return r.get(m,r.get('metrics',{}).get(m))
QUALITY_METRICS=('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft')
def accepted_records(records):
    result=copy.deepcopy(records)
    for r in result:
        if not all(m in r.get('accepted_metrics',{}) for m in QUALITY_METRICS):raise ValueError('Missing accepted-policy primary metrics')
        r.update(r['accepted_metrics']);r['_grading_id']='clay_shale_equivalent_v1'
        if r.get('pixel_only'):
            if not all(m in r['pixel_only'].get('accepted_metrics',{}) for m in QUALITY_METRICS):raise ValueError('Missing accepted-policy pixel metrics')
            r['pixel_only'].update(r['pixel_only']['accepted_metrics'])
    return result

def graded_raster(rows,window,policy):
    from harness import score
    if policy=='clay_shale_equivalent_v1':
        from strategies.grading_policy import grading_rows
        rows=grading_rows(rows)
    return score.raster(rows,window)

def group_rows(rows,keys):
    out=defaultdict(list)
    for r in rows:out[tuple(r.get(k,'unknown') for k in keys)].append(r)
    return out

def well_group(r):
    original=r.get('well',r['log'])
    if r['log'] in DEV_GROUPS:return DEV_GROUPS[r['log']]
    found=re.match(r'(ext_(?:utah|nl|idaho)_.+?)_p[0-9]+(?:_.*)?$',r['log'])
    return found.group(1) if found else original

def log_means(rows):
    out=[]
    for (s,suite,log),rr in group_rows(rows,['strategy','suite','log']).items():
        item=dict(strategy=s,suite=suite,log=log,well=well_group(rr[0]),original_well=rr[0].get('well',log),well_group=well_group(rr[0]),family=rr[0].get('family','unknown'),windows=len(rr))
        item.update({m:avg([field(r,m) for r in rr]) for m in METRICS})
        item['failure_count']=sum(bool(r.get('failure') or r.get('error')) for r in rr)
        item['empty_count']=sum(r.get('prediction_count',r.get('n_pred',-1))==0 for r in rr)
        item['budget_exceed_count']=sum((field(r,'seconds') or 0)>60 for r in rr)
        item['timed_case_count']=sum(field(r,'seconds') is not None for r in rr)
        item['unknown_time_count']=len(rr)-item['timed_case_count']
        out.append(item)
    return out

def bootstrap(values,seed=1108,draws=10000):
    if not values:return [None,None]
    a=np.asarray(values,float);rng=np.random.default_rng(seed)
    means=np.mean(a[rng.integers(0,len(a),size=(draws,len(a)))],axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def summary(rows,keys):
    data=[]
    for key,rr in group_rows(rows,keys).items():
        r=dict(zip(keys,key));r['logs']=len(rr)
        for m in METRICS:r[m]=avg([x[m] for x in rr])
        r['score_ci95']=bootstrap([x['score'] for x in rr if x['score'] is not None])
        for f in ('failure_count','empty_count','budget_exceed_count','windows','timed_case_count','unknown_time_count'):r[f]=sum(x.get(f,0) for x in rr)
        data.append(r)
    return data

def pairs(rows):
    result=[];strategies=sorted(set(x['strategy'] for x in rows))
    for suite in sorted(set(x['suite'] for x in rows)):
        for a,b in itertools.combinations(strategies,2):
            delta=[]
            for ra,rb in paired_case_rows(rows,a,b,suite):
                d=dict(ra)
                for metric in METRICS:
                    va,vb=field(ra,metric),field(rb,metric)
                    d[metric]=va-vb if va is not None and vb is not None else None
                delta.append(d)
            deltas=log_means(delta)
            scores=[d['score'] for d in deltas if d['score'] is not None]
            wg=group_rows([d for d in deltas if d['score'] is not None],['well'])
            wa=[np.array([d['score'] for d in rr]) for rr in wg.values()]
            sensitivity=[None,None]
            if wa:
                rng=np.random.default_rng(1108); means=[]
                for draw in rng.integers(0,len(wa),size=(10000,len(wa))):means.append(float(np.concatenate([wa[i] for i in draw]).mean()))
                sensitivity=np.quantile(means,[.025,.975]).tolist()
            result.append(dict(suite=suite,a=a,b=b,logs=len(scores),wells=len(wa),mean_delta=avg(scores),ci95=bootstrap(scores),well_ci95=sensitivity,per_log=deltas,paired_cases=len(delta)))
    return result

def failure_reason(r):
    if r.get('failure') or r.get('error'):return 'The pipeline failed: '+str(r.get('error',r.get('failure_stage','see run record'))).split('\n')[-1][:160]
    details=str(diagnostics(r) or r.get('stages',{})).lower()
    if 'empty legend' in details or r.get('legend_count')==0 or native_legend_status(r)[0]=='missing_native_legend':return 'The native legend has no entries; the pipeline cannot name the rock.'
    if (field(r,'row_acc') or 0)<.1:return 'Most filled truth rows have a different label or no predicted label.'
    if (field(r,'boundary_f1') or 0)<.25:return 'Most truth contacts have no matching predicted contact.'
    if (field(r,'row_acc') or 0)<(field(r,'boundary_f1') or 0):return 'Label errors cause more loss than contact errors.'
    return 'Missing or extra contacts reduce the interval score.'

def font(size):
    for p in ('C:/Windows/Fonts/arial.ttf','/mnt/c/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:return ImageFont.truetype(p,size)
        except OSError:pass
    return ImageFont.load_default()

def resolve(p):
    p=Path(p)
    if p.is_absolute() or p.exists():return p
    return ROOT/p

def rows_file(p):
    from harness.io import load_rows
    return load_rows(resolve(p)) or []

def render_case(r,target):
    from harness import io,score,viz
    source=r.get('image_path',r.get('image'))
    ann=r.get('annotation_path',r.get('annotation'))
    if source and ann:
        boxes=json.loads(resolve(ann).read_text(encoding='utf-8'))['bounding_boxes']
        log=io.Log(Image.open(resolve(source)),boxes,r['log'])
    else:
        old=io.DATASET
        if r.get('dataset_root'):io.DATASET=resolve(r['dataset_root'])
        try:log=io.open_log(r['log'])
        finally:io.DATASET=old
    try:
        truth=r.get('truth_rows')
        if truth is None:truth=rows_file(r.get('truth_path',r.get('truth_csv')))
        pred=r.get('pred_rows',r.get('prediction_rows'))
        if pred is None:pred=rows_file(r.get('prediction_path',r.get('prediction_csv',r.get('pred_csv'))))
        window=tuple(r.get('window',[r.get('y0'),r.get('y1')]))
        y0,y1=map(int,window)
        # A bounded 1200-pixel detail starts at the row with most label errors.
        gt=graded_raster(truth,window,r.get('_grading_id'));pr=graded_raster(pred,window,r.get('_grading_id'))
        error=((gt!='')&(gt!=pr)).astype(int)
        detail=min(1200,y1-y0)
        sums=np.convolve(error,np.ones(detail,dtype=int),'valid') if len(error)>detail else [error.sum()]
        a=y0+int(np.argmax(sums));b=a+detail
        if r.get('display_detail_window'):
            a,b=map(int,r['display_detail_window'])
            if not y0<=a<b<=y1:raise ValueError('Requested detail is outside the scored window')
            detail=b-a
        height=max(520,min(1500,detail));scale=height/detail
        canvas=Image.new('RGB',(1300,height+140),'white');d=ImageDraw.Draw(canvas)
        d.text((25,16),label(strategy(r))+' / '+r['suite'],fill='#182e3b',font=font(22))
        short=r['log'] if len(r['log'])<103 else r['log'][:100]+'...'
        d.text((25,49),short,fill='#182e3b',font=font(13))
        d.text((25,71),f'Detail: {a}–{b}; window: {y0}–{y1}; grade: '+('accepted equivalence' if r.get('_grading_id') else 'strict reference'),fill='#425560',font=font(14))
        top=119
        col=Image.fromarray(io.read_rgb(log,'interpreted_lithology_column',a,b,scale=min(1,scale)))
        col=col.resize((250,height),Image.Resampling.NEAREST);canvas.paste(col,(80,top))
        d.text((80,97),'Source column',fill='black',font=font(15))
        d.text((365,97),'Truth',fill='black',font=font(15));d.text((650,97),'Prediction',fill='black',font=font(15))
        for rr,x in ((truth,365),(pred,650)):
            for row in rr:
                c=max(a,int(row['top_y']));e=min(b,int(row['bottom_y']))
                if e<=c:continue
                y=top+round((c-a)*scale);z=top+round((e-a)*scale)
                d.rectangle((x,y,x+260,max(y,z-1)),fill=viz.colour(row['lithology']))
                d.line((x,y,x+260,y),fill='#364550',width=1)
                if z-y>=18:d.text((x+6,y+2),str(row['lithology'])[:34],fill='black',font=font(13))
        for i in range(6):
            yy=top+round(i*height/5);d.text((8,yy-7),str(round(a+i*detail/5)),fill='#425560',font=font(12))
        er=np.full((detail,18,3),220,np.uint8);mask=gt[a-y0:b-y0]!='';same=gt[a-y0:b-y0]==pr[a-y0:b-y0]
        er[mask&same]=[20,140,100];er[mask&~same]=[196,50,48]
        canvas.paste(Image.fromarray(er).resize((18,height),Image.Resampling.NEAREST),(932,top))
        # Full-window overview prevents the error detail from hiding overall behaviour.
        oscale=height/(y1-y0)
        overview=Image.fromarray(io.read_rgb(log,'interpreted_lithology_column',y0,y1,scale=min(1,oscale)))
        overview=overview.resize((80,height),Image.Resampling.NEAREST)
        canvas.paste(overview,(990,top))
        d.text((990,76),'Full window overview',fill='#182e3b',font=font(15))
        for text,x in [('Source',990),('Truth',1085),('Pred.',1180)]:d.text((x,97),text,fill='black',font=font(13))
        for rr,x in ((truth,1085),(pred,1180)):
            for row in rr:
                c=max(y0,int(row['top_y']));e=min(y1,int(row['bottom_y']))
                if e<=c:continue
                yy=top+round((c-y0)*oscale);zz=top+round((e-y0)*oscale)
                d.rectangle((x,yy,x+80,max(yy,zz-1)),fill=viz.colour(row['lithology']))
        da=top+round((a-y0)*oscale);db=top+round((b-y0)*oscale)
        d.rectangle((984,da,1268,db),outline='#c34b18',width=3)
        canvas.save(target)
        return {'path':target.relative_to(HERE).as_posix(),'detail_window':[a,b],'scored_window':[y0,y1],'status':'ok'}
    finally:log.close()

CSS='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:#183142;font:16px/1.5 Arial,sans-serif;background:#fff}main{max-width:1180px;margin:auto;padding:30px 22px 70px}h1{font-size:clamp(30px,5vw,48px);line-height:1.08;max-width:820px;letter-spacing:-.025em;margin:8px 0 22px}h2{font-size:25px;margin:42px 0 12px}h3{font-size:19px;margin:22px 0 10px}p{max-width:78ch}a{color:#075f9c;text-underline-offset:3px}nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #b9ccd6;padding:12px 0 18px}.note{background:#f0f5f8;padding:16px;max-width:90ch}.warn{background:#fff1df}.muted{color:#435e70}.table-wrap{overflow-x:auto;border:1px solid #b9ccd6;margin:12px 0 22px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:right;padding:11px 12px;border-bottom:1px solid #dce5ea;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf3f7;color:#233d4c}tr:last-child td{border-bottom:0}tbody tr:hover{background:#f3f7fa}.best{font-weight:bold;color:#006458;background:#e1f2ea}.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:13px;background:#e6eff6}.chart{max-width:850px}.bar-row{display:grid;grid-template-columns:140px 1fr 64px;gap:10px;align-items:center;margin:9px 0}.bar-track{background:#e9eff3;height:27px}.bar{height:27px;min-width:0}.bar-value{font-variant-numeric:tabular-nums;text-align:right}details{margin:15px 0;border-bottom:1px solid #ccd9e1;padding-bottom:14px}summary{cursor:pointer;font-weight:600;line-height:1.6}figure{margin:20px 0 36px;max-width:1300px}figure img{display:block;width:100%;height:auto;border:1px solid #c6d3dc}figcaption{margin:9px 0;color:#304d5e}code{font-size:.9em;word-break:break-all}pre{white-space:pre-wrap;word-break:break-word;background:#f1f5f7;padding:16px;font-size:12px}ul{padding-left:23px;max-width:90ch}.negative{color:#a12b20}.positive{color:#006858}:focus-visible{outline:3px solid #d2630e;outline-offset:3px}::selection{background:#ffe0a9}@media(max-width:500px){main{padding:20px 14px 50px}h2{font-size:22px}th,td{padding:9px 10px}.bar-row{grid-template-columns:105px 1fr 48px;font-size:13px}nav{gap:12px;font-size:14px}figure{margin-left:-6px;margin-right:-6px}}'''

def metric_table(data,first='strategy',unit='Logs'):
    out=['<div class="table-wrap"><table><thead><tr>',f'<th>{esc(first.replace("_"," ").title())}</th>']
    out += ['<th>'+x+'</th>' for x in [unit,'Score','Row acc.','Boundary F1','ARI','Mixed run s','Depth MAE ft','Preprocess s','Total s','Task wall s','Failed','&gt;60 s','Timed cases','Unknown time']]
    out.append('</tr></thead><tbody>')
    best=max((x['score'] for x in data if x['score'] is not None),default=None)
    for r in data:
        out.append('<tr><td>'+esc(label(r[first]))+'</td><td>'+str(r['logs'])+'</td>')
        for m in METRICS:
            cls=' class="best"' if m=='score' and r[m]==best else ''
            out.append('<td'+cls+'>'+num(r[m],2 if m in ('seconds','depth_mae_ft') else 3)+'</td>')
        out+=['<td>'+str(r['failure_count'])+'</td><td>'+str(r['budget_exceed_count'])+'</td><td>'+str(r['timed_case_count'])+'/'+str(r['windows'])+'</td><td>'+str(r['unknown_time_count'])+'</td></tr>']
    return ''.join(out)+'</tbody></table></div>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scores',type=Path,default=HERE/'scores.json');ap.add_argument('--skip-images',action='store_true');args=ap.parse_args()
    raw=json.loads(args.scores.read_text(encoding='utf-8'));rows=[dict(r) for r in raw.get('records',raw.get('results',raw.get('rows',raw.get('cases',[]))))]
    manifest=json.loads((HERE/'manifest_1108.json').read_text(encoding='utf-8'))
    try:
        validation=validate_records(raw,manifest)
        audit=json.loads((HERE/'audit_saved_predictions.json').read_text(encoding='utf-8'))
        validation['independent_csv_replay']=validate_audit(audit,raw['records'],HERE/'manifest_1108.json')
        validation['audited_prediction_bytes']=validate_prediction_bytes(audit,raw['records'])
        integrity=json.loads((HERE/'result_audit.json').read_text(encoding='utf-8'))
        validation['run_integrity']=require_integrity_audit(integrity,raw,manifest,HERE/'manifest_1108.json')
        validation['accepted_policy_replay']=require_policy(raw,audit,ROOT)
        accepted_view=accepted_records(rows)
    except Exception as exc:
        (HERE/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scoreboard incomplete</title><h1>No champion can be declared</h1><p>The saved results did not pass the required decision gate.</p><p>'+esc(str(exc))+'</p></html>',encoding='utf-8')
        (HERE/'decision_gate.json').write_text(json.dumps({'status':'invalid','error':str(exc)},indent=2),encoding='utf-8')
        raise
    (HERE/'decision_gate.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    strict_rows=copy.deepcopy(rows)
    rows=accepted_view
    mapping=build_dev_map(manifest)
    global DEV_GROUPS
    DEV_GROUPS=mapping['log_to_group']
    (HERE/'reviewed_dev_wells.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    for r in rows:
        r['strategy']=strategy(r)
        if r.get('suite')=='human_unqa':r['suite']='hadrian_unqa'
    strict_logs=log_means(strict_rows)
    strict_summaries=summary(strict_logs,['suite','strategy'])
    strict_paired=pairs(strict_rows)
    logs=log_means(rows);summaries=summary(logs,['suite','strategy']);family=summary(logs,['suite','family','strategy']);paired=pairs(rows)
    # Remove repeated evidence before making the descriptive overall.
    # Priority keeps QA and gold truth instead of silver on the same source page.
    order={'human_qa':0,'human_gold':1,'viridien_eval':2,'ext_dev':3,'hadrian_unqa':4}
    bylog=defaultdict(list)
    for x in logs:
        if x['suite']!='hadrian_unqa':bylog[(x['strategy'],x['log'])].append(x)
    unique=[min(rr,key=lambda r:order.get(r['suite'],99)) for rr in bylog.values()]
    overall=summary(unique,['strategy'])
    strict_bylog=defaultdict(list)
    for r in strict_logs:
        if r['suite']!='hadrian_unqa':strict_bylog[(r['strategy'],r['log'])].append(r)
    strict_overall=summary([min(rr,key=lambda r:order.get(r['suite'],99)) for rr in strict_bylog.values()],['strategy'])
    log06_records=[r for r in rows if r['suite']=='viridien_eval' and 'log06' in r['log']]
    log06_parts=summary(log_means(log06_records),['strategy'])
    log06_strict_parts=summary(log_means([r for r in strict_rows if r['suite']=='viridien_eval' and 'log06' in r['log']]),['strategy'])
    log06_labels=[]
    for r in log06_records:
        shares=Counter()
        for interval in rows_file(r['prediction_csv']):shares[interval['lithology']]+=int(interval['bottom_y'])-int(interval['top_y'])
        log06_labels.append(dict(strategy=r['strategy'],top_predictions=[dict(label=lb,pixel_rows=count) for lb,count in shares.most_common(2)]))
    log06_remaining=[]
    from harness import score as shared_score
    for r in log06_records:
        if r['strategy']!='s23_v4dino_identity':continue
        gt=shared_score.raster(rows_file(r['truth_csv']),tuple(r['window']));pred=shared_score.raster(rows_file(r['prediction_csv']),tuple(r['window']))
        log06_remaining.append(dict(strategy=r['strategy'],truth_shale_rows=int(np.sum(gt==shared_score.normalize('Shale'))),predicted_clay_on_shale_rows=int(np.sum((gt==shared_score.normalize('Shale'))&(pred==shared_score.normalize('Clay'))))))
    log06_explanation=dict(prototype_crop_accuracy=.83,prototype_note='The historical 0.83 result used in-bed crops and slash-group membership credit. It is a different task from the strict interval scorer.',strict_label_example=dict(prediction='Clay / Shale',truth='Shale',historical_strict_row_credit=0,accepted_row_credit=1),parts=log06_parts,strict_parts=log06_strict_parts,top_labels=log06_labels,raw_clay_shale_caption_difference=log06_remaining)
    def equal_well(log_data):
        well_rows=[]
        for (strategy_name,well),rr in group_rows(log_data,['strategy','well']).items():
            r=dict(rr[0]);r['log']=well
            r.update({metric:avg([x[metric] for x in rr]) for metric in METRICS})
            for f in ('failure_count','empty_count','budget_exceed_count','windows','timed_case_count','unknown_time_count'):r[f]=sum(x.get(f,0) for x in rr)
            well_rows.append(r)
        return summary(well_rows,['strategy'])
    equal_well_dev=equal_well([r for r in logs if r['suite']=='ext_dev'])
    equal_well_overall=equal_well(unique)
    pixel_rows=[]
    for r in rows:
        q=dict(r)
        if r.get('pixel_only'):
            q.update(r['pixel_only']);q.pop('error',None);q.pop('failure',None)
            q['depth_mae_ft']=None
        pixel_rows.append(q)
    pixel_summaries=summary(log_means(pixel_rows),['suite','strategy'])
    assets=HERE/'images';assets.mkdir(exist_ok=True);worst=[];renders=[]
    grade_views=[]
    for r in log06_records:
        if r['strategy']!='s21_v4dino':continue
        for grade in ('strict','accepted'):
            view=copy.deepcopy(r);view.update(r.get('pixel_only',{}));view['display_detail_window']=[4700,5150]
            if grade=='strict':view.pop('_grading_id',None)
            target=assets/('log06_baseline_'+grade+'_4700_5150.png')
            item=dict(strategy=r['strategy'],grade=grade,source_case_id=r['case_id'])
            if not args.skip_images:
                try:item['render']=render_case(view,target)
                except Exception as exc:item['render']=dict(status='failed',error=f'{type(exc).__name__}: {exc}')
            else:item['render']=dict(status='not_run')
            grade_views.append(item)
    fixed_views=[]
    if 's23_v4dino_identity' in manifest['candidates']:
        for r in log06_records:
            if r['strategy'] not in ('s21_v4dino','s23_v4dino_identity'):continue
            view=dict(r);view.update(r.get('pixel_only',{}));view['display_detail_window']=[4700,5150]
            target=assets/('log06_'+r['strategy']+'_4700_5150.png')
            item=dict(strategy=r['strategy'],source_case_id=r['case_id'],pixel_only_diagnostic=bool(r.get('pixel_only')),scope='Same original Viridien window; both views show rows 4700–5150 with the full-window overview.')
            if not args.skip_images:
                try:item['render']=render_case(view,target)
                except Exception as exc:item['render']=dict(status='failed',error=f'{type(exc).__name__}: {exc}')
            else:item['render']=dict(status='not_run')
            fixed_views.append(item)
    for s in sorted(set(x['strategy'] for x in logs)):
        # Select unique source logs; keep the lowest score across suites for diagnosis.
        candidates={}
        for x in logs:
            if x['strategy']==s and (x['log'] not in candidates or x['score']<candidates[x['log']]['score']):candidates[x['log']]=x
        for rank,x in enumerate(sorted(candidates.values(),key=lambda r:(r['score'],r['log']))[:3],1):
            rr=[r for r in rows if r['strategy']==s and r['suite']==x['suite'] and r['log']==x['log']]
            r=min(rr,key=lambda r:field(r,'score') or 0);rec=dict(strategy=s,rank=rank,log=x['log'],suite=x['suite'],score=x['score'],reason=failure_reason(r))
            target=assets/(s+'_'+str(rank)+'.png')
            if not args.skip_images:
                try:rec['render']=render_case(r,target)
                except Exception as ex:rec['render']={'status':'failed','error':f'{type(ex).__name__}: {ex}'}
            else:rec['render']={'status':'not_run'}
            worst.append(rec);renders.append(rec['render'])
    regimes=runtime_summary(rows);native_status=native_legend_summary(rows)
    failure_evidence=[dict(case_id=r['case_id'],suite=r['suite'],strategy=r['strategy'],log=r['log'],error=r['error'],inference_seconds=r.get('seconds'),failure_phase=r.get('failure_phase'),operational_timeout=r.get('operational_timeout',False),formal_uniform_cap_recheck=r.get('formal_uniform_cap_recheck',False),timeout_receipt=r.get('timeout_receipt')) for r in rows if r.get('error')]
    timing_records=[dict(case_id=r['case_id'],strategy=r['strategy'],seconds=r.get('seconds'),task_wall_seconds=r.get('task_wall_seconds'),diagnostic_seconds=r.get('pixel_only',{}).get('seconds'),process_start_utc=r.get('process_start_utc'),resumed_process=r.get('resumed_process'),first_strategy_call_in_process=r.get('first_strategy_call_in_process'),cache_state='unknown unless explicitly marked in source record') for r in rows]
    timing_summary=dict(total_cases=len(rows),timed_cases=sum(r.get('seconds') is not None for r in rows),unknown_time_cases=sum(r.get('seconds') is None for r in rows),diagnostic_cases=sum(bool(r.get('pixel_only')) for r in rows),processes=sorted(set(r['process_start_utc'] for r in timing_records if r['process_start_utc'])),first_strategy_calls=sum(r.get('first_strategy_call_in_process') is True for r in rows),resume_marked_cases=sum(r.get('resumed_process') is True for r in rows))
    report={'failure_evidence':failure_evidence,'runtime_regimes':regimes,'native_legend_status':native_status,'grading_policy':raw.get('grading_policy'),'grade_comparison_views':grade_views,'strict_reference':{'summary_suite':strict_summaries,'overall':strict_overall,'paired':strict_paired},'class_identity_fix':{'candidate':'s23_v4dino_identity','note':'CLAY_SHALE_FIX.md','effect':'Use measured scores only. The corrected candidate retains separately printed class identities; no extra scorer alias is added. Genuine compound captions remain compound labels.','log06_views':fixed_views},'log06_explanation':log06_explanation,'timing_summary':timing_summary,'timing_records':timing_records,'report_source_sha256':{name:digest(HERE/name) for name in ('build_report.py','report_inputs.py','report_integrity.py','report_policy_gate.py','report_diagnostics.py','rescore_policy.py','reviewed_dev_wells.json')},'equal_well_dev':equal_well_dev,'equal_well_overall':equal_well_overall,'validation':validation,'dev_well_mapping':mapping,'pixel_only_suite':pixel_summaries,'well_group_note':'Use the explicit reviewed DEV name map: KGS, Utah, Idaho Barlow, Flat Bay, and Storm. These are provisional source groups. Original well metadata is retained; source groups are sensitivity units, not certified independent wells.','summary_suite':summaries,'summary_family':family,'per_log':logs,'paired':paired,'overall':overall,'overall_rule':'Equal source-log weight. For a repeated source log, retain human_qa, then human_gold, then viridien_eval, then ext_dev. Exclude Hadrian unQA. Each retained log first averages its suite windows. This removes repeated log evidence and can retain a different extent. Descriptive only.','bootstrap':{'draws':10000,'seed':1108,'unit':'log','interval':'95% percentile','well_sensitivity':'resample wells; retain all logs in each sampled well'},'worst':worst,'render_failures':[r for r in worst+fixed_views+grade_views if r['render']['status']!='ok']}
    (HERE/'report_data.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # All report numbers also live in the requested single score file.
    raw['report']=report;args.scores.write_text(json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lithology scoreboard · 8 November</title><style>'+CSS+'</style><body><main>', '<h1>Local proxy. '+('Four' if len(manifest['candidates'])==4 else 'Three')+' lithology pipelines.</h1><p>Viridien HackTheBurgh · 8 November 2026. Primary LOCAL PROXY: Clay, Shale, Clay/Shale and Clay-Shale are equivalent in both directions. Local proxy score = row accuracy × boundary F1 for each window. Logs have equal weight after their windows are averaged.</p><nav><a href="#suites">Suites</a><a href="#families">Families</a><a href="#paired">Paired differences</a><a href="#worst">Worst logs</a><a href="#method">Method</a><a href="scores.json">Scores JSON</a></nav>']
    parts+=['<h2>Development choice · local proxy</h2><div class="note warn"><p>'+esc(raw.get('decision',{}).get('text','The two Astra reviews are complete within their stated scopes. The parent must assess the complete audited suites before entering a practical development choice. No universal or held-out champion is established.'))+'</p></div>','<h2>Overall, with repeated source logs removed</h2><p class="muted">'+esc(report['overall_rule'])+'</p><div class="chart">']
    for i,r in enumerate(overall):parts+=['<div class="bar-row"><span>'+esc(label(r['strategy']))+'</span><div class="bar-track"><div class="bar" style="width:'+str(100*(r['score'] or 0))+'%;background:'+COLORS[i%len(COLORS)]+'"></div></div><span class="bar-value">'+num(r['score'])+'</span></div>']
    parts+=['</div>',metric_table(overall),'<h2 id="suites">Accepted-policy suite results · local proxy</h2><p>Score is the mean of per-window products within each log, then the mean across logs. It is not the product of the table means. Run time is strategy inference time; preprocessing and total time are in the run records.</p>']
    caveats={'viridien_eval':'13 development regression logs. Gold takes precedence; other truth is silver consensus.','human_qa':'Two QA human-labelled logs. Shenzi may use machine contact positions; its boundary score is not independent evidence.','human_gold':'D-582 log06 supplied gold. This repeats a Viridien source log and is not a new replication.','hadrian_unqa':'Unreviewed Hadrian labels. First composition type is used. Mixed lithology and contact quality remain unverified. Excluded from the descriptive overall.','ext_dev':'External DEV only. Multiple windows from one log have equal weight only after their within-log average. Test and verify were not opened.'}
    for suite,rr in group_rows(summaries,['suite']).items():
        parts+=['<h3>'+esc(suite[0])+'</h3><p class="muted">'+esc(caveats.get(suite[0],''))+'</p>',metric_table(rr)]
    parts+=['<h2>log06 · grading change and inference change</h2><p>This is the original Viridien log06 window against gold intervals. The primary table uses the owner-approved symmetric equivalence among Clay, Shale, Clay/Shale and Clay-Shale. A score change caused by this policy is a grading change; no prediction was rerun. The three score parts must still be read together.</p>',metric_table(log06_parts),'<p>The historical strict grade gave no row credit for “Clay / Shale” against “Shale”. The accepted grade gives full row credit among Clay, Shale, Clay/Shale and Clay-Shale in every direction. Both copied label sequences pass through the unchanged harness scorer; interval contacts, cluster IDs and depths stay intact. The earlier DINO 0.83 result still used crops inside beds, a different task from this interval score.</p><ul>']
    for item in log06_labels:parts+=['<li>'+esc(label(item['strategy']))+': main predicted labels are '+esc(', '.join(x['label'] for x in item['top_predictions']))+'.</li>']
    parts+=['</ul><details><summary>log06 · historical strict reference</summary>',metric_table(log06_strict_parts),'</details><h3>Same baseline prediction · strict and accepted error strips</h3><p>Raw interval labels and source pixels are the same in both views. Only label equality in the error strip changes.</p>']
    for item in grade_views:
        parts+=['<h3>'+esc(item['grade'].title())+' grading</h3>']
        if item['render']['status']=='ok':parts+=['<figure><a href="'+item['render']['path']+'"><img loading="lazy" src="'+item['render']['path']+'" alt="Same baseline log06 prediction with '+esc(item['grade'])+' error grading"></a></figure>']
        else:parts+=['<p>Grade image render '+esc(item['render']['status'])+': '+esc(item['render'].get('error','not run'))+'</p>']
    if fixed_views:
        parts+=['<h2>Separate class identities · measured comparison</h2><p>The frozen DINO baseline and the corrected candidate remain separate rows. S23 is an optional identity experiment that retains separately printed legend captions. It is not needed to earn accepted Clay/Shale equivalence credit. A genuine combined caption remains valid under the accepted grade. The shared scorer is unchanged. <a href="CLAY_SHALE_FIX.md">Read the fix note</a> and <a href="CLAY_SHALE_FIX_MEASURED.md">the measured result and remaining error</a>.</p><p>Both images below show rows 4700–5150 from the same original log06 evaluation window. The right overview shows the complete window. Read the score parts above before judging the visual effect.</p>']
        for remaining in log06_remaining:parts+=['<p>The raw S23 output predicts Clay on '+str(remaining['predicted_clay_on_shale_rows'])+' of '+str(remaining['truth_shale_rows'])+' gold Shale rows in this window. The accepted policy treats those rows as equivalent; this is a raw caption difference, not an accepted-grade error. The earlier small gain was measured under strict grading.</p>']
        for item in fixed_views:
            parts+=['<h3>'+esc(label(item['strategy']))+'</h3>']
            if item['render']['status']=='ok':parts+=['<figure><a href="'+item['render']['path']+'"><img loading="lazy" src="'+item['render']['path']+'" alt="log06 source, truth and '+esc(label(item['strategy']))+' prediction on rows 4700 to 5150"></a><figcaption>'+esc(item['scope'])+'</figcaption></figure>']
            else:parts+=['<p class="note warn">Image render '+esc(item['render']['status'])+': '+esc(item['render'].get('error','not run'))+'</p>']
    parts+=['<h2>Strict reference · local proxy</h2><p>These are the unchanged original strict metrics. The accepted policy above is primary for the owner’s choice.</p><h3>Overall strict reference</h3>',metric_table(strict_overall)]
    for (suite,),rr in group_rows(strict_summaries,['suite']).items():parts+=['<details><summary>'+esc(suite)+' · strict reference</summary>',metric_table(rr),'</details>']
    if failure_evidence:
        parts+=['<h2>Recorded pipeline failures</h2><p>Failed cases retain zero quality metrics. A whole-attempt timeout is separate from inference seconds: the attempt includes its recorded phases, and an unknown inference duration remains unknown. It is not entered as a fast result or as a measured 600-second inference.</p>']
        for failure in failure_evidence:parts+=['<h3>'+esc(label(failure['strategy']))+' · '+esc(failure['suite'])+'</h3><p>'+esc(failure['log'])+'</p><p>'+esc(failure['error'])+'</p><pre>'+esc(json.dumps(failure,ensure_ascii=False,indent=2))+'</pre>']
    parts+=['<h2>Observed timing coverage</h2><p>'+str(timing_summary['timed_cases'])+' of '+str(timing_summary['total_cases'])+' inference calls have a measured time; '+str(timing_summary['unknown_time_cases'])+' are unknown. '+str(timing_summary['diagnostic_cases'])+' cases have a separate pixel-only diagnostic. '+str(timing_summary['first_strategy_calls'])+' calls are marked as the first strategy call in a process; '+str(timing_summary['resume_marked_cases'])+' cases are marked as part of a resumed process.</p><p>First-call markers are not proof of a cold cache. Diagnostic calls can load a model. The JSON retains process, resume, first-call, diagnostic, and task-wall values. Missing markers mean unknown. Time is descriptive; it is not a controlled speed tie-break. Remote CPU records, if present, use decoded lossless windows and have a different scope from local source-page IO and CUDA calls. Preserve their per-record runtime/provenance and do not compare pooled times as a speed tie-break.</p>']
    parts+=['<h3>Separate timing regimes</h3><p>The mixed run-time aggregate above is descriptive and cannot establish efficiency. These rows keep remote crop inference and local source-page calls separate. Raw seconds are unchanged. Missing time means unknown; no speed tie-break is used.</p>',runtime_table(regimes,label),'<h3>Native legend availability</h3><p>Counts use primary diagnostics, or separate pixel-only diagnostics when those provide the available evidence. Local strategy_diagnostics and portable diagnostics fields are both read. A missing native legend is a recorded pipeline limit, not an inference crash. Unknown means no cache evidence was recorded. S19 uses a different legend path.</p>',native_table(native_status,label)]
    parts+=['<h2>Equal-source-group sensitivity</h2><p>The primary table gives each source log equal weight. These tables first average logs within each source group, then give each group equal weight. DEV has five provisional groups. This view does not establish a new-well transfer result.</p><h3>External DEV · equal group weight</h3>',metric_table(equal_well_dev,unit='Source groups'),'<h3>Overall · equal group weight</h3>',metric_table(equal_well_overall,unit='Source groups')]
    parts+=['<h2>Pixel-only diagnostic</h2><p>This table uses the same windows and scorer. When depth calibration fails but a pixel prediction is available, it uses that separate prediction. Other rows keep their primary prediction. A missing diagnostic remains a zero-score failure. These scores do not meet the depth-output requirement and do not replace the primary score.</p>']
    for (suite,),rr in group_rows(pixel_summaries,['suite']).items():parts+=['<details><summary>'+esc(suite)+'</summary>',metric_table(rr),'</details>']
    parts+=['<h2 id="families">Family results</h2><p>Family names come from the supplied metadata. Missing names stay “unknown”. No family is inferred from the score or label.</p>']
    for (suite,fam),rr in group_rows(family,['suite','family']).items():parts+=['<details><summary>'+esc(suite+' · '+str(fam))+'</summary>',metric_table(rr),'</details>']
    parts+=['<h2 id="paired">Paired per-log differences</h2><p>Each value is A minus B. The interval uses 10,000 paired bootstrap draws over complete logs (seed 1108). The second interval resamples complete wells. The reviewed DEV name map groups KGS, Utah, Idaho Barlow, Flat Bay, and Storm. These five groups are provisional source groups; geological independence is not verified. Original well metadata is retained in the JSON. A one-log or one-well interval is degenerate and does not show precision.</p>']
    for p in paired:
        parts+=['<details><summary>'+esc(p['suite']+' · '+label(p['a'])+' minus '+label(p['b']))+' · Δ '+num(p['mean_delta'])+'</summary><p>Log 95% CI ['+num(p['ci95'][0])+', '+num(p['ci95'][1])+']; well 95% CI ['+num(p['well_ci95'][0])+', '+num(p['well_ci95'][1])+']. '+str(p['logs'])+' logs; '+str(p['wells'])+' wells.</p><div class="table-wrap"><table><thead><tr><th>Log / family</th><th>Δ score</th><th>Δ row acc.</th><th>Δ boundary F1</th><th>Δ run s</th></tr></thead><tbody>']
        for r in p['per_log']:parts+=['<tr><td title="'+esc(r['log'])+'">'+esc(r['log'])+'<br><span class="muted">'+esc(r['family'])+'</span></td>'+''.join('<td>'+num(r[m])+'</td>' for m in ('score','row_acc','boundary_f1','seconds'))+'</tr>']
        parts+=['</tbody></table></div></details>']
    parts+=['<h2 id="worst">Three worst source logs for each strategy</h2><p>Logs are ranked by mean score; each source log appears once per strategy. The left image shows up to 1,200 native rows with the most label errors in the worst window. The right overview shows the complete scored window; its orange frame locates the detail. The full scored extent is named on the image. Colour identifies the label. The narrow error strip is green for a matching label, red for a wrong or missing label, and grey for truth blanks.</p>']
    for r in worst:
        parts+=['<h3>'+esc(label(r['strategy']))+' · '+str(r['rank'])+' · score '+num(r['score'])+'</h3><p>'+esc(r['log'])+' · '+esc(r['suite'])+'</p><p>'+esc(r['reason'])+'</p>']
        if r['render']['status']=='ok':parts+=['<figure><a href="'+r['render']['path']+'"><img loading="lazy" src="'+r['render']['path']+'" alt="Source column beside truth and prediction intervals for '+esc(r['log'])+'"></a><figcaption>Open the image to inspect labels at full size. Source and interval bars use the same row scale.</figcaption></figure>']
        else:parts+=['<p class="note warn">Image render '+esc(r['render']['status'])+': '+esc(r['render'].get('error','not run'))+'</p>']
    parts+=['<h2 id="method">Method and limits</h2><p>The local product does not directly include ARI, depth error, runtime or memory; these are separate measures. Row accuracy ignores truth-blank rows. Official correctness is broader than this window grade. A good window score is not proof of full-scan readiness. <a href="readiness_preview.json">Read the bounded readiness diagnostic</a> · <a href="remote/remote_results_shard.json">Remote result shard</a>.</p><p>Both grades are local proxies. The challenge material states correctness and efficiency goals but does not specify this score formula, its weights, boundary tolerance, or Clay/Shale equivalence. This local configuration does not predict judging. <a href="CHALLENGE_ALIGNMENT.md">Read the challenge alignment</a>.</p><ul><li>All candidates receive a common initial depth calibration. s19 can refine it and retain its own output depths. The saved initial calibration can differ from s19 output. Depth MAE uses matched contacts with truth depths; it is not full-track depth error.</li><li>This is an observed development comparison. OCR and depth cache contents were not pinned at every historical read. Later cache snapshots preserve available evidence but cannot prove earlier bytes. It is not a fully frozen, reproducible end-to-end run.</li><li>Timing is observed and depends on process and cache state. Missing inference time is unknown, not a fast success. First-load and resume markers are only known where the run record provides them; a diagnostic can warm a later model call. No controlled cold/warm speed check was run.</li><li>Astra reviewed the baseline and then completed the owner-authorized S23 recheck. The recheck accepted the caption identity fix for this development reevaluation and accepted the case grouping, pairing, depth and descriptive timing fixes. It required a final integrity gate and explicit OCR-cache limits. The parent must verify the local gate fix and final results. Astra did not approve an accuracy gain or a final practical choice. <a href="astra_recheck.md">Read the recheck</a>.</li><li>The accepted policy copies both prediction and truth labels and makes exactly Clay, Shale, Clay/Shale and Clay-Shale one class. It preserves intervals, contacts, cluster IDs and depths, then calls the unchanged shared scorer. Accepted ARI uses the canonical truth class. Raw labels and strict reference metrics are retained.</li><li>The owner configured this grading rule after the Astra recheck. The parent unit-tested it; Astra did not review or approve the new grading configuration.</li><li>A failed pipeline remains in the metric denominator with score, row accuracy, boundary F1 and ARI set to zero. A render failure does not change the score.</li><li>v4 uses a frozen copy of its current code and native legend inputs. Empty native legends on external logs are recorded. This measures the current cache-miss defect.</li><li>DINO uses frozen local weights, native-scale column tiles, and 0.5/1/2× legend swatches. The historical 0.83 crop result used a different task; it cannot be compared directly with this interval score.</li><li>Hadrian has no QA approval. Shenzi contacts may come from machine boundaries. Silver consensus is not independent human truth.</li><li>These are development results. External test and verify remain sealed. An unscored hybrid remains unverified.</li></ul><details><summary>Run provenance and raw checks</summary><pre>'+esc(json.dumps({k:v for k,v in raw.items() if k not in ('results','rows','cases','records','summaries','paired','accepted_summaries','accepted_paired','report')},ensure_ascii=False,indent=2))+'</pre></details><p><a href="evaluation_protocol.json">Declared evaluation method</a> · <a href="scores.json">Every score and paired delta</a> · <a href="report_data.json">Report tables and render records</a></p><p class="muted">Image render failures: '+str(len(report['render_failures']))+' of '+str(len(worst)+len(fixed_views)+len(grade_views))+'.</p></main></body></html>']
    (HERE/'index.html').write_text(''.join(parts),encoding='utf-8')
    print(f'Report built: {len(rows)} windows, {len(logs)} log results, {len(worst)} diagnostic images, {len(report["render_failures"])} render failures')
if __name__=='__main__':main()
