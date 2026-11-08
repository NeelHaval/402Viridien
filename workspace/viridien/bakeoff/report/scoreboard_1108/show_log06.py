"""Show the saved DINO prediction beside source pixels and supplied gold truth."""
from pathlib import Path
import base64
import argparse
import html
import json
import sys
from PIL import Image

R=Path(__file__).resolve().parent
B=R.parents[1]
sys.path.insert(0,str(B))
from harness import io, score
from strategies.grading_policy import grading_rows, score_equivalent

parser=argparse.ArgumentParser()
parser.add_argument('--strategy',choices=['s21_v4dino','s23_v4dino_identity'],default='s21_v4dino')
parser.add_argument('--grading',choices=['strict','equivalent'],default='equivalent')
parser.add_argument('--detail-only',action='store_true')
args=parser.parse_args()
data=json.loads((R/'raw_results.json').read_text(encoding='utf-8'))
rec=next(r for r in data['records'] if r['suite']=='viridien_eval' and 'log06' in r['log'] and r['strategy']==args.strategy)
variant='corrected' if args.strategy=='s23_v4dino_identity' else 'baseline'
variant += '_'+args.grading
truth=io.load_rows(Path(rec['truth_csv']))
pred=io.load_rows(Path(rec['prediction_csv']))
boxes=json.loads(Path(rec['annotation']).read_text(encoding='utf-8'))['bounding_boxes']
log=io.Log(Image.open(rec['image']),boxes,rec['log'])
colors={'Shale':'#d7dce4','Clay':'#f4c478','Clay / Shale':'#f4c478','Siltstone':'#b6d9ee','Sandstone':'#e8dd9a','Sand and Sandstone':'#e8dd9a','SIllty Sandstone':'#e5b3c3'}
assets=R/'explanation_images';assets.mkdir(exist_ok=True)

def image_uri(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')

def bands(rows,window,height):
    out=[];a,b=window
    for row in rows:
        top=max(a,row['top_y']);bottom=min(b,row['bottom_y'])
        if bottom<=top:continue
        y=(top-a)/(b-a)*height;h=(bottom-top)/(b-a)*height
        label=row['lithology'];minimum=48 if len(label)>14 else 24
        content=html.escape(label) if h>=minimum else ''
        out.append(f'<div class="band" style="top:{y:.3f}px;height:{h:.3f}px;background:{colors.get(label,"#ddd")}" title="{html.escape(label)}: {top}–{bottom}">{content}</div>')
    return ''.join(out)

def panel(window,height,key,title,description):
    a,b=window
    pixels=Image.fromarray(io.read_rgb(log,'interpreted_lithology_column',a,b))
    source=assets/(variant+'_'+key+'.png');pixels.save(source)
    equivalent=args.grading=='equivalent'
    gt=score.raster(grading_rows(truth) if equivalent else truth,window)
    pr=score.raster(grading_rows(pred) if equivalent else pred,window)
    import numpy as np
    err=np.full((b-a,8,3),220,np.uint8)
    filled=gt!='';err[filled & (gt==pr)]=[22,143,105];err[filled & (gt!=pr)]=[190,46,40]
    ep=assets/(variant+'_'+key+'_errors.png');Image.fromarray(err).save(ep)
    ticks=''.join(f'<span style="top:{i*height/5:.1f}px">{round(a+i*(b-a)/5)}</span>' for i in range(6))
    return f'<section><h2>{title}</h2><p>{description} Pixel rows {a}–{b}.</p><div class="comparison"><div class="heads"><span>Row</span><span>Scan</span><span>Gold truth</span><span>DINO prediction</span><span></span></div><div class="tracks" style="height:{height}px"><div class="axis">{ticks}</div><img class="source" src="{image_uri(source)}" alt="Scanned lithology column, rows {a} to {b}" style="height:{height}px"><div class="intervals" style="height:{height}px">{bands(truth,window,height)}</div><div class="intervals" style="height:{height}px">{bands(pred,window,height)}</div><img class="errors" src="{image_uri(ep)}" alt="Red is an incorrect label, green is correct, grey is outside gold truth" style="height:{height}px"></div></div></section>'

try:
    example=next((r for r in pred if r['top_y']<=4900<r['bottom_y']),None)
    description='The gold Siltstone bed is 4809–5010. '
    description+=(f"The prediction at row 4900 is {example['top_y']}–{example['bottom_y']}, labelled “{example['lithology']}”." if example else 'There is no prediction at row 4900.')
    details=panel((4700,5150),450,'log06_detail','Close view: the same source rows',description)
    overview='' if args.detail_only else panel(tuple(rec['window']),680,'log06_overview','Full scored window','Blank space in the gold track has no truth label. Error colors use the selected grading rule.')
finally:
    log.close()

css='''body{margin:0;font-family:var(--font-sans,Arial,sans-serif);color:var(--foreground,#183142)}h1{font-size:24px;line-height:1.15;margin:0 0 12px}h2{font-size:19px;margin:24px 0 10px}p{font-size:14px;line-height:1.55;max-width:75ch}section{margin:24px 0 34px}.comparison{overflow-x:auto;padding:4px 0 10px}.heads,.tracks{display:grid;grid-template-columns:42px minmax(90px,1fr) minmax(84px,.85fr) minmax(112px,1fr) 8px;gap:8px;min-width:350px}.heads{font-size:12px;font-weight:600;margin-bottom:8px}.tracks{align-items:start}.axis,.intervals{position:relative}.axis span{position:absolute;right:0;font-size:11px;transform:translateY(-50%)}.intervals{background:var(--secondary,#edf1f4);border:1px solid var(--border,#a9b7bf);overflow:hidden}.band{position:absolute;left:0;right:0;border-top:1px solid #58626a;box-sizing:border-box;color:#17242d;font-size:12px;line-height:1.2;padding:4px 5px;overflow:hidden;overflow-wrap:anywhere}.source{width:100%;image-rendering:auto}.errors{width:8px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}.sw{display:inline-block;width:11px;height:11px;margin-right:5px}.note{border-top:1px solid var(--border,#ccc);padding-top:12px}a{color:var(--primary,#145c9b)}'''
title='DINO prediction beside gold truth'
selected=score_equivalent(pred,truth,tuple(rec['window'])) if args.grading=='equivalent' else rec
metrics=f"On the full scored window: row accuracy {selected['row_acc']:.1%}, boundary F1 {selected['boundary_f1']:.1%}, product {selected['score']:.4f}."
policy_note=('Clay, Shale, Clay/Shale and Clay-Shale receive full label credit in either direction. Raw labels and bed contacts stay unchanged. '
             if args.grading=='equivalent' else 'Historical strict grade: Clay/Shale receives no credit against Shale. ')
comparison=(f'<p><strong>Same saved prediction, different local grade.</strong> Row accuracy: strict {rec["row_acc"]:.1%}; '
            f'approved family {score_equivalent(pred,truth,tuple(rec["window"]))["row_acc"]:.1%}. '
            f'Boundary F1: {rec["boundary_f1"]:.1%} in both. This is a local measure, not a verified challenge grade.</p>')
doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'+css+'</style><body><h1>'+title+'</h1><p>D-582 log06. These are the saved predictions used for the score. Gold truth is the supplied labelled reference. The interval tracks show raw labels.</p>'+comparison+'<div class="legend"><span><i class="sw" style="background:#be2e28"></i>Wrong label</span><span><i class="sw" style="background:#168f69"></i>Correct label</span><span><i class="sw" style="background:#ddd"></i>No gold label</span></div>'+details+overview+'<p class="note">'+policy_note+'“Sand and Sandstone” is credited as “Sandstone” by the existing scorer alias. '+metrics+'</p></body></html>'
output=('log06_dino_distinct' if args.strategy=='s23_v4dino_identity' else 'log06_dino')+'_'+args.grading+('_detail' if args.detail_only else '')+'.html'
(R/output).write_text(doc,encoding='utf-8')
print('Wrote '+output+'; source and error crops: '+str(2 if args.detail_only else 4))
