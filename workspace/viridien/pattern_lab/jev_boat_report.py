"""Render real source boxes and raw/graded interval comparisons on boat."""
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import worker
ROOT=Path(__file__).resolve().parent


def main(case_id):
    start=time.monotonic();io,adapter,api,env=worker.init()
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from harness import score
    from strategies.grading_policy import canonical_grading_label
    out=ROOT/'exports'/case_id;data=worker.read(out/'export.json');result=worker.read(out/'evaluation.json');case=worker.read(ROOT/'cases'/f'{case_id}.json');decision=worker.read(ROOT/'decisions'/f'{case_id}.json')
    truth=io.load_rows(ROOT/'score-inputs'/f'{case_id}.csv');pred=result['rows'];window=tuple(case['window']);col=np.asarray(Image.open(out/'column.png').convert('RGB'))
    font=ImageFont.load_default(size=14);small=ImageFont.load_default(size=12)
    labels=sorted({canonical_grading_label(r['lithology']) for r in truth+pred})
    palette=['#e6ab45','#5c9fce','#75ac6a','#c582ad','#a59ac8','#d77664','#75b6b4','#b4a35e','#9a9da3','#b49174']
    colours={v:palette[n%len(palette)] for n,v in enumerate(labels)}
    g=score.raster(truth,window);p=score.raster(pred,window)
    gg=np.array([canonical_grading_label(v) for v in g],object);pp=np.array([canonical_grading_label(v) for v in p],object);error=(gg!='')&(gg!=pp)
    length=min(700,len(col));counts=np.convolve(error.astype(int),np.ones(length,dtype=int),mode='valid');offset=int(np.argmax(counts)) if counts.size else 0
    def figure(name,top,bottom):
        h=800;width=max(36,min(260,round(col.shape[1]*h/max(1,bottom-top))));im=Image.new('RGB',(960,900),'white');d=ImageDraw.Draw(im)
        d.text((20,8),case['log'][:106],font=font,fill='black');d.text((20,31),f'Rows {top} to {bottom}. Raw source pixels; gold and prediction use the same vertical scale.',font=small,fill='black')
        d.text((48,61),'Source column',font=font,fill='black');d.text((345,61),'Gold truth',font=font,fill='black');d.text((625,61),'Jev prediction',font=font,fill='black')
        crop=Image.fromarray(col[top-window[0]:bottom-window[0]]).resize((width,h),Image.Resampling.NEAREST);im.paste(crop,(50,90))
        for rows,x in [(truth,340),(pred,620)]:
            for row in rows:
                a=max(top,int(row['top_y']));b=min(bottom,int(row['bottom_y']))
                if b<=a:continue
                y0=90+round((a-top)*h/(bottom-top));y1=90+round((b-top)*h/(bottom-top));label=row['lithology'];grade=canonical_grading_label(label)
                d.rectangle((x,y0,x+245,y1),fill=colours.get(grade,'#ddd'),outline='#333')
                if y1-y0>=14:d.text((x+5,y0+1),label[:31],font=small,fill='black')
        for y in np.linspace(top,bottom,9).round().astype(int):
            yy=90+round((y-top)*h/(bottom-top));d.text((2,yy-7),str(y),font=small,fill='black')
        im.save(out/name)
    figure('intervals.png',window[0],window[1]);figure('error-zoom.png',window[0]+offset,window[0]+offset+length)
    log=io.Log(Image.open(case['remote_image']),worker.read(case['remote_annotation'])['bounding_boxes'],case['log']);w,h=log.size;s=1024/max(w,h)
    context=Image.fromarray(io.read_rgb(log,[0,0,w,h],scale=s));d=ImageDraw.Draw(context)
    for name,colour in [('lithology_legend','#d33b4d'),('interpreted_lithology_column','#1877cc'),('depth_column','#1b9657')]:
        box=log.boxes[name];scaled=[round(v*s) for v in box];d.rectangle(scaled,outline=colour,width=3);d.text((max(0,scaled[0]-10),max(0,scaled[1]-14)),name.replace('_',' '),font=small,fill=colour)
    context.save(out/'source-context.png')
    full=io.read_rgb(log,'interpreted_lithology_column',y0=window[0],y1=window[1]);ink=full.max(2)<90;paper=full.min(2)>235;edge=(ink.mean(0)>.6)|(paper.mean(0)>.97);left,right=0,full.shape[1]
    while left<right-4 and edge[left]:left+=1
    while right>left+4 and edge[right-1]:right-=1
    if not np.array_equal(full[:,left:right],col):raise ValueError('Source column coordinate transform mismatch')
    x0=log.boxes['interpreted_lithology_column'][0]+left;x1=x0+col.shape[1];log.close()
    boxes=dict(source_column=[x0,window[0],x1,window[1]],queries=[dict(box=[x0,c['y0'],x1,c['y1']],pixel_sha256=c['pixel_sha256'],blank=c['blank']) for c in data['chunks']],truth=[dict(box=[x0,r['top_y'],x1,r['bottom_y']],raw_label=r['lithology'],grading_label=canonical_grading_label(r['lithology'])) for r in truth],predictions=[dict(box=[x0,r['top_y'],x1,r['bottom_y']],raw_label=r['lithology'],grading_label=canonical_grading_label(r['lithology'])) for r in pred])
    worker.write(out/'boxes.json',boxes)
    errors=Counter((str(a),str(b)) for a,b in zip(gg,pp) if a and a!=b)
    explanation='; '.join(f'{a} to {b or "blank"}: {n} rows' for (a,b),n in errors.most_common(4)) or 'All labelled rows match under the Clay/Shale rule.'
    def image(name):return '<img style="max-width:100%;height:auto" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode('ascii')+'">'
    rows=''.join('<tr><td>'+name+'</td><td>'+f"{m['score']:.4f}"+'</td><td>'+f"{m['row_acc']:.1%}"+'</td><td>'+f"{m['boundary_f1']:.1%}"+'</td></tr>' for name,m in [('Strict',result['strict_harness']),('Clay/Shale equivalent',result['clay_shale_equivalent_v1'])])
    page='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;color:#172331;background:#fff}p{line-height:1.5}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd3dd}code{overflow-wrap:anywhere}h2{margin-top:28px}</style></head><body><h1>Jev end-to-end case</h1><p><code>'+case_id+'</code> · '+html.escape(case['log'])+'</p><table><tr><th>Grading</th><th>Combined score</th><th>Row accuracy</th><th>Boundary F1</th></tr>'+rows+'</table><p>'+html.escape(explanation)+'</p><p>Raw source captions and contacts are preserved. Clay, Shale and approved compound spelling receive equal grading credit. Truth is used only after inference. This is development evidence.</p><p>CPU crop export: '+f"{data['timing_seconds']:.3f}"+' s. Local API control: '+f"{decision['seconds']:.3f}"+' s. CPU interval/scoring step: '+f"{result['seconds']:.3f}"+' s. Shared OCR calibration was cached; queue, transfer and original OCR time are separate.</p><h2>Input boxes</h2>'+image('source-context.png')+'<h2>Whole scored window</h2>'+image('intervals.png')+'<h2>Largest error region</h2>'+image('error-zoom.png')+'<p>Source fingerprint: <code>'+data['source_fingerprint']+'</code></p><p>Scorer SHA-256: <code>'+worker.sha(ROOT/'source/harness/score.py')+'</code><br>Grading SHA-256: <code>'+worker.sha(ROOT/'source/strategies/grading_policy.py')+'</code></p></body></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    worker.write(out/'report-receipt.json',dict(status='ok',case_id=case_id,seconds=time.monotonic()-start,source_fingerprint=env['source_fingerprint'],render_source_sha256=worker.sha(__file__),metric_sha256=worker.sha(ROOT/'source/harness/score.py'),grading_sha256=worker.sha(ROOT/'source/strategies/grading_policy.py'),explanation=explanation,artifacts={name:worker.sha(out/name) for name in ['source-context.png','intervals.png','error-zoom.png','boxes.json','report.html']}))
    print('REPORT',case_id,explanation,flush=True)


if __name__=='__main__':main(sys.argv[1])
