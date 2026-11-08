"""Make a truth-free demo. Run with python -I; all writes stay beside this file."""
import sys
sys.dont_write_bytecode = True
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '3'
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import html
import json
import time
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'external_unlabelled'
sys.path.insert(0, str(ROOT / 'bakeoff'))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from harness import io, legend
from strategies import s04_kmeans_viterbi as strategy
from threadpoolctl import threadpool_limits
cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS = None
# The shared legend parser writes caches relative to io.ROOT.
io.ROOT = HERE / 'cache'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def image_path(name):
    if Path(name).name != name:
        raise ValueError('Invalid image name')
    for ext in ('.tif', '.tiff'):
        path = DATA / 'images' / (name + ext)
        if path.exists():
            return path
    raise FileNotFoundError(name)


def select():
    # Membership is applied before any historical metric is used or retained.
    dev = json.loads((ROOT/'external_tests/eval_set_dev.json').read_text())
    assert len(dev) == 51
    allowed = {r['log'] for r in dev}
    assert not any('forsyth' in n.lower() for n in allowed)
    groups = collections.defaultdict(list)
    with (ROOT/'bakeoff/analysis/transfer/per_window_metrics.csv').open() as f:
        for r in csv.DictReader(f):
            if r['domain'] != 'ext_dev' or r['log'] not in allowed:
                continue
            if r['strategy'] in ('s01', 's02', 's03', 's04', 's11'):
                groups[r['strategy']].append(r)
    ranked = []
    for name, rows in groups.items():
        assert len(rows) == len(dev)
        wells = collections.defaultdict(list)
        for row in rows:
            wells[row['well']].append(float(row['score']))
        ranked.append(dict(strategy=name, windows=len(rows), wells=len(wells),
                           well_macro_score=float(np.mean([np.mean(v) for v in wells.values()])),
                           window_macro_score=float(np.mean([float(r['score']) for r in rows]))))
    ranked.sort(key=lambda r: -r['well_macro_score'])
    assert ranked[0]['strategy'] == 's04'
    dump(HERE/'selection.json', dict(selected='s04', criterion='mean score by well',
        data='current ext_dev, 51 windows, 5 wells', ranking=ranked,
        inherited_weights_used=False, dev_manifest_sha256=sha(ROOT/'external_tests/eval_set_dev.json'),
        note='Fixed strategies only. No fitted weights or confidence calibrator from the old six-well dev set.'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plans():
    result = []
    def add(name, col, leg, companion=None, rotation=0):
        result.append(dict(id=name, column=col, legend=leg,
                           legend_page=companion or name, rotation=rotation,
                           method='Manual box from image pixels. No interval labels.'))
    p='ul_r4_bw_hatch_oregon_dogami_'
    add(p+'CC23-04-65_mud_p001', [700,1110,900,13700], [815,220,1615,435])
    add(p+'CC33-03-65_mud_p001', [720,1520,920,14400], [805,460,1620,685])
    add(p+'CC43-34-65-RD_mud-r_p001', [690,1110,885,9220], [800,225,1615,440])
    add(p+'CoosCounty1-7_mud_p001', [575,1050,745,24300], [665,600,1295,755])
    add(p+'Hammerberg1-RD_mud-R_p001', [705,1570,900,14180], [825,545,1645,775])
    add(p+'Keys1_mud1_p001', [760,1560,960,23650], [875,505,1675,715])
    add(p+'OM12C-03-65_mud_p001', [860,1490,1060,16480], [975,475,1775,685])
    add(p+'CC12-19-65_mud_p001', [425,1390,582,13050], [240,930,1780,1100])
    p='ul_r3_'
    add(p+'oregon_dogami_Adams24-34-75_mud_p001', [660,1950,840,15800], [735,1005,1465,1200])
    add(p+'oregon_dogami_Adams31-34-75-RD_mud_p001', [430,2510,575,15000], [110,1955,1880,2180])
    add(p+'dhi_wister_wister_p002', [765,0,930,2829], [45,1630,1085,2155], p+'dhi_wister_wister_p001')
    add(p+'precision_wl_log-01_p001', [680,1660,800,2200], [825,565,1100,1190])
    add(p+'precision_wl_log-01_p002', [680,0,800,2200], [825,565,1100,1190], p+'precision_wl_log-01_p001')
    add(p+'stratagraph_Measured_Depth_Log_Example_p002', [490,210,640,2200], [875,1010,1210,1830], p+'stratagraph_Measured_Depth_Log_Example_p001')
    add(p+'mississippi_mdl123045000360000a-001_p001', [177,265,217,680], [420,128,640,185])
    add(p+'mississippi_mdl123045000360000a-002_p001', [177,30,217,505], [420,128,640,185],p+'mississippi_mdl123045000360000a-001_p001')
    add(p+'utah_forge_mud_log_p001', [292,1415,470,17500], [585,765,765,1190])
    return result


def font(size):
    # Pillow's bundled font is portable. No machine font path is required.
    return ImageFont.load_default(size=size)


def lines(draw, xy, text, width=54, size=18, fill='#243447'):
    x,y=xy
    for line in textwrap.wrap(str(text), width=width):
        draw.text((x,y),line,font=font(size),fill=fill)
        y += size+5
    return y


COLORS=['#0072b2','#d55e00','#009e73','#cc79a7','#e69f00','#56b4e9','#7361a6','#686868']


def render(log, window, rows, entries, dest, title, warnings):
    y0,y1=window; rgb=io.read_rgb(log,'interpreted_lithology_column',y0,y1)
    canvas=Image.new('RGB',(1000,1280),'white');d=ImageDraw.Draw(canvas)
    lines(d,(24,18),title,65,23)
    d.text((24,84),f'S04 | image rows {y0} to {y1} | Confidence: unverified',font=font(19),fill='#111111')
    d.text((24,117),'Source pixels',font=font(18),fill='black')
    d.text((230,117),'Label overlay',font=font(18),fill='black')
    d.text((445,117),'Predicted intervals',font=font(18),fill='black')
    top=155;h=860
    source=Image.fromarray(rgb).resize((180,h))
    canvas.paste(source,(25,top));canvas.paste(source,(230,top))
    names=sorted({r['lithology'] for r in rows});pal={s:COLORS[i%len(COLORS)] for i,s in enumerate(names)}
    overlay=Image.new('RGBA',(180,h),(0,0,0,0));od=ImageDraw.Draw(overlay)
    for i,r in enumerate(rows):
        a=round((r['top_y']-y0)*h/(y1-y0));b=round((r['bottom_y']-y0)*h/(y1-y0))
        color=pal[r['lithology']];od.rectangle((0,a,27,max(a,b-1)),fill=color)
        od.line((0,a,179,a),fill=color,width=2)
        if b-a>=21:od.text((33,a+2),str(i+1),font=font(16),fill='black',stroke_width=2,stroke_fill='white')
    composite=Image.alpha_composite(source.convert('RGBA'),overlay)
    canvas.paste(composite.convert('RGB'),(230,top))
    yy=top
    for i,r in enumerate(rows):
        if yy>955:
            lines(d,(445,yy),f'{len(rows)-i} more intervals. See the CSV file.',45,17);break
        d.rectangle((445,yy+3,459,yy+17),fill=pal[r['lithology']])
        yy=lines(d,(470,yy),f"{i+1}. {r['lithology']} [{r['top_y']}, {r['bottom_y']})",42,17)+8
    yy=1040
    for warning in warnings[:3]: yy=lines(d,(24,yy),warning,94,17,fill='#922b21')+3
    d.text((24,1210),'No truth is available. Labels need review. Depth is not calibrated.',font=font(17),fill='#111111')
    dest.parent.mkdir(exist_ok=True,parents=True);canvas.save(dest)


def diagnostics(log, row, entries):
    rgb=io.read_rgb(log,'interpreted_lithology_column',row['top_y'],row['bottom_y'])
    q=strategy._feature(rgb)
    refs=np.asarray([strategy._feature(e.swatch) for e in entries])
    # A separate raw descriptor check. This is not S04's cluster confidence.
    distances=np.linalg.norm(refs-q,axis=1);order=np.argsort(distances)
    d0=float(distances[order[0]]);d1=float(distances[order[1]]) if len(order)>1 else None
    return dict(top_y=row['top_y'],bottom_y=row['bottom_y'],lithology=row['lithology'],
                confidence=None, confidence_status='unverified',
                raw_descriptor_nearest=entries[int(order[0])].label,
                raw_descriptor_gap=None if d1 is None else (d1-d0)/max(d1,1e-9),
                raw_descriptor_agrees=entries[int(order[0])].label==row['lithology'])


def validate_rows(rows, window):
    previous=window[0]
    for row in rows:
        if not previous<=row['top_y']<row['bottom_y']<=window[1]:
            raise ValueError('Rows overlap or leave the image window')
        previous=row['bottom_y']


def legend_card(log, entries, dest):
    rgb=io.read_rgb(log,'lithology_legend')
    im=Image.fromarray(rgb);im.thumbnail((950,550))
    canvas=Image.new('RGB',(1000,760+max(1,len(entries))*32),'white')
    d=ImageDraw.Draw(canvas)
    d.text((24,20),'Legend pixels and parsed names',font=font(24),fill='black')
    d.text((24,60),'Names are not corrected by hand. Each name needs review.',font=font(19),fill='black')
    canvas.paste(im,(24,110))
    for i,e in enumerate(entries):
        sw=Image.fromarray(e.swatch);sw.thumbnail((80,28));canvas.paste(sw,(24,710+i*32))
        d.text((120,710+i*32),e.label,font=font(20),fill='black')
    dest.parent.mkdir(exist_ok=True);canvas.save(dest)


def run(limit=None):
    select(); manifest=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8')))
    metadata={r['id']:r for r in manifest}; plan=plans()
    if limit:plan=plan[:limit]
    dump(HERE/'boxes.json',plan)
    started=time.monotonic(); results=[];cards=[]
    for item in plan:
        name=item['id']; source=Image.open(image_path(name));w,h=source.size
        col=list(item['column']);col[3]=min(col[3],h)
        assert 0<=col[0]<col[2]<=w and 0<=col[1]<col[3]<=h
        log=io.Log(source,{'interpreted_lithology_column':col},name)
        warnings=['The boxes were set by hand. Scan drift can move the column outside the box.',
                  'S04 has no calibrated confidence. Each label needs a human check.',
                  'A single label cannot show all rocks in a mixed column.']
        legsource=Image.open(image_path(item['legend_page']))
        leglog=io.Log(legsource,{'lithology_legend':item['legend']},item['legend_page'])
        t=time.monotonic()
        try:
            entries=legend.parse(leglog)
            entries=[e for e in entries if e.is_lithology and e.swatch.size]
            legend_card(leglog,entries,HERE/'legends'/(name+'.png'))
            rec=dict(id=name,source=metadata[name]['source'],well=metadata[name]['well'],strategy='s04',
                     legend_page=item['legend_page'],legend_labels=[e.label for e in entries],
                     confidence='unverified',windows=[],warnings=warnings)
            if not entries:
                rec['status']='blocked_no_usable_legend';rec['intervals']=0
                warnings.insert(0,'The parser found no usable lithology legend. No labels were made.')
                dest=HERE/'overlays'/(name+'_blocked.png')
                render(log,(col[1],min(col[1]+4000,col[3])),[],[],dest,metadata[name]['well'],warnings)
                cards.append((name,dest))
            else:
                wt=time.monotonic()
                with threadpool_limits(limits=3):
                    output=strategy.run(log,(col[1],col[3]),{'legend':entries,'device':'cpu','cache_dir':str(HERE/'cache')})
                allrows=output[0] if isinstance(output,tuple) else output
                validate_rows(allrows,(col[1],col[3]))
                rec['inference_seconds']=time.monotonic()-wt
                allconfidence=[diagnostics(log,row,entries) for row in allrows]
                for y0 in range(col[1],col[3],4000):
                    window=(y0,min(y0+4000,col[3]));wt=time.monotonic()
                    rows=[dict(row,top_y=max(window[0],row['top_y']),bottom_y=min(window[1],row['bottom_y']))
                          for row in allrows if row['top_y']<window[1] and row['bottom_y']>window[0]]
                    dest=HERE/'overlays'/f'{name}_{y0:06d}.png'
                    render(log,window,rows,entries,dest,metadata[name]['well'],warnings)
                    cards.append((name,dest))
                    rec['windows'].append(dict(y0=y0,y1=window[1],intervals=len(rows),
                         seconds=time.monotonic()-wt,overlay=str(dest.relative_to(HERE))))
                io.save_rows(HERE/'predictions'/(name+'.csv'),allrows)
                dump(HERE/'confidence'/(name+'.json'),allconfidence)
                rec.update(status='predicted',intervals=len(allrows),
                    labelled_row_fraction=sum(r['bottom_y']-r['top_y'] for r in allrows)/(col[3]-col[1]))
            rec['seconds']=time.monotonic()-t
        except Exception as error:
            rec=dict(id=name,status='error',error=f'{type(error).__name__}: {error}')
        finally:
            log.close();leglog.close()
        results.append(rec);dump(HERE/'results.json',results)
        print(name,rec['status'],rec.get('intervals'),flush=True)
    done={r['id']:r for r in results}
    inventory=[]
    for row in manifest:
        result=done.get(row['id'])
        inventory.append(dict(row,status=result['status'] if result else 'not_run_missing_reviewed_boxes',
            note='See results.json' if result else 'The input set has no semantic boxes. This demo has no reviewed box for this page.'))
    with (HERE/'inventory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    sheets=[]
    # Two cards per sheet. Each sheet remains readable on a phone at full width.
    for i in range(0,len(cards),2):
        sheet=Image.new('RGB',(1000,2560),'white')
        for j,(_,path) in enumerate(cards[i:i+2]):
            with Image.open(path) as im:sheet.paste(im,(0,j*1280))
        dest=HERE/'contact_sheets'/f'sheet_{i//2+1:03d}.jpg';dest.parent.mkdir(exist_ok=True)
        sheet.save(dest,quality=90);sheets.append(str(dest.relative_to(HERE)))
    import psutil
    summary=dict(input_pages=len(manifest),reviewed_pages=len(plan),
        statuses=dict(collections.Counter(r['status'] for r in results)),
        windows=sum(len(r.get('windows',[])) for r in results),
        intervals=sum(r.get('intervals',0) for r in results),
        contact_sheets=len(sheets),seconds=time.monotonic()-started,
        current_rss_gib=psutil.Process().memory_info().rss/2**30,
        thread_limit=3,strategy='s04',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
          [ROOT/'bakeoff/strategies/s04_kmeans_viterbi.py',ROOT/'bakeoff/harness/legend.py',DATA/'manifest.csv']})
    dump(HERE/'summary.json',summary)
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
      '<title>Unlabelled log demo</title><style>body{margin:16px auto;padding:0 12px;max-width:1000px;background:white;color:#17212b;font:18px system-ui}img{width:100%;height:auto}a{color:#0072b2}figure{margin:24px 0;border-top:1px solid #bbb}p{line-height:1.5}</style>',
      '<h1>Unlabelled log demo</h1><p>S04. All labels need review. Confidence is unverified. No truth was used for these images.</p>',
      '<p><a href="REPORT.md">Read the report</a> · <a href="inventory.csv">Check all input pages</a> · <a href="selection.json">Check strategy selection</a></p>',
      f'<p>{len(manifest)} input pages. {len(plan)} pages have demo boxes. {summary["windows"]} column views are shown. The other pages have no reviewed boxes.</p>',
      '<p>Each source column was processed in one call. The pictures show crops of that result. View edges are not bed boundaries.</p>']
    page.append('<details><summary>Check parsed legends and output files</summary>')
    for r in results:
        name=r['id']
        page.append(f'<p>{html.escape(name)}: {r["status"]} · <a href="legends/{html.escape(name)}.png">Legend</a>')
        if r['status']=='predicted':
            page.append(f' · <a href="predictions/{html.escape(name)}.csv">Intervals</a> · <a href="confidence/{html.escape(name)}.json">Confidence record</a>')
        page.append('</p>')
    page.append('</details>')
    for s in sheets:page.append(f'<figure><a href="{html.escape(s)}"><img loading="lazy" src="{html.escape(s)}" alt="Source pixels with predicted labels"></a></figure>')
    page.append('</html>');(HERE/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.limit)
