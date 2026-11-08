"""Make a truth-free demo. Run with python -I; all writes stay beside this file."""
import sys
sys.dont_write_bytecode = True
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import html
import json
import time
import textwrap
import inspect
from functools import lru_cache
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'external_unlabelled'
sys.path.insert(0, str(ROOT / 'bakeoff'))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from harness import io, legend
from strategies import s19_combine as strategy
from ml.combine import runtime
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
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
    freeze_path = ROOT/'bakeoff/ml/final/FREEZE.json'
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    if frozen['chosen_strategy'] != 's19':
        raise ValueError('The frozen strategy is not S19')
    dump(HERE/'selection.json', dict(selected='s19',
        source='bakeoff/ml/final/FREEZE.json', source_sha256=sha(freeze_path),
        frozen_utc=frozen['frozen_utc'], dependency_hash=frozen['dependency_hash'],
        selection_sweep=False, note='Use the frozen choice. No new selection was run.'))


def configure_runtime():
    # Load unchanged models from their source folder before routing cache writes.
    runtime.load()
    runtime.HERE = HERE/'cache'/'s19'
    runtime._easy_reader.cache_clear()
    # The shared RapidOCR default uses two threads. Keep its same OCR settings.
    @lru_cache(maxsize=1)
    def single_thread_ocr():
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                        det_limit_side_len=3200, det_limit_type='max')
    legend._ocr = single_thread_ocr


class PageAndLegend:
    """Route virtual legend pixels to a companion page without a full-page copy."""
    def __init__(self, source, companion):
        self.source = source
        self.companion = companion
        self.offset = source.size[1]+1
        self.size = (max(source.size[0], companion.size[0]),
                     self.offset+companion.size[1])
        self.mode = 'RGB'

    def crop(self, box):
        x0,y0,x1,y1 = box
        if y0 >= self.offset:
            return io._crop(self.companion, (x0,y0-self.offset,x1,y1-self.offset))
        return io._crop(self.source, box)

    def close(self):
        pass


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
    d.text((24,84),f'S19 | image rows {y0} to {y1} | Confidence: unverified',font=font(19),fill='#111111')
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


def diagnostics(row):
    # Preserve native fields. None of these raw scores is calibrated confidence.
    keys = ('match_cost', 'match_margin', 'boundary_probability')
    return dict(top_y=row['top_y'], bottom_y=row['bottom_y'], lithology=row['lithology'],
                confidence=row.get('confidence'), confidence_status='unverified',
                confidence_source='native' if row.get('confidence') is not None else 'unknown',
                native_scores={k:row[k] for k in keys if k in row})


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


def run():
    process_started=time.monotonic()
    select();configure_runtime(); manifest=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8')))
    metadata={r['id']:r for r in manifest}; plan=plans()
    for folder in ('predictions','confidence','overlays','legends','contact_sheets'):
        path=HERE/folder
        if path.exists(): shutil.rmtree(path)
    dump(HERE/'boxes.json',plan)
    started=time.monotonic(); results=[];cards=[]
    for item in plan:
        name=item['id']; source=Image.open(image_path(name));w,h=source.size
        col=list(item['column']);col[3]=min(col[3],h)
        assert 0<=col[0]<col[2]<=w and 0<=col[1]<col[3]<=h
        source_log=io.Log(source,{'interpreted_lithology_column':col},name)
        warnings=['The boxes were set by hand. Scan drift can move the column outside the box.',
                  'S19 confidence is unverified. Each label needs a human check.',
                  'A single label cannot show all rocks in a mixed column.']
        legsource=Image.open(image_path(item['legend_page']))
        leglog=io.Log(legsource,{'lithology_legend':item['legend']},item['legend_page'])
        routed=PageAndLegend(source_log,leglog)
        virtual_legend=list(item['legend'])
        virtual_legend[1]+=routed.offset;virtual_legend[3]+=routed.offset
        log=io.Log(routed,{'interpreted_lithology_column':col,'lithology_legend':virtual_legend},name)
        t=time.monotonic()
        try:
            entries=runtime.parsed(log)
            entries=[e for e in entries if e.is_lithology and e.swatch.size]
            legend_card(leglog,entries,HERE/'legends'/(name+'.png'))
            rec=dict(id=name,source=metadata[name]['source'],well=metadata[name]['well'],strategy='s19',
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
                with threadpool_limits(limits=1):
                    # Remove only the three-thread resource decorator. Use the same run contract.
                    output=inspect.unwrap(strategy.run)(log,(col[1],col[3]),
                        {'legend':entries,'device':'cpu','calib':None,'cache_dir':str(HERE/'cache')})
                allrows=output[0] if isinstance(output,tuple) else output
                validate_rows(allrows,(col[1],col[3]))
                rec['inference_seconds']=time.monotonic()-wt
                allconfidence=[diagnostics(row) for row in allrows]
                for y0 in range(col[1],col[3],4000):
                    window=(y0,min(y0+4000,col[3]));wt=time.monotonic()
                    rows=[dict(row,top_y=max(window[0],row['top_y']),bottom_y=min(window[1],row['bottom_y']))
                          for row in allrows if row['top_y']<window[1] and row['bottom_y']>window[0]]
                    dest=HERE/'overlays'/f'{name}_{y0:06d}.png'
                    render(log,window,rows,entries,dest,metadata[name]['well'],warnings)
                    cards.append((name,dest))
                    rec['windows'].append(dict(y0=y0,y1=window[1],intervals=len(rows),
                         seconds=time.monotonic()-wt,overlay=dest.relative_to(HERE).as_posix()))
                io.save_rows(HERE/'predictions'/(name+'.csv'),allrows)
                dump(HERE/'confidence'/(name+'.json'),allconfidence)
                rec.update(status='predicted',intervals=len(allrows),
                    labelled_row_fraction=sum(r['bottom_y']-r['top_y'] for r in allrows)/(col[3]-col[1]))
            rec['seconds']=time.monotonic()-t
        except Exception as error:
            rec=dict(id=name,status='error',error=f'{type(error).__name__}: {error}')
        finally:
            log.close();source_log.close();leglog.close()
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
        sheet.save(dest,quality=90);sheets.append(dest.relative_to(HERE).as_posix())
    import psutil
    summary=dict(input_pages=len(manifest),reviewed_pages=len(plan),
        statuses=dict(collections.Counter(r['status'] for r in results)),
        windows=sum(len(r.get('windows',[])) for r in results),
        intervals=sum(r.get('intervals',0) for r in results),
        contact_sheets=len(sheets),seconds=time.monotonic()-started,
        current_rss_gib=psutil.Process().memory_info().rss/2**30,
        thread_limit=1,strategy='s19',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
          [ROOT/'bakeoff/strategies/s19_combine.py',ROOT/'bakeoff/ml/combine/runtime.py',
           ROOT/'bakeoff/ml/combine/selected.json',ROOT/'bakeoff/ml/combine/boundary_net.pt',
           ROOT/'bakeoff/ml/combine/boundary_real.pt',ROOT/'bakeoff/ml/combine/matcher.npz',
           ROOT/'bakeoff/ml/combine/matcher_real.npz',ROOT/'bakeoff/harness/legend_v3.py',
           ROOT/'bakeoff/ml/final/FREEZE.json',DATA/'manifest.csv',HERE/'demo.py']})
    summary.update(predicted_pages=summary['statuses'].get('predicted',0),
        failures=summary['statuses'].get('error',0),
        blocked_pages=sum(v for k,v in summary['statuses'].items() if k.startswith('blocked')),
        skipped_pages=len(manifest)-len(plan),prediction_files=len(list((HERE/'predictions').glob('*.csv'))),
        confidence_files=len(list((HERE/'confidence').glob('*.json'))),
        legend_images=len(list((HERE/'legends').glob('*.png'))),
        overlays=len(list((HERE/'overlays').glob('*.png'))),
        confidence_status='unverified',confidence_source='unknown unless returned by S19')
    summary.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        total_seconds=time.monotonic()-process_started)
    if sys.platform == 'win32':
        summary['peak_rss_gib']=psutil.Process().memory_info().peak_wset/2**30
    else:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary['peak_rss_gib']=peak/(2**30 if sys.platform == 'darwin' else 2**20)
    dump(HERE/'summary.json',summary)
    write_report(summary,results)

    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
      '<title>Unlabelled log demo</title><style>body{margin:16px auto;padding:0 12px;max-width:1000px;background:white;color:#17212b;font:18px system-ui}img{width:100%;height:auto}a{color:#0072b2}figure{margin:24px 0;border-top:1px solid #bbb}p{line-height:1.5}</style>',
      '<h1>Unlabelled log demo</h1><p>S19. All labels need review. Confidence is unverified. No truth was used for these images.</p>',
      '<p><a href="REPORT.md">Read the report</a> · <a href="inventory.csv">Check all input pages</a> · <a href="summary.json">Last summary</a> · <a href="results.json">Page results</a> · <a href="selection.json">Check frozen choice</a> · <a href="baseline_s04/REPORT.md">S04 baseline report</a></p>',
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


def write_report(summary, results):
    text = [
        '# S19 demo on unlabelled pages',
        '',
        'S19 is the frozen choice in `bakeoff/ml/final/FREEZE.json`. '
        'This run used that choice. No new selection was run.',
        '',
        f"The input set has {summary['input_pages']} pages. All {summary['reviewed_pages']} reviewed pages were checked. "
        f"S19 made {summary['intervals']} intervals on {summary['predicted_pages']} pages. "
        f"There were {summary['failures']} errors and {summary['blocked_pages']} blocked pages. "
        f"The other {summary['skipped_pages']} pages have no reviewed boxes. They were skipped.",
        '',
        'The [S04 baseline report](baseline_s04/REPORT.md) and '
        '[S04 summary](baseline_s04/summary.json) are kept with the baseline code and files. '
        'S04 made 64 intervals on 15 pages. It could not use the legends on two Mississippi pages. '
        'S19 made intervals on both pages. This result does not prove that their labels are correct.',
        '',
        '| Page | Result | Legend names | Intervals | Labelled rows |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for r in results:
        text.append(f"| {r['id']} | {r['status']} | {len(r.get('legend_labels',[]))} | "
                    f"{r.get('intervals',0)} | {r.get('labelled_row_fraction',0):.1%} |")
    text += [
        '',
        'All labels need review. There is no interval truth for these pages. '
        'No accuracy or depth error is claimed. Depth fields are empty.',
        '',
        'S19 returns raw match cost, match margin, and boundary probability. '
        'The [confidence files](confidence/) keep these native scores for each interval. '
        'Boundary probability is a boundary score. It is not a probability that the rock name is correct. '
        'The current S19 code returns no interval confidence. Its confidence value is null. '
        'All confidence records are marked unverified. If S19 returns native confidence, the demo will keep it. '
        'No S04 descriptor distance is used as S19 confidence. No S18 confidence model is used.',
        '',
        'The larger interval count needs review. Many short intervals can be false bed boundaries. '
        'OCR can give wrong or incomplete names. The stock v3 parser can clean printed names. '
        'No name was changed by hand. A single label cannot show side-by-side rock mixtures. '
        'Empty tracks, grid lines, and scan drift can affect results. '
        'The reviewed boxes are the same boxes as in the S04 demo.',
        '',
        f"The [offline gallery](index.html) has {summary['windows']} column views and "
        f"{summary['contact_sheets']} contact sheets. There are {summary['overlays']} white PNG overlays, "
        f"{summary['legend_images']} legend images, {summary['prediction_files']} prediction CSV files, "
        f"and {summary['confidence_files']} confidence JSON files. Each supported column was processed in one call. "
        'The pictures are crops of that result. View edges do not add interval boundaries.',
        '',
        'The demo calls the current S19 run function with a page, a full column window, and CPU context. '
        'S19 reads its own v3 legends. A pixel adapter supplies the reviewed companion legend when needed. '
        'The same model files and fixed decoder settings are used. Their hashes are in '
        '[summary.json](summary.json). Only the resource wrapper is removed to set one thread. '
        'The S19 models are loaded before cache paths are moved into this demo folder. '
        'RapidOCR uses the same settings with one thread per pool. Torch and numeric pools use one thread.',
        '',
        f"The last run took {summary['total_seconds']:.2f} seconds after imports. "
        f"Peak RAM was {summary['peak_rss_gib']:.3f} GiB. The limit was 4 GB. "
        'The run used one CPU thread for compute. '
        'See [run.log](run.log) and [resources.txt](resources.txt).',
        '',
        'Only external_unlabelled image pixels and its manifest were used as input data. '
        'No training, model change, threshold change, reserved truth read, benchmark run, or service change was made. '
        'All new files and caches are in this demo folder. No child agent, commit, or push was used.',
        '',
        'From the repository root, use a Python environment with the project packages. '
        'The code and local links work on Windows. No machine font path is needed.',
        '',
        '```text',
        'python -I -B bakeoff/notes/unlabelled_demo/demo.py --strategy s19',
        'python -I -B -m pytest -q -p no:cacheprovider bakeoff/notes/unlabelled_demo/test_artifacts.py --basetemp=bakeoff/notes/unlabelled_demo/pytest_tmp',
        '```',
        '',
        'The artifact checks cover the frozen choice, baseline evidence, page counts, '
        'prediction format, row bounds, native scores, image files, view coverage, and local gallery links. '
        'See [validation.log](validation.log) for the last check result.',
        '',
        f"Last summary: {summary['predicted_pages']} predicted pages; {summary['intervals']} intervals; "
        f"{summary['failures']} errors; {summary['blocked_pages']} blocked pages; {summary['skipped_pages']} skipped pages.",
    ]
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['s19'], default='s19',
                        help='Use frozen S19 on all reviewed external unlabelled pages.')
    parser.parse_args()
    with threadpool_limits(limits=1):
        run()
