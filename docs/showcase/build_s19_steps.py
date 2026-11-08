"""Build eight S19 figures from released scans, with no ground-truth labels.

Run in an environment with the pipeline dependencies installed::
  python docs/showcase/build_s19_steps.py --input-dir /path/to/official-inputs

The input directory must contain <case>.tif and <case>.json for CASES below.
Optional --pipeline-root selects an installed copy of pipeline/s19. Figures use
runtime.run with temporary wrappers to capture actual intermediate arrays. The
wrappers always call the original functions and are restored after inference.
Only small crops and downscaled pages are saved. JSON metrics go to stdout.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CASES = ['c27f24013e5cc155', '49ab92322aa76b26', 'fc5af26c8937097c']
INK = '#172b3a'
MUTED = '#475569'
BLUE = '#1267a5'
RED = '#b32938'
GREEN = '#15734a'
PALETTE = ['#377eb8', '#e67e22', '#4daf4a', '#984ea3', '#d13949', '#a6761d', '#dc78ad', '#64748b', '#17a2a4', '#8d6e63', '#c0a21b', '#5260bc', '#70a330', '#ab5694', '#a1c6de', '#75502d']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', required=True, type=Path)
    ap.add_argument('--pipeline-root', type=Path, default=Path(__file__).resolve().parents[2] / 'pipeline/s19')
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 's19')
    ap.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    args = ap.parse_args()
    sys.path.insert(0, str(args.pipeline_root.resolve()))
    from harness import io, depth
    from ml.combine import runtime
    from ml.boundary.model import raw_features, features
    from ml.integrate.core import entries_only, evidence
    from ml.match.features import describe, matrix
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    Image.MAX_IMAGE_PIXELS = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = lambda n: ImageFont.truetype(args.font, n)
    metrics = {}

    def text(im, xy, value, size=20, fill=INK):
        ImageDraw.Draw(im).text(xy, str(value), font=font(size), fill=fill)

    def canvas(title, caption, height=1000):
        im = Image.new('RGB', (1600, height), 'white')
        text(im, (30, 18), title, 32)
        text(im, (30, 65), caption, 19, MUTED)
        return im

    def save(im, name, info):
        assert im.width <= 1600
        im.save(args.output_dir / name, optimize=True)
        metrics[name] = info
        print(json.dumps({name: info}), flush=True)

    def image(im, rgb, box):
        p = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        p.thumbnail((box[2]-box[0], box[3]-box[1]), Image.Resampling.LANCZOS)
        im.paste(p, (box[0], box[1]))
        return p.size

    def curve(im, values, box, color=BLUE, limits=None):
        v = np.asarray(values)
        lo, hi = limits if limits else (float(v.min()), float(v.max()))
        hi = max(hi, lo+1e-6)
        x0,y0,x1,y1 = box
        draw = ImageDraw.Draw(im)
        draw.rectangle(box, outline='#cbd5e1')
        pts = [(x0+int((float(q)-lo)/(hi-lo)*(x1-x0)), y0+int(i/max(1,len(v)-1)*(y1-y0))) for i,q in enumerate(v)]
        draw.line(pts, fill=color, width=2)
        text(im, (x0,y1+6), f'{lo:.2f}       {hi:.2f}', 15, MUTED)

    logs = []
    for case in CASES:
        path = args.input_dir / (case+'.tif')
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest.startswith(case), (case, digest)
        boxes = json.loads((args.input_dir / (case+'.json')).read_text(encoding='utf-8'))['bounding_boxes']
        logs.append(io.Log(Image.open(path), boxes, case))

    im = canvas('01  Three released pages: pixels and input boxes', 'Blue: legend   |   Red: rock column   |   Green: depth column. Boxes are the supplied annotations.', 1050)
    for i, log in enumerate(logs):
        x = 30+i*525
        text(im, (x,110), log.name, 21)
        # Bounded strip reader avoids a full RGB allocation for the long page.
        scale = min(460/log.size[0], 790/log.size[1])
        rgb = io.read_rgb(log, [0,0,*log.size], scale=scale)
        image(im,rgb,(x,150,x+470,940))
        d = ImageDraw.Draw(im)
        for key,col in [('lithology_legend',BLUE),('interpreted_lithology_column',RED),('depth_column',GREEN)]:
            b = log.boxes[key]
            d.rectangle((x+b[0]*scale,150+b[1]*scale,x+b[2]*scale,150+b[3]*scale),outline=col,width=3)
        text(im,(x,955),f'{log.size[0]} x {log.size[1]} source pixels',18)
    save(im,'01-inputs.png',{'pages':3,'boxes':9,'ids':CASES})

    entries_all = [runtime.parsed(log) for log in logs]
    im = canvas('02  The printed legend supplies the names and cluster IDs', 'Boxes show parser detections. The list gives the OCR label and its cluster ID. No evaluation labels are read.', 1320)
    counts=[]
    for i,(log,entries) in enumerate(zip(logs,entries_all)):
        y=110+i*395
        rgb=io.read_rgb(log,'lithology_legend')
        text(im,(30,y),log.name,21)
        size=image(im,rgb,(30,y+40,920,y+355))
        sx,sy=size[0]/rgb.shape[1],size[1]/rgb.shape[0]
        d=ImageDraw.Draw(im)
        for e in entries:
            dx,dy=log.boxes['lithology_legend'][:2]
            b=[e.swatch_box[0]-dx,e.swatch_box[1]-dy,e.swatch_box[2]-dx,e.swatch_box[3]-dy]
            d.rectangle((30+b[0]*sx,y+40+b[1]*sy,30+b[2]*sx,y+40+b[3]*sy),outline=BLUE,width=2)
            text(im,(30+b[0]*sx,y+40+b[1]*sy),e.id,13,RED)
        for j,e in enumerate(entries):
            xx=950+(j//13)*310
            text(im,(xx,y+38+(j%13)*26),f'{e.id}: {e.label}'[:32],15)
        counts.append(len(entries))
    save(im,'02-legend-parse.png',dict(zip(CASES,counts)))

    log=logs[1]
    # Fixed excerpt, also used for the clustering comparison.
    y0=int(log.boxes['interpreted_lithology_column'][1])+500
    window=(y0,y0+1500)
    captured={}
    def capture(name, fn):
        def wrapped(*a,**kw):
            value=fn(*a,**kw)
            captured[name]=value
            return value
        return wrapped
    with patch.object(runtime,'features',capture('features',runtime.features)), patch.object(runtime,'boundary_probabilities',capture('prob',runtime.boundary_probabilities)), patch.object(runtime,'proposals',capture('proposals',runtime.proposals)):
        rows=runtime.run(log,window,{})
    rgb=io.read_rgb(log,'interpreted_lithology_column',*window)
    prob=captured['prob']; proposals,rules=captured['proposals']
    entries=entries_only(entries_all[1]); _,matcher,cfg=runtime.load()
    raw=raw_features(rgb); feat=captured['features']
    subtitle=f'{log.name} | full-page rows {window[0]}-{window[1]} | native-resolution inference'
    def excerpt(im, x=70):
        p=Image.fromarray(rgb).resize((110,720),Image.Resampling.NEAREST)
        im.paste(p,(x,155))
        text(im,(x,115),'Source',20)
        text(im,(x,890),f'y={window[0]}-{window[1]}',15)
    im=canvas('03  Each row becomes a visual feature vector',subtitle)
    excerpt(im)
    tracks=[('Mean RGB',raw[:,:3].mean(1)),('Dark fraction',raw[:,3]),('H transitions',raw[:,4]),('V transitions',raw[:,5]),('Panel 1 mean (z)',feat[:,6]),('Panel 2 mean (z)',feat[:,12]),('Panel 3 mean (z)',feat[:,18])]
    for i,(name,vals) in enumerate(tracks):
        x=240+i*187
        text(im,(x,120),name,16)
        curve(im,vals,(x,155,x+145,875))
    text(im,(30,940),'32 channels: RGB + ink + transitions, three panels, ink thresholds, and changes at radii 4, 12 and 32.',20)
    save(im,'03-row-features.png',{'rows':len(rgb),'channels':feat.shape[1],'window':window})

    im=canvas('04  BoundaryNet and geometric proposals',subtitle)
    excerpt(im)
    curve(im,prob,(300,155,800,875),limits=(0,1))
    text(im,(300,115),'Boundary probability (mixed CNNs)',21)
    d=ImageDraw.Draw(im)
    for p in proposals[1:-1]:
        yy=155+int(p/len(rgb)*720);d.line((930,yy,1110,yy),fill=BLUE,width=2)
    for p in rules:
        yy=155+int(p/len(rgb)*720);d.line((1200,yy,1470,yy),fill=RED,width=3)
    text(im,(930,115),'PELT + rules',21);text(im,(1200,115),'Drawn rules',21)
    text(im,(30,940),f'{len(proposals)-2} internal proposals; {len(rules)} rules; peak p={prob.max():.3f}. CNN dilations: 1, 2, 4, 8, 16, 32.',20)
    save(im,'04-boundary-probability.png',{'proposals':len(proposals)-2,'rules':len(rules),'max_probability':float(prob.max())})

    # Sample actual decoded segments, not manually chosen lithology names.
    ranked=sorted(rows,key=lambda r:r['bottom_y']-r['top_y'],reverse=True)
    chosen=[]
    for row in ranked:
        if row['cluster_id'] not in {r['cluster_id'] for r in chosen}: chosen.append(row)
        if len(chosen)==6: break
    chosen += [r for r in ranked if r not in chosen][:6-len(chosen)]
    chosen.sort(key=lambda r:r['top_y'])
    im=canvas('05  Match each segment against the page legend', 'FFT hatch, Lab colour, ink, run lengths, LBP and orientation feed the learned pair scorer.', 1180)
    mono=np.mean(np.ptp(rgb.astype(np.int16),axis=2)>12)<.01
    refs=[describe(e.swatch) for e in entries]; matches=[]
    for j,row in enumerate(chosen):
        a,b=row['top_y']-y0,row['bottom_y']-y0
        strip=rgb[a:b]; scores=matcher.predict_proba(matrix(describe(strip,trim=True),refs,mono))[:,1]
        order=np.argsort(scores)[::-1][:3]; yy=115+j*160
        image(im,strip,(35,yy,145,yy+125));text(im,(165,yy),f'Rows {a+y0}-{b+y0}',19)
        for k,index in enumerate(order):
            e=entries[index]; xx=440+k*380
            image(im,e.swatch,(xx,yy,xx+95,yy+82))
            text(im,(xx+110,yy),f'p={scores[index]:.3f}',22)
            text(im,(xx,yy+90),f'{e.id}: {e.label}'[:35],18)
        matches.append([round(float(scores[t]),4) for t in order])
    text(im,(30,1115),'Pair probabilities are independent match scores; they do not sum to one. The decoder then combines segment evidence.',19)
    save(im,'05-legend-matching.png',{'segments':len(chosen),'top3_probabilities':matches})

    colors={e.id:tuple(int(v) for v in e.mean_rgb) for e in entries}
    def decoded(im,x,width=480,labels=True):
        d=ImageDraw.Draw(im)
        for r in rows:
            a=155+(r['top_y']-y0)/len(rgb)*720;b=155+(r['bottom_y']-y0)/len(rgb)*720
            col=colors.get(r['cluster_id'],(200,200,200))
            d.rectangle((x,a,x+100,b),fill=col,outline=INK)
            entry=next((e for e in entries if e.id==r['cluster_id']),None)
            if entry is not None and int(b)-int(a)>2:
                tile=Image.fromarray(entry.swatch)
                tile=tile.resize((98,max(1,round(tile.height*98/tile.width))),Image.Resampling.NEAREST)
                for yy in range(int(a)+1,int(b),tile.height):
                    im.paste(tile.crop((0,0,98,min(tile.height,int(b)-yy))),(x+1,yy))
            if labels and b-a>=19:text(im,(x+115,(a+b)/2-9),f"{r['cluster_id']}: {r['lithology']}",17)
    im=canvas('06  Joint decoding produces the final layers',subtitle)
    excerpt(im);decoded(im,390)
    text(im,(390,115),'Legend swatch + OCR label',21)
    d=ImageDraw.Draw(im)
    boundaries=sorted({r[e] for r in rows for e in ('top_y','bottom_y')})
    for b in boundaries:
        yy=155+(b-y0)/len(rgb)*720;d.line((1170,yy,1480,yy),fill=RED,width=2)
    text(im,(1170,115),'Final edges',21)
    text(im,(30,940),f'{len(rows)} layers; {len(boundaries)} edges. Joint segment DP: min_bed={cfg["min_bed"]} px; blank and Unknown states; preserve drawn rules.',19)
    save(im,'06-decoded-column.png',{'layers':len(rows),'edges':len(boundaries),'min_bed':cfg['min_bed']})

    log=logs[0]
    y0=int(log.boxes['depth_column'][1])+2200
    # Keep OCR cache writes outside the installed pipeline and Git output.
    with patch.object(io, 'ROOT', args.output_dir.parent / 's19-work-cache'):
        calib=runtime.checked_depth(log,depth.calibrate(log))
        cache=io.ROOT/'out/_cache/depth'/f'{log.name}.rapidocr.json'
    data=json.loads(cache.read_text(encoding='utf-8')); points=np.asarray(data['points'])
    residual=(points[:,1]-calib.b)/calib.a-points[:,0]
    im=canvas('07  OCR ticks fit the pixel-to-depth scale',f'{log.name} | d = {calib.a:.8f} y + {calib.b:.5f} | {calib.method}',1050)
    top,bottom=map(int,log.boxes['depth_column'][1::2])
    # Show a bounded crop of the track plus all fit points in the chart.
    drgb=io.read_rgb(log,'depth_column',y0,y0+1500)
    image(im,drgb,(50,150,220,870))
    visible=(points[:,0]>=y0)&(points[:,0]<y0+1500)
    for p in points[visible]:text(im,(230,150+(p[0]-y0)/1500*720),f'{p[1]:g}',18,BLUE)
    d=ImageDraw.Draw(im);box=(490,155,1100,875)
    vmin,vmax=calib.to_depth(top),calib.to_depth(bottom)
    def xy(y,v):return (490+(v-vmin)/(vmax-vmin)*610,155+(y-top)/(bottom-top)*720)
    d.rectangle(box,outline='#cbd5e1');d.line([xy(top,vmin),xy(bottom,vmax)],fill=GREEN,width=3)
    for (y,v),r in zip(points,residual):
        x,yy=xy(y,v);d.ellipse((x-4,yy-4,x+4,yy+4),fill=BLUE if abs(r)<16 else RED)
    text(im,(490,115),'All OCR anchors: depth (x), page row (y)',20)
    text(im,(1170,160),f'{calib.n_points} fit anchors',22)
    text(im,(1170,210),f'RMSE {calib.rmse_px:.3f} px',22)
    text(im,(1170,270),'Residual (pixels)',20)
    curve(im,np.clip(residual,-20,20),(1200,320,1500,800),limits=(-20,20))
    text(im,(30,945),f'Units: {data.get("unit", "unknown")}, as read by OCR. Released sidecar: units not independently verified; ft assumed.',19)
    text(im,(30,978),'Blue points: residual < 16 px. Red points: rejected. Residual trace uses OCR point order, clipped at +/-20 px.',18,MUTED)
    save(im,'07-depth-calibration.png',{'anchors':calib.n_points,'rmse_px':calib.rmse_px,'a':calib.a,'b':calib.b,'ocr_unit':data.get('unit'),'unit_verified':False})

    log=logs[1]
    y0=window[0]
    im=canvas('08  Row clustering cannot supply the printed rock names',subtitle,1050)
    excerpt(im)
    cluster_stats={}
    for j,(name,xdata) in enumerate([('Colour only',raw[:,:3]),('Colour + texture',raw)]):
        z=(xdata-xdata.mean(0))/np.maximum(xdata.std(0),.03)
        with threadpool_limits(limits=2):
            labels=KMeans(n_clusters=len(entries),random_state=19,n_init=10).fit_predict(z)
        xx=350+j*350
        text(im,(xx,115),name,21)
        strip=np.array([[tuple(bytes.fromhex(PALETTE[int(k)%len(PALETTE)][1:]))] for k in labels],dtype=np.uint8)
        im.paste(Image.fromarray(strip).resize((160,720),Image.Resampling.NEAREST),(xx,155))
        changes=int(np.count_nonzero(np.diff(labels)))
        cluster_stats[name]=changes
        text(im,(xx,900),f'{changes} row label changes',19)
    decoded(im,1080,labels=False);text(im,(1080,115),'S19: legend names',21)
    for i,e in enumerate(entries):text(im,(1210,170+i*30),f'{e.id}: {e.label}'[:29],17)
    text(im,(30,960),f'K-means: k={len(entries)} (legend entries), seed=19. Cluster colours are arbitrary; S19 patterns come from swatches.',19)
    text(im,(30,995),'Hatch ink and width changes can split rows into stripes. This is an unscored diagnostic, not a ground-truth accuracy claim.',18,MUTED)
    save(im,'08-why-clustering-failed.png',{'k':len(entries),'row_label_changes':cluster_stats,'s19_layers':len(rows)})
    for log in logs:log.close()
    print('CHECK: 8 figures built from verified released scans',flush=True)


if __name__ == '__main__':
    main()
