"""Reproducible lithology explainer. Dataset and bakeoff are read only.
Notebook helpers below copied from project/example_notebook.ipynb.
Notebook adaptations: 13 px minimum labels, clamp labels to the canvas,
and optional number labels on the six page overviews (to avoid overlap).
Run: python artifact/make_figs.py [--only NAME ...] [--refresh-jev]
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import csv, json, colorsys, time, io, base64, argparse, traceback, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
Image.MAX_IMAGE_PIXELS = None
BOX_COLORS = {"depth_column": (255,0,0), "interpreted_lithology_column": (0,160,255), "lithology_legend": (255,130,0)}
PALETTE = [(40,160,60),(160,100,40),(240,190,0),(150,60,170),(0,150,150),(230,110,20),(200,60,120),(110,110,110)]
LINE_COLOR = (0,40,200)


def get_font(size):
    # DejaVu ships with matplotlib/most Linux installs; fall back to Pillow's built-in font otherwise
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def crop_rgb(img, box, scale=1.0, strip=2000):
    """Crop box=(x0, y0, x1, y1) out of img as RGB, optionally shrunk by `scale`.
    The crop is converted and shrunk `strip` rows at a time, so even a 1-bit image that is
    100,000+ px tall never has to exist at full size as RGB."""
    x0, y0, x1, y1 = box
    out = Image.new("RGB", (round((x1 - x0) * scale), round((y1 - y0) * scale)))
    for ys in range(y0, y1, strip):
        ye = min(ys + strip, y1)
        part = img.crop((x0, ys, x1, ye)).convert("RGB")
        top, bottom = round((ys - y0) * scale), round((ye - y0) * scale)
        if scale != 1.0:
            part = part.resize((out.width, max(1, bottom - top)), Image.Resampling.BOX)
        out.paste(part, (0, top))
    return out

def draw_boxes(img, boxes, origin=(0, 0), scale=1.0, line_width=4, font_size=16, numbered=False):
    """Draw boxes (given in full-image pixels) on img, a crop that starts at `origin` and was scaled by `scale`."""
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    for i, (name, (x0, y0, x1, y1)) in enumerate(boxes.items()):
        color = BOX_COLORS.get(name, (0, 200, 0))
        rect = [(x0 - origin[0]) * scale, (y0 - origin[1]) * scale, (x1 - origin[0]) * scale - 1, (y1 - origin[1]) * scale - 1]
        if rect[3] < 0 or rect[1] > img.height or rect[2] < 0 or rect[0] > img.width:
            continue  # this box lies outside the crop
        draw.rectangle(rect, outline=color, width=line_width)
        label = str(i+1) if numbered else name
        tx = min(rect[0]+6, img.width-draw.textlength(label,font=font)-6)
        ty = max(rect[1],0)+4+(0 if numbered else (font_size+4)*i)
        if numbered:
            draw.rectangle((tx-2,ty-1,tx+draw.textlength(label,font=font)+2,ty+font_size+3),fill='white')
        draw.text((tx,ty),label,fill=color,font=font)
    return img

def load_results(path):
    """Read a results CSV (top_depth, bottom_depth, lithology, top_y, bottom_y, and optionally cluster_id) into a list of dicts."""
    with open(path, newline="") as f:
        return [
            {"top_depth": float(r["top_depth"]), "bottom_depth": float(r["bottom_depth"]), "lithology": r["lithology"],
             "top_y": int(r["top_y"]), "bottom_y": int(r["bottom_y"]),
             "cluster_id": r.get("cluster_id") or None}  # cluster_id is optional
            for r in csv.DictReader(f)
        ]

def build_color_map(labels):
    """Give every distinct label its own colour, in order of first appearance. Works for any labels:
    the palette is used first, then extra hues are generated, so nothing is hard-coded per lithology."""
    colors = {}
    for label in labels:
        if label in colors:
            continue
        i = len(colors)
        if i < len(PALETTE):
            colors[label] = PALETTE[i]
        else:  # beyond the palette: spread hues with the golden ratio so neighbours stay distinct
            h = (i * 0.618034) % 1.0
            colors[label] = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.8))
    return colors

def overlay_results(img, results, colors, column_box, depth_box, origin, scale, alpha=150, coverage=0.5):
    """Paint every interval over the left `coverage` of the lithology column on `img` (a crop starting at
    `origin`, shrunk by `scale`) in the colour given for its label in `colors`, so the original column stays visible on the right. At every interval
    boundary a thin line runs from the depth column to the lithology column and the boundary depth is written
    to the left of the depth column, and the interval's cluster id (if it has one) to the right of the lithology
    column. Every row of `results` is drawn, however small the interval."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(13)
    x0, x1 = (column_box[0] - origin[0]) * scale, (column_box[2] - origin[0]) * scale
    xm = x0 + (x1 - x0) * coverage  # right edge of the painted part
    xd = (depth_box[0] - origin[0]) * scale  # left edge of the depth column
    boundaries = {}  # y (full-image pixels) -> depths given at that boundary
    for r in results:
        y0, y1 = (r["top_y"] - origin[1]) * scale, (r["bottom_y"] - origin[1]) * scale
        color = colors[r["lithology"]]
        draw.rectangle([x0, y0, xm, y1], fill=color + (alpha,), outline=color + (255,), width=1)
        if r.get("cluster_id") is not None:
            draw.text((x1 + 8, (y0 + y1) / 2), f"cluster {r['cluster_id']}", fill=color + (255,), font=font, anchor="lm",
                      stroke_width=3, stroke_fill=(255, 255, 255, 255))
        boundaries.setdefault(r["top_y"], []).append(r["top_depth"])
        boundaries.setdefault(r["bottom_y"], []).append(r["bottom_depth"])
    for y, depths in boundaries.items():
        yy = (y - origin[1]) * scale
        draw.line([(xd, yy), (x1, yy)], fill=LINE_COLOR + (255,), width=1)
        label = " / ".join(f"{d:g}" for d in dict.fromkeys(depths))  # more than one value only if the depths jump
        draw.text((xd - 6, yy), label, fill=LINE_COLOR + (255,), font=font, anchor="rm",
                  stroke_width=3, stroke_fill=(255, 255, 255, 255))
    print(f"drew {len(results)} intervals and {len(boundaries)} boundary labels")
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

def add_legend(img, colors, clusters=None, row_height=34):
    """Return img with a white bar on top holding one colour swatch per label (wrapping onto more rows if needed).
    `clusters` ({label: [cluster ids]}) adds the cluster ids found for each label next to its name."""
    clusters = clusters or {}
    font = get_font(15)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    slots, x, row = [], 10, 0
    texts = {name: name + (f"  (cluster {', '.join(clusters[name])})" if clusters.get(name) else "") for name in colors}
    for name in colors:
        w = 26 + int(measure.textlength(texts[name], font=font)) + 20
        if x + w > img.width and x > 10:  # no room left on this row: start a new one
            x, row = 10, row + 1
        slots.append((name, x, row))
        x += w
    bar = row_height * (row + 1)
    out = Image.new("RGB", (img.width, img.height + bar), (255, 255, 255))
    out.paste(img, (0, bar))
    draw = ImageDraw.Draw(out)
    for name, x, row in slots:
        y = row * row_height
        draw.rectangle([x, y + 8, x + 18, y + 26], fill=colors[name], outline=(0, 0, 0))
        draw.text((x + 26, y + 17), texts[name], fill=(0, 0, 0), font=font, anchor="lm")
    return out

def visualize_results(image, bboxes, results, scale=0.25, save_path=None, coverage=0.5, colors=None):
    """Plot the results CSV on the whole log image: coloured intervals over the lithology column, boundary lines and
    depths next to the depth column, cluster ids next to the lithology column, and a colour key on top.
    Nothing is cropped. `scale` shrinks the picture (1.0 = original size). Returns a PIL image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if not isinstance(bboxes, dict):
        bboxes = json.loads(Path(bboxes).read_text())
    if not isinstance(results, list):
        results = load_results(results)
    boxes = bboxes.get("bounding_boxes", bboxes)  # accept the whole annotation file or just its boxes
    if colors is None:
        colors = build_color_map(r["lithology"] for r in results)

    view = crop_rgb(image, (0, 0, image.width, image.height), scale)
    view = draw_boxes(view, boxes, scale=scale, line_width=2, font_size=13)
    view = overlay_results(view, results, colors, boxes["interpreted_lithology_column"], boxes["depth_column"],
                           (0, 0), scale, coverage=coverage)
    clusters = {}  # label -> cluster ids seen for it, in order of first appearance
    for r in results:
        if r.get("cluster_id") is not None:
            clusters.setdefault(r["lithology"], [])
            if r["cluster_id"] not in clusters[r["lithology"]]:
                clusters[r["lithology"]].append(r["cluster_id"])
    view = add_legend(view, colors, clusters)
    if save_path is not None:
        view.save(save_path)
    return view


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / 'viridien_challenge/Viridien_HTB_challenge/hack27/project'
DATA = PROJECT / 'dataset'
OUT = ROOT / 'artifact'
IMG = OUT / 'img'
LOG = '0000000009_d-582_0000020435-0001_log06_fe_5in_md'
COLOUR = '60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872'
PAGES = [LOG, COLOUR, '60807402860000_gb506#2_md_mudlog',
         'Critter Creek 201-1807H_Mud_PDF_Vertical.pdf', 'Rio_Nueces_3744_Unit_1_212_MUD_5IN',
         '608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD']
HEX = {'Shale':'#2e7d32','Siltstone':'#f9a825','Sandstone':'#1565c0',
       'Sand and Sandstone':'#1565c0','Clay':'#8d6e63','Limestone':'#7e57c2',
       'Dolomite':'#00838f','Anhydrite':'#c62828','Halite':'#ad1457','Coal':'#424242',
       'Silty Sandstone':'#ef6c00', 'No label':'#bdbdbd'}
COLORS = {k:ImageColor.getrgb(v) for k,v in HEX.items()}
Y0, Y1 = 4409, 7012
GB = [4809,5010,6211,6411,6611,6811]
MANIFEST = {}
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def page(name):
    a=json.loads((DATA/'bbox_annotation'/f'{name}.json').read_text())
    return Image.open(DATA/'images'/a['image_name']), a['bounding_boxes']


def record(name, caption, **numbers):
    p=IMG/name
    with Image.open(p) as im: size=list(im.size)
    MANIFEST[name]={'caption':caption,'width_px':size[0],'height_px':size[1],**numbers}
    (OUT/'manifest.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')
    print(name, flush=True)


def savefig(fig,name,caption,**numbers):
    fig.savefig(IMG/name,dpi=100)
    plt.close(fig)
    record(name,caption,**numbers)


def titled(im,title,subtitle=''):
    out=Image.new('RGB',(im.width,im.height+72),'white')
    out.paste(im,(0,72)); d=ImageDraw.Draw(out)
    d.text((12,8),title,font=get_font(18),fill='black')
    d.text((12,38),subtitle,font=get_font(13),fill='#424242')
    return out


def normlabel(x):
    x=x.strip()
    return 'Sandstone' if x.lower() in ['sand and sandstone','sandstone','sand'] else x


def rows_at(rows,start,end):
    arr=np.full(end-start,'No label',dtype=object)
    for r in rows:
        a=max(start,int(r['top_y'])); b=min(end,int(r['bottom_y']))
        if b>a: arr[a-start:b-start]=normlabel(r['lithology'])
    return arr


def boundaries(labels,start):
    return (np.flatnonzero(labels[1:]!=labels[:-1])+1+start).tolist()


def boundary_score(pred,gold,tolerance=20):
    pairs=sorted((abs(p-g),i,j) for i,p in enumerate(pred) for j,g in enumerate(gold) if abs(p-g)<=tolerance)
    pi=set(); gi=set()
    for _,i,j in pairs:
        if i not in pi and j not in gi: pi.add(i); gi.add(j)
    tp=len(pi); precision=tp/len(pred) if pred else 0; recall=tp/len(gold) if gold else 0
    return {'boundary_f1':2*precision*recall/(precision+recall) if precision+recall else 0,
            'boundary_precision':precision,'boundary_recall':recall,'matched_boundaries':tp,
            'predicted_boundaries':len(pred),'gold_boundaries':len(gold),'tolerance_px':tolerance}


def rgbstrip(labels):
    return np.array([COLORS[x] for x in labels],dtype=np.uint8)[:,None,:]


def show_column(ax,crop,start=Y0,end=Y1,title='Scan'):
    ax.imshow(crop,extent=[0,1,end,start],aspect='auto',interpolation='nearest')
    ax.set_title(title); ax.set_xticks([]); ax.set_ylabel('Image row (px)')


def goldlines(axes,solid=(),start=Y0,end=Y1):
    for ax in axes:
        ax.set_ylim(end,start)
        for y in GB:
            if start<y<end: ax.axhline(y,color='#555555',lw=1,ls=(0,(4,4)),alpha=.8)
        for y in solid: ax.axhline(y,color='#c62828',lw=1.2)


def key(fig,labels=('Shale','Siltstone','Sandstone'),note='Dashed = gold boundary'):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=HEX[k],label=k) for k in labels],loc='lower center',
               ncol=len(labels),frameon=False,bbox_to_anchor=(.5,.005))
    fig.text(.02,.055,note,fontsize=11,color='#444444')


def notebook_figs():
    image,boxes=page(LOG); gold=load_results(ROOT/'bakeoff/gt/gold'/f'{LOG}.csv')
    cmap={k:COLORS[k] for k in dict.fromkeys(r['lithology'] for r in gold)}
    full=visualize_results(image,boxes,gold,scale=.1,colors=cmap)
    full.save(IMG/'f01_notebook_full.png')
    record('f01_notebook_full.png','Official notebook visualization of all supplied gold intervals, whole page.',
           scale=.1,intervals=len(gold),source_size=list(image.size))
    box=(boxes['depth_column'][0]-150,4300,boxes['interpreted_lithology_column'][2]+260,7100)
    zoom=crop_rgb(image,box,.5)
    zoom=draw_boxes(zoom,boxes,origin=box[:2],scale=.5,line_width=2,font_size=13)
    visible=[r for r in gold if r['bottom_y']>box[1] and r['top_y']<box[3]]
    zoom=overlay_results(zoom,visible,cmap,boxes['interpreted_lithology_column'],boxes['depth_column'],box[:2],.5)
    zoom=add_legend(zoom,cmap)
    zoom.save(IMG/'f02_notebook_zoom.png')
    record('f02_notebook_zoom.png','Notebook overlay zoom with depths, cluster IDs and gold boundaries.',scale=.5,crop_box=list(box),intervals_visible=len(visible))
    for i,name in enumerate(PAGES,1):
        im,b=page(name)
        bottom=min(im.height,round(b['interpreted_lithology_column'][1]+1.2*im.width))
        scale=700/im.width
        v=crop_rgb(im,(0,0,im.width,bottom),scale)
        # Exact notebook draw_boxes rendering; numbered labels are placed in a margin
        # where the three adjacent long field names cannot collide.
        v=draw_boxes(v,b,scale=scale,line_width=2,font_size=15,numbered=True)
        foot=Image.new('RGB',(700,92),'white'); d=ImageDraw.Draw(foot)
        for j,field in enumerate(b): d.text((12,5+j*28),f'{j+1}  {field}',fill=BOX_COLORS[field],font=get_font(15))
        total=Image.new('RGB',(700,v.height+92),'white'); total.paste(v); total.paste(foot,(0,v.height))
        total=titled(total,f'Page regions | {i} of 6',name[:78])
        fname=f'f03_boxes_{i}.png'; total.save(IMG/fname)
        record(fname,'Header and first 1.2 page-widths of log, with official annotation boxes.',page=name,scale=scale,crop_bottom_y=bottom,boxes=b)
    cache=ROOT/'bakeoff/out/_cache/legend'
    for i,name in enumerate([LOG,COLOUR,PAGES[4]],1):
        src=cache/f'{name}.debug.png'
        if src.exists():
            im=Image.open(src).convert('RGB')
            # Preserve the supplied diagnostic, adding an accessible index below;
            # source debug labels can be tiny or overlap on the Rio legend.
            resized=im.width!=1400
            if resized: im=im.resize((1400,round(im.height*1400/im.width)),Image.Resampling.NEAREST)
            metadata=cache/f'{name}.json'
            entries=json.loads(metadata.read_text()).get('entries',[]) if metadata.exists() else []
            if entries:
                import textwrap
                rows=(len(entries)+3)//4
                panel=Image.new('RGB',(1400,60+rows*50),'white'); d=ImageDraw.Draw(panel)
                d.text((16,10),'Parser output index (IDs match the copied diagnostic above)',fill='black',font=get_font(18))
                for j,e in enumerate(entries):
                    col=j//rows; row=j%rows
                    text=f'{e["id"]}: {e["label"]}'
                    lines=textwrap.wrap(text,width=32)
                    for q,line in enumerate(lines):
                        d.text((16+col*346,48+row*50+q*20),line,fill='black',font=get_font(15))
                joined=Image.new('RGB',(1400,im.height+panel.height),'white'); joined.paste(im); joined.paste(panel,(0,im.height)); im=joined
            fname=f'f05_legend_parse_{i}.png'; im.save(IMG/fname)
            record(fname,'Existing legend-parser diagnostic copied from read-only cache, with a readable output index appended.',
                   source=str(src.relative_to(ROOT)),resized=resized,parser_entries=len(entries))


def comparison():
    im,b=page(LOG); start,end=4404,11195
    gold=load_results(ROOT/'bakeoff/gt/gold'/f'{LOG}.csv')
    with open(ROOT/'bakeoff/gt/silver/A'/f'{LOG}.csv') as f: silver=list(csv.DictReader(f))
    g=rows_at(gold,start,end); s=rows_at(silver,start,end)
    valid=(g!='No label') & (s!='No label'); gv=g!='No label'
    accuracy=float(np.mean(g[valid]==s[valid])); allgold=float(np.mean(g[gv]==s[gv]))
    lo=max(start,min(int(r['top_y']) for r in gold)); hi=min(end,max(int(r['bottom_y']) for r in gold))
    # Only real lithology transitions inside the common gold domain; endpoints excluded.
    pg=boundaries(rows_at(gold,lo,hi),lo); ps=boundaries(rows_at(silver,lo,hi),lo)
    ps=[y for y in ps if rows_at(silver,y-1,y+1).tolist().count('No label')==0]
    score=boundary_score(ps,pg)
    error=np.full((end-start,1,3),COLORS['No label'],dtype=np.uint8)
    error[valid,0]=np.where((g[valid]==s[valid])[:,None],np.array([46,125,50]),np.array([198,40,40]))
    crop=crop_rgb(im,(b['interpreted_lithology_column'][0],start,b['interpreted_lithology_column'][2],end))
    fig,axes=plt.subplots(1,4,figsize=(10.5,11.5),gridspec_kw={'width_ratios':[1.8,1,1,.5]})
    show_column(axes[0],crop,start,end)
    for ax,arr,title in zip(axes[1:],[rgbstrip(g),rgbstrip(s),error],['GOLD','Labeller A','Agreement']):
        ax.imshow(arr,extent=[0,1,end,start],aspect='auto',interpolation='nearest'); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Labeller A vs gold | row accuracy {accuracy:.1%} | boundary F1 {score["boundary_f1"]:.3f}',fontsize=15,y=.98)
    fig.text(.5,.935,'Rows 4404–11195 | boundary tolerance 20 px | green same · red different · grey no label',ha='center',fontsize=11)
    fig.subplots_adjust(left=.1,right=.96,top=.90,bottom=.12,wspace=.35)
    key(fig,note='Accuracy uses rows labelled by both; grey includes rows outside supplied gold.')
    savefig(fig,'f04_labeller_vs_gold.png','Scan, gold, labeller A, and agreement on aligned image rows.',
            row_accuracy=accuracy,row_accuracy_all_gold=allgold,comparison_rows=int(valid.sum()),gold_rows=int(gv.sum()),
            total_rows=end-start,unlabelled_gold_rows=int((~gv).sum()),labeller_coverage_on_gold=float(valid.sum()/gv.sum()),
            gold_boundary_y=pg,labeller_boundary_y=ps,**score)


def demo_crop():
    im,b=page(LOG); x0,_,x1,_=b['interpreted_lithology_column']
    return crop_rgb(im,(x0+8,Y0,x1-8,Y1))


def colour_runs():
    t=time.perf_counter(); im,b=page(COLOUR); start,end=2343,8343
    x0,_,x1,_=b['interpreted_lithology_column']; crop=crop_rgb(im,(x0+2,start,x1-2,end))
    a=np.asarray(crop); hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    valid=(hsv[:,:,1]>65)&(hsv[:,:,2]>60)
    # Quantized hue mode per row excludes black, white and grey scanner marks.
    hue=np.full(end-start,-1,dtype=int)
    for i in range(len(a)):
        bins=(hsv[i,:,0][valid[i]]//10).astype(int)
        if len(bins): hue[i]=np.bincount(bins,minlength=18).argmax()
    from scipy.ndimage import median_filter
    stable=median_filter(hue,size=15,mode='nearest')
    # Absorb runs shorter than 15 rows until the run encoding is stable.
    for _ in range(10):
        cuts=np.r_[0,np.flatnonzero(stable[1:]!=stable[:-1])+1,len(stable)]
        changed=False
        for j,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            if hi-lo<15:
                stable[lo:hi]=stable[lo-1] if lo else stable[hi]; changed=True
        if not changed: break
    bounds=boundaries(stable,start)
    rgb=np.full((len(a),1,3),245,dtype=np.uint8)
    for i,h in enumerate(stable):
        if h>=0: rgb[i,0]=cv2.cvtColor(np.uint8([[[h*10+5,220,220]]]),cv2.COLOR_HSV2RGB)[0,0]
    fig,ax=plt.subplots(1,3,figsize=(9,11),gridspec_kw={'width_ratios':[1,.7,1.5]})
    show_column(ax[0],crop,start,end,'Colour scan')
    ax[1].imshow(rgb,aspect='auto',extent=[0,1,end,start],interpolation='nearest'); ax[1].set_title('Dominant hue')
    ax[2].plot(stable,np.arange(start,end),color='#424242',lw=1); ax[2].set_title('Colour changes'); ax[2].set_xlabel('Hue bin (−1 = no colour)')
    for aa in ax[1:]: aa.set_ylim(end,start); aa.set_yticks([])
    ax[1].set_xticks([])
    for y in bounds: ax[2].axhline(y,color='#c62828',lw=.8)
    fig.suptitle(f'Colour runs | {len(bounds)} changes after minimum 15-row runs',fontsize=15)
    fig.subplots_adjust(top=.91,bottom=.08,left=.11,right=.97,wspace=.35)
    sec=time.perf_counter()-t
    savefig(fig,'d_s1_colour_runs.png','Dominant non-black/non-white colour and stable colour-run boundaries on the colour log.',
            seconds=sec,page=COLOUR,top_y=start,bottom_y=end,min_run_rows=15,boundary_y=bounds,
            boundary_count=len(bounds),valid_colour_row_fraction=float((hue>=0).mean()))


def row_features(gray):
    ink=(gray<160).astype(np.uint8)
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((13,1),np.uint8))
    return uniform_filter1d(np.stack([ink.mean(1),horizontal.mean(1),vertical.mean(1)],axis=1),40,axis=0)


def features_pelt():
    import ruptures as rpt
    from sklearn.preprocessing import StandardScaler
    t=time.perf_counter(); crop=demo_crop(); gray=np.asarray(crop.convert('L'))
    feat=row_features(gray); z=StandardScaler().fit_transform(feat)
    penalty=3*np.log(len(z))*3
    pts=rpt.Pelt(model='l2',min_size=80,jump=5).fit(z).predict(pen=penalty)
    pred=[Y0+p for p in pts[:-1]]; score=boundary_score(pred,GB)
    fig,ax=plt.subplots(1,4,figsize=(12.8,10),gridspec_kw={'width_ratios':[.9,1,1,1]})
    show_column(ax[0],crop)
    for j,title in enumerate(['Ink fraction','Horizontal strokes','Vertical strokes']):
        ax[j+1].plot(feat[:,j],np.arange(Y0,Y1),color=['#424242','#7e57c2','#00838f'][j],lw=1)
        ax[j+1].set_title(title); ax[j+1].set_xlabel('Fraction of row'); ax[j+1].set_yticks([])
    goldlines(ax,pred)
    fig.suptitle(f'Row features + PELT | boundary F1 {score["boundary_f1"]:.3f} at ±20 px',fontsize=16)
    fig.text(.5,.92,'40-row smoothing | red solid = PELT | grey dashed = gold',ha='center',fontsize=12)
    fig.subplots_adjust(top=.87,bottom=.1,left=.08,right=.98,wspace=.30)
    savefig(fig,'d_s2_features_pelt.png','Ink and directional stroke densities with unsupervised PELT change points.',
            seconds=time.perf_counter()-t,smoothing_rows=40,penalty=float(penalty),min_segment_rows=80,
            predicted_boundary_y=pred,gold_boundary_y=GB,**score)


def legend_swatches():
    im,_=page(LOG); legend=crop_rgb(im,(1943,1513,3035,2655))
    gray=np.asarray(legend.convert('L')); binary=(gray<180).astype(np.uint8)*255
    # Close small breaks in scanned rectangle borders before finding contours.
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if 140<w<210 and 85<h<155 and .95<w/h<2.3:
            if not any(abs(x-a)<12 and abs(y-b)<12 for a,b,_,_ in rects): rects.append((x,y,w,h))
    # Locate the requested grid cells by proximity, using detected border rectangles.
    cells={'Clay':(405,410),'Shale':(660,410),'Siltstone':(910,410),'Sand and Sandstone':(405,675)}
    selected={}; boxes={}
    for name,(cx,cy) in cells.items():
        x,y,w,h=min(rects,key=lambda r:(r[0]+r[2]/2-cx)**2+(r[1]+r[3]/2-cy)**2)
        assert abs(x+w/2-cx)<80 and abs(y+h/2-cy)<80, (name,rects)
        box=(x+8,y+8,x+w-8,y+h-8)
        selected[name]=legend.crop(box); boxes[name]=[1943+box[0],1513+box[1],1943+box[2],1513+box[3]]
    return selected,boxes


def template_ncc():
    t=time.perf_counter(); crop=demo_crop(); gray=np.asarray(crop.convert('L'))
    swatches,boxes=legend_swatches()
    fig=plt.figure(figsize=(13.4,10.5)); gs=fig.add_gridspec(2,5,height_ratios=[.16,1],width_ratios=[1,1,1,1,1])
    ax0=fig.add_subplot(gs[1,0]); show_column(ax0,crop)
    peaks={}
    for j,(name,swatch) in enumerate(swatches.items(),1):
        thumb=fig.add_subplot(gs[0,j]); thumb.imshow(swatch); thumb.axis('off'); thumb.set_title(name,color=HEX[name],fontsize=12)
        templ=np.asarray(swatch.convert('L')); match=cv2.matchTemplate(gray,templ,cv2.TM_CCOEFF_NORMED)
        # Horizontal maximum makes the response invariant to pattern phase in x.
        curve=match.max(1); yy=Y0+np.arange(len(curve))+templ.shape[0]/2
        curve=uniform_filter1d(curve,15)
        aa=fig.add_subplot(gs[1,j]); aa.plot(curve,yy,color=HEX[name],lw=1); aa.set_title('NCC response'); aa.set_xlabel('Correlation'); aa.set_yticks([])
        goldlines([aa]); peaks[name]=float(curve.max())
    goldlines([ax0]); fig.suptitle('Legend templates | normalized cross-correlation down the scan',fontsize=16,y=.99)
    fig.subplots_adjust(top=.91,bottom=.09,left=.07,right=.99,wspace=.33,hspace=.3)
    fig.text(.07,.025,'Border-stripped swatches detected by contours | dashed lines = gold boundaries',fontsize=12)
    savefig(fig,'d_s3_template_ncc.png','Four real legend templates and their normalized cross-correlation responses.',
            seconds=time.perf_counter()-t,swatch_boxes=boxes,max_ncc=peaks,horizontal_reduction='maximum',curve_smoothing_rows=15)


def kmeans_viterbi():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from skimage.feature import local_binary_pattern
    from scipy.optimize import linear_sum_assignment
    t=time.perf_counter(); crop=demo_crop(); a=np.asarray(crop.convert('L'))
    starts=np.arange(0,len(a),20); features=[]
    for start in starts:
        tile=a[start:start+20]; sx=cv2.Sobel(tile,cv2.CV_32F,1,0); sy=cv2.Sobel(tile,cv2.CV_32F,0,1)
        lbp=local_binary_pattern(tile,8,1,'uniform'); hist=np.histogram(lbp,bins=np.arange(11),density=True)[0]
        features.append(np.r_[(tile<160).mean(),np.abs(sx).mean()/255,np.abs(sy).mean()/255,hist])
    z=StandardScaler().fit_transform(features); km=KMeans(3,n_init=20,random_state=27).fit(z)
    dist=km.transform(z)**2; emission=-dist/(2*z.shape[1])
    trans=np.log(np.full((3,3),.01)+np.eye(3)*.97)
    dp=emission[0]-np.log(3); back=np.zeros((len(z),3),int)
    for i in range(1,len(z)):
        candidates=dp[:,None]+trans; back[i]=candidates.argmax(0); dp=candidates.max(0)+emission[i]
    path=np.zeros(len(z),int); path[-1]=dp.argmax()
    for i in range(len(z)-1,0,-1): path[i-1]=back[i,path[i]]
    gold=load_results(ROOT/'bakeoff/gt/gold'/f'{LOG}.csv'); g=rows_at(gold,Y0,Y1)
    # Cluster IDs carry no lithology semantics. Hungarian alignment is for display only.
    names=['Shale','Siltstone','Sandstone']; matrix=np.zeros((3,3),int)
    raw=np.repeat(km.labels_,20)[:len(a)]; smooth=np.repeat(path,20)[:len(a)]
    for c in range(3):
        for j,n in enumerate(names): matrix[c,j]=np.sum((raw==c)&(g==n))
    ii,jj=linear_sum_assignment(-matrix); mapping={int(i):names[j] for i,j in zip(ii,jj)}
    r=np.array([mapping[int(i)] for i in raw]); s=np.array([mapping[int(i)] for i in smooth])
    fig,axes=plt.subplots(1,4,figsize=(10,10),gridspec_kw={'width_ratios':[1.6,1,1,1]})
    show_column(axes[0],crop)
    for ax,lab,title in zip(axes[1:],[g,r,s],['Gold','K-means (k=3)','Viterbi (p=.98)']):
        ax.imshow(rgbstrip(lab),extent=[0,1,Y1,Y0],aspect='auto',interpolation='nearest'); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    goldlines(axes); fig.suptitle('Tile texture clustering | persistence smooths the label sequence',fontsize=15)
    fig.subplots_adjust(top=.9,bottom=.13,left=.1,right=.97,wspace=.35)
    key(fig,note='Cluster colours aligned to gold after fitting, for display; this is not a trained classifier.')
    savefig(fig,'d_s4_kmeans_viterbi.png','20-row ink/Sobel/LBP tiles clustered into three states, then Viterbi smoothed.',
            seconds=time.perf_counter()-t,tile_rows=20,tiles=len(z),feature_dimensions=z.shape[1],self_transition=.98,
            raw_transitions=len(boundaries(raw,Y0)),smoothed_transitions=len(boundaries(smooth,Y0)),
            cluster_to_display_label=mapping,raw_row_accuracy=float((r==g).mean()),viterbi_row_accuracy=float((s==g).mean()),
            mapping_uses_gold_for_display=True)


def dinov2():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    t=time.perf_counter(); crop=demo_crop()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; GPU demo cannot run')
    torch.cuda.reset_peak_memory_stats(); device='cuda'
    processor=AutoImageProcessor.from_pretrained('facebook/dinov2-small',local_files_only=True)
    model=AutoModel.from_pretrained('facebook/dinov2-small',local_files_only=True).to(device).eval()
    starts=list(range(0,crop.height-64+1,16)); embeddings=[]
    infer=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(starts),24):
            tiles=[crop.crop((0,y,crop.width,y+64)) for y in starts[start:start+24]]
            inputs=processor(images=tiles,return_tensors='pt').to(device)
            output=model(**inputs).last_hidden_state[:,0]
            embeddings.append(torch.nn.functional.normalize(output,dim=1).cpu().numpy())
    torch.cuda.synchronize(); infer_seconds=time.perf_counter()-infer
    e=np.concatenate(embeddings); distance=1-(e[1:]*e[:-1]).sum(1)
    yy=Y0+(np.array(starts[1:])+np.array(starts[:-1]))/2+32
    peaks,_=find_peaks(distance,distance=5,prominence=.015)
    pred=np.rint(yy[peaks]).astype(int).tolist(); score=boundary_score(pred,GB)
    memory=torch.cuda.max_memory_allocated()/1024**2
    seconds=time.perf_counter()-t
    fig,ax=plt.subplots(1,2,figsize=(10,10),gridspec_kw={'width_ratios':[1,3]})
    show_column(ax[0],crop); ax[1].plot(distance,yy,color='#1565c0',lw=1.2)
    ax[1].scatter(distance[peaks],yy[peaks],c='#c62828',s=35,zorder=3,label='Distance peaks')
    ax[1].set_xlabel('1 − cosine similarity of adjacent tile embeddings'); ax[1].set_title('Neighbour embedding distance'); ax[1].legend(frameon=False)
    goldlines(ax); ax[1].set_yticks([])
    fig.suptitle(f'DINOv2-small | {seconds:.2f} s | peak GPU memory {memory:.0f} MiB',fontsize=16)
    fig.text(.5,.925,'64-row tiles · stride 16 · red dots = peaks · dashed = gold boundaries',ha='center',fontsize=12)
    fig.subplots_adjust(top=.87,bottom=.09,left=.1,right=.97,wspace=.25)
    savefig(fig,'d_s5_dinov2.png','Cached DINOv2-small CUDA embeddings: neighbour cosine distance and detected peaks.',
            seconds=seconds,inference_seconds=infer_seconds,gpu_peak_memory_mib=memory,gpu=torch.cuda.get_device_name(0),
            tiles=len(starts),tile_rows=64,stride_rows=16,embedding_dimensions=e.shape[1],
            peak_boundary_y=pred,peak_prominence=.015,**score)
    del model, inputs, output
    torch.cuda.empty_cache()


# Descriptions transcribed by visual inspection of the actual log06 legend.
# These descriptions are identical across the three calls and contain no crop truth.
CRITERIA={
    'Limestone':'Rectangular brickwork: horizontal courses and alternating vertical joints.',
    'Dolomite':'Slanted brickwork: horizontal courses joined by diagonal leaning joints.',
    'Anhydrite':'Sparse parallel diagonal strokes slanting upward to the right.',
    'Halite':'Large repeated inverted L or right-angle corner strokes, separated by white space.',
    'Coal':'Solid black fill, with almost no white space.',
    'Clay':'Thin short horizontal dashes, staggered in sparse rows with generous white gaps.',
    'Shale':'Staggered rows of somewhat longer, heavier horizontal dashes; very similar to Clay.',
    'Siltstone':'Groups of narrow upright vertical strokes or paired vertical dashes on white.',
    'Silty Sandstone':'Horizontal dash-and-dot pattern: broken horizontal strokes interspersed with dots.',
    'Sand and Sandstone':'Small dark dots distributed as a regular dense stipple on white.'}


def read_jev_key():
    env=ROOT.parents[1]/'misc/.env'
    for line in env.read_text(encoding='utf-8-sig').splitlines():
        k,sep,v=line.strip().partition('=')
        if sep and k.strip()=='JEV_API_KEY': return v.strip().strip('"').strip("'")
    raise RuntimeError('JEV_API_KEY entry was not found')


def jev(refresh=False):
    t=time.perf_counter(); im,b=page(LOG); x0,_,x1,_=b['interpreted_lithology_column']
    cache=OUT/'jev_responses.json'
    results=json.loads(cache.read_text()) if cache.exists() and not refresh else []
    used_cache=len(results)==3
    intervals=[('Shale',4409,4809),('Siltstone',4809,5010),('Sandstone',6611,6811)]
    crops=[]; crop_boxes=[]
    for truth,lo,hi in intervals:
        box=(x0+22,lo+35,x1-22,hi-35)
        crops.append(crop_rgb(im,box).resize((512,512),Image.Resampling.NEAREST)); crop_boxes.append(list(box))
    if len(results)<3:
        api_key=read_jev_key()
        for i in range(len(results),3):
            buf=io.BytesIO(); crops[i].save(buf,format='PNG')
            body={'model':'jev-latest','state':{'crop':'img1'},
                  'assets':{'img1':{'type':'image','format':'png','data_base64':base64.b64encode(buf.getvalue()).decode()}},
                  'questions':{'lith':{'type':'choice','instructions':
                      'Classify the lithology pattern in state.crop by matching its visible strokes or dots to the legend criteria. Ignore scan noise and use the closest visual pattern.',
                      'criteria':CRITERIA}}}
            req=urllib.request.Request('https://api.typesafe.ai/v1/systemone',data=json.dumps(body).encode(),
                 headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json','TypeSafe-Danger-Zone-Images':'true'},method='POST')
            call=time.perf_counter()
            try:
                # The runner's environment proxy is closed; direct HTTPS works.
                # Use a fresh Request, since a failed proxy call mutates its host.
                # This retains normal TLS certificate verification.
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                response=opener.open(req,timeout=55)
                with response: payload=json.load(response)
            except Exception as exc:
                # Do not serialize request objects or authentication headers.
                reason=getattr(exc,'reason','')
                raise RuntimeError(f'JEV call {i+1} failed ({type(exc).__name__}: {type(reason).__name__}: {reason})') from None
            latency=time.perf_counter()-call
            answer=payload['answers']['lith']
            results.append({'truth':intervals[i][0],'crop_box':crop_boxes[i],'latency_seconds':latency,'answer':answer})
            cache.write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(f'JEV call {i+1}: {latency:.2f} seconds',flush=True)
    fig,axes=plt.subplots(3,2,figsize=(13.8,12),gridspec_kw={'width_ratios':[1,2]})
    names=list(CRITERIA)
    for i,(crop,result) in enumerate(zip(crops,results)):
        truth=result['truth']; ans=result['answer']; probs=ans['probabilities']
        if isinstance(probs,list):
            if probs and isinstance(probs[0],dict): probs={v.get('choice',v.get('name')):v.get('probability',v.get('confidence')) for v in probs}
            else: probs=dict(zip(names,probs))
        values=[float(probs.get(n,0)) for n in names]
        axes[i,0].imshow(crop,interpolation='nearest'); axes[i,0].axis('off'); axes[i,0].set_title(f'Truth: {truth}',fontsize=14)
        labels=[n+('  ← truth' if normlabel(n)==truth else '') for n in names]
        axes[i,1].barh(labels,values,color=[HEX[n] for n in names],height=.72)
        axes[i,1].invert_yaxis(); axes[i,1].set_xlim(0,1); axes[i,1].set_xlabel('Returned probability')
        axes[i,1].set_title(f'JEV: {ans["choice"]} | latency {result["latency_seconds"]:.2f} s',fontsize=14)
        for j,v in enumerate(values):
            if v>.001: axes[i,1].text(min(v+.01,.92),j,f'{v:.1%}',va='center',fontsize=10)
    fig.suptitle('JEV | real choice calls against the ten legend patterns',fontsize=17,y=.985)
    fig.subplots_adjust(left=.025,right=.97,top=.93,bottom=.065,wspace=.65,hspace=.47)
    savefig(fig,'d_s8_jev.png','Three gold-interval crops, nearest-neighbour upscaled to 512 px, and real JEV probabilities.',
            seconds=time.perf_counter()-t,model='jev-latest',criteria=CRITERIA,calls=results,
            correct=sum(normlabel(r['answer']['choice'])==r['truth'] for r in results),
            call_count=len(results),total_api_seconds=sum(r['latency_seconds'] for r in results),cached_responses=used_cache,
            crop_resize=[512,512],resize_method='NEAREST')


def failed(name,exc):
    # A missing external service produces an explicit diagnostic, never invented data.
    msg=str(exc)[:200]
    im=Image.new('RGB',(1100,280),'white'); d=ImageDraw.Draw(im)
    d.text((25,25),f'{name}: could not complete',fill='#c62828',font=get_font(22))
    d.text((25,85),type(exc).__name__,fill='black',font=get_font(18))
    for j,line in enumerate([msg[k:k+90] for k in range(0,len(msg),90)]):
        d.text((25,125+j*28),line,fill='black',font=get_font(15))
    im.save(IMG/name)
    record(name,'Execution failed; diagnostic shown instead of fabricated results.',status='failed',error=msg)


def main():
    global MANIFEST
    parser=argparse.ArgumentParser(); parser.add_argument('--only',nargs='+'); parser.add_argument('--refresh-jev',action='store_true')
    args=parser.parse_args(); IMG.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists(): MANIFEST=json.loads((OUT/'manifest.json').read_text())
    tasks=[('notebook',notebook_figs,None),('comparison',comparison,'f04_labeller_vs_gold.png'),
           ('s1',colour_runs,'d_s1_colour_runs.png'),('s2',features_pelt,'d_s2_features_pelt.png'),
           ('s3',template_ncc,'d_s3_template_ncc.png'),('s4',kmeans_viterbi,'d_s4_kmeans_viterbi.png'),
           ('s5',dinov2,'d_s5_dinov2.png'),('s8',lambda:jev(args.refresh_jev),'d_s8_jev.png')]
    for tag,fn,name in tasks:
        if args.only and tag not in args.only: continue
        print('START',tag,flush=True)
        try: fn()
        except Exception as exc:
            print('FAILED',tag,type(exc).__name__,str(exc)[:200],flush=True)
            if name: failed(name,exc)
            else: raise
    print('Done:',len(MANIFEST),'figures',flush=True)


if __name__=='__main__': main()
