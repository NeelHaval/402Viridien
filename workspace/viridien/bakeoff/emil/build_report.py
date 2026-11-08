"""Build standalone flat scientific figures and a fully machine-readable report."""
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from emil.bootstrap import installed_arrow
installed_arrow()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from harness import io, score
from emil import pipeline as p

OUT = HERE/'report'
MODES = ['detect','labels','s11']
COLORS = ['#2878a8','#e8a035','#75608c']


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def dump(path,data): path.write_text(json.dumps(clean(data),indent=2),encoding='utf-8')


def mean(records):
    keys=['row_acc','boundary_f1','cluster_ari','seconds','peak_rss_gb','peak_gpu_gb','depth_mae_ft']
    return dict(n_windows=len(records),errors=sum('error' in r for r in records),**{k:float(np.mean([r[k] for r in records if r.get(k) is not None])) if any(r.get(k) is not None for r in records) else None for k in keys})


def plot_sources(groups, filename='per_source.png', subtitle=None):
    fig,axes=plt.subplots(1,3,figsize=(18,5),facecolor='white')
    names=list(groups)
    display={'kgs':'Kansas (KGS)','idaho':'Idaho','utah':'Utah','nl':'Newfoundland'}
    labels={'detect':'s12 detect (fair)','labels':'s12 labels (oracle)','s11':'s11'}
    x=np.arange(len(names))
    for ax,metric,title in zip(axes,['row_acc','boundary_f1','cluster_ari'],['Row accuracy','Boundary F1','Cluster ARI']):
        for i,mode in enumerate(MODES):
            values=[groups[n][mode][metric] for n in names]
            ax.bar(x+(i-1)*.24,values,width=.23,color=COLORS[i],label=labels[mode])
        ax.set_xticks(x,[display.get(n,n) for n in names],rotation=25,ha='right'); ax.set_ylim(-.05,1.02)
        ax.set_title(title);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle(subtitle or 'Same-window direct scoring: held-out Viridien pages and external dev / test\nDepth OCR bypassed equally; shared preprocessing failures reported separately',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/filename,dpi=170);plt.close(fig)


def palette(log,labels):
    result={}
    for e in labels['legend']:
        x0,y0,x1,y1=map(int,e['box'])
        inset=min(3,max(0,(min(x1-x0,y1-y0)-1)//2))
        rgb=io.read_rgb(log,[x0+inset,y0+inset,x1-inset,y1-inset])
        colour=np.median(rgb.reshape(-1,3),axis=0).astype(np.uint8)
        result[score.normalize(e['label_printed'])]=colour
    result['']=np.array([245,245,245],np.uint8)
    return result


def panel(domain,record,labels,shared=False):
    root=HERE/'results_live'/domain
    name,key=record['log'],record['key']
    if domain=='external': io.DATASET=p.WORK/'external_tests'
    else: io.DATASET=Path(read(HERE/'dataset_paths.json')['viridien'])
    log=io.open_log(name)
    try:
        full=tuple(record['window'])
        truth=io.load_rows(root/'truth_csv'/(key+'.csv'))
        predictions={mode:io.load_rows(root/mode/'csv'/(key+'.csv')) or [] for mode in MODES}
        if domain=='viridien':
            # Show the transition-rich portion after the initial long salt layer.
            layers=labels['layers']
            starts=[l['top_px'] for l in layers if l['top_px']>full[0]+1000]
            start=starts[len(starts)//3] if starts else full[0]
            win=(start,min(full[1],start+3500))
            # Show the largest observed oracle improvement, if there is one.
            g=score.raster(truth,full)
            d=score.raster(predictions['detect'],full)
            o=score.raster(predictions['labels'],full)
            gain=((o==g)&(g!='')).astype(int)-((d==g)&(g!='')).astype(int)
            length=min(3500,len(gain));cs=np.r_[0,np.cumsum(gain)]
            totals=cs[length:]-cs[:-length]
            if totals.max()>0:
                start=full[0]+int(np.argmax(totals));win=(start,start+length)
        else:win=full
        rgb=io.read_rgb(log,'interpreted_lithology_column',*win)
        if shared:
            strategies=dict(detect='s12_emil',labels='oracle_emil',s11='s11_fft_legend')
            predictions={mode:io.load_rows(io.ROOT/'out'/strategies[mode]/'external'/(key+'.csv')) or [] for mode in MODES}
        colours=palette(log,labels)
        width=220;height=min(1800,max(600,win[1]-win[0]));header=80
        canvas=Image.new('RGB',(5*width+80,height+header+70),'white');draw=ImageDraw.Draw(canvas)
        try:font=ImageFont.truetype('DejaVuSans.ttf',15)
        except OSError:font=ImageFont.load_default()
        draw.text((15,8),name,fill='black',font=font)
        if shared:draw.text((15,51),'Shared harness outputs (including depth preprocessing failures)',fill='black',font=font)
        titles=['Scan column','Truth','s12-detect','s12-labels (oracle)','s11']
        draw.text((15,30),f'Rows {win[0]}:{win[1]}; red ticks = unmatched predicted or true boundaries (harness tolerance)',fill='black',font=font)
        scan=Image.fromarray(rgb).resize((width-25,height),Image.Resampling.BOX)
        canvas.paste(scan,(20,header))
        truth_b=score.boundaries(truth,win)
        for i,title in enumerate(titles):draw.text((20+i*width,header-24),title,fill='black',font=font)
        for i,rows in enumerate([truth,*[predictions[m] for m in MODES]],1):
            track_colours=dict(colours)
            if i>1:
                entries=read(root/MODES[i-2]/'legend_full'/(key+'.json'))
                track_colours.update(palette(log,dict(legend=[dict(box=e['box'],label_printed=e['lithology']) for e in entries])))
            raster=score.raster(rows,win)
            band=np.full((len(raster),width-30,3),255,np.uint8)
            for label in set(raster):
                band[raster==label]=track_colours.get(label,np.array([210,210,210],np.uint8))
            im=Image.fromarray(band).resize((width-30,height),Image.Resampling.NEAREST)
            canvas.paste(im,(20+i*width,header))
            for row in rows:
                if row['bottom_y']<=win[0] or row['top_y']>=win[1]:continue
                yy=header+int((max(win[0],row['top_y'])-win[0])*height/(win[1]-win[0]))
                draw.line((20+i*width,yy,20+i*width+width-30,yy),fill='#bbbbbb',width=1)
                if row['bottom_y']-row['top_y']>max(80,(win[1]-win[0])//35):
                    yy=header+int((max(win[0],row['top_y'])-win[0])*height/(win[1]-win[0]))+2
                    draw.text((23+i*width,yy),row['lithology'][:22],fill='black',font=font,stroke_width=1,stroke_fill='white')
            if i>1:
                pred_b=score.boundaries(rows,win);tol=max(4,min(20,.01*(win[1]-win[0])))
                candidates=sorted((abs(a-b),a,b) for a in pred_b for b in truth_b if abs(a-b)<=tol)
                matched_p=set();matched_g=set()
                for _,a,b in candidates:
                    if a not in matched_p and b not in matched_g:matched_p.add(a);matched_g.add(b)
                for a in set(pred_b)-matched_p:
                    yy=header+round((a-win[0])*height/(win[1]-win[0]));draw.line((i*width+20,yy,i*width+45,yy),fill='#d62728',width=2)
                for a in set(truth_b)-matched_g:
                    yy=header+round((a-win[0])*height/(win[1]-win[0]));draw.line(((i+1)*width-35,yy,(i+1)*width-10,yy),fill='#d62728',width=2)
        filename=('panel_shared_' if shared else 'panel_')+key+'.png'
        canvas.save(OUT/filename)
        return dict(domain=domain,log=name,key=key,window=list(win),file=filename,shared_harness=shared)
    finally:log.close()


def overlay(domain,record,labels):
    io.DATASET=p.WORK/'external_tests' if domain=='external' else Path(read(HERE/'dataset_paths.json')['viridien'])
    log=io.open_log(record['log'])
    try:
        box=log.boxes['lithology_legend'];rgb=io.read_rgb(log,box)
        found=read(HERE/'results_live'/domain/'detect/legend_full'/(record['key']+'.json'))
        fig,axes=plt.subplots(2,1,figsize=(18,max(6,12*rgb.shape[0]/rgb.shape[1])),facecolor='white')
        for ax,entries,title,colour in [(axes[0],found,'Detected boxes + EasyOCR names','#d62728'),(axes[1],[dict(box=e['box'],lithology=e['label_printed']) for e in labels['legend']],'True boxes + exact names','#16803c')]:
            ax.imshow(rgb,interpolation='nearest')
            for e in entries:
                x0,y0,x1,y1=e['box'];x0-=box[0];x1-=box[0];y0-=box[1];y1-=box[1]
                ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=colour,linewidth=1.4))
                ax.text(x0,max(0,y0-2),e['lithology'],fontsize=6,color=colour,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.2))
            ax.set_title(f'{title} ({len(entries)})');ax.axis('off')
        fig.suptitle(record['log'],fontsize=10);fig.tight_layout()
        filename='legend_'+record['log']+'.png';fig.savefig(OUT/filename,dpi=180);plt.close(fig)
        return filename
    finally:log.close()



if __name__ == '__main__':
    from emil.live_report import main
    main()
