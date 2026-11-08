"""Measure lithology columns; truth is used only to describe bed distributions."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import sys
import json
import csv
import numpy as np
import pandas as pd
from PIL import Image
HERE=Path(__file__).resolve().parent
BAKE=HERE.parents[1]
REPO=BAKE.parent
sys.path.insert(0,str(BAKE))
os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig')
from harness import io,score
Image.MAX_IMAGE_PIXELS=None

def sample_stats(log,window):
    y0,y1=window
    parts=[]
    # Native pixels, bounded strip sampling, at most 2048 rows per window.
    for top in np.unique(np.linspace(y0,max(y0,y1-256),8,dtype=int)):
        parts.append(io.read_rgb(log,'interpreted_lithology_column',int(top),min(int(top)+256,y1)))
    rgb=np.concatenate(parts).astype(np.int16)
    ink=rgb.min(axis=2)<245
    chrom=(rgb.max(axis=2)-rgb.min(axis=2)>25)&ink
    return dict(ink_share=float(ink.mean()),colour_ink_share=float(chrom.sum()/max(1,ink.sum())),pattern_ink_share=float((ink&~chrom).sum()/max(1,ink.sum())),colour_row_share=float((chrom.mean(axis=1)>.05).mean()))

def cache_data(name,external):
    roots=[HERE/'runtime/out/_cache'] if external else [BAKE/'out/_cache']
    result={}
    for root in roots:
        lp=root/'legend'/(name+'.json')
        dp=root/'depth'/(name+'.rapidocr.json')
        if lp.exists():
            entries=json.loads(lp.read_text())['entries']
            result['legend_entries']=sum(e.get('is_lithology',True) for e in entries)
            result['legend_labels']=[e['label'] for e in entries if e.get('is_lithology',True)]
        if dp.exists():result['calibration']=json.loads(dp.read_text())['calibration']
    return result

def run():
    vir=json.loads((BAKE/'eval_set.json').read_text())
    dev=json.loads((REPO/'external_tests/eval_set_dev.json').read_text())
    names={i['log'] for i in dev}
    with (REPO/'external_tests/manifest.csv').open(newline='') as f:
        manifest={r['id']:r for r in csv.DictReader(f) if r['id'] in names}
    rows=[];beds=[];galleries={}
    dataset=Path(os.environ.get('LITHOLOGY_VIRIDIEN_DATASET',REPO.parent/'dataset')).expanduser()
    # All 51 images for broad image-only coverage; scored bed stats use the fixed 13.
    io.DATASET=dataset
    vir_items={i['log']:i for i in vir}
    for domain,items,root in [('viridien_all',[dict(log=n) for n in io.list_logs()],dataset),('ext_dev',dev,REPO/'external_tests')]:
        io.DATASET=root
        for item in items:
            name=item['log'];external=domain=='ext_dev'
            with_log=io.open_log(name)
            try:
                box=with_log.boxes['interpreted_lithology_column']
                # For eval logs use eval window; otherwise sample full column.
                winitem=item if external else vir_items.get(name,{})
                window=(winitem.get('y0',box[1]),winitem.get('y1',box[3]))
                cache=cache_data(name,external)
                calib=cache.get('calibration',{})
                ppf=float(manifest[name]['px_per_ft']) if external else (1/abs(calib['a']) if calib.get('a') else np.nan)
                meta=manifest[name] if external else {}
                rec=dict(domain=domain,log=name,y0=window[0],y1=window[1],source=meta.get('source','viridien'),well=meta.get('well',name),in_eval=name in vir_items,width_px=box[2]-box[0],page_mode=with_log.mode,one_bit=int(with_log.mode=='1'),px_per_ft=ppf,legend_entries=cache.get('legend_entries',np.nan),**sample_stats(with_log,window))
                truthpath=REPO/'external_tests/truth'/(name+'.csv') if external else BAKE/'gt/silver/B'/(name+'.csv')
                truth=io.load_rows(truthpath) if external or name in vir_items else None
                px=[];ft=[]
                if truth is not None:
                    labels={score.normalize(x) for x in cache.get('legend_labels',[])}
                    total=covered=0
                    for r in truth:
                        a=max(window[0],r['top_y']);b=min(window[1],r['bottom_y'])
                        if b<=a:continue
                        total+=b-a
                        if score.normalize(r['lithology']) in labels:covered+=b-a
                        # Exclude window-clipped beds to avoid systematically short tails.
                        if r['top_y']<window[0] or r['bottom_y']>window[1]:continue
                        thickness=r['bottom_y']-r['top_y']
                        feet=thickness/ppf if np.isfinite(ppf) else np.nan
                        px.append(thickness);ft.append(feet)
                        beds.append(dict(domain='ext_dev' if external else 'viridien_eval',source=rec['source'],log=name,thickness_px=thickness,thickness_ft=feet))
                    rec['legend_truth_coverage']=covered/total if total else np.nan
                rec['median_bed_px']=float(np.median(px)) if px else np.nan
                rec['median_bed_ft']=float(np.nanmedian(ft)) if ft and np.isfinite(ft).any() else np.nan
                rows.append(rec)
                # One bounded crop per source/log for transparent visual comparison.
                if external and rec['source'] not in galleries:
                    crop=io.read_rgb(with_log,'interpreted_lithology_column',window[0],min(window[1],window[0]+900))
                    galleries[rec['source']]=(name,crop)
            finally:with_log.close()
    df=pd.DataFrame(rows);df.to_csv(HERE/'image_statistics.csv',index=False)
    pd.DataFrame(beds).to_csv(HERE/'bed_statistics.csv',index=False)
    evaldf=df[(df.domain=='viridien_all') & df.in_eval].copy();evaldf['source']='viridien_eval'
    fulldf=df[df.domain=='viridien_all'].copy();fulldf['source']='viridien_all_51'
    summary=pd.concat([df[df.domain=='ext_dev'],evaldf,fulldf]).groupby('source').agg(n_pages=('log','nunique'),n_windows=('log','size'),n_wells=('well','nunique'),width_px=('width_px','median'),px_per_ft=('px_per_ft','median'),colour_ink_share=('colour_ink_share','median'),pattern_ink_share=('pattern_ink_share','median'),ink_share=('ink_share','median'),legend_entries=('legend_entries','median'),legend_n_available=('legend_entries','count'),calibration_n_available=('px_per_ft','count'),one_bit_share=('one_bit','mean'),median_bed_px=('median_bed_px','median'),median_bed_ft=('median_bed_ft','median'),legend_truth_coverage=('legend_truth_coverage','mean'))
    summary.to_csv(HERE/'domain_summary.csv')
    # Exploratory visual neighbours use no target labels or accuracy results.
    features=['width_px','px_per_ft','colour_ink_share','ink_share']
    reference=evaldf.copy()
    ref=reference[features].to_numpy(dtype=float,copy=True)
    ref[:,:2]=np.log1p(ref[:,:2])
    center=np.nanmedian(ref,axis=0);scale=np.nanstd(ref,axis=0);scale[scale<1e-6]=1
    neighbours=[]
    for source,group in df[df.domain=='ext_dev'].groupby('source'):
        v=group[features].median().to_numpy(dtype=float,copy=True);v[:2]=np.log1p(v[:2])
        distance=np.sqrt(np.nanmean(((ref-v)/scale)**2,axis=1))
        for idx in np.argsort(distance)[:3]:
            neighbours.append(dict(source=source,log=reference.iloc[idx]['log'],distance=float(distance[idx]),**{k:float(reference.iloc[idx][k]) for k in features}))
    pd.DataFrame(neighbours).to_csv(HERE/'visual_neighbours.csv',index=False)
    plot(df,summary,representatives(df),neighbours,dataset)
    print(summary.to_string())

def representatives(df):
    galleries={}
    io.DATASET=REPO/'external_tests'
    for source,group in df[df.domain=='ext_dev'].groupby('source'):
        center=group[['colour_ink_share','ink_share']].median()
        selected=group.loc[((group[['colour_ink_share','ink_share']]-center).abs().sum(axis=1)).idxmin()]
        log=io.open_log(selected.log)
        try:
            y0=int(selected.y0);y1=int(selected.y1)
            top=max(y0,(y0+y1-900)//2)
            crop=io.read_rgb(log,'interpreted_lithology_column',top,min(y1,top+900))
            galleries[source]=(selected.log,crop)
        finally:log.close()
    return galleries

def redraw():
    df=pd.read_csv(HERE/'image_statistics.csv')
    summary=pd.read_csv(HERE/'domain_summary.csv',index_col=0)
    ref=df[(df.domain=='viridien_all')&df.in_eval]
    features=['width_px','px_per_ft','colour_ink_share','ink_share']
    x=ref[features].to_numpy(dtype=float,copy=True);x[:,:2]=np.log1p(x[:,:2])
    scale=np.nanstd(x,axis=0);scale[scale<1e-6]=1
    neighbours=[]
    for source,group in df[df.domain=='ext_dev'].groupby('source'):
        v=group[features].median().to_numpy(dtype=float,copy=True);v[:2]=np.log1p(v[:2])
        distance=np.sqrt(np.nanmean(((x-v)/scale)**2,axis=1))
        for idx in np.argsort(distance)[:3]:
            neighbours.append(dict(source=source,log=ref.iloc[idx].log,distance=float(distance[idx]),**{k:float(ref.iloc[idx][k]) for k in features}))
    pd.DataFrame(neighbours).to_csv(HERE/'visual_neighbours.csv',index=False)
    dataset=Path(os.environ.get('LITHOLOGY_VIRIDIEN_DATASET',REPO.parent/'dataset')).expanduser()
    plot(df,summary,representatives(df),neighbours,dataset)

def plot(df,summary,galleries,neighbours,dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    cats=['viridien_eval','kgs','idaho','nl','utah']
    fields=[('px_per_ft','Pixels per foot'),('width_px','Column width (px)'),('colour_ink_share','Colour share of ink'),('median_bed_px','Median complete bed (px)'),('median_bed_ft','Median complete bed (ft)'),('legend_entries','Parsed lithology legend entries')]
    fig,axes=plt.subplots(6,1,figsize=(6,20),layout='constrained')
    for ax,(field,label) in zip(axes,fields):
        data=[df[(df.domain=='viridien_all')&df.in_eval][field].dropna()]+[df[df.source==c][field].dropna() for c in cats[1:]]
        ax.boxplot(data,tick_labels=['Viridien','KGS','Idaho','NL','Utah'],showfliers=False,patch_artist=True,boxprops={'facecolor':'#DCEAF4'},medianprops={'color':'#D55E00','linewidth':2})
        ax.set(ylabel=label)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Lithology-column domain gap\nPer-window statistics; medians and quartiles',fontsize=15)
    fig.savefig(HERE/'domain_gap.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(6,14),layout='constrained')
    io.DATASET=dataset
    for row,source in enumerate(['kgs','idaho','nl','utah']):
        name,crop=galleries[source]
        ax=axes[row,0];ax.imshow(crop);ax.set_title(f'{source.upper()} dev\n{name}',fontsize=9);ax.axis('off')
        match=next(n for n in neighbours if n['source']==source)
        log=io.open_log(match['log'])
        try:
            item=df[df.log==match['log']].iloc[0]
            crop=io.read_rgb(log,'interpreted_lithology_column',int(item.y0),min(int(item.y1),int(item.y0)+900))
        finally:log.close()
        ax=axes[row,1];ax.imshow(crop);ax.set_title('Nearest Viridien example\n'+match['log'][:35],fontsize=9);ax.axis('off')
    fig.suptitle('Native-aspect column samples\nNeighbours from image statistics, not accuracy',fontsize=14)
    fig.savefig(HERE/'column_examples.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    if '--plots-only' in sys.argv:redraw()
    else:run()
