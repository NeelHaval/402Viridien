"""Create label-A interval CSVs from visually verified fill patterns.

This is intentionally a one-off labelling aid: the lithology maps below were read
from each log's legend and the boundaries are taken from per-row fill colour.
"""
import csv, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_challenge\Viridien_HTB_challenge\hack27\project\dataset")
DEST = ROOT / "gt" / "silver" / "A"
DEST.mkdir(parents=True, exist_ok=True)

# Legend text, exactly as visually printed.  Palette colours are unambiguous
# backgrounds; black is hatch/grid ink and is removed before the row mode.
MAPS = {
 "60807402860000_gb506#2_md_mudlog": ([((128,255,128),"Clay/Shale"),((255,255,0),"Sandstone")], 5),
 "60811405120000_1inch tvd mudlog": ([((192,192,192),"Clay/Shale"),((255,255,0),"Sandstone"),((255,128,0),"Siltstone")], 5),
 "60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872": ([((0,255,0),"Clay Shale"),((255,255,0),"Coarse sandstone")], 5),
 "608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD": ([((0,255,0),"Clay/Shale"),((255,255,135),"Sandstone")], 5),
 "60817411400100_mc761_1_st1_mudlog_1md_17133_22886": ([((121,125,118),"Shale"),((240,240,0),"Sandstone"),((246,133,66),"Siltstone")], 5),
 "Critter Creek 201-1807H_Mud_PDF_Vertical.pdf": ([((249,125,1),"SILTSTONE"),((249,246,0),"SANDSTONE")], 5),
 "MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft": ([((0,255,0),"Clay-S")], 5),
 "Rio_Nueces_3744_Unit_1_212_MUD_5IN": ([((143,143,79),"Clay-S"),((175,207,223),"Limestone-L")], 5),
 "Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft": ([((0,255,0),"Clay-shale"),((255,255,0),"Fine sandstone")], 5),
}

def image_path(n):
    for x in ('.tif','.tiff'):
        p=DATA/'images'/(n+x)
        if p.exists(): return p

def clean_runs(labels, min_len):
    # Eliminate one-to-four-pixel drawing artefacts, retaining thin real beds.
    lab=labels.copy()
    for _ in range(4):
        cuts=np.r_[0,np.flatnonzero(lab[1:]!=lab[:-1])+1,len(lab)]
        changed=False
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b-a >= min_len: continue
            left=lab[a-1] if a else None; right=lab[b] if b<len(lab) else None
            if left is not None and (right is None or left==right): lab[a:b]=left; changed=True
            elif right is not None and left is None: lab[a:b]=right; changed=True
        if not changed: break
    return lab

def rgb_labels(a, palette, min_len):
    # modal non-ink colour per row; nearest palette tolerates scan/palette shifts
    p=np.array([x[0] for x in palette], dtype=float)
    ans=[]
    for row in a:
        pix=row.reshape(-1,3)
        keep=(pix.max(1)>55) & (pix.sum(1)<735) # discard black hatch and white page
        if not keep.any(): ans.append(-1); continue
        u,c=np.unique(pix[keep],axis=0,return_counts=True)
        v=u[c.argmax()]
        ans.append(int(np.argmin(((p-v)**2).sum(1))))
    ans=np.array(ans)
    # Missing rows are ink-only: interpolate locally before smoothing.
    good=np.flatnonzero(ans>=0)
    if len(good)==0: return ans
    for i in range(len(ans)):
        if ans[i]<0: ans[i]=ans[good[np.argmin(abs(good-i))]]
    ans=median_filter(ans,size=5,mode='nearest')
    return clean_runs(ans,min_len)

def bw_labels(a):
    # Horizontal dashes (Shale) carry much more ink than dotted Sand; blank is no ink.
    ink=(a==0).mean(1)
    # A 61-row profile covers one complete hatch repeat, rather than splitting
    # a dash's white interline into an artificial blank interval.
    den=np.convolve(ink,np.ones(61)/61,mode='same')
    out=np.where(den<0.035,-1,np.where(den<0.12,1,0)) # 0 Shale, 1 Sand and Sandstone
    out=median_filter(out,size=15,mode='nearest')
    return clean_runs(out,12)

def main():
    for e in json.loads((ROOT/'eval_set.json').read_text()):
        n,y0,y1=e['log'],e['y0'],e['y1']
        boxes=json.loads((DATA/'bbox_annotation'/(n+'.json')).read_text())['bounding_boxes']
        x0,_,x1,_=boxes['interpreted_lithology_column']
        with Image.open(image_path(n)) as im:
            if n.startswith('0000000009'):
                a=np.array(im.crop((x0+8,y0,x1-8,y1)).convert('1'))
                lab=bw_labels(a); names=["Shale","Sand and Sandstone"]
            elif n=="60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378":
                lab=np.full(y1-y0,-1); names=[]
            else:
                palette,m=MAPS[n]
                a=np.array(im.crop((x0+3,y0,x1-3,y1)).convert('RGB'))
                lab=rgb_labels(a,palette,m); names=[z[1] for z in palette]
        cuts=np.r_[0,np.flatnonzero(lab[1:]!=lab[:-1])+1,len(lab)]
        rows=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            if lab[a]>=0: rows.append((y0+int(a),y0+int(b),names[lab[a]]))
        with (DEST/(n+'.csv')).open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['top_y','bottom_y','lithology']); w.writerows(rows)
        print(n, len(rows), sorted(set(r[2] for r in rows)))

if __name__=='__main__': main()
