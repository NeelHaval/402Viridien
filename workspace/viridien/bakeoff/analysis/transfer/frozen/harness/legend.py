"""Rectangle based legend discovery, OCR, and inspectable disk cache."""
from dataclasses import dataclass
from functools import lru_cache
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw
from . import io

VERSION = 13

@dataclass
class LegendEntry:
    id: int
    label: str
    swatch_box: list
    swatch: np.ndarray
    mean_rgb: tuple
    is_lithology: bool = True

@lru_cache(maxsize=1)
def _ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2,
                    det_limit_side_len=3200, det_limit_type='max')

def ocr(rgb, scale=3):
    """OCR boxes in input coordinates. Upscaling preserves tiny label glyphs."""
    big = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = _ocr()(big)
    return [(np.asarray(box, dtype=float)/scale, str(label), float(conf))
            for box, label, conf in (result or [])]

def _rectangles(rgb,kernel=3,colour_ink=True):
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = ((np.min(rgb,axis=2) < 225) if colour_ink else (gray<210)).astype('uint8')*255
    closed=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (max(9, w*.012) <= bw <= w*.24 and max(9, h*.015) <= bh <= h*.18): continue
        if not .35 <= bw/bh <= 12: continue
        area = cv2.contourArea(contour)
        if area/(bw*bh) < .65: continue
        # A rectangle has substantial ink along each edge, including hollow swatches.
        patch = ink[y:y+bh, x:x+bw] > 0
        edges = [patch[:2].mean(),patch[-2:].mean(),patch[:,:2].mean(),patch[:,-2:].mean()]
        if min(edges) < .18 or (edges[0]+edges[1])/2<.30: continue
        boxes.append((x,y,x+bw,y+bh))
    # Unbordered colour swatches and dense pattern rectangles form solid components.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colour = ((hsv[:,:,1]>65)&(hsv[:,:,2]>45)).astype('uint8')*255
    colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, np.ones((1,5),np.uint8))
    for contour in cv2.findContours(colour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,bw,bh = cv2.boundingRect(contour)
        if max(9,w*.012)<=bw<=w*.24 and 8<=bh<=h*.18 and .35<=bw/bh<=12:
            if cv2.contourArea(contour)/(bw*bh)>.55: boxes.append((x,y,x+bw,y+bh))
    # Keep outer contour of nested duplicates. Suppress a frame containing multiple cells.
    kept=[]
    for box in sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
        x,y,x1,y1=box
        if any(min(x1,b[2])-max(x,b[0])>0 and min(y1,b[3])-max(y,b[1])>0
               and max(0,min(x1,b[2])-max(x,b[0]))*max(0,min(y1,b[3])-max(y,b[1]))/
               min((x1-x)*(y1-y),(b[2]-b[0])*(b[3]-b[1]))>.7 for b in kept): continue
        kept.append(box)
    return sorted(kept,key=lambda b:(b[1]//8,b[0]))

def _pattern_boxes(rgb,words):
    """Find unframed pattern rectangles after removing OCR text components."""
    mask=(np.min(rgb,axis=2)<235).astype('uint8')*255
    for p,t,c in words:
        cv2.fillPoly(mask,[np.round(p).astype(np.int32)],0)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,9),np.uint8))
    h,w=mask.shape;result=[]
    for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,bw,bh=cv2.boundingRect(contour)
        if max(15,w*.025)<bw<w*.20 and 7<=bh<h*.18 and 1.3<bw/bh<13:
            if cv2.contourArea(contour)/(bw*bh)>.55 and np.mean(np.min(rgb[y:y+bh,x:x+bw],axis=2)<225)>.60:result.append((x,y,x+bw,y+bh))
    return result

def _wide_grid(rgb,boxes):
    """Unbordered Rock Types panels include sparse symbols in rectangular cells."""
    if len(boxes)<6:return boxes,False
    dims=np.asarray([(b[2]-b[0],b[3]-b[1]) for b in boxes])
    bw,bh=np.median(dims,axis=0)
    if bw/bh<3:return boxes,False
    boxes=[b for b in boxes if np.mean(np.min(rgb[b[1]:b[3],b[0]:b[2]],axis=2)<225)>.65]
    if not boxes:return [],True
    bw,bh=np.median([(b[2]-b[0],b[3]-b[1]) for b in boxes],axis=0)
    xs=[];ys=[]
    for axis,values,spread in [(0,xs,bw*.3),(1,ys,bh*.4)]:
        for b in sorted(boxes,key=lambda b:b[axis]):
            v=b[axis]
            if not values or abs(v-values[-1])>spread:values.append(v)
    if len(xs)<2 or len(ys)<3:return boxes,False
    result=[]
    for i,x in enumerate(xs):
        right=xs[i+1]-5 if i+1<len(xs) else rgb.shape[1]-5
        for y in ys:
            x1=min(rgb.shape[1],int(x+bw));y1=min(rgb.shape[0],int(y+bh))
            cell=rgb[y:y1,x:x1];text=rgb[max(0,y-2):min(rgb.shape[0],y1+3),x1+2:right]
            if text.size and np.mean(np.min(text,axis=2)<200)>.008 and np.mean(np.min(cell,axis=2)<225)>.005:
                result.append((int(x),int(y),x1,y1))
            elif cell.size:
                # A compact filled symbol (notably coal) has its label within
                # the otherwise full-width swatch cell.
                left=(np.min(cell[:,:max(1,int(bw*.25))],axis=2)<200).astype('uint8')
                count,_,stats,_=cv2.connectedComponentsWithStats(left,8)
                for sx,sy,sw,sh,area in stats[1:]:
                    if sx<3 and sh>bh*.65 and area/(sw*sh)>.72:
                        result.append((int(x+sx),int(y+sy),int(x+sx+sw),int(y+sy+sh)))
                        break
    return result,True

def _complete_grid(rgb,boxes):
    """Recover faint empty rectangles using repeated detected column dimensions."""
    if len(boxes)<4:return boxes
    ink=np.min(rgb,axis=2)<220
    extra=[]
    floor=max(b[3] for b in boxes)
    for x,y,x1,y1 in boxes:
        bw,bh=x1-x,y1-y
        if bw<50 or bh<35:continue
        profile=ink[:,x:x1].mean(axis=1)
        positions=np.flatnonzero(profile>.23)
        for yy in positions:
            for top in (int(yy),int(yy-bh+1)):
                bottom=top+bh
                if top<floor+bh*.7 or bottom>ink.shape[0]:continue
                patch=ink[top:bottom,x:x1]
                if patch[4:-4,4:-4].size and patch[4:-4,4:-4].mean()>.02:continue
                if max(patch[:3].mean(),patch[-3:].mean())<.2:continue
                if min(patch[:3].mean(),patch[-3:].mean())<.035:continue
                if min(patch[:,:3].mean(),patch[:,-3:].mean())<.035:continue
                b=(x,top,x1,bottom)
                if not any(abs(top-q[1])<bh*.5 and abs(x-q[0])<bw*.5 for q in boxes+extra):extra.append(b)
    return sorted(boxes+extra,key=lambda b:(b[1]//8,b[0]))

def _missing_cells(rgb,boxes):
    if len(boxes)<8:return boxes
    bw,bh=np.median([(b[2]-b[0],b[3]-b[1]) for b in boxes],axis=0)
    xs=[];ys=[]
    for axis,values,tol in [(0,xs,bw*.25),(1,ys,bh*.3)]:
        for b in sorted(boxes,key=lambda b:b[axis]):
            if not values or b[axis]-values[-1]>tol:values.append(b[axis])
    xs=[x for x in xs if sum(abs(b[0]-x)<bw*.25 for b in boxes)>=3]
    if len(xs)<2:return boxes
    result=list(boxes)
    for x in xs:
        for y in ys:
            if any(abs(b[0]-x)<bw*.35 and abs(b[1]-y)<bh*.4 for b in result):continue
            x1=min(rgb.shape[1],int(x+bw));y1=min(rgb.shape[0],int(y+bh))
            patch=np.min(rgb[y:y1,x:x1],axis=2)<225
            if patch.size and patch.mean()>.025:
                edges=[patch[:3].mean(),patch[-3:].mean(),patch[:,:3].mean(),patch[:,-3:].mean()]
                if sum(v>.2 for v in edges)>=3 or patch.mean()>.70:result.append((int(x),int(y),x1,y1))
    return result

def _label(box, boxes, words, shape):
    x,y,x1,y1=box
    bw,bh=x1-x,y1-y
    right= min([b[0]-2 for b in boxes if b[0]>x1+2 and abs((b[1]+b[3]-y-y1)/2)<bh*.8] or [shape[1]])
    right=min(right,x1+bw*5)
    below= min([b[1]-1 for b in boxes if b[1]>y1+2 and abs((b[0]+b[2]-x-x1)/2)<bw*.7] or [shape[0]])
    candidates=[]
    for poly,text,conf in words:
        cx,cy=poly.mean(axis=0)
        if x1-1<=cx<right and poly[:,0].min()>=x1-4 and poly[:,0].max()<=right+4 and y-bh*.1<=cy<=y1+bh*.1:
            candidates.append((poly[0,1],poly[0,0],text))
    if not candidates:
        for poly,text,conf in words:
            cx,cy=poly.mean(axis=0)
            if x-bw*.2<=cx<=x1+bw*.2 and y1<=cy<min(below,y1+bh*1.3):
                candidates.append((poly[0,1],poly[0,0],text))
    return ' '.join(c[2] for c in sorted(candidates)).strip()

def _excluded(label):
    return bool(re.search(r'\b(fossil\w*|accessor\w*|modifier\w*|core|dst|shoe|survey|test|fit|lot|rft|mud\s*(loss|gain)|gas|oil|gaz|huile|show\w*|foramin\w*|diatoms?|pyrite|glauconite|laminate beddings|not available|no sample|casing|unknown)\b',label,re.I))

def parse(log):
    cache=io.ROOT/'out'/'_cache'/'legend'
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/(log.name+'.json')
    data={}
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('version')==VERSION:
            return [LegendEntry(e['id'],e['label'],e['swatch_box'],np.asarray(Image.open(cache/e['swatch_file']).convert('RGB')),tuple(e['mean_rgb']),e['is_lithology']) for e in data['entries']]
    rgb=io.read_rgb(log,'lithology_legend')
    words=[(np.asarray(w['box']),w['text'],w['confidence']) for w in data.get('ocr',[])] or ocr(rgb)
    boxes=[]
    for kernel in (1,3,5,7):
        for colour_ink in (True,False):boxes.extend(_rectangles(rgb,kernel,colour_ink))
    # Glyph holes and bold heading letters can themselves look rectangular.
    boxes=[b for b in boxes if not any(
        p[:,0].min()-1<=(b[0]+b[2])/2<=p[:,0].max()+1 and
        p[:,1].min()-1<=(b[1]+b[3])/2<=p[:,1].max()+1 and
        b[3]-b[1] <= (p[:,1].max()-p[:,1].min())*1.6 and
        np.mean(np.min(rgb[b[1]:b[3],b[0]:b[2]],axis=2)<225)<.60
        for p,t,c in words)]
    patterns=_pattern_boxes(rgb,words)
    for b in patterns:
        if not any(max(0,min(b[2],q[2])-max(b[0],q[0]))*max(0,min(b[3],q[3])-max(b[1],q[1]))/min((b[2]-b[0])*(b[3]-b[1]),(q[2]-q[0])*(q[3]-q[1]))>.5 for q in boxes):boxes.append(b)
    # Suppress internal pattern cells across all closing scales before voting.
    unique=[]
    for b in sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
        if not any(max(0,min(b[2],q[2])-max(b[0],q[0]))*max(0,min(b[3],q[3])-max(b[1],q[1]))/min((b[2]-b[0])*(b[3]-b[1]),(q[2]-q[0])*(q[3]-q[1]))>.70 for q in unique):unique.append(b)
    boxes=unique
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    anchors=[b for b in boxes if (b[2]-b[0])*(b[3]-b[1])>350 and b[3]-b[1]>14 and
             np.mean(hsv[b[1]:b[3],b[0]:b[2],1]>65)>.45]
    # Repeated swatches establish the cell size; remove pattern fragments and
    # frames spanning several cells. Area weighting favours complete rectangles.
    if boxes:
        dimensions=np.asarray([(b[2]-b[0],b[3]-b[1]) for b in boxes],float)
        support=[]
        for width,height in dimensions:
            near=(abs(np.log(dimensions[:,0]/width))<.30)&(abs(np.log(dimensions[:,1]/height))<.30)
            support.append(np.sum(np.prod(dimensions[near],axis=1)))
        typical=dimensions[int(np.argmax(support))]
        if len(anchors)>=2:
            anchored=np.median([(b[2]-b[0],b[3]-b[1]) for b in anchors],axis=0)
            if typical[1]<anchored[1]*.6:typical=anchored
        boxes=[b for b in boxes if .70<(b[2]-b[0])/typical[0]<1.7 and .70<(b[3]-b[1])/typical[1]<1.5]
        boxes=[(b[0],b[1],int(b[0]+typical[0]) if b[2]-b[0]>typical[0]*1.2 else b[2],b[3]) for b in boxes]
        unique=[]
        for b in sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
            if not any(max(0,min(b[2],q[2])-max(b[0],q[0]))*max(0,min(b[3],q[3])-max(b[1],q[1]))/min((b[2]-b[0])*(b[3]-b[1]),(q[2]-q[0])*(q[3]-q[1]))>.45 for q in unique):unique.append(b)
        boxes=unique
    boxes,wide=_wide_grid(rgb,boxes)
    if not wide:boxes=_complete_grid(rgb,_missing_cells(rgb,boxes))
    boxes=sorted(boxes,key=lambda b:(b[1]//5,b[0]))
    headings=[]
    for poly,label,conf in words:
        if re.fullmatch(r'(?:modifiers?|shows?(?: (?:and|&) drilling)?(?: symbols)?|drilling(?: symbols)?|fossils?|accessor(?:y|ies)|technical|mineral|lithology(?: symbols)?|lithological symbols)',label.strip().rstrip(':'),re.I):
            cx,cy=poly.mean(axis=0)
            adjacent=any(b[2]<=cx and abs(cy-(b[1]+b[3])/2)<(b[3]-b[1])*.6 and poly[:,0].min()-b[2]<(b[2]-b[0])*2 for b in boxes)
            if not adjacent:headings.append((poly,label))
    offset=log.boxes['lithology_legend']
    entries=[]
    debug=Image.fromarray(rgb)
    draw=ImageDraw.Draw(debug)
    serial=[]
    for i,box in enumerate(boxes):
        x,y,x1,y1=box
        label=_label(box,boxes,words,rgb.shape)
        # Retry label crops when OCR of the entire grid missed a small label.
        if not label or wide:
            right=min([b[0]-2 for b in boxes if b[0]>x1+2 and abs(b[1]-y)<(y1-y)*.6] or [rgb.shape[1]])
            right=min(right,x1+(x1-x)*5)
            crop=rgb[max(0,y-2):min(rgb.shape[0],y1+3),x1+1:right]
            if crop.size and crop.shape[1]>5:
                crop=cv2.copyMakeBorder(crop,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
                label=' '.join(w[1] for w in ocr(crop))
        swatch=rgb[y+max(2,(y1-y)//14):y1-max(2,(y1-y)//14),x+max(2,(x1-x)//14):x1-max(2,(x1-x)//14)].copy()
        occupied=np.mean(np.min(swatch,axis=2)<225) if swatch.size else 0
        lith=bool(occupied>.008 and bool(label) and not _excluded(label))
        # Headings partition columns at midpoints; closest heading above the cell wins.
        above=[(p,t) for p,t in headings if p[:,1].mean()<y+3]
        if above:
            nearest=min(above,key=lambda pt:abs(pt[0][:,0].mean()-(x+x1)/2)+.25*(y-pt[0][:,1].mean()))
            if re.search(r'modifier|show|drilling|fossil|accessor|technical|mineral',nearest[1],re.I): lith=False
        label=label or ('Empty' if occupied<=.008 else 'Unknown')
        full=[x+offset[0],y+offset[1],x1+offset[0],y1+offset[1]]
        entry=LegendEntry(i,label,full,swatch,tuple(float(v) for v in swatch.mean(axis=(0,1))),lith)
        entries.append(entry)
        filename=log.name+f'.swatch_{i:02d}.png'
        Image.fromarray(swatch).save(cache/filename)
        serial.append(dict(id=i,label=label,swatch_box=full,mean_rgb=entry.mean_rgb,is_lithology=lith,swatch_file=filename))
        draw.rectangle(box,outline='lime' if lith else 'red',width=2)
        draw.text((x,max(0,y-12)),f'{i}: {label}',fill='blue',stroke_width=0)
    debug.save(cache/(log.name+'.debug.png'))
    path.write_text(json.dumps(dict(version=VERSION,entries=serial,ocr=[dict(box=p.tolist(),text=t,confidence=c) for p,t,c in words]),indent=2),encoding='utf-8')
    return entries
