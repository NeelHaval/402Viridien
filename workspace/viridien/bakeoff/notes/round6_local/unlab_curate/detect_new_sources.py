"""Find printed legend frames in new monochrome mud-log headers."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
cv2.setNumThreads(1)


def find_key(header):
    a=np.asarray(header.convert('L'));h,w=a.shape
    ink=(a<180).astype('uint8')*255
    contours,_=cv2.findContours(ink,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if not(.32*w<x<.81*w and 60<y<1300 and 12<=bw<=85 and 12<=bh<=85 and .65<bw/bh<1.55):continue
        if cv2.contourArea(c)/(bw*bh)<.65:continue
        boxes.append((x,y,x+bw,y+bh))
    kept=[]
    for b in sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
        if any(abs((b[0]+b[2]-q[0]-q[2])/2)<10 and abs((b[1]+b[3]-q[1]-q[3])/2)<10 for q in kept):continue
        kept.append(b)
    rows=[]
    for b in kept:
        cy=(b[1]+b[3])/2
        row=next((r for r in rows if abs(np.median([(q[1]+q[3])/2 for q in r])-cy)<18),None)
        if row is None:rows.append([b])
        else:row.append(b)
    good=[r for r in rows if len(r)>=6 and max(b[0] for b in r)-min(b[0] for b in r)>.15*w]
    if good:
        row=max(good,key=lambda r:len(r)+np.median([(b[2]-b[0])*(b[3]-b[1]) for b in r])/10000)
        xs=sorted((b[0]+b[2])/2 for b in row)
        pitch=np.diff(xs)
        # A symbol row has roughly even spacing. Avoid groups of large grid cells.
        if len(pitch) and np.std(pitch)/np.mean(pitch)<.6:
            return dict(style='wege',legend=[max(0,min(b[0] for b in row)-20),max(0,min(b[1] for b in row)-90),
                min(w,max(b[2] for b in row)+30),min(h,max(b[3] for b in row)+110)],symbols=len(row))
    columns=[]
    for b in kept:
        cx=(b[0]+b[2])/2
        if not .45*w<cx<.66*w:continue
        col=next((r for r in columns if abs(np.median([(q[0]+q[2])/2 for q in r])-cx)<20),None)
        if col is None:columns.append([b])
        else:col.append(b)
    good=[r for r in columns if len(r)>=8 and max(b[1] for b in r)-min(b[1] for b in r)>200]
    if good:
        col=max(good,key=len)
        return dict(style='horizon',legend=[max(0,min(b[0] for b in col)-round(.22*w)),
            max(0,min(b[1] for b in col)-90),min(w,max(b[2] for b in col)+25),
            min(h,max(b[3] for b in col)+30)],symbols=len(col))
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('lane',type=Path);a=p.parse_args()
    with (a.source/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    results={};cards=[]
    for i,r in enumerate(rows):
        path=a.source/'images'/f"{r['id']}.png"
        if not path.exists():continue
        with Image.open(path) as im:
            header=im.crop((0,0,im.width,min(2400,im.height))).convert('RGB')
            key=find_key(header)
            if key:
                key.update(image_size=list(im.size),source_image=str(path),source_url=r.get('source_url',''))
                results[r['id']]=key
                crop=header.crop(key['legend']);crop.thumbnail((500,180))
                card=Image.new('RGB',(512,256),'white')
                from PIL import ImageDraw
                d=ImageDraw.Draw(card);d.text((4,3),f"{i} {r['well']} | {key['style']} {key['symbols']} symbols",fill='black')
                d.text((4,17),r['id'],fill='black');card.paste(crop,(4,40));cards.append(card)
        if (i+1)%10==0:print('Header checks',i+1,'keys',len(results),flush=True)
    (a.lane/'auto_source_geometry.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    out=a.lane/'new_source_keys';out.mkdir(exist_ok=True)
    for start in range(0,len(cards),8):
        sheet=Image.new('RGB',(1024,1024),'white')
        for j,card in enumerate(cards[start:start+8]):sheet.paste(card,(j%2*512,j//2*256))
        sheet.save(out/f'{start//8:02d}.jpg',quality=94)
    print('Detected keys',len(results),'of',len(rows),flush=True)


if __name__=='__main__':main()
