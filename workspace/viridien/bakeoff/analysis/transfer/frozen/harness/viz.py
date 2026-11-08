"""Flat, bounded-size comparison figures using strip-safe source reads."""
from pathlib import Path
import colorsys
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import io, score

def colour(label):
    hue=int(hashlib.sha1(score.normalize(label).encode()).hexdigest()[:8],16)/0xffffffff
    return tuple(round(c*255) for c in colorsys.hsv_to_rgb(hue,.32,.95))

def _font(size=13):
    try: return ImageFont.truetype('arial.ttf',size)
    except OSError: return ImageFont.load_default()

def render(log,rows,out_png,gt_rows=None,window=None,stages=None,legend_entries=None,title=''):
    box=log.boxes['interpreted_lithology_column']
    window=window or (box[1],box[3]);y0,y1=map(int,window)
    height=min(2200,max(700,y1-y0));scale=height/(y1-y0)
    stage_items=[(k,v) for k,v in (stages or {}).items() if not k.startswith('_') and isinstance(v,(np.ndarray,Image.Image))]
    canvas=Image.new('RGB',(1700,height+100+(300 if stage_items else 0)),'white')
    d=ImageDraw.Draw(canvas);font=_font();small=_font(11)
    d.text((15,12),title or log.name,fill='black',font=_font(17))
    top=65
    thumb_scale=min(300/log.size[0],height/log.size[1])
    thumb=Image.fromarray(io.read_rgb(log,[0,0,*log.size],scale=thumb_scale))
    canvas.paste(thumb,(10,top))
    for key,c in [('depth_column','blue'),('interpreted_lithology_column','red'),('lithology_legend','green')]:
        b=log.boxes[key];d.rectangle((10+b[0]*thumb_scale,top+b[1]*thumb_scale,10+b[2]*thumb_scale,top+b[3]*thumb_scale),outline=c,width=2)
    d.rectangle((10+box[0]*thumb_scale,top+y0*thumb_scale,10+box[2]*thumb_scale,top+y1*thumb_scale),outline='orange',width=3)
    col=Image.fromarray(io.read_rgb(log,box,y0,y1,scale=min(1,scale)))
    col=col.resize((280,height))
    canvas.paste(col,(345,top))
    d.text((345,43),'Prediction / source',fill='black',font=font)
    d.text((640,43),'Ground truth',fill='black',font=font)
    d.text((840,43),'Error',fill='black',font=font)
    def strip(intervals,x,width):
        for row in intervals:
            a=max(y0,int(row['top_y']));b=min(y1,int(row['bottom_y']))
            if b<=a:continue
            yy=top+round((a-y0)*scale);bottom=top+round((b-y0)*scale)
            d.rectangle((x,yy,x+width,bottom),fill=colour(row['lithology']),outline='white')
            if bottom-yy>11:d.text((x+3,yy+2),str(row['lithology'])[:24],fill='black',font=small)
            d.line((x-5,yy,x+width+3,yy),fill='black')
    strip(rows,345,140)
    if gt_rows is not None:strip(gt_rows,640,185)
    else:d.text((640,top+15),'No GT available',fill='gray',font=font)
    if gt_rows is not None:
        pred=score.raster(rows,window);gt=score.raster(gt_rows,window)
        err=np.full((len(gt),16,3),185,np.uint8)
        err[(gt!='')&(pred==gt)]=(60,180,85)
        err[(gt!='')&(pred!=gt)]=(225,65,65)
        canvas.paste(Image.fromarray(err).resize((16,height),Image.Resampling.NEAREST),(845,top))
    ticks={}
    for row in rows:
        for edge in ('top','bottom'):
            y=int(row[edge+'_y'])
            if y0<=y<=y1:ticks[y]=row.get(edge+'_depth')
    for y,value in sorted(ticks.items()):
        yy=top+round((y-y0)*scale)
        text=f'{y} px' if value is None else f'{value:.1f} ft'
        d.text((875,yy-6),text,fill='black',font=small)
    legend_box=log.boxes['lithology_legend']
    legend_rgb=io.read_rgb(log,legend_box)
    ls=min(650/legend_rgb.shape[1],min(height,1100)/legend_rgb.shape[0])
    legend_img=Image.fromarray(legend_rgb).resize((max(1,round(legend_rgb.shape[1]*ls)),max(1,round(legend_rgb.shape[0]*ls))))
    canvas.paste(legend_img,(1020,top));d.text((1020,43),'Legend (used entries highlighted)',fill='black',font=font)
    used={score.normalize(r['lithology']) for r in rows}
    for entry in legend_entries or []:
        b=entry.swatch_box
        rect=(1020+(b[0]-legend_box[0])*ls,top+(b[1]-legend_box[1])*ls,1020+(b[2]-legend_box[0])*ls,top+(b[3]-legend_box[1])*ls)
        active=score.normalize(entry.label) in used
        d.rectangle(rect,outline='red' if active else ('green' if entry.is_lithology else 'gray'),width=3 if active else 1)
    if stage_items:
        cell=1680//len(stage_items)
        for i,(name,img) in enumerate(stage_items):
            x=10+i*cell;y=height+85
            d.text((x,y),name,fill='black',font=font)
            stage=Image.fromarray(img) if isinstance(img,np.ndarray) else img.copy()
            stage.thumbnail((cell-10,245));canvas.paste(stage.convert('RGB'),(x,y+25))
    path=Path(out_png);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
