"""Strip OCR and robust full-image row to depth calibration."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re
import time
import warnings
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from . import io, legend

VERSION=3
METRES_TO_FEET = 1 / .3048

@dataclass
class Calib:
    a: float
    b: float
    n_points: int
    rmse_px: float
    method: str

    def to_depth(self,y): return float(self.a*float(y)+self.b)

@lru_cache(maxsize=1)
def _easy():
    import easyocr
    import torch
    home_models=__import__('pathlib').Path.home()/'.EasyOCR'/'model'
    models=home_models if (home_models/'craft_mlt_25k.pth').exists() and (home_models/'english_g2.pth').exists() else io.ROOT/'out'/'_cache'/'easyocr'/'model'
    user=io.ROOT/'out'/'_cache'/'easyocr'/'user_network'
    user.mkdir(parents=True,exist_ok=True)
    return easyocr.Reader(['en'],gpu=torch.cuda.is_available(),model_storage_directory=str(models),user_network_directory=str(user))

def _tokens(rgb,engine):
    if engine=='rapidocr': return legend.ocr(rgb,scale=3)
    if engine=='easyocr':
        big=cv2.resize(rgb,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        return [(np.asarray(p)/3,str(t),float(c)) for p,t,c in _easy().readtext(big,allowlist='0123456789,.')]
    raise ValueError('engine must be rapidocr or easyocr')

def _fit(points):
    """Fit in pixel residual space; repeated OCR is not independent evidence.

    Two anchors are allowed only when they are the entire available evidence.
    A pair selected from contradictory readings must be recovered by OCR instead.
    """
    points=np.asarray(points,dtype=float).reshape(-1,2)
    points=np.unique(points[np.isfinite(points).all(axis=1)],axis=0)
    if len(points)<2: raise ValueError('Fewer than two numeric depth anchors found')
    y,d=points[:,0],points[:,1]
    # Search separated point pairs, using a 16px residual tolerance at any scale.
    rng=np.random.default_rng(0)
    best=None
    pairs = ([(i,j) for i in range(len(points)) for j in range(i)]
             if len(points)<60 else
             [rng.choice(len(points),2,replace=False) for _ in range(3000)])
    for i,j in pairs:
        if abs(y[i]-y[j])<max(80,np.ptp(y)*.08): continue
        a=(d[j]-d[i])/(y[j]-y[i])
        if not .0001<a<20: continue
        b=d[i]-a*y[i]
        residual=np.abs((d-b)/a-y)
        inliers=residual<16
        # Counting distinct rows avoids triple votes from OCR orientations.
        rows=np.unique(np.round(y[inliers]/5))
        key=(len(rows),-round(float(np.mean(residual[inliers]**2)), 4),
             float(np.ptp(y[inliers])) if inliers.any() else 0)
        if best is None or key>best[0]: best=(key,a,b,inliers)
    minimum=2 if len(points)==2 else 3
    if best is None or best[0][0]<minimum: raise ValueError('No consistent increasing depth calibration')
    good=best[3]
    reg=LinearRegression().fit(y[good,None],d[good])
    a=float(reg.coef_[0]);b=float(reg.intercept_)
    good=np.abs((d-b)/a-y)<16
    reg=LinearRegression().fit(y[good,None],d[good])
    a=float(reg.coef_[0]);b=float(reg.intercept_)
    rmse=float(np.sqrt(np.mean(((d[good]-b)/a-y[good])**2)))
    if not .0001<a<20 or not np.isfinite([a,b,rmse]).all():
        raise ValueError('Invalid depth slope or residual')
    if np.ptp(y[good])<80 or rmse>16:
        raise ValueError('Insufficient depth anchor span or excessive residual')
    return a,b,int(good.sum()),rmse


def _number(text):
    """Read small depths and decimal metres as well as thousands of feet."""
    token=text.strip().rstrip('-–—').strip()
    token=re.sub(r'(?<=\d)[\s\u00a0](?=\d{3}(?:\D|$))','',token)
    if ',' in token and '.' not in token and re.fullmatch(r'\d+,\d{1,2}',token):
        token=token.replace(',','.')
    else:
        token=token.replace(',','')
    if not re.fullmatch(r'\d{1,6}(?:\.\d+)?',token): return None
    return float(token)


def _units(texts):
    """Explicit depth units outrank units on rate-of-penetration tracks."""
    text=' '.join(texts).casefold()
    for pattern,unit in [
        (r'\b(?:depth|md|tvd)\s*(?:in\s+)?[(:\[]?\s*(?:feet|ft)\b','ft'),
        (r'\b(?:depth|md|tvd)\s*(?:in\s+)?[(:\[]?\s*(?:m|metres?|meters?)\b','m'),
        (r'(?<![a-z])(?:metres?|meters?)\b','m'),
        (r'\b(?:feet|ft)\b','ft'),
        (r'\b(?:m\s*/\s*(?:min|hr|h)|min\s*/\s*m)\b','m')]:
        if re.search(pattern,text): return unit
    return None


def _unit_context(log):
    """Use only image pixels, never file names, truth or neighbouring wells."""
    x0,y0,x1,y1=map(int,log.boxes['depth_column'])
    texts=[]
    # Curve headers and repeated ROP legends lie to the left of the depth track.
    boxes=[[max(0,x0-500),max(0,y0-160),min(log.size[0],x1+30),min(y1,y0+1800)]]
    for box in boxes:
        rgb=io.read_rgb(log,box)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            texts.extend(t for _,t,c in legend.ocr(rgb,scale=2) if c>=.5)
    return _units(texts),texts


def _unique(points):
    result=[]
    for p in sorted(points):
        if not any(abs(p[0]-q[0])<5 and p[1]==q[1] for q in result): result.append(p)
    return result


def _read_points(rgb,top,engine,clean=False):
    if clean:
        # Rules touching glyphs can join all labels into one OCR component.
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        mask=(gray<180).astype(np.uint8)
        rules=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((min(100,len(rgb)),1),np.uint8))
        rgb=rgb.copy();rgb[rules>0]=255
    result=[]
    for rotate in (0,1,3):
        rotated=np.rot90(rgb,rotate).copy() if rotate else rgb
        # Padding protects digits next to the crop edge, especially sparse pages.
        pad=12 if clean else 0
        if pad: rotated=cv2.copyMakeBorder(rotated,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tokens=_tokens(rotated,engine)
        for poly,text,conf in tokens:
            value=_number(text)
            if value is None or conf<.45: continue
            cx,cy=np.asarray(poly).mean(axis=0)-pad
            local_y=cy if rotate==0 else (cx if rotate==1 else rgb.shape[0]-1-cx)
            if 0<=local_y<len(rgb): result.append((float(top+local_y),value))
    return result

def cache_signature(log):
    """Hash the exact track and unit-context pixels in bounded strips."""
    x0,y0,x1,y1=map(int,log.boxes['depth_column'])
    boxes=[[x0,max(0,y0-32),x1,min(log.size[1],y1+1800)],
           [max(0,x0-500),max(0,y0-160),min(log.size[0],x1+30),min(y1,y0+1800)]]
    digest=hashlib.sha256()
    for box in boxes:
        digest.update(str(box).encode())
        for top in range(box[1],box[3],4000):
            digest.update(io.read_rgb(log,box,top,min(top+4000,box[3])).tobytes())
    return [list(log.size),list(log.boxes['depth_column']),digest.hexdigest()]


def calibrate(log,engine='rapidocr'):
    if engine not in ('rapidocr','easyocr'): raise ValueError(engine)
    cache=io.ROOT/'out'/'_cache'/'depth';cache.mkdir(parents=True,exist_ok=True)
    path=cache/(log.name+'.'+engine+'.json')
    signature=cache_signature(log)
    if path.exists():
        try:
            data=json.loads(path.read_text())
            if data.get('version')==VERSION and data.get('signature')==signature:
                calib=Calib(**data['calibration'])
                if np.isfinite([calib.a,calib.b,calib.rmse_px]).all() and calib.a>0:
                    return calib
        except (ValueError,KeyError,TypeError): pass
    start=time.perf_counter()
    x0,y0,x1,y1=map(int,log.boxes['depth_column'])
    points=[]
    # Sample strips across the full column, bounding OCR work even on 300k row logs.
    starts=np.unique(np.linspace(y0,max(y0,y1-1800),min(7,max(1,(y1-y0)//1800)),dtype=int))
    for top in starts:
        bottom=min(int(top)+1800,y1)
        rgb=io.read_rgb(log,[x0,y0,x1,y1],int(top),bottom)
        points.extend(_read_points(rgb,int(top),engine))
    unique=_unique(points)
    try: fit=_fit(unique)
    except ValueError: fit=None
    retried=fit is None or fit[2]<3
    if retried:
        # Annotation often stops at the last rock interval while the printed
        # depth scale continues. Recover those anchors from the same track.
        retry_y0=max(0,y0-32)
        retry_y1=min(log.size[1],y1+1800)
        retry_starts=np.unique(np.linspace(retry_y0,max(retry_y0,retry_y1-1800),
                                          min(7,max(1,int(np.ceil((retry_y1-retry_y0)/1800)))),dtype=int))
        for top in retry_starts:
            rgb=io.read_rgb(log,[x0,retry_y0,x1,retry_y1],int(top),min(int(top)+1800,retry_y1))
            points.extend(_read_points(rgb,int(top),engine,clean=True))
        unique=_unique(points)
        try: fit=_fit(unique)
        except ValueError:
            # Prefer the original two anchors if a retry adds only contradictions.
            if fit is None: raise
    a,b,n,rmse=fit
    if a*y0+b < -max(10,32*a):
        raise ValueError('Depth calibration extrapolates above zero at column top')
    unit,unit_text=_unit_context(log)
    if unit=='m': a*=METRES_TO_FEET;b*=METRES_TO_FEET
    method=engine+('+two-anchor' if n==2 else '+consensus')
    if retried: method+='+clean-retry'
    method+= '+m-to-ft' if unit=='m' else ('+ft' if unit=='ft' else '+ft-assumed')
    calib=Calib(a,b,n,rmse,method)
    path.write_text(json.dumps(dict(version=VERSION,signature=signature,calibration=calib.__dict__,seconds=time.perf_counter()-start,engine=engine,points=unique,unit=unit or 'ft-assumed',unit_text=unit_text),indent=2))
    return calib
