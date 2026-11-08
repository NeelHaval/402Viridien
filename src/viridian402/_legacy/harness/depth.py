"""Strip OCR and robust full-image row to depth calibration."""
from dataclasses import dataclass
from functools import lru_cache
import json
import re
import time
import warnings
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from . import io, legend

VERSION=1

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
    points=np.asarray(points,dtype=float)
    if len(points)<3: raise ValueError('Fewer than three numeric depth tokens found')
    y,d=points[:,0],points[:,1]
    # RANSAC threshold is in depth units. Search slopes from separated point pairs,
    # then use a tolerance corresponding to about 12 pixels for the final fit.
    rng=np.random.default_rng(0)
    best=None
    for _ in range(min(3000,len(points)**2)):
        i,j=rng.choice(len(points),2,replace=False)
        if abs(y[i]-y[j])<max(80,np.ptp(y)*.08): continue
        a=(d[j]-d[i])/(y[j]-y[i])
        if not .0001<a<20: continue
        b=d[i]-a*y[i]
        residual=np.abs((d-b)/a-y)
        inliers=residual<16
        key=(int(inliers.sum()),float(np.ptp(y[inliers])) if inliers.any() else 0)
        if best is None or key>best[0]: best=(key,a,b,inliers)
    if best is None or best[0][0]<3: raise ValueError('No consistent increasing depth calibration')
    model=RANSACRegressor(estimator=LinearRegression(),min_samples=2,
                          residual_threshold=abs(best[1])*16,max_trials=1000,random_state=0)
    # Initial consensus removes repeated header/footer numbers before RANSAC.
    subset=points[best[3]]
    model.fit(subset[:,:1],subset[:,1])
    a=float(model.estimator_.coef_[0]); b=float(model.estimator_.intercept_)
    good=np.abs((d-b)/a-y)<16
    reg=LinearRegression().fit(y[good,None],d[good])
    a=float(reg.coef_[0]);b=float(reg.intercept_)
    rmse=float(np.sqrt(np.mean(((d[good]-b)/a-y[good])**2)))
    if a<=0: raise ValueError('Nonpositive depth slope')
    return a,b,int(good.sum()),rmse

def calibrate(log,engine='rapidocr'):
    if engine not in ('rapidocr','easyocr'): raise ValueError(engine)
    cache=io.ROOT/'out'/'_cache'/'depth';cache.mkdir(parents=True,exist_ok=True)
    path=cache/(log.name+'.'+engine+'.json')
    if path.exists():
        data=json.loads(path.read_text())
        if data.get('version')==VERSION: return Calib(**data['calibration'])
    start=time.perf_counter()
    x0,y0,x1,y1=log.boxes['depth_column']
    points=[]
    # Sample strips across the full column, bounding OCR work even on 300k row logs.
    starts=np.unique(np.linspace(y0,max(y0,y1-1800),min(7,max(1,(y1-y0)//1800)),dtype=int))
    for top in starts:
        bottom=min(int(top)+1800,y1)
        rgb=io.read_rgb(log,[x0,y0,x1,y1],int(top),bottom)
        for rotate in (0,1,3):
            rotated=np.rot90(rgb,rotate).copy() if rotate else rgb
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tokens=_tokens(rotated,engine)
            for poly,text,conf in tokens:
                token=text.strip().replace(',','').replace(' ','')
                if not re.fullmatch(r'\d{3,6}(?:\.\d+)?',token) or conf<.45: continue
                cx,cy=np.asarray(poly).mean(axis=0)
                local_y=cy if rotate==0 else (cx if rotate==1 else rgb.shape[0]-1-cx)
                points.append((float(top+local_y),float(token)))
        # Never use a log name or GT values to choose rotation or depth anchors.
    # Deduplicate detections of the same number in overlapping strips/orientations.
    unique=[]
    for p in sorted(points):
        if not any(abs(p[0]-q[0])<5 and p[1]==q[1] for q in unique): unique.append(p)
    a,b,n,rmse=_fit(unique)
    calib=Calib(a,b,n,rmse,engine+'+ransac')
    path.write_text(json.dumps(dict(version=VERSION,calibration=calib.__dict__,seconds=time.perf_counter()-start,engine=engine,points=unique),indent=2))
    return calib
