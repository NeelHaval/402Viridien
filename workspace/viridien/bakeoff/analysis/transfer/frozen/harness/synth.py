"""Seeded pattern columns with exact interval truth and occasional blank gaps."""
import cv2
import numpy as np
from . import legend

def make(log,n=1,height=4000,seed=0):
    entries=[e for e in legend.parse(log) if e.is_lithology]
    if not entries: raise ValueError('No usable lithology swatches')
    rng=np.random.default_rng(seed)
    width=log.boxes['interpreted_lithology_column'][2]-log.boxes['interpreted_lithology_column'][0]
    result=[]
    for _ in range(n):
        rgb=np.full((height,width,3),255,np.uint8);rows=[];y=0;previous=None
        while y<height:
            bottom=min(height,y+int(rng.integers(90,500)))
            if rng.random()<.12:
                y=bottom;previous=None;continue
            candidates=[e for e in entries if e.id!=previous] or entries
            entry=candidates[int(rng.integers(len(candidates)))]
            swatch=entry.swatch
            tile=np.tile(swatch,(int(np.ceil((bottom-y)/swatch.shape[0])),int(np.ceil(width/swatch.shape[1])),1))
            rgb[y:bottom]=tile[:bottom-y,:width]
            # A drawn line makes every generated boundary intentional.
            rgb[y:min(y+1,bottom)]=30
            rows.append(dict(top_y=y,bottom_y=bottom,lithology=entry.label,cluster_id=entry.id))
            y=bottom;previous=entry.id
        result.append((rgb,rows))
    return result
