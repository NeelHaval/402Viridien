"""Read-only dataset access and interval CSV helpers."""
from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
from emil.paths import dataset_path
DATASET = dataset_path()
FIELDS = ['top_y', 'bottom_y', 'lithology', 'cluster_id', 'top_depth', 'bottom_depth']

@dataclass
class Log:
    image: Image.Image
    boxes: dict
    name: str
    _tiff: object = field(default=None,init=False,repr=False)
    _strip_backend: object = field(default=None,init=False,repr=False)

    @property
    def mode(self): return self.image.mode

    @property
    def size(self): return self.image.size

    def close(self):
        self.image.close()
        if self._tiff is not None:self._tiff.close()

def _crop(log,box):
    """Decode only intersecting TIFF strips; Pillow is the CCITT/virtual fallback.

    Pillow.crop itself loads an entire TIFF. Using segment decoding for the large
    colour pages avoids that hidden full-page allocation as well as RGB copies.
    """
    if log._strip_backend is None:
        log._strip_backend=False
        filename=getattr(log.image,'filename',None)
        if filename:
            try:
                import tifffile
                log._tiff=tifffile.TiffFile(filename)
                page=log._tiff.pages[0]
                if not page.is_tiled and page.planarconfig==1 and page.photometric in (0,1,2,3) and page.rowsperstrip<=4000:
                    # Unsupported codecs (in particular CCITT) fail before any read.
                    page.decode(None,0)
                    log._strip_backend=True
            except (ImportError,ValueError,NotImplementedError):pass
    if not log._strip_backend:return log.image.crop(box).convert('RGB')
    page=log._tiff.pages[0];x0,y0,x1,y1=box
    output=np.empty((y1-y0,x1-x0,3),np.uint8)
    for index in range(y0//page.rowsperstrip,(y1-1)//page.rowsperstrip+1):
        log._tiff.filehandle.seek(page.dataoffsets[index])
        segment,where,_=page.decode(log._tiff.filehandle.read(page.databytecounts[index]),index)
        start=int(where[2]);a=max(y0,start);b=min(y1,start+segment.shape[1])
        data=segment[0,a-start:b-start,x0:x1]
        if page.photometric==3:
            table=(page.colormap>>8).astype(np.uint8).T
            data=table[data[:,:,0]]
        elif data.shape[-1]==1:
            if data.dtype==bool:data=data.astype(np.uint8)*255
            elif data.dtype!=np.uint8:data=np.clip(data,0,255).astype(np.uint8)
            if page.photometric==0:data=255-data
            data=np.repeat(data,3,axis=2)
        output[a-y0:b-y0]=data[:,:,:3]
    return Image.fromarray(output)

def _images():
    return {p.stem: p for p in (DATASET / 'images').iterdir()
            if p.suffix.lower() in ('.tif', '.tiff') and not p.name.endswith('md(1).tif')}

def list_logs():
    return sorted(set(_images()) & {p.stem for p in (DATASET / 'bbox_annotation').glob('*.json')})

def open_log(name):
    with (DATASET / 'bbox_annotation' / (name + '.json')).open(encoding='utf-8') as f:
        data = json.load(f)
    return Log(Image.open(_images()[name]), data['bounding_boxes'], name)

def read_rgb(log, box, y0=None, y1=None, scale=1.0):
    if isinstance(box, str): box = log.boxes[box]
    x0, by0, x1, by1 = map(int, box)
    y0 = max(by0, by0 if y0 is None else int(y0))
    y1 = min(by1, by1 if y1 is None else int(y1))
    if scale <= 0 or x1 <= x0 or y1 <= y0:
        raise ValueError('Empty crop or nonpositive scale')
    width, height = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
    result = np.empty((height, width, 3), dtype=np.uint8)
    for top in range(y0, y1, 4000):
        bottom = min(top+4000, y1)
        dst0 = round((top-y0)*height/(y1-y0))
        dst1 = round((bottom-y0)*height/(y1-y0))
        if dst1 <= dst0: continue
        crop = _crop(log,(x0, top, x1, bottom))
        if crop.size != (width, dst1-dst0):
            crop = crop.resize((width, dst1-dst0), Image.Resampling.LANCZOS)
        result[dst0:dst1] = np.asarray(crop)
    return result

def load_rows(path):
    path = Path(path)
    if not path.exists(): return None
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            item = dict(row)
            for key in ('top_y', 'bottom_y', 'cluster_id'):
                if item.get(key, '') != '': item[key] = int(float(item[key]))
            for key in ('top_depth', 'bottom_depth'):
                if item.get(key, '') != '': item[key] = float(item[key])
                else: item.pop(key, None)
            rows.append(item)
    return rows

def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
