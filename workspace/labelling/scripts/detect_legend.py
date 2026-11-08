"""Detect boxed legend swatches and read adjacent labels locally; never writes labels."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Exclude recognisable non-lithology symbols, without mapping rock names to classes.
ACCESSORY = ('mud gain', 'mud loss', 'oil and gas', 'gas good', 'fresh water',
             'salted water', 'salt water', 'sidewall core', 'casing', 'survey',
             'core', 'mdt', 'fit', 'dry')


def legend_image(cache_dir: Path) -> tuple[Image.Image, list[int]]:
    meta = json.loads((cache_dir / 'meta.json').read_text())
    layer = meta['layers']['legend']
    box = layer['box']
    level = next(v for v in layer['levels'] if v['scale'] == 1)
    image = Image.new('RGB', (box[2]-box[0], box[3]-box[1]), 'white')
    for tile in level['tiles']:
        with Image.open(cache_dir / tile['file']) as part:
            image.paste(part, (tile.get('x0', box[0])-box[0], tile['y0']-box[1]))
    return image, box


def find_swatches(image: Image.Image) -> list[list[int]]:
    gray = np.asarray(image.convert('L'))
    H, W = gray.shape
    ink = (gray < 210).astype(np.uint8)*255
    # Repair short breaks in scanned rectangular borders, including 1-bit pages.
    mask = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if (max(20,W*.03) <= w < W*.45 and max(15,H*.025) <= h < H*.4
                and .65 < w/h < 6 and cv2.contourArea(contour)/(w*h) > .7):
            candidates.append([x,y,x+w,y+h])
    # Ignore pattern cells / inner duplicate contours contained within a swatch.
    boxes = []
    for b in sorted(candidates, key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
        if not any(b[0]>=a[0]-3 and b[1]>=a[1]-3 and b[2]<=a[2]+3 and b[3]<=a[3]+3 for a in boxes):
            boxes.append(b)
    if not boxes:
        return []
    # Repeated swatch dimensions distinguish the legend grid from printed letters.
    def similar(a,b):
        return .75<(a[2]-a[0])/(b[2]-b[0])<1.25 and .75<(a[3]-a[1])/(b[3]-b[1])<1.25
    best = max(boxes, key=lambda b:sum((a[2]-a[0])*(a[3]-a[1]) for a in boxes if similar(a,b)))
    boxes = [a for a in boxes if similar(a,best)]
    boxes.sort(key=lambda b:b[1])
    rows = []
    for box in boxes:
        row = next((r for r in rows if abs(r[0][1]-box[1]) < (box[3]-box[1])*.35), None)
        if row is None:
            rows.append([box])
        else:
            row.append(box)
    return [b for r in rows for b in sorted(r,key=lambda b:b[0])]


def text_regions(boxes, size):
    W,H = size
    gaps = []
    for b in boxes:
        same_row = [a for a in boxes if a[0]>b[0] and abs(a[1]-b[1])<(b[3]-b[1])*.35]
        if same_row:
            gaps.append((min(a[0] for a in same_row)-b[2])/(b[2]-b[0]))
    right = not gaps or float(np.median(gaps)) >= .8
    regions = []
    for x,y,x2,y2 in boxes:
        w,h = x2-x,y2-y
        if right:
            next_x = min((a[0] for a in boxes if a[0]>x and abs(a[1]-y)<h*.35),default=W)
            rect = [x2+max(3,round(w*.08)),max(0,y-round(h*.1)),next_x-3,min(H,y2+round(h*.15))]
        else:
            next_y = min((a[1] for a in boxes if a[1]>y2),default=H)
            rect = [max(0,x-round(w*.15)),y2+4,min(W,x2+round(w*.15)),min(H,next_y-5,y2+round(h*1.15))]
        regions.append((rect,7 if right else 6))
    return regions


def read_text(image, psm):
    binary = shutil.which('tesseract')
    if binary is None:
        return '', 'OCR unavailable: Tesseract is not installed; enter the names in the review.'
    # Upscaling small labels improves recognition; retain spelling and colour words.
    image = ImageOps.expand(image.convert('RGB'),border=8,fill='white')
    if image.height < 100:
        image = image.resize((image.width*3,image.height*3))
    buffer = io.BytesIO(); image.save(buffer,format='PNG')
    try:
        result = subprocess.run([binary,'stdin','stdout','--psm',str(psm)],
                                input=buffer.getvalue(),capture_output=True,timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return '',f'OCR failed: {exc}'
    if result.returncode:
        return '', 'OCR failed; enter the name in the review.'
    return ' '.join(result.stdout.decode('utf-8',errors='replace').split()), None


def detect(cache_dir: Path, ocr=read_text):
    image, frame = legend_image(cache_dir)
    boxes = find_swatches(image)
    suggestions, warnings = [], []
    for box,(region,psm) in zip(boxes,text_regions(boxes,image.size)):
        x,y,x2,y2 = box
        # Blank printed slots are not lithology swatches.
        interior = np.asarray(image.crop((x+4,y+4,x2-4,y2-4)).convert('L'))
        if not interior.size or np.mean(interior<225)<.008:
            continue
        if region[2]<=region[0] or region[3]<=region[1]:
            text,warning='',None
        else:
            text,warning = ocr(image.crop(region),psm)
        if warning and warning not in warnings:
            warnings.append(warning)
        normal = ' '.join(text.lower().replace('_',' ').split()).strip(' .:;')
        accessory = any(normal==term or normal.startswith(term+' ') for term in ACCESSORY)
        suggestions.append({'box':[x+frame[0],y+frame[1],x2+frame[0],y2+frame[1]],
                            'label_printed':text,'include':bool(text) and not accessory,
                            'note':'Likely drilling/accessory symbol' if accessory else 'Check the name and box'})
    if not suggestions:
        warnings.append('No boxed swatches detected. Use Box legend swatch for this layout.')
    return {'suggestions':suggestions,'warnings':warnings}
