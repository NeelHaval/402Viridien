"""Seeded construction tests and real-pattern synthetic stress columns.

Synthetic geometry has exact construction boundaries. Real-pattern labels are
inherited from parent human/vector labels, not independent geological truth.
Every derivative inherits its parent's split and well group.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .data import load_boxes, load_intervals, load_manifest, read_rgb, save_intervals, sha256

STRESSES = ('clean', 'blur', 'noise', 'low-contrast', 'jpeg', 'horizontal-shear', 'downsample', 'occlusion', 'mixed')


def _degrade(rgb, stress, rng):
    if stress not in STRESSES:
        raise ValueError(f'unsupported stress {stress!r}; choices: {", ".join(STRESSES)}')
    image = Image.fromarray(rgb)
    if stress in ('blur', 'mixed'):
        image = image.filter(ImageFilter.GaussianBlur(.75))
    if stress in ('low-contrast', 'mixed'):
        image = ImageEnhance.Contrast(image).enhance(.45)
    if stress == 'downsample':
        size = image.size
        image = image.resize((max(1,size[0]//2),max(1,size[1]//2)), Image.Resampling.BOX).resize(size,Image.Resampling.BILINEAR)
    if stress == 'jpeg':
        buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=40); buffer.seek(0)
        image = Image.open(buffer).convert('RGB')
    output = np.asarray(image).copy()
    if stress in ('noise','mixed'):
        output = np.clip(output.astype(float)+rng.normal(0,10,output.shape),0,255).astype(np.uint8)
    if stress == 'horizontal-shear':
        # Row-preserving distortion: no invalid claim about rotated boundary truth.
        for y in range(len(output)):
            output[y] = np.roll(output[y], round(4*np.sin(y/30)), axis=0)
    if stress == 'occlusion':
        # Narrow vertical scanner streak leaves the underlying construction label defined.
        x = output.shape[1]//2
        output[:,x:x+max(1,output.shape[1]//20)] = 235
    return output


def _tile(swatch, height, width):
    return np.tile(swatch, (max(1,(height+len(swatch)-1)//len(swatch)),
                            max(1,(width+swatch.shape[1]-1)//swatch.shape[1]), 1))[:height,:width]


def _write_example(output, identity, swatches, labels, *, height, seed, split, group, lineage, stress):
    rng = np.random.default_rng(seed)
    yoffset, width = max(180,len(labels)*48+12), 256
    page = np.full((height+yoffset,width,3),255,np.uint8)
    rows, legend = [], []
    minimum = max(8,height//12)
    boundaries = [0]
    while boundaries[-1] < height:
        boundaries.append(min(height,boundaries[-1]+int(rng.integers(minimum,max(minimum+1,height//4)))))
    choices = []
    for a,b in zip(boundaries,boundaries[1:]):
        available = [i for i in range(len(labels)) if not choices or i != choices[-1]] or [0]
        cluster = int(rng.choice(available)); choices.append(cluster)
        page[yoffset+a:yoffset+b,32:128] = _tile(swatches[cluster], b-a, 96)
        rows.append({'top_y':yoffset+a,'bottom_y':yoffset+b,'lithology':labels[cluster],
                     'cluster_id':cluster,'top_depth':round(1000+a*.5,1),'bottom_depth':round(1000+b*.5,1)})
    # Only the column is degraded. A clean source legend creates a controlled match task.
    page[yoffset:,32:128] = _degrade(page[yoffset:,32:128],stress,rng)
    for i,(swatch,label) in enumerate(zip(swatches,labels)):
        box = [160,8+i*48,208,40+i*48]
        page[box[1]:box[3],box[0]:box[2]] = _tile(swatch,32,48)
        legend.append({'id':i,'label':label,'swatch_box':box,'is_lithology':True})
    image = Image.fromarray(page)
    draw = ImageDraw.Draw(image)
    for i,label in enumerate(labels):
        draw.text((211,10+i*48),label[:7],fill='black')
    for yy in range(yoffset,height+yoffset,100):
        draw.text((0,yy),str(int(1000+(yy-yoffset)*.5)),fill='black')
    image_path, annotation_path, truth_path = (output/(identity+suffix) for suffix in ('.png','.json','.csv'))
    image.save(image_path)
    annotation = {'width':width,'height':height+yoffset,'bounding_boxes':{
        'depth_column':[0,yoffset,30,height+yoffset], 'interpreted_lithology_column':[32,yoffset,128,height+yoffset],
        'lithology_legend':[152,0,256,yoffset]}}
    annotation_path.write_text(json.dumps(annotation,indent=2)+'\n',encoding='utf-8')
    save_intervals(truth_path,rows)
    return {'id':identity,'image':image_path.name,'annotation':annotation_path.name,'truth':truth_path.name,
            'source':'synthetic','group':group,'split':split,'window':[yoffset,height+yoffset],
            'truth_window':[yoffset,height+yoffset], 'tags':['synthetic',stress],
            'label_provenance':{'kind':'exact_construction_boundaries','label_semantics':'inherited source label or procedural fixture',
                                'generator':'viridian402.synthetic:v1','seed':seed},
            'lineage':lineage,'legend':legend,'depth_calibration':{'a':.5,'b':1000-yoffset*.5},
            'depth_unit':'ft','depth_datum':'synthetic', 'fresh_holdout':False}


def _finish(output,records,name):
    path = output/'manifest.json'
    path.write_text(json.dumps({'schema_version':1,'name':name,'records':records,
        'evidence_note':'Constructed pattern tests; synthetic scores do not establish real scanned-log quality.'},indent=2)+'\n',encoding='utf-8')
    load_manifest(path)
    return path


def generate_demo(output_dir, seed=0, n=3) -> Path:
    """Tiny offline CI fixture; no network, weights or paid calls."""
    if n < 1 or n > 1000:
        raise ValueError('n must be between 1 and 1000')
    output = Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    y,x = np.indices((24,24))
    swatches = []
    for base, mask in [((205,180,90),(x+y)%9==0),((120,150,175),y%7==0),((180,115,115),(x%8==0)&(y%8<3))]:
        swatch = np.zeros((24,24,3),np.uint8); swatch[:] = base; swatch[mask] = (40,40,40); swatches.append(swatch)
    records=[]
    for i in range(n):
        group=f'procedural-demo:{seed}:{i}'
        records.append(_write_example(output,f'demo-{i:03d}',swatches,['Sandstone','Shale','Siltstone'],height=420,
            seed=seed+i,split='dev',group=group,stress='clean',
            lineage={'parent_id':'procedural-patterns-v1','parent_group':group,'parent_split':'dev','origin':'procedural_fixture'}))
    return _finish(output,records,'small-procedural-demo')


def generate_synthetic(manifest_path, output_dir, seed=0, n=3, height=800, stress='clean') -> Path:
    """Build n new columns using sampled real parent patterns, inheriting splits.

    Parent truth is used only during dataset construction. It is never present
    in a model request. Sample provenance records the exact source crop/hash.
    """
    if n < 1 or n > 1000 or height < 64 or height > 12000:
        raise ValueError('require 1<=n<=1000 and 64<=height<=12000')
    if stress not in STRESSES:
        raise ValueError(f'unsupported stress {stress}')
    source = load_manifest(manifest_path)
    candidates=[r for r in source['records'] if r['truth'] and r['source'] != 'synthetic']
    if not candidates:
        raise ValueError('real-pattern synthesis requires labelled nonsynthetic parents')
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed); records=[]
    for i in range(n):
        parent=candidates[int(rng.integers(len(candidates)))]
        box=load_boxes(parent['annotation'])['interpreted_lithology_column']
        truths=[]
        for original in load_intervals(parent['truth']):
            row={**original,'top_y':max(original['top_y'],parent['window'][0]),
                 'bottom_y':min(original['bottom_y'],parent['window'][1])}
            if row['bottom_y']-row['top_y']>=8:
                truths.append(row)
        by_label={}
        for row in truths:
            by_label.setdefault(row['lithology'],[]).append(row)
        if not by_label:
            raise ValueError(f"no usable source patterns in {parent['id']}")
        labels=sorted(by_label)
        if len(labels)>6:
            labels=sorted(rng.choice(labels,6,replace=False).tolist())
        swatches=[]; crops=[]
        for label in labels:
            row=by_label[label][int(rng.integers(len(by_label[label])))]
            inset=min(3,(box[2]-box[0]-1)//2)
            crop=[box[0]+inset,row['top_y']+2,box[2]-inset,min(row['bottom_y']-2,row['top_y']+66)]
            swatches.append(read_rgb(parent['image'],crop));crops.append({'label':label,'box':crop})
        lineage={'parent_id':parent['id'],'parent_group':parent['group'],'parent_split':parent['split'],
                 'parent_image_sha256':parent['image_sha256'],'parent_annotation_sha256':parent['annotation_sha256'],
                 'parent_truth_sha256':parent['truth_sha256'],'origin':'real_interval_crops','crops':crops,
                 'source_manifest_sha256':sha256(manifest_path),'stress':stress}
        records.append(_write_example(output,f'synthetic-{i:03d}',swatches,labels,height=height,seed=seed+i,
            split=parent['split'],group=parent['group'],lineage=lineage,stress=stress))
    return _finish(output,records,'real-pattern-synthetic-'+stress)
