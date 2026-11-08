import sys,json,csv,math
from pathlib import Path
import fitz
import numpy as np
from PIL import Image
root=Path(sys.argv[1]).resolve(); out=root/'external_tests_round6'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
audit=read(out/'audit.json'); windows=read(out/'eval_set.json')
manifest=list(csv.DictReader((out/'manifest.csv').open(encoding='utf-8')))
checks=[]; total=0; pixels=0
for rec in audit['logs']:
 name=rec['id']; folder=root/rec['extraction']; m=read(folder/'meta.json'); shift=rec['y_shift']
 rows=list(csv.DictReader((out/'truth'/f'{name}.csv').open(encoding='utf-8')))
 box=read(out/'bbox_annotation'/f'{name}.json')['bounding_boxes']['interpreted_lithology_column']
 with fitz.open(root/rec['source_pdf']) as doc:
  scale=m['render']['px_per_pt']; pix=doc[m['page']-1].get_pixmap(matrix=fitz.Matrix(scale,scale),colorspace=fitz.csRGB,alpha=False)
  pdf=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
 with Image.open(out/'images'/f'{name}.tif') as image, Image.open(folder/'image.png') as source:
  assert source.size==pdf.size, name
  # Compare every track pixel with a new source-PDF raster, in bounded strips.
  checked=0
  for a in range(box[1],box[3],256):
   b=min(a+256,box[3]); dest=(box[0],a,box[2],b); orig=(box[0],a-shift,box[2],b-shift)
   assert np.array_equal(np.asarray(image.crop(dest)),np.asarray(pdf.crop(orig))), (name,a)
   checked+=(b-a)*(box[2]-box[0])
  for row in rows:
   a,b=int(row['top_y']),int(row['bottom_y']); assert box[1]<=a<b<=box[3]
   assert float(row['top_depth'])<float(row['bottom_depth'])
   candidates=[v for v in m['intervals_pt'] if round(v['top_pt']*scale)+shift==a and round(v['bottom_pt']*scale)+shift==b]
   assert any(v['lithology']==row['lithology'] for v in candidates), (name,row)
   v=next(v for v in candidates if v['lithology']==row['lithology'])
   dominant=min(v['components'],key=lambda c:(-round(c['width_share'],10),c['lithology'],c['xref'] or 0))
   assert dominant['lithology']==row['lithology']
   assert any(e['kind']=='rock' and e['xref']==v['xref'] and e['label']==row['lithology'] for e in m['legend_entries'])
  for first,second in zip(rows,rows[1:]): assert int(first['bottom_y'])<=int(second['top_y'])
  checks.append(dict(id=name,intervals=len(rows),track_pixels_checked=checked,pdf_raster_equal=True,geometry_and_legend_identity=True))
  total+=len(rows); pixels+=checked
 pdf.close()
for w in windows:
 assert w['log'] in {r['id'] for r in manifest}
 assert 0<w['y1']-w['y0']<=6000
wells=sorted({r['well'] for r in manifest})
split=dict(test=wells,train=[],validation=[],unit='whole well',note='All pages and windows from this well stay in test. No training or tuning permitted.')
(out/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
result=dict(status='PASS',windows=len(windows),wells=wells,families=['Texas Rock Types box / WellSight'],page_logs=len(manifest),truth_intervals=total,track_pixels_checked=pixels,checks=checks,limits=['Dominant visible width for mixed rows; equal-width ties use alphabetical labels.','Depths are fits to printed PDF depth marks, not independent surveyed depths.','Checks establish image and source-tile agreement; no independent geological ground truth is claimed.'])
(out/'pixel_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(f"PASS: {len(windows)} windows, {len(wells)} wells, {len(manifest)} pages, {total} intervals, {pixels} source-matched track pixels")
