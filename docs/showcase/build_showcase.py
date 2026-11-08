"""Make documentation figures from fixed, saved evidence. No model calls."""
from pathlib import Path
import csv
import hashlib
import io
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True, help='Existing approved evidence workspace')
parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
parser.add_argument('--review-dir', type=Path, required=True, help='Local scratch path outside Git')
args = parser.parse_args()
ROOT = args.source_root
REPO = args.repo_root
OUT = REPO / 'docs/showcase'
OUT.mkdir(parents=True, exist_ok=True)
OCR = ROOT / 'management/vultr-setup/test-swarm/ocr'
COMP = ROOT / 'viridien_work/bakeoff/report/scoreboard_1108/benedek_comparison'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
PINS = {
    'multi-tick.tif': '4280a39e932fdf0e0d0a9ac1bca02b7372e41aeb32bf9ef97ffa551d747f109c',
    'multi-tick.bbox.json': 'ab8fd492868643be0e5f107bd07b1e42f6c037bd3eec75fc3e07952fa2e3fd3c',
    'receipts.json': '168c8334486c3caf2302ffce873c113e1daeaf30546e17e7bc448c6d3bd126ec',
    'case.py': '1df5ed5e86c7cde917a58579fe6ebb3345c160dae58c14521df30f2feb9bb20e',
}
assert all(sha(OCR / name) == pin for name, pin in PINS.items())
receipts = json.loads((OCR / 'receipts.json').read_text(encoding='utf-8'))
csv_text = next(r['body'] for r in receipts if r['path'].endswith('/csv'))
rows = list(csv.DictReader(io.StringIO(csv_text)))
assert len(rows) == 4
assert [(int(r['top_y']), int(r['bottom_y'])) for r in rows] == [(400,696),(696,1000),(1000,1304),(1304,1600)]
source = Image.open(OCR / 'multi-tick.tif').convert('RGB')
font = lambda size, bold=False: ImageFont.truetype(str(args.font_dir / ('arialbd.ttf' if bold else 'arial.ttf')), size)
INK, MUTED = '#172633', '#405366'
figures = []

def save(im, name, record):
    path = OUT / name
    im.convert('RGB').save(path, optimize=True)
    assert path.stat().st_size < 500_000
    figures.append({'file': name, 'sha256': sha(path), 'bytes': path.stat().st_size, 'size': list(im.size), **record})

# Keep all three released pages. Use the same preselected middle 768-row excerpt.
manifest = json.loads((COMP / 'visual_manifest.json').read_text(encoding='utf-8'))
canvas = Image.new('RGB', (1760, 1010), 'white')
draw = ImageDraw.Draw(canvas)
draw.text((18,15), 'Released pages: source pixels beside saved S19 intervals', font=font(31,True), fill=INK)
draw.text((18,57), 'Standalone S19, not the app | unscored | S19 depth uncalibrated | vertical axis: full-image row (px)', font=font(22), fill=MUTED)
records=[]
for i, case in enumerate(['c27f24013e5cc155','49ab92322aa76b26','fc5af26c8937097c']):
    rel = 'pred_images/' + case + '_2.png'
    item = next(x for x in manifest['images'] if x['file'].replace('\\','/') == rel)
    path = COMP / rel
    with Image.open(path) as image:
        assert image.size == (1024,868)
        # Exact crop removes only the separate Benedek panel. No rescaling.
        canvas.paste(image.convert('RGB').crop((0,0,568,868)), (16+i*580,100))
    draw.text((18+i*580,975), 'rows ' + str(item['window']), font=font(19), fill=INK)
    records.append({'source': str(path.relative_to(ROOT)).replace('\\','/'), 'source_sha256': sha(path), 'crop_xyxy':[0,0,568,868], 'scale':1, 'case':case, 'window':item['window'], 'upstream_pins':item['hashes']})
save(canvas,'01-released-pages-s19.png',{'kind':'cached-released-page-predictions','truth_available':False,'app_evidence':False,'depth_available':False,'sources':records})

# The printed legend and cluster IDs are real saved bytes, not invented labels.
canvas=Image.new('RGB',(1440,530),'white');draw=ImageDraw.Draw(canvas)
draw.text((24,15),'Generated fixture: printed legend to saved output label and cluster ID',font=font(29,True),fill=INK)
canvas.paste(source.crop((30,1700,650,2100)),(20,83))
draw.text((700,83),'Saved OCR label',font=font(26,True),fill=INK)
draw.text((1160,83),'cluster_id',font=font(26,True),fill=INK)
for y,label in [(157,'Sandstone'),(257,'Shale grey'),(357,'Halite')]:
    ids={r['cluster_id'] for r in rows if r['lithology']==label}; assert len(ids)==1
    draw.text((700,y),label,font=font(29),fill=INK);draw.text((1210,y),next(iter(ids)),font=font(29,True),fill=INK)
draw.text((24,490),'No unknown label occurs in this fixture. Unknown labels must remain explicit on other inputs.',font=font(22),fill=MUTED)
save(canvas,'02-fixture-legend.png',{'kind':'generated-fixture-legend','source':'management/vultr-setup/test-swarm/ocr/multi-tick.tif','source_sha256':PINS['multi-tick.tif'],'crop_xyxy':[30,1700,650,2100],'receipt_sha256':PINS['receipts.json'],'app_image':'sha256:5b938b2669b04dcf86220b6c3bff712904d45cf4fcc29848b9df8ec8bbd65451'})

# Draw only the saved prediction edges. The original four-band pixels stay intact.
canvas=Image.new('RGB',(1500,1030),'white');draw=ImageDraw.Draw(canvas)
draw.text((24,15),'Generated four-band fixture: source and saved app intervals',font=font(32,True),fill=INK)
draw.text((24,60),'Historical image fcb8b3196042 | regression check, not project-scan accuracy | full-image row (px)',font=font(23),fill=MUTED)
crop=(40,380,390,1620);scale=.67
panel=source.crop(crop).resize((235,831),Image.Resampling.LANCZOS)
canvas.paste(panel,(140,135))
draw.text((130,104),'Original pixels',font=font(23,True),fill=INK)
draw.text((540,104),'Saved predicted interval',font=font(23,True),fill=INK)
draw.text((1120,104),'Fixture depth (ft)',font=font(23,True),fill=INK)
boundaries=[int(rows[0]['top_y'])]+[int(r['bottom_y']) for r in rows]
for y in boundaries:
    yy=135+round((y-crop[1])*831/(crop[3]-crop[1]))
    draw.line((110,yy,470,yy),fill='#144f83',width=2)
    draw.text((25,yy-13),str(y),font=font(21),fill=INK)
for row in rows:
    a,b=int(row['top_y']),int(row['bottom_y'])
    yy=135+round(((a+b)/2-crop[1])*831/(crop[3]-crop[1]))
    draw.text((540,yy-45),row['lithology']+'  /  cluster '+row['cluster_id'],font=font(28,True),fill=INK)
    draw.text((540,yy-6),f'y={a}..{b}',font=font(24),fill=MUTED)
    draw.text((1120,yy-25),row['top_depth']+' to '+row['base_depth'],font=font(25),fill=INK)
draw.text((24,986),'Three internal edges: drawn 700 / 1000 / 1300; saved 696 / 1000 / 1304. Maximum offset: 4 px.',font=font(24),fill=INK)
save(canvas,'03-fixture-source-prediction.png',{'kind':'generated-fixture-saved-predictions','source_sha256':PINS['multi-tick.tif'],'crop_xyxy':list(crop),'rendered_crop_size':[235,831],'receipt_sha256':PINS['receipts.json'],'app_image':'sha256:5b938b2669b04dcf86220b6c3bff712904d45cf4fcc29848b9df8ec8bbd65451','app_source_commit':'fcb8b319604200b999902b44753afb55b7da133d','boundary_check':{'drawn':[700,1000,1300],'saved':[696,1000,1304],'max_abs_px':4}})

(OUT/'fixture-api.csv').write_text(csv_text,encoding='utf-8',newline='')
with (OUT/'fixture-harness.csv').open('w',encoding='utf-8',newline='') as f:
    fields=['top_y','bottom_y','lithology','cluster_id','top_depth','bottom_depth']
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
    writer.writerows({k:(r['base_depth'] if k=='bottom_depth' else r[k]) for k in fields} for r in rows)
shutil.copyfile(OCR/'multi-tick.bbox.json',OUT/'fixture-bbox.json')
shutil.copyfile(OCR/'receipts.json',OUT/'fixture-receipts.json')
shutil.copyfile(ROOT/'viridien_work/bakeoff/strategies/challenge_export.py',REPO/'scripts/challenge_export.py')
adapter_sha=sha(REPO/'scripts/challenge_export.py')
(REPO/'docs/export-adapter.sha256').write_text(adapter_sha+'  scripts/challenge_export.py\n',encoding='utf-8',newline='\n')
evidence={'version':'strata-readme-evidence-v1','scope':'private repository documentation; no truth overlays; no final app acceptance','fixture_source_pins':PINS,'historical_app_image':'sha256:5b938b2669b04dcf86220b6c3bff712904d45cf4fcc29848b9df8ec8bbd65451','historical_app_commit':'fcb8b319604200b999902b44753afb55b7da133d','core_candidate_559c99cb4e24_accepted':False,'export_adapter_sha256':adapter_sha,'figures':figures,'saved_files':{name:sha(OUT/name) for name in ['fixture-api.csv','fixture-harness.csv','fixture-bbox.json','fixture-receipts.json']},'selection':'all three released pages, existing fixed middle 768-row excerpts; no labels used for selection','newtest_depth_note':'S19 depths unavailable in the saved comparison. Ignore older HTML boilerplate suggesting otherwise.'}
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8',newline='\n')
contact=Image.new('RGB',(1024,810),'#eeeeee')
for name,xy,box in [('01-released-pages-s19.png',(0,0),(1024,330)),('02-fixture-legend.png',(0,345),(510,450)),('03-fixture-source-prediction.png',(520,345),(500,450))]:
    with Image.open(OUT/name) as im:
        im.thumbnail(box);contact.paste(im,xy)
args.review_dir.mkdir(parents=True, exist_ok=True)
contact.save(args.review_dir/'contact-review.png')
print('FIGURES 3; saved fixture rows 4; internal boundaries 3; max offset 4 px; files under 500KB')
print('EXPORT_ADAPTER_SHA256 '+adapter_sha)
