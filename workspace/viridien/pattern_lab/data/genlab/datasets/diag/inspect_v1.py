"""Frozen v1 diagnosis; run from the workspace with python -X utf8."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from genlab.dataset import *
from genlab.dataset_eval import Features, fit_predict

folder = ROOT / 'v1'
out = ROOT / 'diag'
rows = [json.loads(s) for s in (folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
truth = [r for r in read(folder/'real.json') if r['split'] == 'validation']
train = [r for r in rows if r['format'] == 'strip' and r['split'] == 'train' and r['variant'] == 'augmented']
features = Features(folder)
pred = fit_predict(features, train, truth)
b = DatasetBuilder()
records, failures = [], []
for r, p in zip(truth, pred):
    rgb = np.asarray(Image.open(folder/r['file']).convert('RGB'))
    box = list(r['box']); box[3] = min(box[3], box[1]+480)
    fresh = np.asarray(b.images.crop(b.docs[r['doc']], box).convert('RGB'))
    ink = rgb.max(2) < 80
    rec = dict(doc=r['doc'], annotation_id=r['annotation_id'], name=r['name'], predicted=p,
        box=r['box'], saved_wh=[rgb.shape[1],rgb.shape[0]], frame_equal=bool(np.array_equal(rgb, fresh)),
        white=float((rgb.min(2)>=240).mean()), ink=float(ink.mean()),
        edge_ink=float(np.concatenate([ink[:,:2].ravel(),ink[:,-2:].ravel()]).mean()),
        heavy_rows=int((ink.mean(1)>.65).sum()), heavy_cols=int((ink.mean(0)>.75).sum()),
        outside_column=bool(r['box'][0]<b.docs[r['doc']]['boxes']['interpreted_lithology_column'][0] or
                            r['box'][2]>b.docs[r['doc']]['boxes']['interpreted_lithology_column'][2]))
    records.append(rec)
    if p != r['name']:
        failures.append((r,p,rgb))
# Include every validation document, then three extra shale sink examples.
selected=[]; seen=set()
for f in failures:
    if f[0]['doc'] not in seen:
        selected.append(f); seen.add(f[0]['doc'])
selected += [f for f in failures if all(f[0]['file'] != s[0]['file'] for s in selected)][:10-len(selected)]
for i,(r,p,rgb) in enumerate(selected):
    true_fake=[s for s in rows if s['class_id']==r['class_id'] and s['variant']=='clean'][:2]
    if not true_fake:
        true_fake=[s for s in read(folder/'eval-swatches.json',[]) if s['class_id']==r['class_id'] and s['variant']=='clean'][:2]
    predicted=[s for s in train if s['name']==p][:2]
    sheet=Image.new('RGB',(1000,600),'#dedede'); draw=ImageDraw.Draw(sheet)
    draw.text((10,8),f"v1 failure {i+1}: {r['doc']} {r['annotation_id']}",fill='black')
    draw.text((10,26),f"Truth={r['name']} predicted={p} box={r['box']}",fill='black')
    panels=[('EXACT matcher input (native RGB)',Image.fromarray(rgb)),
            ('EXACT logistic input (12x20), enlarged 8x',Image.fromarray(rgb).resize((12,20),Image.Resampling.BILINEAR).resize((96,160),Image.Resampling.NEAREST))]
    panels += [(f"True fake {s['class_id']}"+(' (swatch; absent v1)' if s['file'].startswith('evaluation/') else ''), Image.open(folder/s['file']).convert('RGB')) for s in true_fake]
    panels += [(f"Pred fake {s['class_id']}", Image.open(folder/s['file']).convert('RGB')) for s in predicted]
    for j,(label,im) in enumerate(panels):
        x=10+(j%3)*330; y=60+(j//3)*270
        draw.text((x,y),label,fill='black')
        # Never rescale native matcher inputs. Very tall crops continue in source PNG.
        sheet.paste(im.crop((0,0,min(310,im.width),min(240,im.height))),(x,y+22))
    sheet.save(out/f'failure-{i+1:02}.png')
    Image.fromarray(rgb).save(out/f'failure-{i+1:02}-exact.png')
_write = __import__('genlab.build',fromlist=['_write'])._write
_write(out/'crop-audit.json',records)
print('PASS exact source-frame/RGB comparisons',sum(r['frame_equal'] for r in records),'/',len(records))
print('height min/median/max',np.percentile([r['saved_wh'][1] for r in records],[0,50,100]))
print('width min/median/max',np.percentile([r['saved_wh'][0] for r in records],[0,50,100]))
print('heavy rails',sum(r['heavy_cols']>0 for r in records),'frame rows',sum(r['heavy_rows']>0 for r in records),'outside column',sum(r['outside_column'] for r in records))
for n in ('shale','anhydrite'):
    rr=[r for r in records if r['name']==n]
    ff=[s for s in train if s['name']==n]
    print(n,'real n',len(rr),'white median',np.median([r['white'] for r in rr]),'fake n',len(ff),
          'fake white median',np.median([(np.asarray(Image.open(folder/s['file'])).min(2)>=240).mean() for s in ff]))
print('sheets',len(selected),'baseline correct',sum(r['name']==p for r,p in zip(truth,pred)))
