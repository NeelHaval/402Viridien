import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from genlab.dataset import *
f=ROOT/'v4';e=read(f/'eval.json')
truth={r['annotation_id']:r for r in read(f/'real.json') if r['split']=='validation'}
models={r['class_id']:r for r in read(f/'stats.json')['classes']}
fake={r['class_id']:r for r in read(f/'eval-swatches.json') if r['variant']=='clean'}
failed=[];seen=set()
for p in e['closed_set_within_document']['matcher']['clean']['predictions']:
    if p['predicted']!=p['actual'] and p['actual'] not in seen:
        failed.append(p);seen.add(p['actual'])
sheet=Image.new('RGB',(1120,240*len(failed)+35),'#dedede');draw=ImageDraw.Draw(sheet)
draw.text((10,7),'v4 remaining failures: real / true native legend / true swatch fake / predicted swatch fake',fill='black')
for i,p in enumerate(failed):
    r=truth[p['annotation_id']];m=models[p['actual']];y=35+240*i
    draw.text((10,y),f"{m['class_id']} {m['name']} => {models[p['predicted']]['name']} seam_risk={m['tile_provenance']['seam_risk']}",fill='black')
    panels=[Image.open(f/r['file']),Image.open(f/m['tile_provenance']['raw_file']),Image.open(f/fake[p['actual']]['file']),Image.open(f/fake[p['predicted']]['file'])]
    for j,im in enumerate(panels):
        draw.text((10+j*275,y+18),f'{im.width} x {im.height} (native)',fill='black')
        sheet.paste(im.crop((0,0,min(260,im.width),min(195,im.height))),(10+j*275,y+35))
sheet.save(ROOT/'diag/v4-remaining-failures.png')
print('PASS remaining-class failure sheet',len(failed))
