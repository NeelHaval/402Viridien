import sys, json, urllib.request, hashlib, collections, re
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from genlab import learn, candidates
import model
OUT = Path(__file__).parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(s): return ''.join(s.casefold().split())
def save(p, v): (OUT/p).write_text(json.dumps(v, indent=2), encoding='utf-8')
board = json.load(urllib.request.urlopen('http://127.0.0.1:8782/board.json'))
state = json.load(urllib.request.urlopen('http://127.0.0.1:8781/state.json'))
save('board-snapshot.json', board); save('state-snapshot.json', state)
policy = read(ROOT/'data/normalized/split-policy.json')
def train(did):
    group=policy['document_groups'][did]
    return int(hashlib.sha256((policy['seed']+':'+group).encode()).hexdigest()[:8],16)%5 != 0
reports={p.stem:read(p) for p in (ROOT/'data/genlab/candidates').glob('*.json') if len(p.stem)==16}
entries={f"{did}:{e['legend_id']}":e for did,r in reports.items() for e in r['entries']}
focus=[e for e in board['entries'] if e['training_allowed'] and e['state'] in ('Failed','Unsure','Rejected')]
names={norm(r['name']):did for did,r in reports.items()}
weights=collections.Counter(); total=0; intervals=collections.defaultdict(list)
with (ROOT/'../bakeoff/labeltool/data/prefill.jsonl').open(encoding='utf-8') as f:
    for line in f:
        g=json.loads(line); did=names.get(norm(g['log']))
        if did and train(did):
            h=max(0,int(g['bottom_y'])-int(g['top_y']));total+=h
            weights[did,norm(g.get('suggestion',''))]+=h
            intervals[did,norm(g.get('suggestion',''))].append(g)
def category(e):
    label=norm(e['label']); note=e['note'].lower()
    if e['state']!='Failed': return 'owner-box / review conflict'
    if any(x in label for x in ('conglomer','breccia','granitewash','finesand','salt-peppersand')): return 'stochastic texture candidate'
    if label in ('congl.sand','congl.sandstone','sand-sis'):return 'damage / composite / needs box'
    if any(x in note for x in ('letter','obscur','dropout','damag','stray','blurred','masking')): return 'damage / composite / needs box'
    return 'period or lattice underdetermined'
for e in focus:
    e['estimated_rows']=weights[e['doc_id'],norm(e['label'])]
    e['training_row_share']=e['estimated_rows']/total
    e['cause']=category(e)
    e['guess_intervals']=intervals[e['doc_id'],norm(e['label'])]
focus.sort(key=lambda e:-e['estimated_rows'])
save('inventory.json',dict(total_training_guessed_rows=total,counts=dict(collections.Counter(e['cause'] for e in focus)),entries=focus))
print('DENOM',total,'FOCUSROWS',sum(e['estimated_rows'] for e in focus))
print('CAUSES',collections.Counter(e['cause'] for e in focus))
for e in focus[:35]:print(e['key'],e['label'],e['state'],e['estimated_rows'],e['cause'])
images=model.Images()
docs={}
def crop(did, box):
    cache=ROOT/'data/genlab/crops'/did/('%d_%d_%d_%d.png'%tuple(box))
    if cache.exists():
        with Image.open(cache) as im:return np.asarray(im.convert('RGB'))
    if did not in docs:docs[did]=read(ROOT/'data/seeds'/f'{did}.json')
    return np.asarray(images.crop(docs[did],box).convert('RGB'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
for start in range(0,len(focus),25):
    sheet=Image.new('RGB',(1350,25*115),'#dddddd'); draw=ImageDraw.Draw(sheet)
    for i,e in enumerate(focus[start:start+25]):
        raw=crop(e['doc_id'],entries[e['key']]['swatch_box'])
        Image.fromarray(raw).save(OUT/f"swatch-{e['key'].replace(':','-')}.png")
        rgb,_=learn.clean(raw)
        e['crop_shape']=list(rgb.shape)
        draw.text((5,i*115+4),f"{e['key']} {e['label']} ({e['estimated_rows']} guessed rows)",font=font,fill='black')
        im=Image.fromarray(rgb); im=im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST)
        sheet.paste(im,(5,i*115+25))
    sheet.save(OUT/f'inventory-{start//25}.png')
if images.current:images.current.close()
save('inventory.json',dict(total_training_guessed_rows=total,counts=dict(collections.Counter(e['cause'] for e in focus)),entries=focus))
