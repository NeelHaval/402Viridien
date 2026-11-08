"""Read-only source audit; all crop cache and outputs stay in this directory."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, json, hashlib, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import model
from genlab import learn, paths, candidates

OUT = Path(__file__).resolve().parent
paths.CROPS = OUT / 'crops'
state = json.loads((ROOT / 'data/genlab/state.json').read_text(encoding='utf-8'))
cards = state['cards']
keys = [
    '5fbeaf2a0650e013:1024', 'd368760651ded938:15',
    '28112eef4c3c645a:19', '5709fe9e86ce942b:7',
    'c524c8e761b99c48:0', '952caeadf1d8467c:1',
    '952caeadf1d8467c:6', 'aea797844b2c0ce1:11',
    '7966a10fba6fa984:1', '57b1857d99779870:1',
    '1bac8dda58a39eaf:1', '952caeadf1d8467c:4',
    'a7ef06d45149d9f6:2',
    'c524c8e761b99c48:2', 'b4a8617d27247cbd:38',
    'b4a8617d27247cbd:44', '16c9064b18b77b61:0',
    'aea797844b2c0ce1:5', '38c9df683c1517bc:12',
    'd368760651ded938:3', '760ae0a5db33bede:0',
    'b4a8617d27247cbd:37', '926182367112f14a:0',
    '5fbeaf2a0650e013:1011', 'aeb414a4367bafd4:7',
    'c524c8e761b99c48:8',
    '926182367112f14a:u0', 'b4a8617d27247cbd:u1',
    '760ae0a5db33bede:u0', 'd2b5e2c76cfa887c:u0',
]
selected = [next(c for c in cards if c['key'] == k) for k in keys]
policy = json.loads((ROOT / 'data/normalized/split-policy.json').read_text())
def split(did):
    group = policy['document_groups'][did]
    value = hashlib.sha256((policy['seed'] + ':' + group).encode()).hexdigest()
    return 'test' if int(value[:8], 16) % 5 == 0 else 'train'
summary = dict(built_at=state['built_at'], cards=len(cards),
    source_counts=dict(collections.Counter(c['source'] for c in cards)),
    sample_keys=keys,
    test_cards=[c['key'] for c in cards if split(c['doc_id']) == 'test'],
    test_donors=[c['key'] for c in cards if c.get('donor') and split(c['donor']['doc_id']) == 'test'],
    independent_scores=sum(bool(c.get('realism', {}).get('held_out')) for c in cards if c.get('realism')),
    reused_scores=sum(bool(c.get('realism', {}).get('reused_training')) for c in cards if c.get('realism')),
    agent_column_evidence=sum(c.get('evidence_kind') == 'column' for c in cards if c['source']=='agent-made'))
(OUT / 'snapshot.json').write_text(json.dumps(dict(summary=summary, cards=selected),indent=2), encoding='utf-8')
print(json.dumps(summary), flush=True)
store, images = model.Store(), model.Images()
crops = learn.Crops(images)
docs = {}
results, panels = [], []
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)

def getdoc(did):
    if did not in docs: docs[did] = store.get(did)
    return docs[did]

def rgb_at(did, box):
    box = list(map(int,box))
    assert (box[2]-box[0])*(box[3]-box[1]) < 300000
    return crops.get(getdoc(did), box)

def describe(tile, reference):
    clean, offset = learn.clean(reference)
    data = dict(tile=tile, bg=learn.background(tile))
    pix = candidates.pixel_match(clean, data)
    dice = candidates.symbol_match(clean, data)
    return dict(pixel=round(pix,5), ink_dice=round(dice,5),
        bg_gap=round(float(np.linalg.norm(learn.background(clean)-data['bg'])),2),
        ink_real=round(float((clean.max(2)<80).mean()),5),
        ink_tile=round(float((tile.max(2)<80).mean()),5),
        shape=list(clean.shape[:2]), offset=list(offset))

for index, card in enumerate(selected):
    did, lid = card['doc_id'], card['legend_id']
    doc = getdoc(did)
    tile = np.asarray(Image.open(ROOT/'data/genlab'/card['tile']).convert('RGB'))
    sample = np.asarray(Image.open(ROOT/'data/genlab'/card['samples'][0]).convert('RGB'))
    legend = next((e for e in doc['legend'] if e['id']==lid), None)
    swatch = rgb_at(did, legend['box']) if legend else None
    reals = [rgb_at(card.get('real_doc_id',did), b) for b in card['real']]
    col = doc['boxes']['interpreted_lithology_column']
    actual_column = [b[1]>=col[1] and b[3]<=col[3] and b[0]>=col[0]-5 and b[2]<=col[2]+5 for b in card['real']]
    result = dict(key=card['key'], source=card['source'],label=card['label'],
        tile_size=list(tile.shape[:2]), real_is_column=actual_column,
        real_doc_id=card.get('real_doc_id',did), real_boxes=card['real'],
        realism=card.get('realism'), method=card.get('method'),
        real_metrics=[describe(tile,r) for r in reals],
        swatch_metrics=describe(tile,swatch) if swatch is not None else None)
    results.append(result)
    stem = f'{index+1:02d}-{did}-{lid}'
    Image.fromarray(tile).save(OUT/(stem+'-tile.png'))
    for j,r in enumerate(reals): Image.fromarray(r).save(OUT/(stem+f'-real{j}.png'))
    if swatch is not None: Image.fromarray(swatch).save(OUT/(stem+'-legend.png'))
    # Every image has the identical 3x nearest-neighbour pixel scale.
    row = Image.new('RGB',(1650,570),'#e8e8e8')
    draw=ImageDraw.Draw(row)
    title = f'{index+1:02d} {card["key"]} {card["source"]} {card["label"][:75]}'
    draw.text((8,4),title,fill='black',font=font)
    vals = [tile, sample, reals[0] if reals else None,
            reals[1] if len(reals)>1 else None, swatch]
    labels = ['Native tile (once)','Generated strip','Real 1 '+('COLUMN' if actual_column and actual_column[0] else 'legend/donor'),
              'Real 2' if len(reals)>1 else 'No second crop','Legend (untrimmed)' if swatch is not None else 'No legend (unlisted)']
    for j,(rgb,label) in enumerate(zip(vals,labels)):
        x=8+j*328
        draw.text((x,30),label,fill='black',font=small)
        if rgb is not None:
            part=Image.fromarray(rgb[:160,:103])
            part=part.resize((part.width*3,part.height*3),Image.Resampling.NEAREST)
            row.paste(part,(x,52))
            draw.text((x,538),f'{rgb.shape[1]} x {rgb.shape[0]} native px',fill='black',font=small)
    panels.append(row)
    print(json.dumps(result),flush=True)
    if len(panels)==5:
        sheet=Image.new('RGB',(1650,570*len(panels)),'white')
        for j,p in enumerate(panels): sheet.paste(p,(0,j*570))
        sheet.save(OUT/f'sheet-{index//5+1}.png')
        panels=[]
    (OUT/'measurements.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if images.current: images.current.close()
print('COMPLETE',len(results),flush=True)
