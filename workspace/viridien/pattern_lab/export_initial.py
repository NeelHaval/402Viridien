"""Export the initial approved Hadrian sources and make a comparison sheet."""
import json
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import Store,Images,DATA,HOME,write
from server import CropSource
from generator.core import generate
store=Store();doc=store.get(store.catalog[0]['id'])
examples=[s for s in doc['segments'] if s.get('example') and s.get('approved')]
body=json.dumps({'document_id':doc['id'],'count_per_example':3,'seed':0}).encode()
request=urllib.request.Request('http://127.0.0.1:8776/api/dataset',data=body,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=60) as response:
    result=json.load(response)
write(DATA/'first-dataset.json',{k:v for k,v in result.items() if k!='manifest'})
canvas=Image.new('RGB',(1000,455),'#f2f5f8');draw=ImageDraw.Draw(canvas)
images=Images();source=CropSource(images,doc)
for i,s in enumerate(examples):
    x=i*200;label=next(e['label_printed'] for e in doc['legend'] if e['id']==s['legend_ids'][0]);draw.text((x+8,8),label,fill='#22374d');draw.text((x+8,31),'Real',fill='#53657a');draw.text((x+96,31),'Generated',fill='#53657a')
    b=s['box'];crop=images.crop(doc,b);small=crop.crop((0,max(0,(crop.height-360)//2),crop.width,min(crop.height,max(0,(crop.height-360)//2)+360)))
    generated,meta=generate(source,b,width=90,height=360)
    canvas.paste(small,(x+8,60));canvas.paste(generated,(x+96,60));draw.text((x+8,433),meta['method'],fill='#53657a')
canvas.save(DATA/'pattern-comparison.png')
print('Initial dataset:',result['samples'],'samples from',result['examples'],'accepted sources.')
print('ZIP:',HOME/'data/exports'/Path(result['url']).name)
print('Comparison:',DATA/'pattern-comparison.png')
