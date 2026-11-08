import sys,json,csv
from pathlib import Path
from PIL import Image
p=Path(sys.argv[1]);base=p.parents[3]
e=json.loads((base/'bakeoff/eval_set.json').read_text(encoding='utf-8'));excluded={x['log'] for x in e}
with (base/'analysis/label_shares_per_log.csv').open(encoding='utf-8') as f:
 rows=list(csv.DictReader(f))
n=next(x['log'] for x in rows if 'Gulf' in x['family'] and x['log'] not in excluded)
f=next((base.parent/'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset/images').glob(n+'.*'))
Image.MAX_IMAGE_PIXELS=None;im=Image.open(f);im.thumbnail((460,850));im.convert('RGB').save(p/'exemplar.png');(p/'exemplar.json').write_text(json.dumps({'log':n,'path':str(f),'non_eval':True}),encoding='utf-8');print(n)
