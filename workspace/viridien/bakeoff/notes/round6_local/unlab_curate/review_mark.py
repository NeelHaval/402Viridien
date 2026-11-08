"""Record only sheets which the operator has just inspected."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('sheets',type=int,nargs='*');a=p.parse_args()
lane=Path(__file__).resolve().parent;out=lane.parents[3]/'external_unlabelled_test'
path=lane/'visual_review.json'
d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'approved_ids':[],'sheets':{}}
items=json.loads((out/'eval_set.json').read_text(encoding='utf-8'))
for n in a.sheets:
    name=f'{n:03d}.jpg';d['sheets'][name]=hashlib.sha256((out/'review_sheets'/name).read_bytes()).hexdigest()
    for item in items[n*4:n*4+4]:
        if item['log'] not in d['approved_ids']:d['approved_ids'].append(item['log'])
path.write_text(json.dumps(d,indent=2),encoding='utf-8')
changed=[]
for sheet in sorted((out/'review_sheets').glob('*.jpg')):
    if d['sheets'].get(sheet.name)!=hashlib.sha256(sheet.read_bytes()).hexdigest():changed.append(sheet.name)
print('Recorded',len(d['approved_ids']),'page approvals; sheets still requiring review:',','.join(changed))
