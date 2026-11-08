"""PNG-only release audit: python -X utf8 -m genlab.tests.audit_dataset v6 --strict."""
from __future__ import annotations
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from scipy.ndimage import label
from genlab.dataset import ROOT,read,pixel_hash
from genlab.dataset_eval import leak_audit


def audit(folder):
    folder = Path(folder)
    result = dict(version=folder.name,leaks=leak_audit(folder))
    rows = [json.loads(s) for s in (folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    reals = read(folder/'real.json')
    classes = read(folder/'stats.json')['classes']
    def shape(row):
        with Image.open(folder/row['file']) as im:
            return im.size
    clean = {(r['class_id'],shape(r)):r for r in rows if r['format']=='strip' and r['variant']=='clean'}
    identical,totals = Counter(),Counter()
    calibration = defaultdict(Counter)
    for row in rows:
        if row['format']=='strip' and row['variant']=='augmented':
            totals[row['split']] += 1
            calibration[row['split']+'/'+row['family']][row.get('augmentation',{}).get('calibration','legacy-unrecorded')] += 1
            paired = clean.get((row['class_id'],shape(row)))
            if paired and paired['pixel_sha256']==row['pixel_sha256']:
                identical[row['split']] += 1
    result['identity'] = dict(identical=dict(identical),total=dict(totals))
    result['calibration_by_split_family'] = {k:dict(v) for k,v in calibration.items()}
    result['gates'] = dict(validation_seam_risk=sum(bool(c.get('tile_provenance',{}).get('seam_risk')) for c in classes if c['split']=='validation'),
        unsafe_rendered=sum(c.get('renderable',True) and bool(c.get('tile_provenance',{}).get('seam_risk')) for c in classes),
        withheld=sum(not c.get('renderable',True) for c in classes))
    support = read(folder/'eval-swatches.json',[])
    finite = [r for r in support if r.get('format')=='finite-legend']
    result['recognition_support'] = dict(finite_classes=len({r['class_id'] for r in finite}),
        finite_records=len(finite),finite_in_training_manifest=sum(r.get('format')=='finite-legend' for r in rows),
        generation_classes=sum(c.get('synthesis_eligible',c.get('renderable',True)) for c in classes))
    result['integrity'] = dict(pixel_mismatches=result['leaks'].get('pixel_mismatches',0),mask_mismatches=0)
    if not result['leaks']['passed']:
        result['passed'] = False
        return result
    for meta in classes:
        if meta.get('render_model'):
            from genlab import learn
            try:
                learn.load_model(meta['render_model'],root=folder,signature=meta['signature'])
            except (ValueError,KeyError,OSError):
                result['integrity']['pixel_mismatches'] += 1
    for row in rows:
        with Image.open(folder/row['file']) as im:
            rgb = np.asarray(im.convert('RGB'))
        result['integrity']['pixel_mismatches'] += pixel_hash(rgb)!=row['pixel_sha256']
        if row['format']=='column':
            with Image.open(folder/row['label_mask_png']) as im:
                mask = np.asarray(im)
            for layer in read(folder/row['truth_json'])['layers']:
                result['integrity']['mask_mismatches'] += not np.all(mask[layer['top']:layer['bottom']]==layer['mask_value'])
    old = ROOT/'v5'
    fixture = next((m for m in read(old/'realism.json',{}).get('measurements',[]) if m['id']=='c72589f5d566d9b5563875b8'),None)
    if fixture:
        from genlab.realism import component_budget
        with Image.open(old/fixture['mask']) as im:
            mask = np.asarray(im)
        with Image.open(old/fixture['colours']) as im:
            colours = np.asarray(im.convert('RGB'))
        capped = component_budget(mask,colours,np.full_like(colours,220),.05)
        result['residual_fixture'] = dict(before_pixels=int((mask>0).sum()),before_components=label(mask)[1],
            after_pixels=int((capped>0).sum()),after_components=label(capped)[1])
    fake = defaultdict(list)
    for row in rows:
        if row['format']=='strip' and row['variant']=='augmented':
            fake[(row['doc'],row['class_id'])].append(row)
    used = set()
    x,y,train = [],[],[]
    occupancy = Counter()
    for real in sorted(reals,key=lambda r:(r['doc'],r['annotation_id'])):
        choices = [f for f in fake[(real['doc'],real['class_id'])] if f['file'] not in used]
        if not choices:
            continue
        f = choices[0]
        used.add(f['file'])
        arrays = []
        for row in (real,f):
            with Image.open(folder/row['file']) as im:
                arrays.append(np.asarray(im.convert('RGB')))
        h = min(144,*map(len,arrays))
        for target,a in zip((1,0),arrays):
            dark = a[:h].max(2)<80
            rail = max(dark[:,0].mean(),dark[:,-1].mean())
            x.append([rail]);y.append(target);train.append(real['split']=='train')
            if real['split']=='validation' and rail>.8:
                occupancy['real' if target else 'fake'] += 1
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    tr = np.array(train)
    x,y = np.array(x),np.array(y)
    if tr.any() and (~tr).any():
        model = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500)).fit(x[tr],y[tr])
        correct = int((model.predict(x[~tr])==y[~tr]).sum())
        result['critic_rail_probe'] = dict(correct=correct,n=int((~tr).sum()),accuracy=100*correct/int((~tr).sum()),occupancy=dict(occupancy))
    result['evaluation'] = read(folder/'eval.json',{}).get('shortcut_gate')
    result['passed'] = result['leaks']['passed'] and not any(result['integrity'].values()) and result['gates']['unsafe_rendered']==0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    parser.add_argument('--strict',action='store_true')
    args = parser.parse_args()
    result = audit(ROOT/args.version)
    print(json.dumps(result,indent=2,default=int))
    print(('PASS' if result['passed'] else 'FAIL'),'dataset audit',args.version)
    if args.strict and not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
