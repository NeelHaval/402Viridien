"""Validate and freeze explicitly reviewed geometry; no model results are read."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
FORBIDDEN=('barlow2-14','marcotte1-12','red-brook-1','emmaward1','flat-bay-3','irvin1-19','forsythb2-17')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('lane',type=Path)
    p.add_argument('--freeze',action='store_true');a=p.parse_args()
    dataset=(a.root/'external_unlabelled_test').resolve()
    items=json.loads((dataset/'eval_set.json').read_text(encoding='utf-8'))
    with (dataset/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        manifest=list(csv.DictReader(f))
    names={r['log'] for r in items};assert len(names)==len(items)==len(manifest)
    assert names=={r['id'] for r in manifest}
    assert not any(t in json.dumps([manifest,items]).casefold() for t in FORBIDDEN)
    source={r['id']:r for r in manifest};hashes={};annotations={}
    for item in items:
        name=item['log'];ann=json.loads((dataset/'bbox_annotation'/f'{name}.json').read_text(encoding='utf-8'))
        assert ann['truth_status']=='absent' and ann['geometry_only']
        boxes=ann['bounding_boxes'];assert {'depth_column','interpreted_lithology_column','lithology_legend'}<=set(boxes)
        path=dataset/'images'/ann['image_name']
        with Image.open(path) as im:
            assert im.size==(ann['width'],ann['height'])
            for box in boxes.values():
                x0,y0,x1,y1=box;assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height,(name,box)
            im.verify()
        assert boxes['interpreted_lithology_column'][1]<=item['y0']<item['y1']<=boxes['interpreted_lithology_column'][3]
        assert boxes['depth_column'][1]<=item['y0']<item['y1']<=boxes['depth_column'][3]
        h=digest(path);assert h==source[name]['image_sha256']
        assert h not in hashes,('Exact duplicate',name,hashes.get(h))
        hashes[h]=name;annotations[name]=ann
    exemplars=json.loads((a.lane/'feature_exemplars.json').read_text(encoding='utf-8'))
    assert not {r['log'] for r in exemplars['exemplars']}&set(exemplars['excluded_eval_logs'])
    scores=json.loads((a.lane/'page_scores.json').read_text(encoding='utf-8'))
    score_ids={r['id'] for r in scores}
    assert names<=score_ids
    reviewed=[]
    if a.freeze:
        review=json.loads((a.lane/'visual_review.json').read_text(encoding='utf-8'))
        assert names==set(review['approved_ids']), 'Review must name exactly the selected pages'
        for sheet,h in review['sheets'].items():
            assert digest(dataset/'review_sheets'/sheet)==h,('Review sheet changed',sheet)
        reviewed=list(review['sheets'])
        # Remove only stale files generated in these three owned directories.
        known={r['id'] for r in scores};removed=0
        for folder in ('images','bbox_annotation','thumbnails'):
            parent=(dataset/folder).resolve()
            assert parent.parent==dataset
            for path in parent.iterdir():
                if path.is_file() and path.stem in known-names:
                    assert path.resolve().parent==parent
                    path.unlink();removed+=1
        for name,ann in annotations.items():
            ann['visual_review']='checked against source, depth, column and legend crops'
            (dataset/'bbox_annotation'/f'{name}.json').write_text(json.dumps(ann,indent=2),encoding='utf-8')
        for r in manifest:r['visual_review']='checked'
        with (dataset/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
        contract=json.loads((dataset/'curation.json').read_text(encoding='utf-8'))
        contract.update(status='FROZEN TEST',frozen_utc=datetime.now(timezone.utc).isoformat(),
                        visual_review_sheets=len(reviewed),stale_generated_files_removed=removed,
                        eval_set_sha256=digest(dataset/'eval_set.json'),manifest_sha256=digest(dataset/'manifest.csv'))
        (dataset/'curation.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    result=dict(pages=len(items),windows=len(items),families=dict(Counter(r['family'] for r in items)),
                wells=len({r['well'] for r in items}),source_urls=len({r['source_url'] for r in manifest}),
                scored_pages=len(scores),non_eval_exemplars=len(exemplars['exemplars']),
                annotations_checked=len(annotations),exact_duplicates=0,forbidden_wells=0,
                truth_files=0,visual_review_sheets=len(reviewed),
                bytes=sum(p.stat().st_size for p in (dataset/'images').glob('*.tif')))
    (a.lane/'set_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SET CHECK: '+json.dumps(result),flush=True)


if __name__=='__main__':main()
