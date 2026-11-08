"""Validate the converted lane outputs without reading any evaluation truth elsewhere."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = 160_000_000


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(truth, unlabelled):
    manifest = read_csv(truth/'manifest.csv')
    items = json.loads((truth/'eval_set.json').read_text())
    assert len({r['id'] for r in manifest}) == len(manifest)
    ids = {r['id'] for r in manifest}
    assert {item['log'] for item in items} == ids
    total = 0
    for record in manifest:
        name = record['id']
        data = json.loads((truth/'bbox_annotation'/f'{name}.json').read_text())
        with Image.open(truth/'images'/f'{name}.tif') as im:
            assert im.size == (data['width'],data['height'])
            for x0,y0,x1,y1 in data['bounding_boxes'].values():
                assert 0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height
            assert im.mode == 'RGB'
            # Decode strips including a final strip, without a full-page RGB copy.
            for y in range(0, im.height, 4000):
                strip = im.crop((0,y,im.width,min(y+4000,im.height)))
                strip.load()
                strip.close()
        box = data['bounding_boxes']['interpreted_lithology_column']
        rows = read_csv(truth/'truth'/f'{name}.csv')
        assert len(rows) == int(record['truth_intervals'])
        previous = box[1]
        for row in rows:
            top,bottom = int(row['top_y']),int(row['bottom_y'])
            assert previous <= top < bottom <= box[3]
            assert float(row['top_depth']) < float(row['bottom_depth'])
            assert row['lithology'] not in ('','(blank)')
            previous = bottom
        for item in [i for i in items if i['log']==name]:
            assert box[1] <= item['y0'] < item['y1'] <= box[3]
            assert any(int(r['top_y']) < item['y1'] and int(r['bottom_y']) > item['y0'] for r in rows)
        total += len(rows)
        assert record['source_url'] and record['licence_note']
    ul = read_csv(unlabelled/'manifest.csv')
    assert len({r['id'] for r in ul}) == len(ul)
    assert {p.stem for p in (unlabelled/'images').glob('*.tif')} == {r['id'] for r in ul}
    drafts = 0
    for record in ul:
        name = record['id']
        with Image.open(unlabelled/'images'/f'{name}.tif') as im:
            assert im.width*im.height <= 32_020_000
            for y in range(0,im.height,4000):
                strip = im.crop((0,y,im.width,min(y+4000,im.height)))
                strip.load()
                strip.close()
            draft = unlabelled/'bbox_annotation'/f'{name}.json'
            if draft.exists():
                annotation = json.loads(draft.read_text())
                assert annotation['draft'] is True
                for x0,y0,x1,y1 in annotation['bounding_boxes'].values():
                    assert 0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height
                drafts += 1
        assert record['source_url'] and record['has_legend'] == 'True'
    result = dict(truth_logs=len(manifest), truth_windows=len(items), truth_intervals=total,
                  unlabelled_pages=len(ul), unlabelled_by_family=dict(Counter(r['layout_family_guess'] for r in ul)),
                  draft_boxes=drafts, status='passed')
    (truth/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    if not sys.flags.isolated:
        raise SystemExit('Use python -I')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--unlabelled', type=Path, required=True)
    args = parser.parse_args()
    validate(args.truth,args.unlabelled)
