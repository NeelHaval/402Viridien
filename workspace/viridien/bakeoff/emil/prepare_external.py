"""Convert trusted extraction metadata + exact CSV truth; no PDF code is loaded."""
import csv
import json
import math
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    external = WORK / 'external_tests'
    audit = json.loads((external / 'audit.json').read_text())['logs']
    windows = json.loads((external / 'eval_set.json').read_text())
    view = HERE / 'external_view'
    index = []
    for entry in audit:
        name = entry['id']
        folder = WORK / entry['extraction']
        meta = json.loads((folder / 'meta.json').read_text())
        ann = json.loads((external / 'bbox_annotation' / f'{name}.json').read_text())
        rocks = [e for e in meta['legend_entries'] if e['kind'] == 'rock']
        assert all((folder / e['swatch_png']).exists() for e in rocks)
        scale = meta['render']['px_per_pt']
        boxes = [e[k] for e in rocks for k in ('bbox_pt', 'label_bbox_pt')]
        # Exact crop origin used in build_testset.legend_crop, without reopening PDFs.
        left = max(0, math.floor((min(b[0] for b in boxes)-3)*scale))
        top = max(0, math.floor((min(b[1] for b in boxes)-18)*scale))
        dx, dy = (10-left, -top) if entry['y_shift'] else (0, 0)
        legend = []
        for i, e in enumerate(rocks):
            b = e['bbox_pt']
            legend.append(dict(id=i, label_printed=e['label'], box=[
                math.floor(b[0]*scale)+dx, math.floor(b[1]*scale)+dy,
                math.ceil(b[2]*scale)+dx, math.ceil(b[3]*scale)+dy]))
        ids = {e['label_printed']: e['id'] for e in legend}
        with (external / 'truth' / f'{name}.csv').open(newline='', encoding='utf-8') as f:
            truth = list(csv.DictReader(f))
        layers = []
        for win in [w for w in windows if w['log'] == name]:
            for row in truth:
                a, b = max(win['y0'], int(row['top_y'])), min(win['y1'], int(row['bottom_y']))
                if b > a:
                    label = row['lithology']
                    assert label in ids, (name, label)
                    layers.append(dict(top_px=a, bottom_px=b, composition=[dict(legend_id=ids[label], type=label, proportion=1.0)]))
        labels = dict(image_name=f'{name}.tif', legend=legend, layers=sorted(layers, key=lambda r:r['top_px']),
                      provenance=dict(extraction=entry['extraction'], y_shift=entry['y_shift'], windows=[w for w in windows if w['log']==name]))
        dump(HERE / 'external_labels' / f'{name}.json', labels)
        dump(view / 'test' / f'{name}.json', labels)
        dump(view / 'dataset/bbox_annotation' / f'{name}.json', ann)
        dst = view / 'dataset/images' / f'{name}.tif'
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try: os.link(external / 'images' / f'{name}.tif', dst)
            except OSError: shutil.copyfile(external / 'images' / f'{name}.tif', dst)
        index.append(dict(log=name, n_layers=len(layers), n_legend=len(legend), extraction=str(folder.relative_to(WORK)), box_shift=[dx,dy]))
    dump(view / 'splits.json', dict(train=[], test=[x['log'] for x in index]))
    dump(HERE / 'external_conversion.json', index)
    print(f'Converted {len(index)} pages and {len(windows)} windows')


if __name__ == '__main__': main()
