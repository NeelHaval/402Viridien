"""Small deterministic checks; no dataset truth or model calls."""
from pathlib import Path
import numpy as np
from PIL import Image
from llm_label import (BASE, boundary_signal, depth, encode_rows,
                       normalize_intervals, parse_json, save_csv, tile_image, tiles)

for height in [2400, 2608, 6791]:
    spans = tiles(4000, 4000 + height)
    assert spans[0][0] == 4000 and spans[-1][1] == 4000 + height
    assert all(400 <= b-a <= 600 for a,b in spans)
    assert all(left[1]-right[0] == 60 for left,right in zip(spans,spans[1:]))
rgb = np.full((580, 80, 3), 245, np.uint8)
rgb[290:] = (90,110,140)
signal = boundary_signal(rgb)
assert abs(int(signal.argmax())-290) <= 1
vocab = {'entries': [{'label': 'Clay'}, {'label': 'Sand'}]}
rows = normalize_intervals({'intervals': [
    {'top_y': 4000, 'bottom_y': 4280, 'label': 'clay', 'confidence': .8},
    {'top_y': 4280, 'bottom_y': 4580, 'label': 'Sand', 'confidence': .9}]},
    4000,4580,vocab,signal,4000)
assert rows[0]['bottom_y'] == 4290 == rows[1]['top_y']
calib = depth.Calib(.1,200,4,1,'synthetic')
labels = np.array(['Clay']*290+['Sand']*290,dtype=object)
agreed = np.ones(580,dtype=bool); agreed[200:250] = False
encoded = encode_rows(labels,np.ones(580),4000,calib,agreed)
assert len(encoded) == 4 and encoded[-1]['bottom_y'] == 4580
assert parse_json('```json\n{"entries":[]}\n```') == {'entries': []}
tmp=BASE/'out/_checks'
tmp.mkdir(parents=True,exist_ok=True)
panel=Image.new('RGB',(800,850),'white')
tile_image(rgb,4000,4580,4000,panel,tmp/'tile.png',1)
im=Image.open(tmp/'tile.png')
assert all(512<=s<=1600 for s in im.size)
save_csv(tmp/'consensus.csv',encoded,True)
assert ',false' in (tmp/'consensus.csv').read_text()
print('PASS: coverage, exact 60-row overlaps, image dimensions, boundary snapping, agreement runs, CSV and JSON.')
