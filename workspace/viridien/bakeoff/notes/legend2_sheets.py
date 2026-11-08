"""Numbered contact sheet per v2 group (pictures at one height, names underneath) for LLM checkers."""
import json
from PIL import Image, ImageDraw
D = 'analysis/legend_families2/'
G = json.load(open(D + 'groups.json', encoding='utf-8'))['groups']
for g in G:
    cells, x, y, W = [], 0, 0, 1200
    for it in g['items']:
        im = Image.open(D + it['file']); w = min(im.width, 140)
        if x + max(w, 120) > W:
            x, y = 0, y + 48 + 34
        cells.append((it, im.crop((0, 0, w, 48)), x, y)); x += max(w, 120) + 8
    s = Image.new('RGB', (W, y + 48 + 34), 'white'); d = ImageDraw.Draw(s)
    for it, im, cx, cy in cells:
        d.text((cx, cy), str(it['k']), fill=(157, 48, 48)); s.paste(im, (cx, cy + 12))
        d.text((cx, cy + 62), ', '.join(it['names'])[:24], fill='black')
    s.save(D + f"g{g['id']:03d}.png"); g['sheet'] = f"g{g['id']:03d}.png"
json.dump(dict(json.load(open(D + 'groups.json', encoding='utf-8')), groups=G), open(D + 'groups.json', 'w', encoding='utf-8'), indent=1)
print(len(G), 'sheets')
