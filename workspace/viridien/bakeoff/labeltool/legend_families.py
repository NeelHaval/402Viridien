"""Legend-only clustering: every distinct legend picture in the 51 logs, grouped by drawn shape (JEV, names
hidden; analysis/pattern_families.json) and then by fill colour. Shows the names each log gave a picture,
so look-alikes with different names stand out.

    python -X utf8 -B labeltool/legend_families.py  -> analysis/legend_families/{groups.json, g###.png, s###.png}
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'legend_families'
NAMES = {0: 'black', 1: 'dark'}


def colour_key(rgb):
    r, g, b = (int(v) for v in rgb)
    if max(r, g, b) - min(r, g, b) < 40:
        return 'white' if r > 200 else 'grey' if r > 90 else 'black'
    q = tuple(v // 86 for v in (r, g, b))  # 3 levels per channel
    return 'rgb%d%d%d' % q


def swatch(log, sid):
    for e in jev2.Context.get(log)['tiles']:
        if e['id'] == sid:
            return e['swatch'][..., :3]
    return None


def main():
    data = json.loads((ROOT / 'analysis' / 'pattern_families.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for rep in data['reps']:
        m0 = rep['members'][0]
        sw = swatch(m0['log'], m0['id'])
        if sw is None or not sw.size:
            continue
        im = Image.fromarray(sw)
        k = max(1, min(3, 90 // max(1, im.height)))
        im = im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST)
        f = f"s{rep['n']:03d}.png"
        im.save(OUT / f)
        names = sorted({m['label'] for m in rep['members']})
        rgb = [int(v) for v in np.asarray(sw).reshape(-1, 3).mean(0)]
        groups[(rep['pattern'], colour_key(rep['desc'].get('rgb') or rgb))].append(dict(
            n=rep['n'], file=f, names=names, logs=len({m['log'] for m in rep['members']}),
            members=[[m['log'], m['id'], m['label']] for m in rep['members']], jev_conf=rep.get('conf'),
            ink=rep['desc'].get('ink', 0), w=im.width, h=im.height))
    out = []
    for gi, ((pattern, colour), items) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        items.sort(key=lambda x: x['ink'])
        for i, it in enumerate(items, 1):
            it['k'] = i
        distinct = sorted({n for it in items for n in it['names']})
        # Sheet for the LLM: numbered swatches with their names underneath.
        W, H = 1100, 0
        cells, x, y, rowh = [], 0, 0, 0
        for it in items:
            w, h = it['w'], it['h'] + 30
            if x + w > W:
                x, y, rowh = 0, y + rowh + 8, 0
            cells.append((it, x, y)); x += max(w, 110) + 8; rowh = max(rowh, h)
        sheet = Image.new('RGB', (W, y + rowh + 8), 'white')
        d = ImageDraw.Draw(sheet)
        for it, cx, cy in cells:
            sheet.paste(Image.open(OUT / it['file']), (cx, cy + 12))
            d.text((cx, cy), str(it['k']), fill=(157, 48, 48))
            d.text((cx, cy + 14 + it['h']), ', '.join(it['names'])[:30], fill='black')
        sheet.save(OUT / f'g{gi:03d}.png')
        out.append(dict(id=gi, pattern=pattern, colour=colour, sheet=f'g{gi:03d}.png', items=items,
                        names=distinct, name_conflict=len(distinct) > 1))
    (OUT / 'groups.json').write_text(json.dumps(dict(groups=out, pictures=sum(len(g['items']) for g in out)), indent=1), encoding='utf-8')
    print(len(out), 'groups,', sum(len(g['items']) for g in out), 'pictures,', sum(g['name_conflict'] for g in out), 'with more than one name')


if __name__ == '__main__':
    main()
