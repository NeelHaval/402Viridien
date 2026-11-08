"""Numbered contact sheets at one isotropic scale, with an authoritative index."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from PIL import Image, ImageDraw
from synth.common import DATA, read_json, write_json, local_file, relative, tile, font, short, pid_slug


def wrapped(draw, text, xy, width, size=18):
    x, y = xy
    line = ''
    for word in text.split():
        if draw.textlength(line + word, font=font(size)) > width and line:
            draw.text((x, y), line, fill='black', font=font(size)); y += size+5; line = ''
        line += word+' '
    draw.text((x, y), line, fill='black', font=font(size))
    return y + size + 5


def make_sheets(pattern, candidates, folder=DATA / 'sheets', max_width=1600):
    folder.mkdir(parents=True, exist_ok=True)
    candidates = candidates[:30]
    maxw = max((c['col_w'] for c in candidates), default=100)
    maxh = max((c['h'] for c in candidates), default=100)
    scale = min(2, 240/maxw, 320/maxh)
    cellw = max(245, math.ceil(maxw*scale)+24)
    cellh = math.ceil(maxh*scale)+90
    cols = min(6, max(1, (max_width-24)//cellw))
    # At most two pages, with a practical reviewer canvas height of ~1800px.
    rows = math.ceil(len(candidates)/cols)
    split = rows * cellh + 210 > 1800
    pages = 2 if split and len(candidates) > 1 else 1
    perpage = math.ceil(len(candidates)/pages)
    numbers, files = {}, []
    for page in range(pages):
        batch = candidates[page*perpage:(page+1)*perpage]
        width = min(max_width, max(700, cols*cellw+24))
        canvas = Image.new('RGB', (width, 230+max(1, math.ceil(len(batch)/cols))*cellh), 'white')
        d = ImageDraw.Draw(canvas)
        y = wrapped(d, pattern['label'], (16, 12), width-190, 26)
        y = wrapped(d, pattern['family'], (16, y), width-190, 19)
        y = wrapped(d, pattern['legend_type'], (16, y), width-190, 15)
        d.text((16, y+4), f"Target {pattern['n_target']} | {len(candidates)} candidates | scale {scale:.3f}x | page {page+1}/{pages}", font=font(16), fill='black')
        if pattern['swatches']:
            entry = pattern['swatches'][0]
            with Image.open(local_file(entry.get('native_file', entry['file']))) as sw:
                canvas.paste(tile(sw.convert('RGB'), 150, 140), (width-166, 12))
        if not batch:
            wrapped(d, 'No candidates: '+pattern.get('status', 'not mined'), (16, 200), width-32)
        filename = pid_slug(pattern['pid']) + (f'-{page+1}' if pages > 1 else '') + '.png'
        path = folder / filename
        files.append(relative(path))
        for i, c in enumerate(batch):
            number = page*perpage+i+1
            x, y = 12+(i%cols)*cellw, 200+(i//cols)*cellh
            d.text((x, y), str(number), font=font(30), fill='black')
            d.text((x+52, y+4), c['source'], font=font(15), fill='black')
            d.text((x, y+36), short(c['log']), font=font(13), fill='black')
            with Image.open(local_file(c['file'])) as crop:
                crop = crop.convert('RGB')
                size = (max(1, round(crop.width*scale)), max(1, round(crop.height*scale)))
                canvas.paste(crop.resize(size, Image.Resampling.NEAREST), (x, y+58))
            numbers[str(number)] = dict(c, number=number, sheet=relative(path), sheet_page=page+1,
                                        display_scale=scale, display_xy=[x, y+58], display_size=list(size))
        canvas.save(path)
    return dict(sheets=files, candidates=numbers)


def main():
    catalog = read_json(DATA / 'catalog.json')
    if not catalog: raise SystemExit('Run synth/patterns.py first')
    candidates = read_json(DATA / 'candidates.json', {})
    index = {p['pid']: make_sheets(p, candidates.get(p['pid'], [])) for p in catalog['patterns']}
    write_json(DATA / 'sheets' / 'index.json', index)
    print(f'{len(index)} patterns; {sum(len(v["sheets"]) for v in index.values())} sheets; '
          f'{sum(len(v["candidates"]) for v in index.values())} numbered candidates')


if __name__ == '__main__': main()
