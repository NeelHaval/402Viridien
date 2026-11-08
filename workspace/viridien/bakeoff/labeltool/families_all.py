"""Split the whole dataset's lithology columns into families of look-alike patterns, for quick review.

1. Sample chunks down every column (CHUNK rows every STEP rows), skipping blank paper.
2. Describe each chunk by fill colour (coarse colour histogram) and ink texture (density, run lengths, row period).
3. Hard split by main fill colour, then k-means on texture inside each colour.
4. Per family: a contact sheet of up to 12 chunks from as many logs as possible, plus that family's
   legend look-alikes (swatches whose colour matches), for the owner and an LLM to check.

    python -X utf8 -B labeltool/families_all.py   -> analysis/families_all/{index.html, families.json, f###.png}
"""
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2, suggest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'families_all'
CHUNK, STEP = 48, 400


def runs(mask):
    """Mean run length of True along the last axis."""
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    starts, ends = np.where(d == 1), np.where(d == -1)
    n = len(starts[0])
    return (ends[1] - starts[1]).mean() if n else 0.0


def describe(rgb):
    a = rgb[:, 4:-4].astype(int)
    s = a.sum(2)
    ink = s < 200
    paper = s > 700
    fill = ~ink & ~paper
    q = (a // 64)  # 4 levels per channel -> 64 bins
    codes = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    hist = np.bincount(codes[fill], minlength=64).astype(float)
    hist = hist / hist.sum() if hist.sum() else hist
    main = int(hist.argmax()) if hist.sum() and fill.mean() > 0.15 else -1  # -1 = black/white pattern
    prof = ink.mean(1) - ink.mean()
    spec = np.abs(np.fft.rfft(prof))[1:]
    period = len(prof) / (spec.argmax() + 1) if spec.size and spec.max() > 0 else 0
    tex = np.array([ink.mean() * 4, runs(ink) / 10, runs(ink.T) / 10, runs(~ink & ~paper | paper) / 40,
                    min(period, 48) / 24, ink.mean(0).std() * 4, ink.mean(1).std() * 4])
    return main, np.concatenate([hist * 2, tex])


def row_codes(rgb):
    """Per-row dominant fill colour code (same 64 bins as describe), -1 for black/white rows,
    smoothed with a 9-row majority so dots and dashes do not break a bed into pieces."""
    a = rgb[:, 4:-4].astype(int)
    s = a.sum(2)
    fill = (s >= 200) & (s <= 700)
    q = a // 64
    codes = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    n = len(a)
    raw = np.full(n, -1)
    for i in range(n):
        if fill[i].mean() > 0.15:
            raw[i] = np.bincount(codes[i][fill[i]], minlength=64).argmax()
    sm = raw.copy()
    for i in range(n):
        w = raw[max(0, i - 4):i + 5]
        v, c = np.unique(w, return_counts=True)
        sm[i] = v[c.argmax()]
    return sm


def homogeneous(rgb):
    """True when the top and bottom halves of a chunk look like the same pattern (no bed boundary inside)."""
    a = rgb[:, 4:-4].astype(int)
    h = len(a) // 2
    halves = []
    for part in (a[:h], a[h:]):
        s = part.sum(2)
        ink = s < 200
        fill = (s >= 200) & (s <= 700)
        q = part // 64
        codes = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        hist = np.bincount(codes[fill], minlength=64).astype(float)
        hist = hist / hist.sum() if hist.sum() else hist
        halves.append((ink.mean(), runs(ink), (s > 700).mean(), hist, fill.mean()))
    (i0, r0, p0, h0, f0), (i1, r1, p1, h1, f1) = halves
    if abs(i0 - i1) > 0.07 or abs(p0 - p1) > 0.15 or abs(f0 - f1) > 0.15:
        return False
    if max(r0, r1) > 1 and max(r0, r1) / max(0.5, min(r0, r1)) > 1.6:
        return False
    return np.abs(h0 - h1).sum() < 0.6


def sample(name):
    """Chunks placed only inside runs of one fill colour, then checked half-against-half, so a chunk
    never straddles two beds."""
    ctx = jev2.Context.get(name)
    box = ctx['log'].boxes['interpreted_lithology_column']
    y0, y1 = int(box[1]) + 200, int(box[3])
    with ctx['lock']:
        col = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=y0, y1=y1)[..., :3]
    if col.shape[1] < 20:
        return []
    codes = row_codes(col)
    cut = np.flatnonzero(np.diff(codes)) + 1
    out, stats = [], Counter()
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(codes)]):
        a, b = a + 4, b - 4  # keep off the boundary itself
        if b - a < CHUNK:
            stats['short_run'] += 1
            continue
        for y in range(a + ((b - a - CHUNK) % STEP) // 2, b - CHUNK + 1, STEP):
            rgb = col[y:y + CHUNK]
            s = rgb[:, 4:-4].astype(int).sum(2)
            if (s > 700).mean() > 0.9:
                stats['blank'] += 1
                continue
            if not homogeneous(rgb):
                stats['mixed'] += 1
                continue
            main, f = describe(rgb)
            out.append(dict(log=name, y=y0 + y, main=main, f=f))
    sample.stats.update(stats)
    return out


sample.stats = Counter()


def legend_swatches(name):
    out = []
    for t in jev2.Context.get(name)['tiles']:
        main, _ = describe(np.pad(t['swatch'][..., :3], ((0, 0), (4, 4), (0, 0)), constant_values=255)) if t['swatch'].shape[1] > 2 else (-1, None)
        out.append(dict(label=t['label'], main=main, swatch=t['swatch'][..., :3]))
    return out


def main():
    from sklearn.cluster import KMeans
    names = sorted(p.stem for p in (ROOT / 'labeltool' / 'data' / 'logs').glob('*.json'))
    chunks, legends = [], {}
    for n in names:
        try:
            c = sample(n)
            legends[n] = legend_swatches(n)
        except Exception as e:
            print('skip', n[:50], type(e).__name__, str(e)[:80])
            continue
        chunks += c
        print(f'{n[:60]:60s} {len(c)} chunks')
    by_colour = defaultdict(list)
    for i, c in enumerate(chunks):
        by_colour[c['main']].append(i)
    fams = []
    for colour, idx in by_colour.items():
        if len(idx) < 6:
            continue
        X = np.stack([chunks[i]['f'] for i in idx])
        k = max(1, min(16, len(idx) // 40))
        lab = KMeans(k, n_init=4, random_state=0).fit_predict(X) if k > 1 else np.zeros(len(idx), int)
        for j in range(k):
            m = [idx[t] for t in np.flatnonzero(lab == j)]
            if len(m) >= 6:
                fams.append(dict(colour=colour, members=m))
    fams.sort(key=lambda f: -len(f['members']))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fi, fam in enumerate(fams):
        m = fam['members']
        X = np.stack([chunks[i]['f'] for i in m])
        centre = X.mean(0)
        order = np.argsort(((X - centre) ** 2).sum(1))
        picked, seen = [], Counter()
        for rounds in range(3):  # spread across logs: one per log first
            for t in order:
                i = m[t]
                if seen[chunks[i]['log']] == rounds and i not in picked:
                    picked.append(i); seen[chunks[i]['log']] += 1
                if len(picked) >= 12:
                    break
            if len(picked) >= 12:
                break
        tiles = []
        for i in picked:
            c = chunks[i]
            ctx = jev2.Context.get(c['log'])
            with ctx['lock']:
                rgb = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=c['y'], y1=c['y'] + CHUNK)
            im = Image.fromarray(rgb[:, :, :3])
            k = max(1, min(3, 120 // max(1, im.width)))
            tiles.append(im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST))
        W = sum(t.width for t in tiles) + 6 * len(tiles)
        H = max(t.height for t in tiles)
        sheet = Image.new('RGB', (W, H + 14), 'white')
        d = ImageDraw.Draw(sheet)
        x = 0
        for n_, t in enumerate(tiles):
            sheet.paste(t, (x, 14)); d.text((x + 2, 1), str(n_ + 1), fill='black'); x += t.width + 6
        sheet.save(OUT / f'f{fi:03d}.png')
        logs = Counter(chunks[i]['log'] for i in m)
        # Legend look-alikes: entries of member logs whose main colour matches this family's colour.
        cands = Counter()
        for lg in logs:
            for s in legends.get(lg, []):
                if s['main'] == fam['colour']:
                    cands[s['label']] += logs[lg]
        rows.append(dict(id=fi, size=len(m), logs=len(logs), colour=fam['colour'], sheet=f'f{fi:03d}.png',
                         chunks=[dict(log=chunks[i]['log'], top_y=chunks[i]['y'], bottom_y=chunks[i]['y'] + CHUNK) for i in picked],
                         legend_lookalikes=[l for l, _ in cands.most_common(8)], top_logs=[l for l, _ in logs.most_common(3)],
                         members=[[chunks[m[t]]['log'], chunks[m[t]]['y']] for t in order]))
    (OUT / 'families.json').write_text(json.dumps(dict(chunks=len(chunks), families=rows), indent=1), encoding='utf-8')
    e = html.escape
    page = [f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Pattern families</title>
<style>body{{font:14px/1.4 system-ui,sans-serif;margin:12px;color:#17212b}}h1{{font-size:20px}}.f{{border:1px solid #c5ccd3;border-radius:6px;padding:8px;margin:0 0 10px}}
.f img{{max-width:100%;height:auto;image-rendering:pixelated;display:block}}.m{{font-size:12px;color:#55616b}}</style>
<h1>{len(rows)} pattern families from {len(chunks)} chunks of {len(names)} logs</h1>
<p class=m>Each row: up to 12 chunks from different logs that the clustering put together. Number = chunk. Should all look like one pattern.</p>"""]
    for r in rows:
        page.append(f"<div class=f><b>Family {r['id']}</b> · {r['size']} chunks in {r['logs']} logs · legend entries with this colour: {e(', '.join(r['legend_lookalikes']) or 'black/white or none')}"
                    f"<img src='{r['sheet']}' alt='family {r['id']}'></div>")
    (OUT / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(len(chunks), "chunks ->", len(rows), "families; rejected:", dict(sample.stats))


if __name__ == '__main__':
    main()
