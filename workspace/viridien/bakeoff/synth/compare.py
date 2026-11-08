"""Build a portable, offline blind comparison: python -B -m synth.compare.

Images, CSS and JavaScript are embedded in compare.html. The adjacent JSON
contains measured differences/provenance; compare.png is a PIL visual audit.
"""
import argparse
import base64
from collections import defaultdict
import hashlib
import html
from io import BytesIO
import json
import numpy as np
from PIL import Image, ImageDraw
from .common import HOME, read_json, write_json, local_file, font
from .texstats import load_stats, texture_stats, aggregate, boundary_stats
from .procgen import render_pattern, render_column, motif


def embedded(rgb):
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')


def image_tag(rgb, zoom=1, label=''):
    h, w = rgb.shape[:2]
    return f'<img src="{embedded(rgb)}" width="{w*zoom}" height="{h*zoom}" alt="{html.escape(label)}" loading="lazy">'


def read_source(source, height=80, offset=0):
    if source['kind'] == 'crop':
        with Image.open(local_file(source['file'])) as image:
            rgb = np.asarray(image.convert('RGB'))
        offset = min(offset, max(0, len(rgb)-height))
        return rgb[offset:offset+height].copy()
    from harness import io
    log = io.open_log(source['log'])
    try:
        top = min(source['top_y']+offset, max(source['top_y'], source['bottom_y']-height))
        return io.read_rgb(log, 'interpreted_lithology_column', top, min(top+height, source['bottom_y']))
    finally:
        log.close()


def choose_patterns(data, n_common=16):
    patterns = data['patterns']
    looks = read_json(HOME/'data/lookalikes.json', {})
    pair_candidates = []
    for pid, pairs in looks.items():
        for other in pairs:
            q = other['pid']
            if pid < q and pid in patterns and q in patterns:
                merit = sum(patterns[k]['source_kind'] == 'labelled_native' for k in (pid, q))
                penalty = sum(patterns[k]['swatch_colour_conflict'] for k in (pid, q))
                pair_candidates.append((-merit, penalty, pid, q))
    extra, pairs = [], []
    for _, _, pid, q in sorted(pair_candidates):
        if pid not in extra and q not in extra:
            extra.extend([pid, q])
            pairs.append([pid, q])
        if len(extra) >= 4:
            break
    def rank(pid):
        p = patterns[pid]
        labelled_rows = sum(s['bottom_y']-s['top_y'] for s in p['sources'] if s['kind'] == 'native')
        return (p['share'], p['source_kind'] == 'labelled_native', labelled_rows, p['pixels'])
    available = sorted((p for p in patterns if p not in extra), key=rank, reverse=True)
    common, families = [], set()
    for pid in available:
        family = patterns[pid]['family']
        if family not in families:
            common.append(pid)
            families.add(family)
    common.extend(pid for pid in available if pid not in common)
    return common[:n_common]+extra, extra, pairs


def stat_rows(real, generated, real_boundary, generated_boundary):
    keys = [('Dash median', 'horizontal_ink', 'median'), ('Dash p90', 'horizontal_ink', 'p90'),
            ('Gap median', 'horizontal_gap', 'median'), ('Row spacing', 'row_spacing', 'median'),
            ('Ink share', 'ink_share', None), ('Lab mean', 'lab_mean', None)]
    rows = []
    for title, key, field in keys:
        a, b = (real[key][field], generated[key][field]) if field else (real[key], generated[key])
        denom = max(float(np.linalg.norm(a)), .01)
        error = float(np.linalg.norm(np.array(b)-a))/denom
        rows.append(dict(stat=title, real=a, generated=b, relative_error=error, off=error > .2))
    if real_boundary and generated_boundary:
        a, b = float(np.median(real_boundary)), float(np.median(generated_boundary))
        error = abs(b-a)/max(a, 1)
        rows.append(dict(stat='Boundary width', real=a, generated=b, relative_error=error, off=error>.2))
    else:
        rows.append(dict(stat='Boundary width', real=None, generated=None, relative_error=None, off=False))
    return rows


def format_number(value):
    if value is None: return 'unresolved / unavailable'
    if isinstance(value, list): return ', '.join(f'{v:.1f}' for v in value)
    return f'{value:.3g}'


def boundary_chunks(pid, data, rng):
    from harness import io
    p = data['patterns'][pid]
    candidates = p['boundaries']
    fallback = False
    if not candidates:
        seen = {}
        for q in data['patterns'].values():
            if q['family'] == p['family']:
                for c in q['boundaries']:
                    seen[(c['log'], c['boundary_y'])] = c
        candidates = list(seen.values())
        fallback = True
    real, fake, provenance, real_width, fake_width = [], [], [], [], []
    if not candidates:
        return real, fake, provenance, real_width, fake_width, 'No labelled contact is available for this family.'
    order = rng.permutation(len(candidates))
    for i in range(3):
        c = candidates[int(order[i % len(order)])]
        log = io.open_log(c['log'])
        try:
            rgb = io.read_rgb(log, 'interpreted_lithology_column', c['top_y'], c['bottom_y'])
        finally:
            log.close()
        y = c['boundary_y']-c['top_y']
        gen = render_column(dict(width=rgb.shape[1], grid=False, beds=[
            dict(pid=c['left_pid'], height=y), dict(pid=c['right_pid'], height=len(rgb)-y)]), int(rng.integers(2**32)))
        real.append(rgb)
        fake.append(gen)
        provenance.append(c)
        r, g = boundary_stats(rgb, y), boundary_stats(gen, y)
        if r['profile_width'] is not None: real_width.append(r['profile_width'])
        if g['profile_width'] is not None: fake_width.append(g['profile_width'])
    note = 'Contacts from other patterns in this family.' if fallback else 'Contacts involving this pattern.'
    if len(candidates) < 3: note += f' Only {len(candidates)} distinct contacts; examples repeat.'
    return real, fake, provenance, real_width, fake_width, note


def build(output=HOME/'out/compare.html', seed=20261107, n_common=16):
    output = local_file(str(output)) if not hasattr(output, 'resolve') else output.resolve()
    output.relative_to(HOME)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = load_stats()
    chosen, extras, pairs = choose_patterns(data, n_common)
    rng = np.random.default_rng(seed)
    cards, trials, report, previews = [], [], [], []
    for number, pid in enumerate(chosen):
        p = data['patterns'][pid]
        sources = [s for s in p['sources'] if s['kind'] == ('native' if p['source_kind'] == 'labelled_native' else 'crop')]
        substantial = [s for s in sources if s.get('h', s['bottom_y']-s['top_y']) >= 40]
        if substantial: sources = substantial
        order = rng.permutation(len(sources))
        real, fake, provenance = [], [], []
        for i in range(4):
            s = sources[int(order[i % len(order)])]
            rgb = read_source(s, offset=i*13)
            matching = [v for v, origin in zip(p['variants'], p['variant_sources'])
                        if origin['log'] == s['log'] and origin.get('top_y') == s.get('top_y')
                        and origin.get('file') == s.get('file')]
            appearance = dict(p, variants=matching or p['variants'])
            generated = render_pattern(pid, rgb.shape[0], rgb.shape[1], int(rng.integers(2**32)), stats=appearance)
            real.append(rgb)
            fake.append(generated)
            provenance.append(s)
        rb, gb, contacts, rw, gw, note = boundary_chunks(pid, data, rng)
        measured_real, measured_fake = aggregate([texture_stats(im) for im in real]), aggregate([texture_stats(im) for im in fake])
        rows = stat_rows(measured_real, measured_fake, rw, gw)
        real_side = int(rng.integers(2))
        ab = [real[0], fake[0]] if real_side == 0 else [fake[0], real[0]]
        trial_id = hashlib.sha256((str(seed)+pid).encode()).hexdigest()[:16]
        trials.append(dict(id=trial_id, pid=pid, real=real_side))
        h = html.escape
        blind = ''.join(f'<button class="pick" data-trial="{number}" data-side="{j}" aria-label="Choose {chr(65+j)} as real"><b>{chr(65+j)}</b>{image_tag(im, label=f"Candidate {chr(65+j)}")}</button>' for j, im in enumerate(ab))
        grids = ''
        for zoom in (1, 3):
            columns = ''.join('<div><h4>'+label+'</h4><div class="strip">'+''.join(image_tag(im, zoom, label) for im in images)+'</div></div>' for label, images in [('Real', real), ('Generated', fake)])
            grids += f'<h3>{zoom}× native pixels</h3><div class="paired">{columns}</div>'
        borders = ''.join('<div><h4>'+label+'</h4><div class="strip">'+''.join(image_tag(im, 1, label+' contact') for im in images)+'</div></div>' for label, images in [('Real', rb), ('Generated', gb)])
        table = ''.join('<tr class="'+('off' if row['off'] else '')+'"><th scope="row">'+row['stat']+'</th><td>'+format_number(row['real'])+'</td><td>'+format_number(row['generated'])+'</td><td>'+('over 20%' if row['off'] else '—' if row['relative_error'] is None else f"{row['relative_error']*100:.0f}%")+'</td></tr>' for row in rows)
        with Image.open(local_file(p['swatch_file'])) as image:
            swatch = np.asarray(image.convert('RGB'))
        legend_style = motif(dict(p, **p['swatch_stats'], swatch_colour_conflict=False, source_kind='legend'))
        geometry_conflict = legend_style != motif(p)
        warning = '<p class="warning">Indexed crop colour or geometry conflicts with the legend. These crops have not been verified.</p>' if p['swatch_colour_conflict'] or geometry_conflict else ''
        badge = 'Look-alike pair' if pid in extras else 'Common-pattern selection'
        cards.append(f'''<article id="pattern-{number}"><p class="eyebrow">{number+1} / {len(chosen)} · {badge} · {h(p['family'])}</p>
<h2>{h(p['label'])}</h2><p class="pid">{h(pid)}</p>{warning}
<p>Choose the real image. Both images have identical native dimensions.</p><div class="blind">{blind}</div>
<p class="answer" id="answer-{number}" aria-live="polite">Unanswered</p>
<details><summary>Inspect samples, legend, contacts and statistics</summary><p>Source: {h(p['source_kind'])}. Four native chunks per side; no stretching. Generated chunks use the measured appearance of the matching source, with fresh procedural geometry/noise. Sources may repeat when fewer than four exist.</p>
<div class="legend">{image_tag(swatch, label='Legend swatch')}<p>Legend topology: {h(legend_style)}; rendered topology: {h(motif(p))}.<br>Legend row spacing {p['swatch_stats']['row_spacing']['median']:.1f}px; column {p['row_spacing']['median']:.1f}px.</p></div>{grids}
<h3>Bed contacts</h3><p>{h(note)}</p><div class="paired">{borders}</div>
<div class="table-scroll"><table><caption>Measured displayed chunks; lengths in native pixels. Highlighted rows exceed 20% relative error.</caption><thead><tr><th>Statistic</th><th>Real</th><th>Generated</th><th>Error</th></tr></thead><tbody>{table}</tbody></table></div>
<p>Boundary width is the local 10–90% texture profile width. Unresolved profiles are omitted. Contact rules may be present; patterns are clipped at the exact bed row.</p></details></article>''')
        report.append(dict(pid=pid, source_kind=p['source_kind'], swatch_conflict=p['swatch_colour_conflict'], geometry_conflict=geometry_conflict, real_sources=provenance,
                           contacts=contacts, boundary_note=note, rows=rows, motif=motif(p)))
        previews.append((p, real, fake, swatch, rb, gb))
    digest = hashlib.sha256((HOME/'data/texstats.json').read_bytes()).hexdigest()[:12]
    script = (HOME/'compare.js').read_text(encoding='utf-8').replace('__TRIALS__', json.dumps(trials)).replace('__VERSION__', f'{digest}-{seed}')
    css = (HOME/'compare.css').read_text(encoding='utf-8')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lithology blind comparison</title><style>{css}</style></head><body>
<header><h1>Lithology: real or generated?</h1><p>{len(chosen)} patterns across {len(data['families'])} families. Pick the real chunk in each pair; inspect details after making your choice.</p>
<p>Common selection covers families, then ranks by catalogue row share and available labelled rows. Four extra patterns come from recorded look-alike pairs. Unverified crop conflicts are flagged.</p>
<p id="score" role="status">0 correct / 0 answered</p><button id="export">Export results as JSON</button><button id="reset">Reset score</button><p id="storage" aria-live="polite"></p></header>
<main>{''.join(cards)}</main><footer>Seed {seed} · measurements {digest} · offline, embedded images · scores stay in this browser</footer><script>{script}</script></body></html>'''
    output.write_text(page, encoding='utf-8')
    write_json(output.with_suffix('.json'), dict(seed=seed, stats_digest=digest, lookalike_pairs=pairs, patterns=report))
    # Raster audit of every displayed sample at 1x and 3x, using the same arrays.
    canvas = Image.new('RGB', (1520, len(previews)*440), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (p, real, fake, swatch, rb, gb) in enumerate(previews):
        y = i*440
        draw.text((12, y+8), p['pid'], fill='black', font=font(16))
        draw.text((12, y+34), 'REAL', fill='black', font=font(14))
        draw.text((755, y+34), 'GENERATED', fill='black', font=font(14))
        for x, collection in ((12, real), (755, fake)):
            dx = x
            for rgb in collection:
                canvas.paste(Image.fromarray(rgb), (dx, y+58))
                dx += rgb.shape[1]+8
            zoom = Image.fromarray(collection[0]).resize((collection[0].shape[1]*3, collection[0].shape[0]*3), Image.Resampling.NEAREST)
            canvas.paste(zoom, (x, y+150))
        draw.line((0, y+435, 1520, y+435), fill='#cccccc')
    canvas.save(output.with_suffix('.png'))
    print(f'{len(chosen)} patterns; {len(extras)} look-alike extras; {sum(row["off"] for p in report for row in p["rows"])} statistic flags; {output}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=__import__('pathlib').Path, default=HOME/'out/compare.html')
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    build(args.output, args.seed)
