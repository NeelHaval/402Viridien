"""Conservative legend/column verification. Run explicitly to refresh all reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import learn, paths, render, score, selection
from generator._texture import estimate_period, trim_frame
from .swatch import learn_swatch, remove_frame

VERSION = 5
TIERS = ('human', 'verified', 'borrowed', 'swatch', 'ambiguous', 'no', 'not-lithology')


def signature(doc, groups):
    approved = [(lid, learn.signature(exs)) for (did, lid), exs in groups.items() if did == doc['id']]
    prefill_hash=hashlib.sha256(paths.PREFILL.read_bytes()).hexdigest() if paths.PREFILL.exists() else None
    payload = [VERSION, doc['legend'], doc['boxes']['interpreted_lithology_column'], sorted(approved),prefill_hash]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def swatch(rgb):
    return learn_swatch(rgb)


def choose_tile(crops, doc, exs, model, data, swdata, risk):
    """Reserve depth intervals before selection; require foreground fidelity."""
    train, _ = learn.reserve_evidence(exs)
    if swdata is not None and not risk:
        risk = not all(score.fidelity(learn.clean(crops.get(doc,e['box']))[0],swdata)['acceptable']
                       for e in train)
    probe=data
    if data is not None and data['kind']=='quilt' and data.get('strip_origins'):
        # selection.crop_scores asks for a local viewport; its API is outside
        # this module's ownership. Translate a temporary probe only.
        origin=data['strip_origins'][0]
        probe=dict(data,strip_origins=[x-origin for x in data['strip_origins']],x_origin=0)
    selected,chosen=selection.choose(crops,doc,train,model,probe,swdata,risk)
    return selected,(data if chosen is probe else chosen)


def name_key(name):
    key = ' '.join(re.findall(r'[a-z0-9]+', name.casefold()))
    return {'sand and sandstone': 'sandstone', 'dolostone': 'dolomite'}.get(key, key)


def not_lithology(label, rgb):
    name = name_key(label)
    if re.search(r'\b(shows?|fluorescence|cuts?|stains?|fossils?|accessor(?:y|ies))\b', name):
        return 'Name describes a show, fossil, or accessory symbol.'
    if re.fullmatch(r'(oil|gas)( good| moderate| poor)?|dry|salted water', name):
        return 'Name describes a fluid/show rather than lithology.'
    if re.fullmatch(r'(pyrite|chert nodules?)( accessory| accessories)?', name):
        return 'Named accessory (bulk chert lithologies are retained).'
    white = (rgb.min(2) >= 240)
    yy, xx = np.where(~white)
    if white.mean() >= .97 and len(yy) >= 2:
        if np.ptp(yy) < rgb.shape[0] * .55 and np.ptp(xx) < rgb.shape[1] * .55:
            return 'At least 97% white with one small central isolated symbol.'
    return None


def pixel_match(rgb, data, bg=None, palette=None, minimum=0.0, distance=45):
    """Exact best cyclic RGB-distance match, with a background gate.

    Seed the incumbent with FFT alignment, then eliminate phases only when
    their mismatch count cannot beat it. Every native crop pixel contributes.
    With a minimum, scores below that threshold may return zero (twin test only).
    """
    data['_last_alignment'] = None
    if not rgb.size or np.linalg.norm((learn.background(rgb) if bg is None else bg) - data['bg']) >= 45:
        return 0.0
    tile = data['tile'].astype(np.float32)
    py, px = tile.shape[:2]
    h, w = rgb.shape[:2]
    ys, xs = np.indices((h, w))
    yy, xx = ys.ravel() % py, xs.ravel() % px
    actual = rgb.reshape(-1, 3).astype(np.float32)
    colours, inverse = palette if palette is not None else np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    if len(colours) <= 64:
        # Scanned TIFFs usually have a small palette. One circular correlation
        # per observed colour counts the exact distance predicate for ALL phases.
        # This avoids a phases x pixels search on non-repeating swatches.
        hits = np.zeros((py, px), np.float64)
        positions = yy * px + xx
        spectra = data.setdefault('_colour_fft_' + str(distance), {})
        for i, colour in enumerate(colours):
            key = tuple(map(int, colour))
            if key not in spectra:
                diff = tile - colour.astype(np.float32)
                compatible = (np.einsum('ijk,ijk->ij', diff, diff) < distance ** 2).astype(float)
                spectra[key] = np.fft.rfft2(compatible) if compatible.any() else None
            if spectra[key] is None:
                continue
            folded = np.bincount(positions[inverse == i], minlength=py * px).reshape(py, px)
            hits += np.fft.irfft2(np.conj(np.fft.rfft2(folded)) * spectra[key], s=(py, px))
        data['_last_alignment'] = tuple(map(int,np.unravel_index(hits.argmax(),hits.shape)))
        return float(np.rint(hits.max()) / len(actual))
    # Fold the crop to seed a good phase even when it is smaller than the tile.
    sums = np.zeros((py * px, 3), np.float64)
    counts = np.bincount(yy * px + xx, minlength=py * px)
    np.add.at(sums, yy * px + xx, actual)
    folded = (sums / np.maximum(1, counts[:, None])).reshape(py, px, 3)
    dy, dx, _ = learn._align(folded, tile)
    sy, sx = -dy, -dx
    diff = tile[(yy + sy) % py, (xx + sx) % px] - actual
    best = int((np.einsum('ij,ij->i', diff, diff) < distance ** 2).sum())
    if best == len(actual):
        return 1.0
    best = max(best, int(np.ceil(minimum * len(actual))) - 1)
    ay, ax = np.indices((py, px))
    ay, ax = ay.ravel(), ax.ravel()
    # For noisy colour scans, correlate conservative RGB cubes first. A tile
    # pixel can match a cube only if its distance to that cube is <45. These
    # counts are upper bounds, so discarding a phase cannot hide a better match.
    if minimum < .90:
        quantized = (rgb.reshape(-1, 3) // 32).astype(int)
        codes = quantized[:, 0] * 64 + quantized[:, 1] * 8 + quantized[:, 2]
        bounds = np.zeros((py, px), np.float64)
        spectra = data.setdefault('_bounds_fft_' + str(distance), {})
        for code in np.unique(codes):
            key = int(code)
            if key not in spectra:
                low = np.array([key // 64, (key // 8) % 8, key % 8]) * 32
                gap = np.maximum(np.maximum(low - tile, tile - (low + 31)), 0)
                mask = (np.einsum('ijk,ijk->ij', gap, gap) < distance ** 2).astype(float)
                spectra[key] = np.fft.rfft2(mask)
            folded = np.bincount((yy * px + xx)[codes == code], minlength=py * px).reshape(py, px)
            bounds += np.fft.irfft2(np.conj(np.fft.rfft2(folded)) * spectra[key], s=(py, px))
        possible = np.rint(bounds).ravel() > best
        ay, ax = ay[possible], ax[possible]
    misses = np.zeros(len(ay), np.int32)
    # A fixed shuffle encounters ink and background early, without subsampling.
    order = np.random.default_rng(0).permutation(len(actual))
    for start in range(0, len(order), 64):
        ids = order[start:start + 64]
        diff = tile[(ay[:, None] + yy[ids]) % py, (ax[:, None] + xx[ids]) % px] - actual[ids]
        misses += (np.einsum('ijk,ijk->ij', diff, diff) >= distance ** 2).sum(1).astype(np.int32)
        keep = misses < len(actual) - best
        ay, ax, misses = ay[keep], ax[keep], misses[keep]
        if not len(ay):
            break
    if len(misses):
        best = max(best, len(actual) - int(misses.min()))
    result = best / len(actual)
    return result if result >= minimum else 0.0


def symbol_match(rgb, data):
    """Best cyclic ink Dice, so empty background cannot approve wrong symbols."""
    from .score import foreground
    tile = foreground(data['tile'], data.get('bg'))
    actual = foreground(rgb)
    py, px = tile.shape
    yy, xx = np.indices(actual.shape)
    positions = (yy % py * px + xx % px).ravel()
    folded = np.bincount(positions, weights=actual.ravel(), minlength=py*px).reshape(py, px)
    all_pixels = np.bincount(positions, minlength=py*px).reshape(py, px)
    spectrum = np.fft.rfft2(tile.astype(float))
    hits = np.fft.irfft2(np.conj(np.fft.rfft2(folded)) * spectrum, s=(py, px))
    predicted = np.fft.irfft2(np.conj(np.fft.rfft2(all_pixels)) * spectrum, s=(py, px))
    denominator = predicted + actual.sum()
    if not actual.any() and not tile.any():
        return 1.0
    return float(np.max(2 * np.maximum(0, hits) / np.maximum(1, denominator)))


def trim_evidence_symbols(rgb, data):
    """Trim a neighbouring motif even when it shares the layer's background."""
    phase = data.get('_last_alignment')
    h,w = rgb.shape[:2]
    if phase is None or h<18:
        return rgb,0
    tile = data['tile']
    py,px = tile.shape[:2]
    dy,dx = phase
    fake = tile[(np.arange(h)[:,None]+dy)%py,(np.arange(w)[None,:]+dx)%px]
    actual_ink, expected_ink = rgb.max(2)<80, fake.max(2)<80
    counts = actual_ink.sum(1)+expected_ink.sum(1)
    dice = 2*(actual_ink & expected_ink).sum(1)/np.maximum(1,counts)
    dice[counts==0] = 1
    # Only trim edges around an already coherent central motif.
    middle = slice(h//3,2*h//3)
    denom = counts[middle].sum()
    if denom and 2*(actual_ink[middle]&expected_ink[middle]).sum()/denom < .80:
        return rgb,0
    bad = dice < .5
    top,bottom = 0,h
    if bad[:min(8,h//3)].any():
        top = int(np.flatnonzero(bad[:h//3])[-1])+1
    if bad[-min(8,h//3):].any():
        bottom = h-h//3+int(np.flatnonzero(bad[-(h//3):])[0])
    return rgb[top:bottom].copy(),top


def boxes_for(doc, prefill):
    """At most 28 suggested intervals per label plus 12 whole-column probes.

    Use a central native 64-pixel lane: edge annotations cannot become evidence.
    The evidence boxes record precisely the pixels compared and learned from.
    """
    x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
    mid = (x0 + x1) // 2
    x0, x1 = max(x0 + 2, mid - 32), min(x1 - 2, mid + 32)
    output = {}
    scans = []
    for y in np.linspace(y0, max(y0, y1 - 96), 12).astype(int):
        scans.append([x0, int(y), x1, min(y1, int(y) + 96)])
    for entry in doc['legend']:
        rows = [r for r in prefill if learn.norm(r.get('suggestion')) == learn.norm(entry['label_printed'])]
        if len(rows) > 28:
            rows = [rows[i] for i in np.linspace(0, len(rows) - 1, 28).astype(int)]
        boxes = []
        for r in rows:
            a, b = max(y0, int(r['top_y'])), min(y1, int(r['bottom_y']))
            if b - a >= 24:
                centre = (a + b) // 2
                boxes.append([x0, max(a, centre - 48), x1, min(b, centre + 48)])
        output[entry['id']] = [list(b) for b in dict.fromkeys(tuple(b) for b in boxes + scans) if b[3] - b[1] >= 24]
    return output


def depth_ranges(evidence):
    """Overlapping or touching windows are one range, never independent votes."""
    ranges = []
    for e in sorted(evidence, key=lambda e: e['box'][1]):
        a, b = e['box'][1::2]
        if ranges and a <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
    return ranges


def analyze(doc, crops, groups, prefill):
    started = time.time()
    entries, textures, cleaned = [], {}, {}
    for e in doc['legend']:
        rgb, data, repeat = swatch(crops.get(doc, e['box']))
        cleaned[e['id']], textures[e['id']] = rgb, data
        entries.append(dict(legend_id=e['id'], label=e['label_printed'], swatch_box=e['box'],
                            repeat=repeat, cycles=repeat['observed_cycles'], evidence=[], twins=[],
                            seam_risk=repeat['seam_risk'], seam_x=repeat['seam_x'], seam_y=repeat['seam_y'],
                            exclusion=not_lithology(e['label_printed'], rgb)))
    for i, e in enumerate(entries):
        for other in entries[i + 1:]:
            if e['exclusion'] or other['exclusion']:
                continue
            a, b = textures[e['legend_id']], textures[other['legend_id']]
            ab = pixel_match(a['tile'], b, minimum=.95)
            if ab < .95:
                continue
            ba = pixel_match(b['tile'], a, minimum=.95)
            if min(ab, ba) >= .95:
                e['twins'].append(dict(legend_id=other['legend_id'], label=other['label'], match=round(ab, 4), reverse=round(ba, 4)))
                other['twins'].append(dict(legend_id=e['legend_id'], label=e['label'], match=round(ba, 4), reverse=round(ab, 4)))
    pools = boxes_for(doc, prefill)
    all_boxes = list(dict.fromkeys(tuple(b) for pool in pools.values() for b in pool))
    entry_by_id = {e['legend_id']: e for e in entries}
    for box in all_boxes:
        rgb, (dl, dt) = learn.clean(crops.get(doc, box))
        bg = learn.background(rgb)
        palette = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
        matches = sorted(((pixel_match(rgb, textures[e['legend_id']], bg, palette), i, e['legend_id'])
                          for i, e in enumerate(entries) if not e['exclusion']), reverse=True)
        if not matches:
            continue
        best, _, lid = matches[0]
        runner = matches[1][0] if len(matches) > 1 else 0.0
        # Cross-label probes may verify an entry even if the saved guess was wrong.
        if best >= .90 and best - runner >= .08:
            rgb, trim_top = trim_evidence_symbols(rgb,textures[lid])
            dt += trim_top
            if rgb.shape[0]<12:
                continue
            clean_box = [box[0]+dl, box[1]+dt, box[0]+dl+rgb.shape[1], box[1]+dt+rgb.shape[0]]
            entry_by_id[lid]['evidence'].append(dict(box=clean_box, match=round(best, 5),
                                                   runner_up=round(runner, 5), margin=round(best - runner, 5)))
    docs = {doc['id']: dict(doc=doc)}
    learned = {}
    for e in entries:
        lid = e['legend_id']
        # Bound evidence per entry, spread over depth rather than first rows.
        ev = sorted(e['evidence'], key=lambda e: e['box'][1])
        if len(ev) > 40:
            ev = [ev[i] for i in np.linspace(0, len(ev) - 1, 40).astype(int)]
        e['evidence'] = ev
        e['depth_ranges'] = depth_ranges(ev)
        e['matches'] = len(ev)
        e['candidates_scored'] = len(all_boxes)
        human = groups.get((doc['id'], lid), [])
        if e['exclusion']:
            tier, reason = 'not-lithology', e['exclusion']
        elif human:
            tier, reason = 'human', f'{len(human)} human-approved single-label boxes.'
        elif e['twins']:
            tier, reason = 'ambiguous', 'Swatch is indistinguishable from ' + ', '.join(t['label'] for t in e['twins']) + '.'
        elif len(ev) >= 3 and len(e['depth_ranges']) >= 2:
            tier, reason = 'verified', f'{len(ev)} exclusive column matches across {len(e["depth_ranges"])} separate depth ranges.'
        elif not e['seam_risk']:
            tier, reason = 'swatch', 'Seam tests pass on both axes; requires a confident reviewer verdict.'
        else:
            tier, reason = 'no', f'Only {len(ev)} exclusive matches across {len(e["depth_ranges"])} ranges; {e["cycles"]:g} swatch cycles.'
        e.update(tier=tier, reason=reason)
        if tier in ('verified', 'human'):
            exs = human or examples(doc, e)
            m, data = learn.learn(crops, docs, (doc['id'], lid), exs)
            if data is not None:
                m, data = choose_tile(crops, doc, exs, m, data, textures[lid], e['seam_risk'])
            e['model'] = m
            if data is None and tier == 'verified':
                e.update(tier='no', reason='Column evidence did not survive the existing tile learner.')
            elif data is not None:
                learned[lid] = data
        if e['tier'] == 'swatch':
            learned[lid] = textures[lid]
            e['model'] = swatch_model(doc, e, textures[lid])
    return dict(version=VERSION, doc_id=doc['id'], name=doc['name'], signature=signature(doc, groups),
                seconds=round(time.time() - started, 2), entries=entries), learned, cleaned, textures


def swatch_model(doc, entry, data):
    h, w = data['tile'].shape[:2]
    return dict(key=f'{doc["id"]}:{entry["legend_id"]}', doc_id=doc['id'],
                legend_id=entry['legend_id'], label=entry['label'], examples=0,
                sources=[], suspects=[], suspect_count=0, kind='tile', tile_size=[w, h],
                agreement=None, anchored=False, period=dict(y=h, x=w), reason=entry['reason'])


def examples(doc, entry):
    return [dict(id=f'machine-{entry["legend_id"]}-{i}', doc_id=doc['id'], legend_id=entry['legend_id'],
                 label=entry['label'], box=e['box'], provenance='machine-verified', flagged=False)
            for i, e in enumerate(entry['evidence'])]


def save_data(path, data):
    arrays = dict(bg=data['bg'], phase=data.get('phase', (0, 0)), anchored=data.get('anchored', False))
    for key in ('anchored_x','anchored_y','x_repeat','x_origin','strip_origins','max_layer_height','native_extent','quilt_local'):
        if key in data:
            arrays[key] = data[key]
    if data['kind'] == 'tile':
        arrays['tile'] = data['tile']
    else:
        arrays.update({f'strip_{i}': s for i, s in enumerate(data['strips'])})
    # Write through a unique temporary file: the live builder may read concurrently.
    import uuid
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_data(path):
    with np.load(path, allow_pickle=False) as a:
        d = dict(bg=a['bg'].copy(), phase=tuple(a['phase']), anchored=bool(a['anchored']))
        for key in ('anchored_x','anchored_y','x_repeat','x_origin','strip_origins','max_layer_height','native_extent','quilt_local'):
            if key in a:
                d[key] = a[key].tolist() if a[key].ndim else a[key].item()
        if 'tile' in a:
            d.update(kind='tile', tile=a['tile'].copy())
        else:
            d.update(kind='quilt', strips=[a[k].copy() for k in sorted(a.files) if k.startswith('strip_')])
    return d


def reports():
    for path in sorted(paths.CANDIDATES.glob('*.json')):
        if path.name not in ('summary.json', 'review-queue.json'):
            yield json.loads(path.read_text(encoding='utf-8'))


def donor_pool():
    pool = []
    for report in reports():
        if not report.get('training_allowed', True) or report.get('version') != VERSION:
            continue
        if learn.generation_split(dict(id=report['doc_id'],name=report['name']))[1]!='train':
            continue
        vp = paths.VERDICTS / (report['doc_id'] + '.json')
        verdicts = json.loads(vp.read_text(encoding='utf-8')) if vp.exists() else {}
        for e in report['entries']:
            if e['tier'] not in ('human', 'verified') or 'data_file' not in e:
                continue
            if verdicts.get(str(e['legend_id']), {}).get('verdict') == 'reject':
                continue
            data = load_data(paths.CANDIDATES / e['data_file'])
            if data['kind'] == 'tile':
                pool.append(dict(doc_id=report['doc_id'], name=report['name'], entry=e, data=data))
    return pool


def find_donor(doc_id, entry, rgb, donors):
    """Only explicit names, real RGB colours and >=.92 best cyclic agreement."""
    if entry['tier'] in ('human', 'verified', 'ambiguous', 'not-lithology') or entry['matches']:
        return None
    best = None
    bg = learn.background(rgb)
    palette = None
    for donor in donors:
        if donor['doc_id'] == doc_id or name_key(donor['entry']['label']) != name_key(entry['label']):
            continue
        if np.linalg.norm(bg - donor['data']['bg']) > 12:
            continue
        if symbol_match(rgb, donor['data']) < .80:
            continue
        if palette is None:
            palette = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
        match = pixel_match(rgb, donor['data'], bg=bg, palette=palette, minimum=.92, distance=20)
        if match >= .92 and (best is None or match > best[0]):
            best = match, donor
    return best


def borrow(doc, report, learned, cleaned, donors):
    for e in report['entries']:
        hit = find_donor(doc['id'], e, cleaned[e['legend_id']], donors)
        if not hit:
            continue
        match, donor = hit
        de = donor['entry']
        e.update(tier='borrowed', donor=dict(doc_id=donor['doc_id'], legend_id=de['legend_id'],
                 name=donor['name'], label=de['label'], tier=de['tier'], match=round(match, 5),
                 evidence=de.get('human_evidence') or de['evidence'], data_file=de['data_file']),
                 reason=f'Name and colour swatch match donor at {match:.3f}.')
        e['model'] = swatch_model(doc, e, donor['data'])
        learned[e['legend_id']] = donor['data']


def persist_models(doc, report, learned):
    for e in report['entries']:
        if e['legend_id'] not in learned:
            continue
        data = learned[e['legend_id']]
        h = hashlib.sha256(json.dumps(e['model'], sort_keys=True).encode())
        for a in ([data['tile']] if data['kind'] == 'tile' else data['strips']):
            h.update(a.tobytes())
        name = f'{doc["id"]}-{e["legend_id"]}-{h.hexdigest()[:16]}.npz'
        save_data(paths.CANDIDATES / name, data)
        e['data_file'] = name


def sheets(doc, report, learned, cleaned, textures, crops, docs=None):
    """Native swatches at 3x; evidence and generated strips share display width."""
    import textwrap
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    except OSError:
        font = ImageFont.load_default(size=17)
    rows = []
    for e in report['entries']:
        lid = e['legend_id']
        sw = Image.fromarray(cleaned[lid])
        sw = sw.resize((sw.width * 3, sw.height * 3), Image.Resampling.NEAREST)
        crop_width = sw.width
        pics = [sw]
        evidence = e.get('human_evidence') or e['evidence']
        evidence_doc = doc
        if e['tier'] == 'borrowed':
            evidence = e['donor']['evidence']
            evidence_doc = (docs or {})[e['donor']['doc_id']]['doc']
        for ev in evidence[:4]:
            im = Image.fromarray(crops.get(evidence_doc, ev['box']))
            # Limit to a readable native evidence window, retaining its box in JSON.
            if im.height > 140:
                im = im.crop((0, 0, im.width, 140))
            pics.append(im.resize((im.width*3, im.height*3), Image.Resampling.NEAREST))
        data = learned.get(lid, textures[lid])
        rails = dict(x0=0, x1=64, mask=[False] * 64, ink=[0, 0, 0])
        for seed in range(2):
            im = Image.fromarray(render.sample(data, rails, 120, np.random.default_rng(seed)))
            pics.append(im.resize((im.width*3, im.height*3), Image.Resampling.NEAREST))
        height = max(240, max(im.height for im in pics) + 40)
        width = max(1400, 410 + sum(im.width + 16 for im in pics))
        row = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(row)
        text = f'{lid}: {e["label"]}\n{e["tier"]} | cycles {e["cycles"]:g} | matches {e["matches"]} | ranges {len(e["depth_ranges"])}'
        text += '\n' + '\n'.join(textwrap.wrap(e['reason'], 44))
        text += f'\nSeam X {e["seam_x"]["match"]:.3f} ({e["seam_x"]["method"]})'
        text += f'\nSeam Y {e["seam_y"]["match"]:.3f} ({e["seam_y"]["method"]}) | risk {e["seam_risk"]}'
        if e['tier'] == 'borrowed':
            text += f'\nDonor {e["donor"]["doc_id"]}:{e["donor"]["legend_id"]}'
        if e['evidence']:
            text += f'\nbest match {max(v["match"] for v in e["evidence"]):.3f}'
        draw.multiline_text((12, 12), text, font=font, fill='black', spacing=5)
        x = 410
        for i, im in enumerate(pics):
            label = 'Legend' if i == 0 else ('Generated' if i >= len(pics) - 2 else
                    ('Donor column' if e['tier']=='borrowed' else 'Target column'))
            label += f' {im.width//3}x{im.height//3} @3x'
            draw.text((x, 8), label, font=font, fill='black')
            row.paste(im, (x, 35))
            x += im.width + 16
        draw.line((0, height - 1, width, height - 1), fill='#cccccc')
        rows.append((e, row))
    pages, batch, height = [], [], 0
    for e, row in rows:
        if batch and height + row.height > 4000:
            pages.append(batch)
            batch, height = [], 0
        batch.append((e, row))
        height += row.height
    if batch:
        pages.append(batch)
    files = []
    for i, batch in enumerate(pages):
        path = paths.CANDIDATES / (doc['id'] + ('' if i == 0 else f'-{i + 1}') + '.png')
        page = Image.new('RGB', (max(r.width for _, r in batch), sum(r.height for _, r in batch)), 'white')
        y = 0
        for row_index, (e, row) in enumerate(batch):
            e.update(sheet_png=path.name, row_index=row_index, row_y=y, row_height=row.height)
            page.paste(row, (0, y))
            y += row.height
        page.save(path)
        files.append(path.name)
    return files


def refresh(doc, crops, groups, prefill, *, donors=None, docs=None, training_allowed=True):
    from .build import _write
    report, learned, cleaned, textures = analyze(doc, crops, groups, prefill)
    paths.CANDIDATES.mkdir(parents=True, exist_ok=True)
    report['training_allowed'] = training_allowed
    previous = paths.CANDIDATES / (doc['id'] + '.json')
    old_entries = {e['legend_id']: e for e in json.loads(previous.read_text(encoding='utf-8'))['entries']} if previous.exists() else {}
    for e in report['entries']:
        lid = e['legend_id']
        old = old_entries.get(lid, {})
        e['previous_tier'] = old.get('previous_tier', old.get('tier'))
        raw = crops.get(doc, e['swatch_box'])
        legacy, _ = trim_frame(raw)
        legacy, _ = learn.clean(legacy)
        legacy, _ = learn.clean(legacy.transpose(1, 0, 2))
        legacy = legacy.transpose(1, 0, 2)
        _, residual = remove_frame(legacy)
        e['legacy_frame_seam_fixed'] = bool(any(residual[::2]) and any(e['repeat']['frame_trim'][::2]))
        if e['tier'] == 'human':
            e['human_evidence'] = [dict(box=x['box']) for x in groups[(doc['id'], lid)]
                                   if x['id'] in e.get('model', {}).get('sources', [])]
    if donors:
        borrow(doc, report, learned, cleaned, donors)
    persist_models(doc, report, learned)
    report['sheets'] = sheets(doc, report, learned, cleaned, textures, crops, docs)
    _write(paths.CANDIDATES / (doc['id'] + '.json'), report)
    return report


def summary():
    from .build import _write
    documents, total = [], Counter({tier: 0 for tier in TIERS})
    queue = []
    fixed = fixed_safe = 0
    for report in reports():
        counts = Counter(e['tier'] for e in report['entries'])
        total.update(counts)
        for e in report['entries']:
            if e.get('previous_tier') == 'swatch-only' and e.get('legacy_frame_seam_fixed'):
                fixed += 1
                fixed_safe += not e['seam_risk']
            if e['tier'] in ('verified', 'borrowed', 'swatch'):
                queue.append(dict(doc_id=report['doc_id'], legend_id=e['legend_id'], label=e['label'],
                                  tier=e['tier'], sheet_png=e['sheet_png'], row_index=e['row_index']))
        documents.append(dict(doc_id=report['doc_id'], name=report['name'], seconds=report['seconds'],
                              counts={tier: counts[tier] for tier in TIERS}))
    result = dict(overall=dict(total), documents=documents, legacy_swatch_frame_fixes=fixed,
                  legacy_swatch_frame_fixes_seam_safe=fixed_safe)
    _write(paths.CANDIDATES / 'summary.json', result)
    _write(paths.CANDIDATES / 'review-queue.json', queue)
    return result


def _scan_one(doc, groups, prefill, training_allowed):
    """A worker owns its TIFF reader; documents and output files are disjoint."""
    import model
    images = model.Images()
    try:
        return refresh(doc, learn.Crops(images), groups, prefill, training_allowed=training_allowed)
    finally:
        if images.current:
            images.current.close()


def recovery_donor(card, target_x0=None, donor_x0=None):
    """Borrow the full bound model; translate only the explicit lane origin."""
    data=learn.load_model(card['render_model'],signature=card['signature'])
    if target_x0 is not None:
        if donor_x0 is None:raise ValueError('Donor lane origin required for translation')
        delta=target_x0-donor_x0
        data=dict(data,x_origin=data.get('x_origin',0)+delta,
                  phase=(data.get('phase',(0,0))[0],data.get('phase',(0,0))[1]+delta))
        if 'strip_origins' in data:data['strip_origins']=[v+delta for v in data['strip_origins']]
        if 'native_extent' in data:data['native_extent']=[v+delta for v in data['native_extent']]
    return data


def freeze_donor_tile(card,data):
    """Recovery evidence must outlive the generated-preview pruning cycle."""
    tile=data['tile'];h,w=tile.shape[:2]
    pixel=hashlib.sha256(f'{w}x{h}:'.encode()+tile.tobytes()).hexdigest()
    folder=paths.OUT/'recover'/'donor-assets';folder.mkdir(parents=True,exist_ok=True)
    file=folder/(hashlib.sha256(card['key'].encode()).hexdigest()[:16]+'-'+pixel+'.png')
    if not file.exists():Image.fromarray(tile).save(file)
    return dict(tile_path=file.relative_to(paths.OUT).as_posix(),
                tile_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),model_signature=card['signature'])


def recover():
    """Isolated, train-only recovery; no production admission or guessed truth."""
    import model
    from .build import _write
    from .reading import ReadOnlyStore
    root = paths.OUT/'recover'
    inventory = json.loads((paths.OUT/'critique'/'s2-failed'/'inventory.json').read_text(encoding='utf-8'))
    state = json.loads(paths.STATE.read_text(encoding='utf-8'))
    store = ReadOnlyStore(model.Store())
    docs = {c['id']:store.get(c['id']) for c in store.catalog}
    _,human_groups=learn.human_examples(store)
    images = model.Images()
    crops = learn.Crops(images)
    policy = json.loads((paths.DATA/'normalized'/'split-policy.json').read_text(encoding='utf-8'))
    index = json.loads((paths.DATA/'normalized'/'current.json').read_text(encoding='utf-8'))
    # Recompute occurrence weights with the *generation* split, including fixed
    # validation exclusion. The old failed-only list missed dominant rejects.
    prefill=score.load_prefill();names={learn.norm(d['name']):did for did,d in docs.items()}
    weights=Counter();guesses={};denominator=0
    for name,rows in prefill.items():
        did=names.get(learn.norm(name))
        if did not in docs or learn.generation_split(docs[did],policy)[1]!='train':continue
        for row in rows:
            w=max(0,int(row['bottom_y'])-int(row['top_y']));denominator+=w
            lids=[e['id'] for e in docs[did]['legend'] if learn.norm(e['label_printed'])==learn.norm(row.get('suggestion'))]
            for lid in lids:
                weights[(did,lid)]+=w/len(lids);guesses.setdefault((did,lid),[]).append(row)
    entries={e['key']:dict(e) for e in inventory['entries']}
    generated={c['key'] for c in state['cards']}
    for (did,lid),weight in weights.items():
        key=f'{did}:{lid}'
        vp=paths.VERDICTS/(did+'.json');verdicts=json.loads(vp.read_text(encoding='utf-8')) if vp.exists() else {}
        if key not in generated or verdicts.get(str(lid),{}).get('verdict')=='reject':
            native=next(e for e in docs[did]['legend'] if e['id']==lid)
            if not_lithology(native['label_printed'],np.full((2,2,3),200,np.uint8)):continue
            entries.setdefault(key,dict(key=key,doc_id=did,legend_id=lid,label=native['label_printed'],
                cause='owner-box / review conflict',note='Rejected or missing dominant class; preserve owner verdict; refit only accepted class-labelled crops',
                owner_verdict=verdicts.get(str(lid)),guess_intervals=guesses.get((did,lid),[])))
    for e in entries.values():
        e['estimated_rows']=weights[(e['doc_id'],e['legend_id'])]
        e['training_row_share']=e['estimated_rows']/max(1,denominator)
    inventory=dict(entries=sorted(entries.values(),key=lambda e:(
        0 if e['estimated_rows']>0 and human_groups.get((e['doc_id'],e['legend_id'])) else 1,
        -e['estimated_rows'],e['key'])),total_training_guessed_rows=denominator)
    _write(root/'inventory-r2.json',inventory)
    fingerprint=hashlib.sha256(json.dumps([learn.MODEL_VERSION,
        [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (render.__file__,learn.__file__,__file__)],inventory,policy,index.get('snapshot'),
        [(did,d.get('revision')) for did,d in docs.items()],
        [(c['key'],c.get('signature'),c.get('tile'),c.get('render_model'),
          hashlib.sha256((paths.OUT/c['tile']).read_bytes()).hexdigest() if c.get('tile') and (paths.OUT/c['tile']).exists() else None,
          hashlib.sha256((paths.OUT/c['render_model']['asset']['path']).read_bytes()).hexdigest() if c.get('render_model') else None)
         for c in state['cards']]],sort_keys=True).encode()).hexdigest()
    previous=root/'search-results.json'
    if previous.exists() and (root/'review-queue.json').exists():
        saved=json.loads(previous.read_text(encoding='utf-8'))
        if saved.get('fingerprint')==fingerprint:
            print('PASS recovery: unchanged evidence; retained completed searches',flush=True)
            return
    forbidden = {r['pixel_sha256'] for r in index['records']
                 if r.get('split')=='test' or not r.get('usable_for_generation',False)}
    def eligible(did):
        return did in docs and learn.generation_split(docs[did],policy)[1]=='train'
    def pixel_hash(rgb):
        h,w=rgb.shape[:2]
        return hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()
    def source(did,lid,boxes,kind):
        group,split = learn.generation_split(docs[did],policy)
        hashes = [pixel_hash(crops.get(docs[did],b)) for b in boxes]
        if any(h in forbidden for h in hashes):
            return None
        return dict(doc_id=did,legend_id=lid,source_group=group,split=split,
                    source_revision=docs[did].get('revision'),boxes=boxes,pixel_sha256=hashes,kind=kind)
    pool=[];excluded=[]
    for c in state['cards']:
        if not eligible(c['doc_id']):
            excluded.append(dict(key=c['key'],reason='held-out source well'))
            continue
        if not learn.lineage_allowed(c,docs):
            excluded.append(dict(key=c['key'],reason='missing or revoked model ancestry'))
            continue
        # A swatch/approved tile is not donor column evidence. Lettering is not
        # an interchangeable geological glyph even if its RGB agreement is 1.
        if not c.get('tile') or c.get('reference_kind', 'target column' if c['source'] in ('human','machine-verified') else 'legend')!='target column':
            continue
        if any(t in name_key(c['label']).split() for t in ('granite','gr','congl')):
            continue
        boxes=c.get('real',[])
        if not boxes:
            continue
        lineage=source(c['doc_id'],c['legend_id'],boxes,'donor column')
        if lineage is None:
            continue
        try:donor=recovery_donor(c)
        except ValueError:continue
        lineage.update(freeze_donor_tile(c,donor),proposal_round=c.get('proposal_round'))
        lineage.update(learn.ancestry_metadata(c))
        tile=donor['tile']
        if pixel_hash(tile) in forbidden:
            continue
        if not any((b[3]-b[1])>=2*tile.shape[0] for b in boxes):
            continue
        pool.append((c,donor,lineage))
    leads_path=paths.OUT/'critique'/'s2-failed'/'lookalike-results.json'
    leads=json.loads(leads_path.read_text(encoding='utf-8')).get('pairs',[]) if leads_path.exists() else []
    linked={(p['target'],p['donor']) for p in leads}
    queue=[];owner=[];searches=[]
    try:
        for entry in inventory['entries']:
            did,lid=entry['doc_id'],entry['legend_id'];key=entry['key'];doc=docs[did]
            folder=root/did;folder.mkdir(parents=True,exist_ok=True)
            native=next(e for e in doc['legend'] if e['id']==lid)
            group,split=learn.generation_split(doc,policy)
            proposal=dict(doc_id=did,legend_id=lid,label=entry['label'],round=1,status='failed',
                method='reconstructed',tile_png=f'legend-{lid}-tile.png',period=dict(x=None,y=None),
                column_boxes=[],swatch_match=None,column_match=None,
                recovery=dict(method='needs-evidence',donor=None,lineage=[],evidence=[],
                    axes=dict(x='unknown',y='unknown'),class_identity='unconfirmed',
                    estimated_rows=entry['estimated_rows'],share_of_column_rows=entry['training_row_share'],
                    occurrence='machine guessed' if entry['estimated_rows'] else 'unknown occurrence',
                    source_split=split,production_eligible=False))
            record=proposal['recovery'];data=None;raw=None;reals=[];hits=[]
            if not eligible(did):
                record.update(method='deferred-held-out',reason='Validation/test target: no pixels read, fitted, or recovered')
            else:
                raw,_=learn.clean(crops.get(doc,native['box']))
                lineage=source(did,lid,[native['box']],'target legend')
                if lineage is None:
                    record.update(method='quarantined',reason='Source pixels match excluded normalized record')
                else:
                    record['lineage']=[lineage]
                    for card,donor,ancestor in pool:
                        if card['doc_id']==did or not (name_key(card['label'])==name_key(entry['label']) or (key,card['key']) in linked):
                            continue
                        if np.linalg.norm(learn.background(raw)-donor['bg'])>12:
                            continue
                        rgb=pixel_match(raw,donor,distance=20);dice=symbol_match(raw,donor)
                        hits.append(dict(donor=card['key'],rgb=rgb,foreground_dice=dice))
                        if rgb>=.92 and dice>=.8 and (data is None or rgb>record.get('donor_match',0)):
                            donor_x0=docs[card['doc_id']]['boxes']['interpreted_lithology_column'][0]
                            target_x0=doc['boxes']['interpreted_lithology_column'][0]
                            data=recovery_donor(card,target_x0,donor_x0)
                            record.update(method='donor-native',donor=dict(doc_id=card['doc_id'],legend_id=card['legend_id']),
                                donor_match=rgb,lineage=[lineage,ancestor],axes=dict(x='borrowed',y='borrowed'),
                                evidence=[dict(reference='target legend',rgb=rgb,foreground_dice=dice)],
                                class_identity='legend appearance only')
                            record['placement_translation']=dict(donor_lane_x0=donor_x0,target_lane_x0=target_x0,dx=target_x0-donor_x0,dy=0)
                    if data is None and entry['cause']=='stochastic texture candidate':
                        data=render.glyph_model(raw,'texture')
                        if data:
                            record.update(method='stochastic-glyph-bootstrap',axes=dict(x='stochastic',y='stochastic'),
                                evidence=[dict(reference='target legend',complete_glyphs=len(data['glyphs']),density=data['density'])],
                                reason='Native glyph pool and density; target-column occurrence and spacing unconfirmed')
                    if key=='5709fe9e86ce942b:6':
                        old=json.loads((paths.PROPOSALS/did/f'legend-{lid}.json').read_text(encoding='utf-8'))
                        boxes=old.get('column_boxes',[])
                        reals=[learn.clean(crops.get(doc,b))[0] for b in boxes]
                        if reals:
                            px=learn.horizontal_period(reals[0])
                            # The short fit box aliases the alternating 35-row
                            # course at 18. Retain the explicit 35-row hypothesis
                            # and validate it on the other four native boxes.
                            yp=(old.get('period') or {}).get('y') or estimate_period(reals[0],0)['pixels']
                            if px and len(reals[0])>=yp:
                                tile=reals[0][:yp,:px].copy()
                                data=dict(kind='tile',tile=tile,bg=learn.background(tile),phase=(0,0),anchored=False)
                                evidence=[score.fidelity(c,data) for c in reals[1:]]
                                record.update(method='column-geometry',donor=None,axes=dict(x='observed column',y='observed column'),
                                    evidence=evidence,fit_boxes=boxes[:1],reserved_boxes=boxes[1:],
                                    class_identity='requires one owner label',reason='Column spacing overrides rejected legend spacing; class is unconfirmed')
                                ancestor=source(did,lid,boxes,'unlabelled target column')
                                if ancestor:
                                    record['lineage']=[lineage,ancestor]
                                else:
                                    data=None;record.update(method='quarantined')
                                proposal['column_boxes']=boxes
                    owner_exs=human_groups.get((did,lid),[])
                    if owner_exs and entry['estimated_rows']>0 and entry['cause']=='owner-box / review conflict':
                        # Fit exclusively on owner-labelled training crops; reserve
                        # depth-disjoint evidence before fitting and never tune it.
                        fit_ex=max(owner_exs,key=lambda e:((e['box'][3]-e['box'][1])*(e['box'][2]-e['box'][0]),str(e['id'])))
                        held_exs=[e for e in owner_exs if not learn.boxes_overlap(e['box'],fit_ex['box'],512)]
                        fit_exs=[fit_ex] if held_exs else owner_exs
                        fitted,data=learn.learn(crops,{did:dict(doc=doc)},(did,lid),fit_exs)
                        if held_exs:
                            fitted.update(heldout_ids=[e['id'] for e in held_exs],heldout_boxes=[e['box'] for e in held_exs])
                        if data is not None:
                            reals=[learn.clean(crops.get(doc,e['box']))[0] for e in owner_exs]
                            boxes=[e['box'] for e in owner_exs]
                            fit_inputs=list(fitted['training_boxes'])+[b for b in fitted.get('native_source_boxes',[]) if b not in fitted['training_boxes']]
                            ancestor=source(did,lid,fit_inputs,'owner-labelled target column')
                            if ancestor:
                                record.update(method='owner-column-refit',lineage=[ancestor],
                                    fit_boxes=fitted['training_boxes'],reserved_boxes=fitted['heldout_boxes'],
                                    class_identity='owner-labelled training crops',axes=dict(x='native owner crop',y=fitted.get('repeat_support')),
                                    evidence=score.realism(fitted,data,crops,doc,owner_exs),
                                    owner_verdict=entry.get('owner_verdict'),reason='Replacement for review; original rejection remains in force')
                                proposal['column_boxes']=boxes
                            else:data=None;record.update(method='quarantined')
            if data is not None:
                preview_x0=data.get('x_origin',0)
                preview_width=(doc['boxes']['interpreted_lithology_column'][2]-doc['boxes']['interpreted_lithology_column'][0]
                               if record['method']=='owner-column-refit' else 96)
                if data['kind']=='tile':
                    tile=data['tile'];proposal['period']=dict(x=tile.shape[1] if data.get('x_repeat',True) else None,y=tile.shape[0])
                elif data['kind'] in ('texture','jitter'):
                    tile=render.layer(data,0,96,0,240,np.random.default_rng(7))
                    np.savez_compressed(folder/f'legend-{lid}-glyphs.npz',bg=data['bg'],density=data['density'],
                                        large_density=data.get('large_density',0),small_density=data.get('small_density',data['density']),
                                        **{f'glyph_{i}':g for i,g in enumerate(data['glyphs'])})
                    record['texture_model']=f'legend-{lid}-glyphs.npz'
                    record['tile_is_preview_only']=True
                else:
                    tile=render.layer(data,preview_x0,preview_x0+preview_width,0,240,np.random.default_rng(7))
                    record['tile_is_preview_only']=True
                sig=hashlib.sha256(json.dumps([key,learn.MODEL_VERSION,record['lineage']],sort_keys=True).encode()).hexdigest()[:16]
                proposal['render_model']=learn.export_model(data,folder,sig,proposal)
                Image.fromarray(tile).save(folder/proposal['tile_png'])
                proposal['status']='made';proposal['swatch_match']=pixel_match(raw,dict(tile=tile,bg=learn.background(tile)))
                if len(reals)>1:
                    proposal['column_match']=float(np.median([score.fidelity(r,dict(tile=tile,bg=learn.background(tile)))['pixel'] for r in reals[1:]]))
                preview=[render.layer(data,preview_x0,preview_x0+preview_width,0,240,np.random.default_rng(s)) for s in (7,11)]
                panes=[('Target legend',raw)]+[(f'Target column {i}',r[:160]) for i,r in enumerate(reals[:2])]+[(f'Generated {s}',v) for s,v in zip((7,11),preview)]
                if record.get('donor'):
                    donor=record['donor'];ancestor=record['lineage'][-1]
                    donor_rgb=crops.get(docs[donor['doc_id']],ancestor['boxes'][0])[:160]
                    panes.insert(1,(f'Donor {donor["doc_id"]}',donor_rgb))
                sheet=Image.new('RGB',(sum(max(a.shape[1]*3,180)+16 for _,a in panes),max(a.shape[0]*3 for _,a in panes)+55),'#dddddd')
                draw=ImageDraw.Draw(sheet);x=0
                for title,a in panes:
                    draw.text((x+3,4),f'{title} {a.shape[1]}x{a.shape[0]} @3x',fill='black')
                    im=Image.fromarray(a);sheet.paste(im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST),(x,45))
                    x+=max(a.shape[1]*3,180)+16
                sheet_path=folder/f'legend-{lid}-sheet.png';sheet.save(sheet_path)
                record['sheet_png']=sheet_path.relative_to(root).as_posix()
                record['seeds_distinct']=not np.array_equal(*preview)
                if data['kind']=='texture':
                    observed=[float(score.foreground(v,data['bg']).mean()) for v in preview]
                    record.update(generated_density=observed,
                        density_relative_error=max(abs(v-data['density'])/max(data['density'],1e-9) for v in observed),
                        density_note='Packing may underfill dense or incomplete glyph pools; independent review is required')
                    from scipy.ndimage import label,find_objects
                    def large_share(rgb):
                        mask=score.foreground(rgb,data['bg']);labels,_=label(mask,structure=np.ones((3,3)));large=0
                        for i,sl in enumerate(find_objects(labels),1):
                            if sl and max(sl[0].stop-sl[0].start,sl[1].stop-sl[1].start)>=8:large+=int((labels[sl]==i).sum())
                        return large/max(1,int(mask.sum()))
                    source_share=large_share(raw);shares=[large_share(v) for v in preview]
                    record.update(glyph_evidence=data.get('glyph_evidence'),source_large_ink_share=source_share,
                        generated_large_ink_share=shares,
                        morphology_alarm=bool(any(abs(v-source_share)/max(.001,source_share)>.2 for v in shares) or
                                              not data.get('glyph_evidence',{}).get('morphology_supported',False)))
            elif eligible(did) and record['method'] not in ('quarantined',):
                record['reason']=entry['note']
                record['prior_only']=True
                record['alternative_spacing']='Unknown axes remain unmeasured; conventions are not column evidence'
            historical_review=(key=='4a056c8ff25cf749:5' and data is None and
                               (folder/proposal['tile_png']).exists() and (folder/f'legend-{lid}-sheet.png').exists())
            if historical_review:
                # Preserve the critic's existing review submission without
                # laundering its opaque historical donor into eligible pixels.
                donor_id='7b0d6b0a3e062ece'
                proposal.update(status='made',donor=dict(doc_id=donor_id,ancestry_status='unknown'),
                                ancestor_docs=sorted([did,donor_id]))
                record.update(method='donor-native-historical-review',review_only=True,load_blocked=True,
                    ancestry_status='unknown',tile_is_preview_only=True,
                    historical_preview_sha256=hashlib.sha256((folder/proposal['tile_png']).read_bytes()).hexdigest(),
                    sheet_png=f'{did}/legend-{lid}-sheet.png',
                    reason='Existing critic-reviewed preview only. Donor proposal ancestry is missing; no rendering payload or ingestion eligibility. Recover donor history and confirm target class before generation.')
            needs_label=eligible(did) and (entry['cause']=='owner-box / review conflict' or entry['estimated_rows']>0 or entry['cause']=='damage / composite / needs box')
            if needs_label:
                col=doc['boxes']['interpreted_lithology_column']
                guesses=entry.get('guess_intervals',[])
                approximate=(proposal['column_boxes'] or [[col[0],int(guesses[0]['top_y']) if guesses else col[1],col[2],min(col[3],int(guesses[0]['top_y'])+160) if guesses else min(col[3],col[1]+160)]])[0]
                owner.append(dict(key=key,doc_id=did,legend_id=lid,label=entry['label'],estimated_rows=entry['estimated_rows'],
                    share_of_column_rows=entry['training_row_share'],approximate_box=approximate,
                    page_fraction=approximate[1]/doc['height'],
                    request='Confirm class in one box spanning complete motifs and missing repeat axes' if proposal['column_boxes'] else
                            'First confirm occurrence; if present label one complete motif box; otherwise absent/unknown decision',
                    position_basis='existing unlabelled column evidence' if proposal['column_boxes'] else 'unverified machine guess' if guesses else 'column start; no located occurrence'))
            proposal['notes']=record.get('reason','Independent review required; no production admission')
            _write(folder/f'legend-{lid}.json',proposal)
            queue.append(dict(key=key,doc_id=did,legend_id=lid,status=proposal['status'],method=record['method'],
                proposal=(folder/f'legend-{lid}.json').relative_to(root).as_posix(),sheet_png=record.get('sheet_png'),
                share_of_column_rows=entry['training_row_share'],estimated_rows=entry['estimated_rows'],
                review='ready' if key in {'4a056c8ff25cf749:5','5709fe9e86ce942b:6','57b1857d99779870:12','b4a8617d27247cbd:18','b4a8617d27247cbd:27'} and (data is not None or historical_review) else
                       'pending motif review' if data is not None else 'needs evidence',
                ready=bool(key in {'4a056c8ff25cf749:5','5709fe9e86ce942b:6','57b1857d99779870:12','b4a8617d27247cbd:18','b4a8617d27247cbd:27'} and (data is not None or historical_review)),
                ancestry_status='unknown' if historical_review else 'declared' if data is not None else 'unresolved',
                load_blocked=historical_review,
                production_eligible=False))
            searches.append(dict(key=key,hits=hits,excluded_target=not eligible(did),missing_axes=record['axes']))
    finally:
        if images.current:
            images.current.close()
    owner.sort(key=lambda e:(-e['share_of_column_rows'],e['key']))
    _write(root/'review-queue.json',dict(schema='recovery-1',total=len(queue),entries=queue,
        training_row_denominator=inventory['total_training_guessed_rows'],nominal_rows=sum(e['estimated_rows'] for e in queue),
        recovered_geological_rows=0,note='Proposals await independent review; machine row weights are not occurrence truth'))
    _write(root/'needs-owner-box.json',dict(entries=owner,position_warning='Approximate positions without class evidence may show another fill'))
    _write(root/'search-results.json',dict(searches=searches,excluded_donors=excluded,fingerprint=fingerprint,
        eligible_donors=len(pool),rerun_when='New source box, donor, annotation, or changed source hash'))
    print('PASS recovery:',json.dumps(dict(total=len(queue),made=sum(e['status']=='made' for e in queue),
          methods=dict(Counter(e['method'] for e in queue)),owner_requests=len(owner),eligible_donors=len(pool))),flush=True)


def main():
    import model
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--doc', help='Document id (omit to explicitly refresh all documents)')
    ap.add_argument('--workers', type=int, default=1, help='Independent document scanner processes (default: 1)')
    ap.add_argument('--recover',action='store_true',help='Build isolated recovery proposals and triage queue')
    args = ap.parse_args()
    if args.recover:
        recover()
        return
    store = model.Store()
    docs, groups = learn.human_examples(store)
    if args.doc and args.doc not in docs:
        ap.error('Unknown document id: ' + args.doc)
    images = model.Images()
    crops = learn.Crops(images)
    prefill = {learn.norm(name): rows for name, rows in score.load_prefill().items()}
    try:
        selected = [(did, info) for did, info in docs.items() if not args.doc or did == args.doc]
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_scan_one, info['doc'],
                           {k:v for k,v in groups.items() if k[0]==did},
                           prefill.get(learn.norm(info['name']), []), info['training_allowed'])
                           for did, info in selected]
                for future in as_completed(futures):
                    report = future.result()
                    print(f'{report["doc_id"]} {report["name"]}: {dict(Counter(e["tier"] for e in report["entries"]))} ({report["seconds"]}s)', flush=True)
        else:
            for did, info in selected:
                report = refresh(info['doc'], crops, groups, prefill.get(learn.norm(info['name']), []),
                                 training_allowed=info['training_allowed'])
                print(f'{did} {info["name"]}: {dict(Counter(e["tier"] for e in report["entries"]))} ({report["seconds"]}s)', flush=True)
        # Resolve donors only after every local report exists: independent of catalog order.
        donors = donor_pool()
        from .build import _write
        for report in list(reports()):
            did = report['doc_id']
            if args.doc and did != args.doc:
                continue
            doc = docs[did]['doc']
            cleaned, textures, learned = {}, {}, {}
            for e in report['entries']:
                lid = e['legend_id']
                cleaned[lid], textures[lid], _ = swatch(crops.get(doc, e['swatch_box']))
                if 'data_file' in e:
                    learned[lid] = load_data(paths.CANDIDATES / e['data_file'])
            borrow(doc, report, learned, cleaned, donors)
            persist_models(doc, report, learned)
            report['sheets'] = sheets(doc, report, learned, cleaned, textures, crops, docs)
            _write(paths.CANDIDATES / (did + '.json'), report)
    finally:
        if images.current:
            images.current.close()
    result = summary()
    print('PASS candidates:', json.dumps(result['overall']), f'documents={len(result["documents"])}', flush=True)


if __name__ == '__main__':
    main()
