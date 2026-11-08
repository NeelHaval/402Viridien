"""Measure native lithology textures without changing dataset files.

python -B -m synth.texstats
All distances are native pixels; Lab uses CIE L*=0..100, a*, b*.
Only scalar distributions and spectra are saved, never source pixel templates.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import csv
import hashlib
import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks
from .common import HOME, ROOT, ANALYSIS, read_json, write_json, local_file, norm


def lab_image(rgb):
    return cv2.cvtColor(np.asarray(rgb, np.float32) / 255, cv2.COLOR_RGB2LAB)


def ink_mask(rgb):
    light = lab_image(rgb)[..., 0]
    background = float(np.percentile(light, 85))
    mask = light < min(65., background - 18.)
    if mask.mean() < .003 and np.std(light.mean(1)) > max(.8, light.std()*.4):
        threshold, _ = cv2.threshold(np.clip(light*2.55, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        mask = light < threshold/2.55+.4
    return mask


def runs(line, value=True, censored=False):
    line = np.asarray(line, bool) == value
    edges = np.diff(np.r_[False, line, False].astype(int))
    a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if censored:
        keep = (a > 0) & (b < len(line))
        a, b = a[keep], b[keep]
    return (b - a).tolist()


def distribution(values):
    v = np.asarray(values, float)
    if not len(v):
        return dict(n=0, median=0., p90=0., mean=0., std=0., quantiles=[], histogram={})
    keys, counts = np.unique(np.round(v, 3), return_counts=True)
    return dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                mean=float(v.mean()), std=float(v.std()),
                quantiles=np.quantile(v, np.linspace(0, 1, 33)).tolist(),
                histogram={str(k): int(c) for k, c in zip(keys, counts)})


def dominant_fill(values):
    """Reject ink fringes and adjacent-bed contamination from fill moments."""
    if len(values) < 8: return values
    bins = np.floor(values/5).astype(np.int64)
    codes = bins[:, 0]*65536+(bins[:, 1]+128)*256+bins[:, 2]+128
    unique, counts = np.unique(codes, return_counts=True)
    center = np.median(values[codes == unique[counts.argmax()]], axis=0)
    keep = np.linalg.norm(values-center, axis=1) < 14
    return values[keep] if keep.mean() > .5 else values


def row_centers(mask):
    profile = mask.mean(1).astype(float)
    if profile.max(initial=0) < .02:
        return np.array([], int)
    # Dense horizontal rules are measured separately, not mistaken for dash rows.
    profile[profile > .92] = 0
    smoothed = cv2.GaussianBlur(profile[:, None], (1, 3), .55)[:, 0]
    peaks, _ = find_peaks(smoothed, prominence=max(.015, smoothed.max() * .12), distance=2)
    return peaks


def texture_stats(rgb):
    rgb = np.asarray(rgb, np.uint8)
    mask_full = ink_mask(rgb)
    h, w = mask_full.shape
    margin = max(1, round(w * .08))
    occ = mask_full.mean(0)
    left = np.flatnonzero(occ[:margin] > .78)
    right = np.flatnonzero(occ[-margin:] > .78)
    lo = int(left[-1] + 1) if len(left) else 0
    hi = w - (int(margin - right[0]) if len(right) else 0)
    if hi - lo < w * .65:
        lo, hi = 0, w
    rgb = rgb[:, lo:hi]
    mask = mask_full[:, lo:hi]
    rules = mask.mean(1) > .92
    starts = np.flatnonzero(rules & np.r_[True, ~rules[:-1]])
    full_row_hatch = bool(rules.mean() > .10 or (len(starts) >= 3 and np.median(np.diff(starts)) < 24))
    hatch_rows = rules.copy() if full_row_hatch else np.zeros(h, bool)
    if full_row_hatch:
        rules[:] = False
    clean = mask.copy()
    clean[rules] = False
    horizontal = [v for row in clean for v in (runs(row, True, True) if not row.all() else [len(row)])]
    gaps = [v for row in clean if row.any() for v in runs(row, False, True)]
    vertical = [v for col in clean.T for v in runs(col, True, True)]
    vgap = [v for col in clean.T if col.any() for v in runs(col, False, True)]
    centers = row_centers(clean)
    if full_row_hatch:
        edges = np.diff(np.r_[False, hatch_rows, False].astype(int))
        centers = ((np.flatnonzero(edges == 1)+np.flatnonzero(edges == -1)-1)/2).round().astype(int)
    spacing = np.diff(centers).tolist()
    core_runs = [v for y in centers for v in runs(clean[y], True, True)]
    stagger, wobble, ragged = [], [], []
    for y0, y1 in zip(centers, centers[1:]):
        a = clean[max(0, y0-1):y0+2].mean(0)
        b = clean[max(0, y1-1):y1+2].mean(0)
        if a.std() > .01 and b.std() > .01:
            n = min(hi-lo-1, max(2, round(np.median(horizontal or [4]) + np.median(gaps or [4]))))
            shifts = np.arange(-n//2, n//2+1)
            scores = [np.dot(a-a.mean(), np.roll(b-b.mean(), int(s))) for s in shifts]
            stagger.append(float(-shifts[int(np.argmax(scores))]))
    # Row-local ink centroid motion and edge-count variation quantify roughness.
    for y in centers:
        near = clean[max(0, y-3):min(h, y+4)]
        yy, xx = np.nonzero(near)
        if len(xx) > 4:
            count = np.bincount(xx, minlength=rgb.shape[1])
            slopes = (np.bincount(xx, weights=yy, minlength=rgb.shape[1])[count > 0]/count[count > 0]).tolist()
            lower, upper = np.full(rgb.shape[1], len(near)), np.zeros(rgb.shape[1])
            np.minimum.at(lower, xx, yy)
            np.maximum.at(upper, xx, yy)
            ragged.extend((upper[count > 0]-lower[count > 0]+1).tolist())
            if len(slopes) > 1:
                wobble.extend(np.abs(np.diff(slopes)).tolist())
    lab = lab_image(rgb)
    safe_fill = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3) > 1.4
    if safe_fill.sum() < max(8, mask.size*.025): safe_fill = ~mask
    fill = lab[safe_fill]
    ink = lab[mask]
    if not len(fill):
        fill = lab.reshape(-1, 3)
    fill = dominant_fill(fill)
    mu = fill.mean(0)
    covariance = np.cov(fill.T) if len(fill) > 1 else np.zeros((3, 3))
    residual = lab[..., 0] - mu[0]
    residual[~safe_fill] = 0
    ps = np.abs(np.fft.rfft2(residual - residual.mean())) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(rgb.shape[1])[None, :]
    radius = np.sqrt(fx*fx + fy*fy)
    spectrum = [float(ps[(radius >= a) & (radius < b)].mean())
                if np.any((radius >= a) & (radius < b)) else 0.
                for a, b in zip(np.linspace(0, .71, 17)[:-1], np.linspace(0, .71, 17)[1:])]
    total = sum(spectrum)
    spectrum = [v/total if total else 0. for v in spectrum]
    smooth = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), 3)
    corr = []
    for axis in (0, 1):
        a, b = np.take(residual, range(residual.shape[axis]-1), axis), np.take(residual, range(1, residual.shape[axis]), axis)
        corr.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > .001 and b.std() > .001 else 0.)
    rule_centers = np.flatnonzero(rules & np.r_[True, ~rules[:-1]])
    binary = bool(np.all((rgb == 0) | (rgb == 255)) and np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))
    components = cv2.connectedComponentsWithStats(clean.astype(np.uint8), 8)[2][1:]
    components = components[(components[:, 4] >= 2) & (components[:, 2] < max(3, rgb.shape[1]*.9))]
    # A colour scan may contain multiple population/rock proportions. Keep
    # sample appearances separate so generation does not average red and green.
    # A mud-log column can show proportions of two rocks side by side. Model
    # those as colour bands, not a Gaussian blend of unrelated fill colours.
    column_fill = []
    for x in range(rgb.shape[1]):
        vals = lab[:, x][safe_fill[:, x]]
        column_fill.append(np.median(vals, 0) if len(vals) else mu)
    column_fill = np.array(column_fill)
    if len(column_fill) >= 5:
        column_fill = cv2.medianBlur(column_fill.astype(np.float32)[None], 5)[0]
    cuts = np.flatnonzero(np.linalg.norm(np.diff(column_fill, axis=0), axis=1) > 18)+1
    edges = [0]+cuts.tolist()+[rgb.shape[1]]
    bands = []
    if len(edges) > 10: edges = [0, rgb.shape[1]]
    for a, b in zip(edges, edges[1:]):
        values = lab[:, a:b][safe_fill[:, a:b]]
        if len(values) < 2: values = fill
        values = dominant_fill(values)
        bands.append(dict(fraction=(b-a)/rgb.shape[1], lab_mean=values.mean(0).tolist(),
                          covariance=(np.cov(values.T) if len(values)>1 else np.zeros((3, 3))).tolist()))
    return dict(height=h, width=w, pixels=h*w, horizontal_ink=distribution(horizontal),
                horizontal_gap=distribution(gaps), dash_core=distribution(core_runs), vertical_ink=distribution(vertical),
                vertical_gap=distribution(vgap), row_spacing=distribution(spacing),
                stagger=distribution(stagger), wobble=distribution(wobble),
                edge_raggedness=distribution(ragged), ink_share=float(clean.mean()),
                lab_mean=lab.mean((0, 1)).tolist(), fill_lab_mean=mu.tolist(),
                fill_lab_covariance=covariance.tolist(), ink_lab_mean=(ink.mean(0) if len(ink) else np.zeros(3)).tolist(),
                noise_spectrum=spectrum, noise_correlation=corr, fade_std=float(smooth.std()),
                scan_mode='1-bit' if binary else ('greyscale' if np.max(np.ptp(rgb.astype(int), axis=2)) < 3 else 'colour'),
                antialias_share=float(((lab[..., 0] > 5) & (lab[..., 0] < 95)).mean()),
                borders=dict(left=lo, right=w-hi), grid_spacing=distribution(np.diff(rule_centers)),
                grid_thickness=distribution(runs(rules)),
                fill_bands=bands, full_row_hatch=full_row_hatch,
                short_core_fraction=float(np.mean(np.array(core_runs) < np.median(core_runs)*.35)) if core_runs else 0.,
                channel_binary=bool(np.all((rgb == 0) | (rgb == 255))),
                hard_edges=bool(len(np.unique(rgb[..., 0].astype(np.int32)*65536+rgb[..., 1].astype(np.int32)*256+rgb[..., 2])) <= 16),
                component_width=distribution(components[:, 2] if len(components) else []),
                component_height=distribution(components[:, 3] if len(components) else []))


def aggregate(records):
    """Pool empirical distributions; retain within/between-sample colour variance."""
    if not records:
        return None
    out = dict(records[0])
    weights = np.array([r['pixels'] for r in records], float)
    weights /= weights.sum()
    for key, value in out.copy().items():
        if isinstance(value, dict) and 'histogram' in value:
            histogram = defaultdict(int)
            for r in records:
                for k, n in r[key]['histogram'].items():
                    histogram[k] += n
            vals = np.repeat([float(k) for k in histogram], list(histogram.values()))
            out[key] = distribution(vals)
        elif isinstance(value, (list, int, float)) and key not in ('pixels', 'fill_bands', 'full_row_hatch', 'channel_binary', 'hard_edges'):
            out[key] = np.average([r[key] for r in records], axis=0, weights=weights).tolist()
    means = np.array([r['fill_lab_mean'] for r in records])
    delta = means - out['fill_lab_mean']
    out['fill_lab_covariance'] = np.average([r['fill_lab_covariance'] for r in records], axis=0, weights=weights).__add__(
        np.einsum('n,ni,nj->ij', weights, delta, delta)).tolist()
    out['borders'] = {k: float(np.average([r['borders'][k] for r in records], weights=weights)) for k in ('left', 'right')}
    out['scan_mode'] = max(set(r['scan_mode'] for r in records), key=lambda m: sum(w for r, w in zip(records, weights) if r['scan_mode'] == m))
    out['pixels'] = sum(r['pixels'] for r in records)
    out['samples'] = len(records)
    return out


def boundary_stats(rgb, y):
    """10–90% profile of a local two-texture feature projection, in native rows.

    Texture darkness and colour are combined. Indistinguishable contacts return
    unresolved=True instead of claiming a measured zero-width contact.
    """
    mask = ink_mask(rgb)
    margin = max(1, rgb.shape[1]//12)
    lab = lab_image(rgb[:, margin:-margin])
    features = np.c_[lab.mean(1), mask[:, margin:-margin].mean(1)*100]
    n = len(rgb)
    left, right = np.median(features[:max(1, y//2)], 0), np.median(features[min(n-1, y+(n-y)//2):], 0)
    direction = right-left
    denom = float(direction @ direction)
    unresolved = denom < 4
    profile = np.clip((features-left) @ direction/max(denom, 1e-9), 0, 1)
    # Search around the annotated row, allowing small annotation offsets.
    a, b = max(0, y-8), min(n, y+9)
    widths = []
    for center in range(a, b):
        lows = np.flatnonzero(profile[max(0, center-8):center+1] <= .1)
        highs = np.flatnonzero(profile[center:min(n, center+9)] >= .9)
        if len(lows) and len(highs):
            widths.append(center+int(highs[0]) - (max(0, center-8)+int(lows[-1])))
    width = min(widths) if widths and not unresolved else None
    density = mask[:, margin:-margin].mean(1)
    local = density[max(0, y-3):min(n, y+4)] > .88
    line_runs = runs(local)
    # A contact cuts strokes when ink touches the contact and has no matching
    # continuation on the opposite side (exclude full-width contact rules).
    cut = np.logical_xor(mask[max(0, y-1)], mask[min(n-1, y)]).mean()
    return dict(profile_width=width, unresolved=unresolved, drawn_line=bool(line_runs),
                line_thickness=max(line_runs, default=0), cut_share=float(cut))


def trusted_intervals():
    manifest = read_json(ANALYSIS/'mix_manifest.json', {})
    sources = {}
    for path in sorted((ROOT/'labelling/out').glob('*/consensus.csv')):
        name = manifest.get(path.parent.name, {}).get('log')
        if not name:
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = [dict(r, source='consensus') for r in csv.DictReader(stream) if r.get('agreed', '').lower() == 'true']
        sources[name] = rows
    for path in sorted((ROOT/'gt/gold').glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            sources[path.stem] = [dict(r, source='gold') for r in csv.DictReader(stream)]
    for name, rows in sources.items():
        for row in rows:
            row['top_y'], row['bottom_y'] = int(float(row['top_y'])), int(float(row['bottom_y']))
        rows.sort(key=lambda r: r['top_y'])
    return sources


def boundary_summary(values, heights=()):
    widths = [v['profile_width'] for v in values if v['profile_width'] is not None]
    return dict(n=len(values), resolved=len(widths), profile_width=distribution(widths),
                line_probability=float(np.mean([v['drawn_line'] for v in values])) if values else 0.,
                line_thickness=distribution([v['line_thickness'] for v in values if v['drawn_line']]),
                cut_share=distribution([v['cut_share'] for v in values]),
                sliver_frequency=float(np.mean(np.asarray(heights) <= 6)) if len(heights) else 0.,
                bed_height=distribution(heights))


def trim_native(rgb):
    """Keep a stable fill stretch when an annotation includes a contact fringe."""
    if len(rgb) < 24: return rgb, 0
    lab, mask = lab_image(rgb), ink_mask(rgb)
    fallback = np.median(lab[~mask], 0) if (~mask).any() else np.median(lab.reshape(-1, 3), 0)
    rows = np.array([np.median(row[~m], 0) if (~m).any() else fallback for row, m in zip(lab, mask)])
    center = np.median(rows, 0)
    stable = np.linalg.norm(rows-center, axis=1) < 20
    edge = np.diff(np.r_[False, stable, False].astype(int))
    a, b = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    if not len(a): return rgb, 0
    chosen = int(np.argmax(b-a))
    if b[chosen]-a[chosen] >= max(12, len(rgb)*.4):
        return rgb[a[chosen]:b[chosen]], int(a[chosen])
    return rgb, 0


def structure_stats(rgb):
    mask = ink_mask(rgb)
    h, w = mask.shape
    margin = max(1, round(w*.08))
    occupancy = mask.mean(0)
    left, right = np.flatnonzero(occupancy[:margin] > .8), np.flatnonzero(occupancy[-margin:] > .8)
    dense = mask[:, margin:-margin].mean(1) > .94
    starts = np.flatnonzero(dense & np.r_[True, ~dense[:-1]])
    distances = np.diff(starts)
    # Frequent full-width hatches must not become a depth grid. A grid
    # requires repeated, widely separated rules in a long native strip.
    gaps = distances[(distances >= 40) & (distances <= 800)]
    return dict(borders=dict(left=int(left[-1]+1) if len(left) else 0,
                             right=int(margin-right[0]) if len(right) else 0),
                grid_spacing=distribution(gaps), grid_thickness=distribution(runs(dense)), rows=h)


def structure_summary(records):
    return dict(borders={k: float(np.median([r['borders'][k] for r in records])) if records else 0.
                         for k in ('left', 'right')},
                grid_spacing=distribution([v for r in records for v in r['grid_spacing']['quantiles']]),
                grid_thickness=distribution([v for r in records for v in r['grid_thickness']['quantiles']]))


def build(output=HOME/'data/texstats.json'):
    from harness import io
    catalog = read_json(HOME/'data/catalog.json')['patterns']
    index = read_json(HOME/'data/sheets/index.json')
    input_files = sorted({c['file'] for v in index.values() for c in v.get('candidates', {}).values()})
    inputs = dict(index_sha256=hashlib.sha256(__import__('json').dumps(index, sort_keys=True).encode()).hexdigest(),
                  crop_files=input_files, crop_patterns=sum(bool(v.get('candidates')) for v in index.values()),
                  crop_entries=sum(len(v.get('candidates', {})) for v in index.values()))
    intervals = trusted_intervals()
    patterns = {}
    family_records, family_bounds, family_heights = defaultdict(list), defaultdict(list), defaultdict(list)
    lookup = {(name, norm(p['label'])): p for p in catalog for name in p['logs']}
    native, contacts, structures = defaultdict(list), defaultdict(list), defaultdict(list)
    errors = []
    crop_cache = {}
    for name, rows in intervals.items():
        try:
            log = io.open_log(name)
            try:
                box = log.boxes['interpreted_lithology_column']
                p = next((p for p in catalog if name in p['logs']), None)
                if p and rows:
                    start = max(box[1], rows[0]['top_y'])
                    end = min(box[3], start+4000)
                    if end > start:
                        sample = io.read_rgb(log, box, start, end)
                        structures[p['family']].append(dict(log=name, top_y=start, bottom_y=end, **structure_stats(sample)))
                for row in rows:
                    p = lookup.get((name, norm(row['lithology'])))
                    if p is None:
                        continue
                    pid = p['pid']
                    family_heights[p['family']].append(row['bottom_y']-row['top_y'])
                    if len(native[pid]) < 6 and row['bottom_y']-row['top_y'] >= 8:
                        inset = min(3, (row['bottom_y']-row['top_y'])//5)
                        top = max(box[1], row['top_y']+inset)
                        bottom = min(box[3], row['bottom_y']-inset, top+240)
                        if bottom-top >= 4:
                            rgb = io.read_rgb(log, box, top, bottom)
                            rgb, inset_rows = trim_native(rgb)
                            top += inset_rows
                            bottom = top+len(rgb)
                            native[pid].append((texture_stats(rgb), dict(log=name, top_y=top, bottom_y=bottom, source=row['source'])))
                for left, right in zip(rows, rows[1:]):
                    if right['top_y']-left['bottom_y'] > 6 or norm(left['lithology']) == norm(right['lithology']):
                        continue
                    p = lookup.get((name, norm(left['lithology'])))
                    q = lookup.get((name, norm(right['lithology'])))
                    if p is None or q is None:
                        continue
                    y = right['top_y']
                    top, bottom = max(box[1], y-24), min(box[3], y+24)
                    if top >= y or bottom <= y:
                        continue
                    rgb = io.read_rgb(log, box, top, bottom)
                    measure = boundary_stats(rgb, y-top)
                    record = dict(log=name, top_y=top, bottom_y=bottom, boundary_y=y, left_pid=p['pid'], right_pid=q['pid'],
                                  source=left['source'], **measure)
                    contacts[p['pid']].append(record)
                    contacts[q['pid']].append(record)
                    family_bounds[p['family']].append(record)
            finally:
                log.close()
        except (OSError, ValueError, KeyError) as exc:
            errors.append(dict(log=name, error=str(exc)))
    for pattern_index, p in enumerate(catalog):
        pid = p['pid']
        crops = list(index.get(pid, {}).get('candidates', {}).values())
        records = []
        for crop in crops:
            file = crop['file']
            if file not in crop_cache:
                with Image.open(local_file(file)) as image:
                    crop_cache[file] = texture_stats(np.asarray(image.convert('RGB')))
            records.append(crop_cache[file])
        if not records and not native[pid]:
            continue
        native_stats = [v[0] for v in native[pid]]
        # Trusted intervals override embedding-only candidates.
        measured = aggregate(native_stats or records)
        sw = p['swatches'][0]
        swfile = sw.get('native_file') or sw['file']
        with Image.open(local_file(swfile)) as image:
            swstats = texture_stats(np.asarray(image.convert('RGB')))
        color_distance = float(np.linalg.norm(np.array(swstats['fill_lab_mean'])-measured['fill_lab_mean']))
        measured.update(pid=pid, label=p['label'], family=p['family'], share=p['share'],
                        source_kind='labelled_native' if native_stats else 'indexed_crops_unverified',
                        crop_stats=aggregate(records), native_stats=aggregate(native_stats),
                        swatch_stats=swstats, swatch_file=swfile,
                        swatch_colour_conflict=color_distance > 25, swatch_delta_lab=color_distance,
                        variants=native_stats or records,
                        variant_sources=[v[1] for v in native[pid]] if native_stats else crops,
                        sources=[dict(c, kind='crop') for c in crops]+[dict(v[1], kind='native') for v in native[pid]],
                        boundaries=contacts[pid], boundary=boundary_summary(contacts[pid]))
        patterns[pid] = measured
        family_records[p['family']].extend(native_stats or records)
        if (pattern_index+1) % 50 == 0:
            print(f'Measured {pattern_index+1}/{len(catalog)} catalogue patterns; {len(crop_cache)} distinct crops', flush=True)
    families = {}
    for family, records in family_records.items():
        families[family] = dict(texture=aggregate(records), boundary=boundary_summary(family_bounds[family], family_heights[family]),
                                structure=structure_summary(structures[family]), structure_sources=structures[family])
    for p in patterns.values():
        if not p['boundary']['n']:
            p['boundary'] = dict(families[p['family']]['boundary'], fallback='family')
    output_data = dict(version=1, units='native pixels; CIE Lab D65', seed=0,
                       methodology='Censored interior runs; row-peak spacing; adjacent-row cyclic phase; radial fill PSD; local 10–90 contact projection. Gold supersedes consensus per log. No search-pool pseudo labels.',
                       patterns=patterns, families=families, errors=errors, inputs=inputs,
                       limitations=['Indexed mining crops are unverified; colour conflicts with legends are flagged.',
                                    'Boundaries with indistinguishable feature profiles remain unresolved.',
                                    'Family boundary fallbacks are explicit; no unobserved contact is claimed measured.'])
    write_json(output, output_data)
    print(f'{len(patterns)} patterns, {len(families)} families; {sum(len(v) for v in family_bounds.values())} boundaries; {len(errors)} errors')
    return output_data


@lru_cache(maxsize=1)
def load_stats():
    data = read_json(HOME/'data/texstats.json')
    if not data:
        raise FileNotFoundError('Run python -B -m synth.texstats first')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=HOME/'data/texstats.json')
    build(parser.parse_args().output)
