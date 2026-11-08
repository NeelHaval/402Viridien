"""Seeded, native-resolution vector lithology renderer. No image/crop reads.

render_pattern(pid, height, width, seed) -> uint8 HxWx3 RGB
render_column({'width': W, 'beds': [{'pid': P, 'height': H}, ...]}, seed)
also accepts explicit contiguous top_y/bottom_y beds. One-row beds are valid.
Statistics contain colour moments, empirical lengths and spectra, never tiles.
"""
import math
import cv2
import numpy as np
from .texstats import load_stats


def sample(stat, rng, default=1, low=.1, high=None):
    values = stat.get('quantiles', [])
    result = float(np.interp(rng.random(), np.linspace(0, 1, len(values)), values)) if values else default
    return float(np.clip(result, low, high if high is not None else max(low, result)))


def motif(stats):
    """Legend topology; native components take precedence for flagged bad labels."""
    label = stats['pid'].split('/')[-1]
    d = stats['horizontal_ink']['median']
    v = stats['vertical_ink']['median']
    component_w = stats['component_width']['median']
    component_h = stats['component_height']['median']
    if stats['ink_share'] < .008 and not stats['horizontal_ink']['n']:
        return 'fill'
    brick_label = 'limestone' in label or 'dolomit' in label
    if stats.get('full_row_hatch') and (not brick_label or stats.get('swatch_colour_conflict')):
        return 'rules'
    if 'd-582' in stats['pid']:
        if label in ('clay', 'shale') and stats.get('source_kind') == 'indexed_crops_unverified' and component_h > max(8, component_w*1.5):
            return 'hooks'
        if label == 'siltstone': return 'ticks'
        if 'sand' in label: return 'dots'
        if 'dolomit' in label: return 'slanted_bricks'
        if 'limestone' in label: return 'bricks'
        if 'halite' in label: return 'hooks'
        if 'anhydr' in label: return 'crosses'
        return 'dashes'
    if stats.get('swatch_colour_conflict'):
        if d >= 7:
            return 'dashes'
        if component_h >= 4 and component_w >= 4:
            return 'hooks'
        return 'dots'
    if 'dolomit' in label:
        return 'slanted_bricks'
    if 'limestone' in label:
        return 'bricks'
    if 'halite' in label:
        return 'hooks'
    if 'sand' in label or label in ('silt', 'silstone'):
        return 'dots'
    if 'siltstone' in label and v > d*1.7:
        return 'ticks'
    if 'tuff' in label or 'volcan' in label:
        return 'chevrons'
    if label == 'claym':
        return 'ticked_dashes'
    return 'dashes'


def _field(height, width, spectrum, rng):
    noise = rng.normal(size=(height, width))
    fy, fx = np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fx*fx+fy*fy)
    bins = np.minimum(15, (radius/.71*16).astype(int))
    gain = np.sqrt(np.maximum(0, np.asarray(spectrum)[bins]))
    noise = np.fft.irfft2(np.fft.rfft2(noise)*gain, s=(height, width)).real
    noise -= noise.mean()
    return noise/max(noise.std(), 1e-9)


def _fill(stats, height, width, rng):
    if len(stats.get('fill_bands', [])) > 1:
        out = np.empty((height, width, 3), np.uint8)
        x = 0
        total = 0.
        for band in stats['fill_bands']:
            total += band['fraction']
            end = min(width, round(total*width))
            if end > x:
                single = dict(stats, fill_bands=[], fill_lab_mean=band['lab_mean'], fill_lab_covariance=band['covariance'])
                out[:, x:end] = _fill(single, height, end-x, rng)
            x = end
        return out
    mean = np.array(stats['fill_lab_mean'], float)
    cov = np.array(stats['fill_lab_covariance'], float)
    eigenvalues, vectors = np.linalg.eigh(cov)
    root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    if np.max(eigenvalues) < .0001:
        lab = np.broadcast_to(mean, (height, width, 3)).copy()
    else:
        fields = np.stack([_field(height, width, stats['noise_spectrum'], rng) for _ in range(3)], -1)
        lab = mean + fields @ root.T
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(rgb*255), 0, 255).astype(np.uint8)


def _stroke(mask, x, y, length, thickness, angle, rng, wobble, roughness):
    n = max(2, int(length/2)+1)
    t = np.linspace(0, length, n)
    # Low amplitude correlated motion changes edges without destroying continuity.
    drift = np.cumsum(rng.normal(0, min(.16, wobble*.2), n))
    xs = x + np.cos(angle)*t - np.sin(angle)*drift
    ys = y + np.sin(angle)*t + np.cos(angle)*drift
    normal = np.array([-np.sin(angle), np.cos(angle)])
    path = np.c_[xs, ys]
    # Flat caps and independently roughened edges match printed dashes better
    # than the round caps supplied by an ordinary stroked polyline.
    edge_noise = rng.normal(0, roughness, (n, 1))
    upper = path+normal*(thickness/2+edge_noise)
    lower = path-normal*(thickness/2+rng.normal(0, roughness, (n, 1)))
    polygon = np.rint(np.r_[upper, lower[::-1]]*4).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)


def _render(stats, height, width, rng, borders=True):
    if height <= 0 or width <= 0:
        raise ValueError('height and width must be positive')
    height, width = int(height), int(width)
    if stats.get('variants'):
        weights = np.array([v['pixels'] for v in stats['variants']], float)
        weights /= weights.sum()
        variant = stats['variants'][int(rng.choice(len(weights), p=weights))]
        conflict = np.linalg.norm(np.array(variant['fill_lab_mean'])-stats['swatch_stats']['fill_lab_mean']) > 25
        stats = dict(stats, **variant)
        stats['swatch_colour_conflict'] = bool(conflict)
    rgb = _fill(stats, height, width, rng)
    style = motif(stats)
    mask = np.zeros((height*4, width*4), np.uint8)
    dash_stat = stats['dash_core'] if stats.get('dash_core', {}).get('n') else stats['horizontal_ink']
    dash = max(1., dash_stat['median'])
    gap = max(1., stats['horizontal_gap']['median'])
    spacing = max(2., stats['row_spacing']['median'] or stats['vertical_gap']['median']+1)
    thick = max(.65, stats['vertical_ink']['median'])
    if style == 'ticks':
        thick = max(.6, stats['horizontal_ink']['median']*.8)
        length = max(2., stats['vertical_ink']['median'])
        pitch = max(thick+1, dash+gap)
    elif style == 'dots':
        thick = max(.7, stats['vertical_ink']['median'] or dash)
        length, pitch = thick, dash+gap
    else:
        # Horizontal run lengths count the core of an antialiased stroke.
        thick = max(.65, min(thick, spacing*.55))
        length, pitch = dash, dash+gap
        if style in ('hooks', 'crosses', 'bricks', 'slanted_bricks'):
            thick = min(thick, max(.65, stats['horizontal_ink']['median']))
    phase = rng.uniform(0, pitch)
    y = rng.uniform(-spacing, 0)
    row = 0
    wobble = stats['wobble']['median']
    roughness = .13 if stats['scan_mode'] == '1-bit' else .025
    core_histogram = dash_stat.get('histogram', {})
    short_core_fraction = stats.get('short_core_fraction',
        sum(n for value, n in core_histogram.items() if float(value) < dash*.35)/max(1, dash_stat['n']))
    while style != 'fill' and y < height+spacing:
        stagger = stats['stagger']['median'] if stats['stagger']['n'] else pitch/2
        if style == 'dots' and abs(stagger) < .5: stagger = pitch/2
        stagger += rng.normal(0, .12)
        phase = (phase+stagger) % pitch
        x = -pitch+phase
        row_jitter = rng.normal(0, min(.15, stats['row_spacing']['std']*.02))
        while x < width+pitch:
            if style == 'ticks':
                line_len = sample(stats['vertical_ink'], rng, length, max(1, length*.65), length*1.4)
                line_t = sample(stats['horizontal_ink'], rng, thick, max(.65, thick*.7), thick*1.3)
                _stroke(mask, x, y+row_jitter, line_len, line_t*.82, math.pi/2, rng, wobble, roughness)
                dx = line_t+sample(stats['horizontal_gap'], rng, gap, max(1, gap*.5), gap*1.7)
            elif style == 'dots':
                diameter = sample(stats['horizontal_ink'], rng, dash, .8, max(1., dash*1.5))
                if stats.get('channel_binary') and stats['scan_mode'] != '1-bit':
                    cv2.rectangle(mask, (round((x-diameter/2)*4), round((y+row_jitter-thick/2)*4)),
                                  (round((x+diameter/2)*4), round((y+row_jitter+thick/2)*4)), 255, -1)
                else:
                    cv2.ellipse(mask, (round(x*4), round((y+row_jitter)*4)),
                                (max(1, round(diameter*2)), max(1, round(thick*2))),
                                float(rng.uniform(-20, 20)), 0, 360, 255, -1)
                dx = diameter+sample(stats['horizontal_gap'], rng, gap, max(1, gap*.5), gap*1.8)
            else:
                line_len = sample(dash_stat, rng, length, max(1, length*.65), length*1.5)
                thickness_stat = stats['horizontal_ink'] if style in ('hooks', 'crosses') else stats['vertical_ink']
                line_t = sample(thickness_stat, rng, thick, max(.65, thick*.6), thick*1.3)
                line_t = min(line_t, spacing*.55)*.93
                if style == 'rules':
                    _stroke(mask, -4, y, width+8, line_t, 0, rng, wobble, 0)
                    break
                elif style in ('bricks', 'slanted_bricks'):
                    _stroke(mask, x, y, pitch, line_t, 0, rng, wobble, 0)
                    _stroke(mask, x, y, spacing, line_t, math.pi/2 if style == 'bricks' else 2.1, rng, wobble, 0)
                elif style == 'hooks':
                    size = max(2, stats['component_width']['median']-line_t)
                    _stroke(mask, x, y, size, line_t, 0, rng, wobble, 0)
                    _stroke(mask, x, y, max(2, stats['component_height']['median']-line_t), line_t, -math.pi/2, rng, wobble, 0)
                    line_len = size
                elif style == 'chevrons':
                    _stroke(mask, x, y, line_len, line_t, -.55, rng, wobble, 0)
                    _stroke(mask, x+line_len*.85, y-line_len*.5, line_len, line_t, .55, rng, wobble, 0)
                elif style == 'crosses':
                    _stroke(mask, x, y, line_len, line_t, 0, rng, wobble, 0)
                    _stroke(mask, x+line_len*.5, y-spacing*.2, spacing*.4, line_t, math.pi/2, rng, wobble, 0)
                else:
                    _stroke(mask, x, y, line_len, line_t, 0, rng, wobble, roughness)
                    if style == 'ticked_dashes':
                        _stroke(mask, x+line_len*.3, y, max(1.2, spacing*.35), max(.65, line_t*.6), -math.pi/2, rng, wobble, 0)
                    if line_len > 6 and rng.random() < min(.08, short_core_fraction):
                        cut = round((x+line_len*rng.uniform(.2, .8))*4)
                        cv2.rectangle(mask, (cut, round((y-line_t)*4)),
                                      (cut+max(1, round(line_t)), round((y+line_t)*4)), 0, -1)
                dx = line_len+sample(stats['horizontal_gap'], rng, gap, max(1, gap*.5), gap*1.8)
            # A shared lattice preserves row-to-row alignment; run/gap draws
            # vary stroke extent with only slow drift of the printing lattice.
            x += max(1, .75*pitch+.25*dx + rng.normal(0, min(.35, gap*.025)))
        # A continuous lattice with small row jitter, independently clipped by beds.
        y += spacing + rng.normal(0, min(.22, stats['row_spacing']['std']*.025))
        row += 1
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA).astype(float)/255
    if stats['scan_mode'] == '1-bit':
        alpha = (alpha > .35).astype(float)
    elif stats.get('hard_edges'):
        alpha = (alpha > .55).astype(float)
    else:
        # Scanner dropout and grain act on generated strokes, never sampled ink.
        rough = min(.09, stats['wobble']['mean']*.015)
        alpha *= np.clip(1+rng.normal(0, rough, alpha.shape), 0, 1)
    ink = cv2.cvtColor(np.array([[stats['ink_lab_mean']]], np.float32), cv2.COLOR_LAB2RGB)[0, 0]*255
    rgb = np.rint(rgb*(1-alpha[..., None])+ink*alpha[..., None]).astype(np.uint8)
    if borders:
        _borders(rgb, stats, rng)
    return _encode(rgb, stats)


def _encode(rgb, stats):
    if stats['scan_mode'] == '1-bit':
        value = (rgb.mean(2) >= 180).astype(np.uint8)*255
        return np.repeat(value[..., None], 3, 2)
    if stats.get('channel_binary'):
        return (rgb >= 128).astype(np.uint8)*255
    return rgb


def _borders(rgb, stats, rng):
    for edge in ('left', 'right'):
        width = round(stats['borders'][edge])
        if width:
            if edge == 'left': rgb[:, :width] = 0
            else: rgb[:, -width:] = 0


def render_pattern(pid, height, width, seed, stats=None):
    stats = stats if stats is not None else load_stats()['patterns'].get(pid)
    if stats is None:
        raise KeyError(f'No measured native texture for {pid}')
    return _render(stats, height, width, np.random.default_rng(seed))


def finish_column(rgb, beds, seed, grid=True, borders=True, models=None):
    """Apply measured contact rules and family rails; no blending across beds."""
    rng = np.random.default_rng(seed)
    data = load_stats()
    patterns = data['patterns'] if models is None else models
    contacts = []
    for bed in beds[1:]:
        if not bed.get('pid') or bed['pid'] not in patterns:
            continue
        stats = patterns[bed['pid']]
        boundary = stats['boundary']
        y = int(bed['top_y'])
        if rng.random() < boundary['line_probability']:
            t = max(1, round(sample(boundary['line_thickness'], rng)))
            rgb[y:min(len(rgb), y+t)] = 0
            contacts.append(dict(row=y, thickness=t))
    first = next((b for b in beds if b.get('pid') in patterns), None)
    if first:
        stats = patterns[first['pid']]
        family = data['families'][stats['family']].get('structure', data['families'][stats['family']]['texture']) if models is None else stats.get('structure', stats)
        period = family['grid_spacing']['median']
        if grid and period > 6:
            thickness = max(1, round(family['grid_thickness']['median']))
            for y in range(int(rng.integers(max(1, round(period)))), len(rgb), max(1, round(period))):
                rgb[y:y+thickness] = 0
        if borders:
            _borders(rgb, stats, rng)
        rgb = _encode(rgb, stats)
    return rgb, contacts


def render_column(spec, seed):
    width = int(spec['width'])
    beds, cursor = [], 0
    for item in spec['beds']:
        bed = dict(item)
        top = int(bed.get('top_y', cursor))
        bottom = int(bed.get('bottom_y', top+int(bed.get('height', 0))))
        if top != cursor or bottom <= top:
            raise ValueError('Beds must be positive, contiguous and start at row zero')
        beds.append(dict(bed, top_y=top, bottom_y=bottom))
        cursor = bottom
    if not beds or width <= 0:
        raise ValueError('A column needs beds and a positive width')
    if 'height' in spec and int(spec['height']) != cursor:
        raise ValueError('height does not match bed extents')
    rng = np.random.default_rng(seed)
    canvas = np.full((cursor, width, 3), 255, np.uint8)
    for bed in beds:
        if bed.get('pid'):
            stats = load_stats()['patterns'][bed['pid']]
            canvas[bed['top_y']:bed['bottom_y']] = _render(stats, bed['bottom_y']-bed['top_y'], width, rng, borders=False)
    return finish_column(canvas, beds, int(rng.integers(2**32)), spec.get('grid', True), spec.get('borders', True))[0]
