"""Independent nuisance draws, explicit parameters, and row-preserving skew."""
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

PROBABILITIES = dict(column_borders=.65, depth_grid=.55, blank_gaps=.18,
    text_overlay=.20, annotation_overlay=.15, symbol_overlay=.15,
    gradational_boundaries=.35, interbedded=.20, boundary_lines=.55,
    colour_jitter=.55, fading=.35, binarisation=1., scan_noise=.45,
    jpeg=.20, lzw=.20, skew=.25)


def draw_nuisances(rng, clean=False, mono=False, overrides=None):
    probabilities = dict(PROBABILITIES)
    probabilities['binarisation'] = 1. if mono else 0.
    probabilities.update(overrides or {})
    if any(not 0 <= float(p) <= 1 for p in probabilities.values()):
        raise ValueError('Nuisance probabilities must lie in [0,1]')
    return {name: dict(probability=float(p), selected=bool(rng.random() < p) and
                      (not clean or name == 'binarisation'), applied=False)
            for name, p in probabilities.items()}


def apply_nuisances(rgb, beds, assets, flags, rng):
    rgb = rgb.copy()
    h, w = rgb.shape[:2]
    for name in ('text_overlay', 'annotation_overlay', 'symbol_overlay'):
        flag = flags[name]
        if not flag['selected']:
            continue
        candidates = assets['overlays']
        if name == 'text_overlay':
            candidates = [v for v in candidates if v[0].shape[1] >= v[0].shape[0]]
        elif name == 'symbol_overlay':
            candidates = [v for v in candidates if .4 < v[0].shape[1] / v[0].shape[0] < 2]
        if not candidates:
            flag['reason'] = 'No suitable connected ink component in donor non-lithology strip'
            continue
        flag['placements'] = []
        for _ in range(int(rng.integers(1, 5))):
            patch, box = candidates[int(rng.integers(len(candidates)))]
            ph, pw = patch.shape[:2]
            if pw > w or ph > h:
                continue
            y, x = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
            mask = patch.min(axis=2) < 210
            rgb[y:y + ph, x:x + pw][mask] = patch[mask]
            flag['placements'].append(dict(log=assets['name'], source_box=box, box=[x, y, x + pw, y + ph]))
        flag['applied'] = bool(flag['placements'])
    flag = flags['gradational_boundaries']
    if flag['selected']:
        ranges = []
        for left, right in zip(beds, beds[1:]):
            if not left['lithology'] or not right['lithology'] or rng.random() > .4:
                continue
            n = min(int(rng.integers(4, 31)), left['bottom_y'] - left['top_y'],
                    right['bottom_y'] - right['top_y'])
            if n < 4:
                continue
            y = right['top_y']
            lo = y - n // 2
            old = rgb[lo:lo + n].copy()
            other = np.concatenate([rgb[y:y + n // 2], rgb[y - (n - n // 2):y]])
            p = np.linspace(0, 1, n)[:, None]
            # Ordered probability on the two sides, with a stochastic dither.
            chance = np.where(np.arange(n)[:, None] < n // 2, p, 1 - p)
            mask = rng.random((n, w)) < chance
            rgb[lo:lo + n] = np.where(mask[..., None], other, old)
            ranges.append([lo, lo + n])
        flag.update(applied=bool(ranges), ranges=ranges, truth='nominal contact center')
    flag = flags['boundary_lines']
    if flag['selected']:
        ys = [b['top_y'] for b in beds[1:] if rng.random() < .45]
        for y in ys:
            rgb[y:y + 1] = rng.integers(25, 100)
        flag.update(applied=bool(ys), rows=ys, per_boundary_probability=.45)
    flag = flags['depth_grid']
    if flag['selected']:
        k = max(10, assets['grid_k'])
        phase = int(rng.integers(k))
        patch = assets['grid']
        patch = np.tile(patch, (1, (w + patch.shape[1] - 1) // patch.shape[1], 1))[:, :w]
        for y in range(phase, h, k):
            rgb[y:y + 1] = np.minimum(rgb[y:y + 1], patch)
        flag.update(applied=True, k=k, phase=phase, log=assets['name'], source_y=assets['grid_source_y'])
    flag = flags['column_borders']
    if flag['selected']:
        thickness = int(rng.integers(1, 3))
        rgb[:, :thickness] = 40
        rgb[:, -thickness:] = 40
        flag.update(applied=True, width=thickness)
    flag = flags['colour_jitter']
    if flag['selected']:
        gains = rng.uniform(.93, 1.07, 3)
        offset = rng.uniform(-5, 5, 3)
        rgb = np.clip(rgb.astype(float) * gains + offset, 0, 255).astype(np.uint8)
        flag.update(applied=True, gains=gains.tolist(), offset=offset.tolist())
    flag = flags['fading']
    if flag['selected']:
        ends = rng.uniform(.02, .22, 2)
        fade = np.linspace(*ends, h)[:, None, None]
        rgb = np.clip(rgb * (1 - fade) + 255 * fade, 0, 255).astype(np.uint8)
        flag.update(applied=True, ends=ends.tolist())
    flag = flags['scan_noise']
    if flag['selected']:
        sigma = float(rng.uniform(1, 5))
        rgb = np.clip(rgb.astype(float) + rng.normal(0, sigma, rgb.shape), 0, 255).astype(np.uint8)
        speckles = rng.random((h, w)) < .0006
        rgb[speckles] = rng.choice([0, 255], size=(int(speckles.sum()), 1))
        flag.update(applied=True, sigma=sigma, speckle_probability=.0006)
    flag = flags['skew']
    if flag['selected']:
        degrees = float(rng.uniform(-.29, .29))
        shear = np.tan(np.deg2rad(degrees))
        rgb = cv2.warpAffine(rgb, np.array([[1., shear, -shear * h / 2], [0, 1., 0]]),
                             (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        flag.update(applied=True, degrees=degrees, transform='horizontal shear; row coordinates unchanged')
    flag = flags['jpeg']
    if flag['selected']:
        quality = int(rng.integers(45, 86))
        stream = BytesIO()
        Image.fromarray(rgb).save(stream, format='JPEG', quality=quality)
        stream.seek(0)
        with Image.open(stream) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        flag.update(applied=True, quality=quality)
    flag = flags['binarisation']
    if flag['selected']:
        rgb = np.repeat(np.asarray(Image.fromarray(rgb).convert('L'))[..., None] < 180, 3, axis=2)
        rgb = np.where(rgb, 0, 255).astype(np.uint8)
        flag.update(applied=True, threshold=180)
    # Blank gaps are explicit empty-label truth intervals. Their rows remain
    # precisely empty even after scan/codec perturbations.
    gaps = [b for b in beds if not b['lithology']]
    for bed in gaps:
        rgb[bed['top_y']:bed['bottom_y']] = 255
    flags['blank_gaps'].update(applied=bool(gaps), ranges=[[b['top_y'], b['bottom_y']] for b in gaps])
    return rgb
