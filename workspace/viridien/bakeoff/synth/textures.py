"""Native-scale crop quilting and real-page nuisance assets."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from harness import io, legend, legend_v2
from .statistics import ROOT, read_json, resolve_crop


@lru_cache(maxsize=256)
def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def fit_width(rgb, width, rng):
    """Only a >15% mismatch permits isotropic resampling; otherwise crop/tile."""
    h, w = rgb.shape[:2]
    scale = width / w if abs(width - w) > .15 * w else 1.
    if scale != 1:
        # One affine scale in both axes, including subpixel sampling. The output
        # height is a rounded canvas extent, not a separate vertical scale.
        rgb = cv2.warpAffine(rgb, np.array([[scale, 0, 0], [0, scale, 0]], float),
                             (width, max(2, round(h * scale))), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    # Full-width approved crops include the source's column rails. Shift only
    # the texture interior, otherwise those rails turn into moving black bars.
    # Strip only persistent ink in the outer 8%; record the native-pixel cut.
    occupancy = (rgb.min(axis=2) < 110).mean(axis=0)
    margin = max(1, round(rgb.shape[1] * .08))
    left_ink = np.flatnonzero(occupancy[:margin] > .75)
    right_ink = np.flatnonzero(occupancy[-margin:] > .75)
    trim_left = int(left_ink[-1] + 1) if len(left_ink) else 0
    trim_right = int(margin - right_ink[0]) if len(right_ink) else 0
    if trim_left + trim_right < rgb.shape[1] // 3:
        rgb = rgb[:, trim_left:rgb.shape[1] - trim_right if trim_right else None]
    else:
        trim_left = trim_right = 0
    phase = int(rng.integers(rgb.shape[1]))
    rgb = np.roll(rgb, phase, axis=1)
    if rgb.shape[1] < width:
        rgb = np.tile(rgb, (1, (width + rgb.shape[1] - 1) // rgb.shape[1], 1))
    offset = int(rng.integers(max(1, rgb.shape[1] - width + 1)))
    return rgb[:, offset:offset + width], dict(source_size=[w, h], scale_x=scale,
        scale_y=scale, isotropic_exception=scale != 1, phase_x=phase, crop_x=offset,
        fitted_size=[width, int(rgb.shape[0])], rail_trim_left=trim_left, rail_trim_right=trim_right)


def minimum_cut(error):
    """Minimum-error connected seam across columns in a vertical overlap."""
    height, width = error.shape
    cost = error[:, 0].copy()
    back = np.zeros((height, width), np.int32)
    for x in range(1, width):
        options = np.stack([np.r_[np.inf, cost[:-1]], cost, np.r_[cost[1:], np.inf]])
        choices = options.argmin(axis=0)
        back[:, x] = np.arange(height) + choices - 1
        cost = error[:, x] + options[choices, np.arange(height)]
    seam = np.empty(width, np.int32)
    seam[-1] = int(cost.argmin())
    for x in range(width - 1, 0, -1):
        seam[x - 1] = back[seam[x], x]
    return seam


def quilt(crops, approved_path, height, width, rng):
    if not crops:
        raise ValueError('No approved crops in requested hold-out pool')
    canvas = np.empty((height, width, 3), np.uint8)
    used, filled = [], 0
    while filled < height:
        crop = crops[int(rng.integers(len(crops)))]
        path = resolve_crop(crop['file'], approved_path)
        native = pixels(str(path))
        if native.shape[1] != int(crop['col_w']) or native.shape[0] != int(crop['h']):
            raise ValueError('Crop dimensions differ from approved manifest: ' + str(path.name))
        patch, transform = fit_width(native, width, rng)
        overlap = min(24, filled, max(0, len(patch) // 3))
        start = filled - overlap
        count = min(len(patch), height - start)
        if overlap:
            # Search candidate start rows by average row error, then find a
            # connected pixel seam for the best candidate overlap.
            candidates = range(min(32, max(1, len(patch) - overlap)))
            offset = min(candidates, key=lambda y: float(np.square(
                canvas[start:filled].astype(float) - patch[y:y + overlap]).mean()))
            patch = patch[offset:]
            count = min(len(patch), height - start)
            error = np.square(canvas[start:filled].astype(float) - patch[:overlap]).mean(axis=2)
            seam = minimum_cut(error)
            mask = np.arange(overlap)[:, None] >= seam[None, :]
            canvas[start:filled] = np.where(mask[..., None], patch[:overlap], canvas[start:filled])
            canvas[filled:start + count] = patch[overlap:count]
            transform.update(overlap_rows=overlap, source_y_offset=offset,
                             seam_min=int(seam.min()), seam_max=int(seam.max()))
        else:
            canvas[start:start + count] = patch[:count]
            transform.update(overlap_rows=0, source_y_offset=0)
        used.append(dict(file=crop['file'], log=crop['log'], top_y=crop.get('top_y'),
                         bottom_y=crop.get('bottom_y'), output_y=start, rows=count, **transform))
        if start + count <= filled:
            raise ValueError('Crop too short for overlap quilting')
        filled = start + count
    return canvas, used


def donor_assets(name, cache_root):
    """Read small real-page regions and existing legend caches, never whole RGB pages."""
    log = io.open_log(name)
    try:
        lb = log.boxes['lithology_legend']
        legend_rgb = io.read_rgb(log, lb)
        entries = None
        for cache in [ROOT / 'out' / '_cache' / 'legend_v2', ROOT / 'out' / '_cache' / 'legend',
                      ROOT / 'classify' / 'out' / '_cache' / 'legend']:
            data = read_json(cache / (name + '.json'))
            if data and all((cache / e['swatch_file']).exists() for e in data['entries']):
                entries = [legend.LegendEntry(e['id'], e['label'], e['swatch_box'],
                    pixels(str(cache / e['swatch_file'])).copy(), tuple(e['mean_rgb']), e['is_lithology'])
                    for e in data['entries']]
                break
        if entries is None:
            old_cache = legend_v2.CACHE
            try:
                legend_v2.CACHE = Path(cache_root) / 'donor_legend'
                entries = legend_v2.parse(log)
            finally:
                legend_v2.CACHE = old_cache
        box = log.boxes['interpreted_lithology_column']
        width = box[2] - box[0]
        top, bottom = box[1], min(box[3], box[1] + 4000)
        sample = io.read_rgb(log, box, top, bottom)
        darkness = (sample.min(axis=2) < 110).mean(axis=1)
        selected = np.flatnonzero(darkness > .7)
        if not len(selected):
            selected = np.array([int(darkness.argmax())])
        centers = selected[np.r_[True, np.diff(selected) > 1]]
        distances = np.diff(centers)
        distances = distances[(distances >= 10) & (distances <= 400)]
        k = int(np.median(distances)) if len(distances) else 200
        grid_y = int(selected[len(selected) // 2])
        grid = sample[grid_y:grid_y + 1].copy()
        # The nearest side strip is outside the annotated lithology box. Pick
        # connected ink components as genuine text/mark/symbol overlays.
        side_width = max(32, min(180, width * 2))
        right_space = max(0, log.size[0] - box[2] - 8)
        left_space = max(0, box[0] - 8)
        if right_space >= side_width or right_space >= left_space:
            x0 = box[2] + 8
            side_width = min(side_width, right_space)
        else:
            side_width = min(side_width, left_space)
            x0 = box[0] - side_width - 8
        side_box = [x0, top, min(log.size[0], x0 + side_width), bottom]
        side = io.read_rgb(log, side_box) if side_width else np.full((1, 1, 3), 255, np.uint8)
        ink = (side.min(axis=2) < 180).astype(np.uint8)
        ink = cv2.dilate(ink, np.ones((2, 3), np.uint8))
        components = cv2.connectedComponentsWithStats(ink)[2][1:]
        overlays = []
        for x, y, w, h, area in components:
            if 3 <= w <= min(width, 100) and 4 <= h <= 65 and area >= 6:
                overlays.append((side[y:y + h, x:x + w].copy(),
                                 [int(x0 + x), int(top + y), int(x0 + x + w), int(top + y + h)]))
        return dict(name=name, legend_rgb=legend_rgb, legend_box=list(lb), entries=entries,
                    width=width, grid=grid, grid_k=k, grid_source_y=top + grid_y,
                    overlays=overlays[:200], mode=log.mode,
                    border=sample[:, :1].copy(), side_box=side_box)
    finally:
        log.close()
