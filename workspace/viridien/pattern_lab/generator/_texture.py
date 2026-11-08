"""Pixel copying, repeat measurement, and minimum-error seam quilting."""

import numpy as np


def trim_frame(rgb):
    """Remove thin, continuous dark rails at the outer edge only."""
    height, width = rgb.shape[:2]
    dark = rgb.max(axis=2) < 65
    trims = []
    for occupancy, extent in ((dark.mean(axis=0), width), (dark.mean(axis=1), height)):
        limit = min(4, max(0, (extent - 8) // 2), max(1, round(extent * .06)))
        low = high = 0
        while low < limit and occupancy[low] >= .96:
            low += 1
        while high < limit and occupancy[extent - high - 1] >= .96:
            high += 1
        # A solid swatch is texture, rather than an outer frame.
        if np.mean(occupancy) > .8:
            low = high = 0
        trims.append((low, high))
    (left, right), (top, bottom) = trims
    return rgb[top:height - bottom, left:width - right].copy(), [left, top, right, bottom]


def estimate_period(rgb, axis):
    """Find an integer raster repeat; reject weak and blank-overlap matches."""
    values = np.moveaxis(rgb, axis, 0)
    # Bound work for long source intervals without resampling their pixels.
    if len(values) > 1024:
        start = (len(values) - 1024) // 2
        values = values[start:start + 1024]
    if values.shape[1] > 256:
        step = max(1, values.shape[1] // 256)
        values = values[:, ::step]
    values = values.astype(np.float32)
    size = len(values)
    energy = float(np.var(values, axis=0).mean())
    if energy < .01:
        return dict(pixels=1, confidence=1.0, observed_cycles=size, constant_axis=True)
    minimum_overlap = min(6, max(2, size // 3))
    maximum = min(256, size - minimum_overlap)
    scores = {}
    centre = values.mean(axis=0, keepdims=True)
    for lag in range(1, maximum + 1):
        a, b = values[:-lag], values[lag:]
        # Uniform fragments can give an exact but unsupported long repeat.
        evidence = (float(np.square(a - centre).mean()) + float(np.square(b - centre).mean())) / 2
        if evidence < energy * .15:
            continue
        error = float(np.square(a - b).mean())
        scores[lag] = 1.0 - error / (2 * energy)
    candidates = [lag for lag, score in scores.items()
                  if lag >= 2 and score >= .82
                  and score >= scores.get(lag - 1, -float("inf"))
                  and score >= scores.get(lag + 1, -float("inf"))]
    if not candidates:
        return dict(pixels=None, confidence=round(max(0., max(scores.values(), default=0.)), 6),
                    observed_cycles=0, constant_axis=False)
    best = max(scores[lag] for lag in candidates)
    # Prefer a supported fundamental repeat to its longer multiples.
    period = min(lag for lag in candidates if scores[lag] >= max(.82, best - .025))
    return dict(pixels=int(period), confidence=round(float(scores[period]), 6),
                observed_cycles=round(size / period, 3), constant_axis=False)


def minimum_cut(error):
    """Return a connected minimum-cost seam across the overlap columns."""
    height, width = error.shape
    cost = error[:, 0].astype(np.float64)
    back = np.zeros((height, width), dtype=np.int16)
    positions = np.arange(height)
    for x in range(1, width):
        options = np.stack((np.r_[np.inf, cost[:-1]], cost, np.r_[cost[1:], np.inf]))
        choices = options.argmin(axis=0)
        back[:, x] = positions + choices - 1
        cost = error[:, x] + options[choices, positions]
    seam = np.empty(width, dtype=np.int16)
    seam[-1] = int(cost.argmin())
    for x in range(width - 1, 0, -1):
        seam[x - 1] = back[seam[x], x]
    return seam


def _repeat(values, target, period, rng):
    """Choose a native tile where the measured repeat matches its next copy."""
    size = len(values)
    if period == 1:
        origin = int(rng.integers(size))
    else:
        # Compare short strips around each candidate tile boundary. No blend.
        span = min(4, size - period)
        candidates = range(max(1, size - period - span + 1))
        errors = []
        for start in candidates:
            a = values[start:start + span].astype(np.float32)
            b = values[start + period:start + period + span].astype(np.float32)
            errors.append(float(np.square(a - b).mean()))
        threshold = min(errors) + 1e-6
        best = [i for i, error in enumerate(errors) if error <= threshold]
        origin = int(best[int(rng.integers(len(best)))])
    phase = int(rng.integers(period))
    tile = values[origin:origin + period]
    indices = (np.arange(target) + phase) % period
    return tile[indices].copy(), dict(method="native-repeat", tile_origin=origin,
                                     phase=phase, tile_length=period, seams=0)


def extend_axis(rgb, target, axis, repeat, rng, variation):
    """Extend one axis. Every output pixel is copied from a source pixel."""
    values = np.moveaxis(rgb, axis, 0)
    size = len(values)
    if target <= size:
        origin = int(rng.integers(size - target + 1))
        result = values[origin:origin + target].copy()
        return np.moveaxis(result, 0, axis), dict(method="native-crop", source_offset=origin, seams=0)
    period = repeat["pixels"]
    if period is not None and (variation == 0 or repeat["constant_axis"]):
        result, record = _repeat(values, target, period, rng)
        return np.moveaxis(result, 0, axis), record
    if size < 4:
        # A seam cannot be measured in a one-to-three-pixel exemplar.
        result = values[np.arange(target) % size].copy()
        return np.moveaxis(result, 0, axis), dict(method="short-source-tile", tile_length=size, seams=0)
    patch_size = max(3, round(size * .8))
    overlap = min(20, max(1, patch_size // 5))
    starts = np.unique(np.linspace(0, size - patch_size, min(40, size - patch_size + 1)).astype(int))
    canvas = np.empty((target, *values.shape[1:]), dtype=np.uint8)
    origin = int(starts[int(rng.integers(len(starts)))])
    count = min(patch_size, target)
    canvas[:count] = values[origin:origin + count]
    filled = count
    placements = [dict(output_start=0, source_start=origin, length=count, overlap=0)]
    while filled < target:
        start = filled - overlap
        scores = []
        old = canvas[start:filled].astype(np.float32)
        for candidate in starts:
            new = values[candidate:candidate + overlap].astype(np.float32)
            scores.append(float(np.square(old - new).mean()))
        best = min(scores)
        # Variation changes only the choice of a near-matching real patch.
        tolerance = max(1e-6, (float(np.mean(scores)) - best) * variation * .35)
        choices = [i for i, error in enumerate(scores) if error <= best + tolerance]
        origin = int(starts[choices[int(rng.integers(len(choices)))]])
        count = min(patch_size, target - start)
        patch = values[origin:origin + count]
        difference = old - patch[:overlap].astype(np.float32)
        error = np.square(difference).mean(axis=2)
        seam = minimum_cut(error)
        mask = np.arange(overlap)[:, None] >= seam[None, :]
        canvas[start:filled] = np.where(mask[..., None], patch[:overlap], canvas[start:filled])
        canvas[filled:start + count] = patch[overlap:count]
        placements.append(dict(output_start=start, source_start=origin, length=count, overlap=overlap,
                               seam_min=int(seam.min()), seam_max=int(seam.max()),
                               match_error=round(scores[int(np.where(starts == origin)[0][0])], 4)))
        filled = start + count
    return np.moveaxis(canvas, 0, axis), dict(method="minimum-error-quilting", seams=len(placements) - 1,
                                            patch_length=patch_size, placements=placements)
