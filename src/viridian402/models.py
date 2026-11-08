"""Small local pixel baseline and importable model plugin contract.

Plugins implement predict(request: dict, config: dict) -> list[interval].
The request contains image paths and public boxes, never truth or split details.
The baseline finds boundaries and clusters pixels automatically. Without an OCR
plugin or explicitly supplied diagnostic legend it honestly returns Unknown.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .data import load_boxes, read_rgb


def resolve_model(spec: str):
    if spec == 'builtin:baseline':
        return baseline
    if spec == 'builtin:empty':
        return lambda request, config: []
    if spec in ('legacy:s11', 'builtin:legacy-s11'):
        from .legacy import predict
        return predict
    if ':' not in spec or spec.startswith('builtin:'):
        raise ValueError('model must be builtin:baseline, legacy:s11 or module:function')
    module, function = spec.rsplit(':', 1)
    model = getattr(importlib.import_module(module), function)
    if not callable(model):
        raise TypeError(f'{spec} is not callable')
    return model


def _smooth(values, radius):
    if radius <= 0:
        return values
    padded = np.pad(values, ((radius, radius), (0, 0)), mode='edge')
    sums = np.vstack([np.zeros((1, values.shape[1])), padded.cumsum(axis=0)])
    return (sums[2*radius+1:] - sums[:-2*radius-1])/(2*radius+1)


def _descriptor(rgb):
    values = rgb.astype(np.float32)/255
    gray = values.mean(2)
    # Colour, occupancy, directional changes, and row-density periodicity.
    profile = (gray < .8).mean(1)
    profile = profile-profile.mean()
    denom = float(profile @ profile) + 1e-8
    correlations = [float(profile[:-lag] @ profile[lag:])/denom if len(profile) > lag else 0
                    for lag in (2, 4, 8, 16)]
    return np.r_[np.median(values.reshape(-1, 3), axis=0),
                 (gray < .8).mean(), np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0,
                 np.abs(np.diff(gray, axis=0)).mean() if len(gray) > 1 else 0, correlations]


def baseline(request: dict, config: dict) -> list[dict]:
    """Detect mean-colour/ink transitions then cluster colour and texture."""
    with Image.open(request['image']) as image:
        boxes = load_boxes(request['annotation'], image.size)
    x0, _, x1, _ = boxes['interpreted_lithology_column']
    y0, y1 = map(int, request['window'])
    rgb = read_rgb(request['image'], (x0, y0, x1, y1),
                   max_window_pixels=int(config.get('max_window_pixels', 12_000_000)),
                   max_decode_bytes=int(config.get('max_decode_bytes', 512_000_000)))
    inset = max(0, min(rgb.shape[1]//10, (rgb.shape[1]-1)//2))
    inner = rgb[:, inset:rgb.shape[1]-inset or None]
    values = inner.astype(np.float32)/255
    gray = values.mean(2)
    features = np.c_[np.median(values, axis=1), (gray < .8).mean(1), gray.std(1)]
    radius = int(config.get('smooth_radius', 2))
    if not 0 <= radius <= 100:
        raise ValueError('smooth_radius must be in [0,100]')
    features = _smooth(features, radius)
    change = np.linalg.norm(np.diff(features, axis=0), axis=1)
    threshold = float(config.get('boundary_threshold', .10))
    min_height = int(config.get('min_height', 4))
    if min_height < 1 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('positive min_height and boundary_threshold required')
    candidates = np.flatnonzero(change >= threshold)+1
    # Collapse a smoothed transition to its largest local change.
    points = []
    for bucket in np.split(candidates, np.flatnonzero(np.diff(candidates) > max(1, radius*2))+1):
        if len(bucket):
            points.append(int(bucket[np.argmax(change[bucket-1])]))
    # Full-width dark horizontal rules are additional generic boundary evidence.
    dark = (gray < .20).mean(1) > .90
    points.extend(int(v) for v in np.flatnonzero(dark & ~np.r_[False, dark[:-1]]))
    boundaries = [0]
    for point in sorted(set(points)):
        if point-boundaries[-1] >= min_height and len(rgb)-point >= min_height:
            boundaries.append(point)
    boundaries.append(len(rgb))
    references = []
    for entry in request.get('legend', []):
        if not entry.get('is_lithology', True) or not entry.get('label'):
            continue
        if 'swatch_box' not in entry:
            continue
        swatch = read_rgb(request['image'], entry['swatch_box'], max_decode_bytes=int(config.get('max_decode_bytes', 512_000_000)))
        references.append((str(entry['label']), _descriptor(swatch)))
    clusters, rows = [], []
    cluster_threshold = float(config.get('cluster_threshold', .24))
    blank_threshold = float(config.get('blank_threshold', .985))
    for a, b in zip(boundaries, boundaries[1:]):
        patch = inner[a:b]
        # Exclude only almost entirely unprinted white stretches.
        if float((patch.mean(2) > 248).mean()) >= blank_threshold:
            continue
        descriptor = _descriptor(patch)
        distances = [float(np.linalg.norm(descriptor-center)) for center in clusters]
        if not distances or min(distances) > cluster_threshold:
            cluster = len(clusters)
            clusters.append(descriptor)
        else:
            cluster = int(np.argmin(distances))
        lithology = 'Unknown'
        if references:
            match = int(np.argmin([np.linalg.norm(descriptor-ref) for _, ref in references]))
            lithology = references[match][0]
        row = {'top_y': y0+a, 'bottom_y': y0+b, 'lithology': lithology, 'cluster_id': cluster}
        depth = request.get('depth', request.get('depth_calibration'))
        if depth is not None:
            slope, intercept = float(depth['a']), float(depth['b'])
            if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
                raise ValueError('depth calibration requires finite a>0 and b')
            row.update(top_depth=round(slope*(y0+a)+intercept, 1), bottom_depth=round(slope*(y0+b)+intercept, 1))
        rows.append(row)
    return rows


