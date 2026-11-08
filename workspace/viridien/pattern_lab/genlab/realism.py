"""Fit scan corruption to native crops; no TIFF access in this module."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def component_budget(mask, colours, clean, fraction=.05):
    """Keep whole components or reject; never turn a shape into pepper noise."""
    from scipy.ndimage import label
    labels, n = label(mask > 0)
    out = np.zeros_like(mask, np.uint8)
    budget = int(clean.shape[0] * clean.shape[1] * fraction)
    background = np.median(clean.reshape(-1, 3), axis=0).mean()
    for index in range(1, n + 1):
        take = labels == index
        size = int(take.sum())
        # Broad motif disagreement and isolated dots are not verified dirt.
        if 2 <= size <= budget and float(colours[take].mean()) < background - 25:
            out[take] = 255
            budget -= size
    return out


def stroke(rgb, delta):
    """Change native vertical ink weight by exactly one row, without wrapping."""
    if not delta or len(rgb) < 2:
        return rgb.copy()
    dark = rgb.max(2) < 80
    previous = np.zeros_like(dark)
    previous[1:] = dark[:-1]
    out = rgb.copy()
    if delta > 0:
        take = previous & ~dark
        shifted = np.concatenate([rgb[:1], rgb[:-1]])
        out[take] = shifted[take]
    else:
        take = dark & ~previous
        bg = np.median(rgb[~dark], axis=0) if (~dark).any() else np.array([255]*3)
        out[take] = bg
    return out


def aligned(tile, real):
    """Exact least-squares best cyclic phase, using folded FFT correlations."""
    ph, pw = tile.shape[:2]
    yy, xx = np.indices(real.shape[:2])
    pos = ((yy % ph) * pw + xx % pw).ravel()
    counts = np.bincount(pos, minlength=ph * pw).reshape(ph, pw)
    cf = np.conj(np.fft.rfft2(counts))
    loss = np.zeros((ph, pw))
    for ch in range(3):
        t = tile[..., ch].astype(float)
        sums = np.bincount(pos, weights=real[..., ch].ravel(), minlength=ph * pw).reshape(ph, pw)
        loss += np.fft.irfft2(cf * np.fft.rfft2(t * t) -
                              2 * np.conj(np.fft.rfft2(sums)) * np.fft.rfft2(t), s=(ph, pw)).real
    dy, dx = np.unravel_index(loss.argmin(), loss.shape)
    return tile[(yy + dy) % ph, (xx + dx) % pw].copy(), [int(dy), int(dx)]


def measure(tile, real):
    fake, phase = aligned(tile, real)
    actual = real.astype(float)
    best = None
    # These are fitting candidates, not an assumed augmentation distribution.
    for radius in (0, .35, .7, 1., 1.4, 2.):
        base = np.asarray(Image.fromarray(fake).filter(ImageFilter.GaussianBlur(radius)), float)
        drift = np.median((actual - base).reshape(-1, 3), axis=0)
        residual = actual - base - drift
        err = np.linalg.norm(residual, axis=2)
        loss = float(np.mean(np.minimum(err, 45)))
        if best is None or loss < best[0]:
            best = loss, radius, drift, residual, err
    _, radius, drift, residual, err = best
    med = np.median(err)
    threshold = max(45., float(med + 3 * 1.4826 * np.median(np.abs(err - med))))
    mask = err > threshold
    # Large disagreement is ambiguous (wrong phase/model or mixed lithology).
    # Keep the measurement but don't paste an entire different pattern.
    usable = float(mask.mean()) <= .35
    good = residual[~mask]
    # MAD collapses whenever most residuals are exactly zero. A 99% winsorized
    # standard deviation retains sparse low-amplitude scanner variation.
    if len(good):
        centered = good - np.median(good, axis=0)
        bound = np.quantile(np.abs(centered), .99, axis=0)
        noise = float(np.sqrt(np.mean(np.clip(centered, -bound, bound) ** 2)))
    else:
        noise = 0.
    rails = []
    for x in sorted(set(range(min(3, real.shape[1]))) | set(range(max(0, real.shape[1]-3), real.shape[1]))):
        if (real[:, x].max(1) < 80).mean() > .8:
            rails.append(dict(side='left' if x < real.shape[1] // 2 else 'right',
                              offset=x if x < real.shape[1] // 2 else real.shape[1]-1-x,
                              colour=np.median(real[:, x], axis=0).astype(int).tolist(), source_x=x))
    return dict(blur=radius, noise=noise, drift=drift.tolist(), phase=phase,
                overlay_fraction=float(mask.mean()), overlay_usable=usable,
                residual_threshold=threshold, rails=rails, noise_estimator='99% winsorized RMS'), (mask.astype(np.uint8) * 255), real.copy()


def augment(clean, measurement, overlay, rng):
    """Appearance corruption only: geometry and class/row labels are unchanged."""
    params = dict(measurement or dict(blur=0., noise=0., drift=[0., 0., 0.], overlay_fraction=0.))
    weighted = stroke(clean, int(params.get('stroke_delta', 0)))
    rgb = np.asarray(Image.fromarray(weighted).filter(ImageFilter.GaussianBlur(params['blur'])), float)
    rgb += np.asarray(params['drift'])
    if params['noise']:
        rgb += rng.normal(0, params['noise'], rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    params['pasted_pixels'] = 0
    if overlay is not None and params.get('overlay_usable', True):
        mask, colours = overlay
        mask = mask.copy()
        for rail in params.get('rails', []):
            mask[:, rail['source_x']] = 0
        # Paste each connected component once at native scale, never modulo a
        # donor height. Skip components that do not fit the target window.
        from scipy.ndimage import find_objects, label
        labels, _ = label(mask > 0)
        events = []
        remaining = int(rgb.shape[0] * rgb.shape[1] * .05)
        for index, box in enumerate(find_objects(labels), 1):
            if box is None:
                continue
            component = labels[box] == index
            h, w = component.shape
            size = int(component.sum())
            if size > remaining or h > len(rgb) or w > rgb.shape[1]:
                continue
            y = int(rng.integers(len(rgb)-h+1))
            x = int(rng.integers(rgb.shape[1]-w+1))
            rgb[y:y+h, x:x+w][component] = colours[box][component]
            params['pasted_pixels'] += int(component.sum())
            remaining -= size
            events.append(dict(y=y, x=x, height=h, width=w, pixels=int(component.sum())))
        params['overlay_components'] = events
    # Rails belong on the column edges, not randomly through its lithology.
    for rail in params.get('rails', []):
        x = rail['offset'] if rail['side'] == 'left' else rgb.shape[1]-1-rail['offset']
        if 0 <= x < rgb.shape[1]:
            rgb[:, x] = rail['colour']
    return rgb, params
