"""Conservative frame cleaning and native, unmirrored swatch repeats."""
from __future__ import annotations

import numpy as np
from collections import Counter

from . import learn
from generator._texture import estimate_period


def remove_frame(rgb):
    """Remove 1–3 px rails within 4 px of an edge, including their margins.

    A rail must be dark over 90% of its central length, unlike the interior.
    Recurrent full ink lines (striped fills), and solid dark fills, are kept.
    Work on both axes before cropping so one rail cannot hide another.
    """
    trims = []
    for axis in (1, 0):
        v = np.moveaxis(rgb, axis, 0)
        central = v[:, 3:-3] if v.shape[1] > 8 else v
        bg = learn.background(rgb)
        neutral = np.ptp(central.astype(float), axis=2) <= 12
        darker = central.mean(2) < min(200, float(np.mean(bg)) - 35)
        occupancy = ((central.max(2) < 80) | (neutral & darker)).mean(1)
        n = len(v)
        inner = occupancy[5:-5] if n > 14 else occupancy[3:-3]
        runs, run = [], 0
        for value in [*inner, 0]:
            if value >= .90:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        typical = Counter(runs).most_common(1)[0][0] if runs else 0
        cuts = []
        for reverse in (False, True):
            occ = occupancy[::-1] if reverse else occupancy
            cut = 0
            for start in range(min(4, max(0, n // 4))):
                if occ[start] < .90:
                    continue
                end = start
                while end < n and occ[end] >= .90:
                    end += 1
                # Repeated interior ink columns/rows are part of the fill.
                thickness = end - start
                if len(inner) and inner.mean() < .8:
                    if not typical and thickness <= 3:
                        cut = end
                    elif typical and thickness != typical:
                        # A thin rail between fill stripes, or a rail touching
                        # a normal stripe: retain the normal stripe thickness.
                        excess = thickness - typical if thickness > typical else thickness
                        if excess <= 3:
                            cut = start + excess
                break
            cuts.append(cut)
        trims.append(cuts)
    (left, right), (top, bottom) = trims
    # When the vertical sides establish an enclosing rectangle, its complete
    # horizontal edge runs are frame too, even when a fill stroke has the same
    # thickness. Removing only the "excess" leaves a second rail in the tile.
    if left and right:
        occupancy = (rgb[:, left:rgb.shape[1]-right].max(2) < 80).mean(1)
        for reverse in (False, True):
            occ = occupancy[::-1] if reverse else occupancy
            run = 0
            while run < len(occ) and occ[run] >= .90:
                run += 1
            if 0 < run <= 3:
                if reverse:
                    bottom = max(bottom, run)
                else:
                    top = max(top, run)
    h, w = rgb.shape[:2]
    return rgb[top:h-bottom, left:w-right].copy(), [left, top, right, bottom]


def edge_continuity(rgb, axis, tolerance=35):
    """Opposite native edge pixels must agree at >=96% (RGB distance <35).

    Used when no repeated overlap proves a wrap. No reflection or blending.
    This is evidence of continuity, not proof of a geological repeat.
    """
    v = np.moveaxis(rgb, axis, 0).astype(float)
    match = float((np.linalg.norm(v[0] - v[-1], axis=-1) < tolerance).mean())
    # Equal black edges can double the width of a line on wrapping. Compare
    # each joined run with runs actually observed inside the swatch.
    dark = v.max(2) < 80
    bad = 0
    for lane in dark.T:
        if lane.all() or not (lane[0] and lane[-1]):
            continue
        head = next((i for i, value in enumerate(lane) if not value), len(lane))
        tail = next((i for i, value in enumerate(lane[::-1]) if not value), len(lane))
        runs, run = [], 0
        for value in [*lane[head:len(lane)-tail], False]:
            if value:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if head + tail > max(runs or [max(head, tail)]):
            bad += 1
    joined = bad / max(1, dark.shape[1])
    return dict(ok=match >= .96 and joined <= .10, match=round(match, 5),
                doubled_run_fraction=round(joined, 5), method='edge-wrap')


def axis_repeat(rgb, axis, edge_guard=3):
    measured = estimate_period(rgb, axis)
    v = np.moveaxis(rgb, axis, 0).astype(float)
    n = len(v)
    # Borders can coincide with normal fill strokes. Infer the period from
    # the interior and extract native pixels there, never average two cycles
    # (which creates gray seams from black and background pixels).
    guard = edge_guard if n > 18 else 0
    core = v[guard:n-guard] if guard else v
    energy = float(np.var(core, axis=0).mean())
    candidates = []
    # Two observed cycles establish the native transition across the seam.
    # Check pixel agreement as well as the estimator's variance correlation.
    for p in range(1, min(256, len(core) - 3) + 1):
        if p == 1 and energy >= .01:
            continue
        hits = np.linalg.norm(core[p:] - core[:-p], axis=-1) < 35
        fraction = float(hits.mean())
        correlation = 1 - float(np.square(core[p:] - core[:-p]).mean()) / max(.01, 2 * energy)
        start = (n-p)//2
        # Adjacent edge colours need not be equal: a real stroke may start or
        # end at the seam. For a short overlap, require both sides of the
        # wrapped transition to be observed in the native pixels.
        boundary = (start > 0 and start+p < n and
                    (np.linalg.norm(v[start-1]-v[start+p-1], axis=-1) < 35).mean() >= .96 and
                    (np.linalg.norm(v[start]-v[start+p], axis=-1) < 35).mean() >= .96)
        continuity = p <= len(core) // 2 or boundary
        if fraction >= .96 and float(hits.mean(1).min()) >= .85 and correlation >= .82 and continuity:
            candidates.append((p, fraction))
    if candidates:
        best = max(s for _, s in candidates)
        p, match = next((p, s) for p, s in candidates if s >= best - .01)
        measured.update(pixels=p, observed_cycles=round(n / p, 3))
        return p, measured, dict(ok=True, match=round(match, 5), method='repeat-overlap')
    seam = edge_continuity(rgb, axis)
    # One complete swatch period remains usable even without two cycles.
    measured.update(pixels=n if seam['ok'] else None,
                    observed_cycles=1 if seam['ok'] else measured['observed_cycles'])
    return n, measured, seam


def learn_swatch(rgb):
    rgb, trims = remove_frame(rgb)
    py, yp, sy = axis_repeat(rgb, 0)
    px, xp, sx = axis_repeat(rgb, 1)
    top, left = (rgb.shape[0]-py)//2, (rgb.shape[1]-px)//2
    tile = rgb[top:top+py, left:left+px].copy()
    ink_rows = (rgb.max(2)<80).mean(1)
    if (((ink_rows<.02)|(ink_rows>.85)).mean()>=.95 and (ink_rows>.85).any() and (ink_rows<.02).any()):
        # Continuous horizontal stripes with a thin vertical scanner dropout:
        # the native row majority establishes x=1 without repeating the scratch.
        tile = learn.supported_vote(tile.transpose(1,0,2)[:,:,None,:])
        px = 1
        xp.update(pixels=1,constant_axis=True,observed_cycles=float(rgb.shape[1]))
        sx = dict(ok=True,match=round(float(((ink_rows<.02)|(ink_rows>.85)).mean()),5),method='row-consensus')
    # An isolated full-width scanner line can sit in the unobserved middle
    # of a >1/2-swatch period and never participate in overlap comparisons.
    bg = learn.background(rgb)
    for axis, seam in ((0, sy), (1, sx)):
        v = np.moveaxis(tile, axis, 0)
        foreign = ((np.linalg.norm(v.astype(float)-bg, axis=2) > 45) & (v.max(2) >= 80)).mean(1)
        if np.any(foreign >= .90):
            seam.update(ok=False, reason='Full scanline has an unexplained non-ink colour.')
    meta = dict(yp, x=xp, frame_trim=trims, seam_x=sx, seam_y=sy,
                seam_risk=not (sx['ok'] and sy['ok']))
    return rgb, dict(kind='tile', tile=tile, bg=learn.background(rgb),
                     anchored=False, phase=(0, 0)), meta
