"""Fast row-wise colour and pattern runs for lithology columns."""

import cv2
import numpy as np
from harness import io

NAME = "S1 colour runs"
NEEDS_GPU = False
PAID_API = None


def _colour_rows(rgb):
    """Return each row's modal quantised RGB colour, or -1 for blank/ink."""
    h, w = rgb.shape[:2]
    x0, x1 = int(w * .15), max(int(w * .15) + 1, int(w * .85))
    a = rgb[:, x0:x1]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    valid = (mx > 45) & ~((mn > 242) & (mx > 248)) & ((mx - mn > 10) | (mx < 220))
    q = (a.astype(np.uint16) >> 4)
    code = (q[:, :, 0] * 256 + q[:, :, 1] * 16 + q[:, :, 2]).astype(np.int32)
    codes = np.full(h, -1, np.int32)
    for y in range(h):
        vals = code[y, valid[y]]
        if vals.size >= max(2, int((x1 - x0) * .025)):
            hist = np.bincount(vals, minlength=4096)
            codes[y] = int(hist.argmax())
    return codes


def _swatch_rgb(entry):
    a = entry.swatch.astype(np.uint8)
    if a.size == 0:
        return np.asarray(entry.mean_rgb, np.float32)
    mx = a.max(axis=2); mn = a.min(axis=2)
    good = (mx > 45) & ~((mn > 242) & (mx > 248)) & ((mx - mn > 10) | (mx < 220))
    if good.any():
        vals = (a[good].astype(np.uint16) >> 4)
        code = vals[:, 0] * 256 + vals[:, 1] * 16 + vals[:, 2]
        c = np.bincount(code, minlength=4096).argmax()
        return np.array([((c >> 8) & 15) * 16 + 8, ((c >> 4) & 15) * 16 + 8, (c & 15) * 16 + 8], np.float32)
    return np.asarray(entry.mean_rgb, np.float32)


def _texture_signature(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < 210).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ax, ay = np.abs(gx), np.abs(gy)
    # Ink density and relative horizontal/vertical edge energy.
    return np.array([ink.mean(), ax.mean() / (ax.mean() + ay.mean() + 1e-4), ay.mean() / (ax.mean() + ay.mean() + 1e-4)], np.float32)


def _runs(labels, min_len):
    """Merge short islands into an adjacent run; retain blank runs."""
    out = labels.copy()
    for _ in range(3):
        starts = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
        ends = np.r_[starts[1:], len(out)]
        changed = False
        for i, (a, b) in enumerate(zip(starts, ends)):
            if b - a >= min_len:
                continue
            left = out[starts[i - 1]] if i else None
            right = out[starts[i + 1]] if i + 1 < len(starts) else None
            if left is not None and left == right:
                out[a:b] = left; changed = True
            elif left is not None and right is not None:
                # Brief unclassified rows within a filled interval are usually pattern ink.
                if out[a] < 0:
                    out[a:b] = left; changed = True
                else:
                    out[a:b] = left if b - a <= min_len else right; changed = True
        if not changed:
            break
    return out


def run(log, window, ctx):
    y0, y1 = map(int, window)
    box = log.boxes['interpreted_lithology_column']
    rgb = io.read_rgb(log, box, y0, y1)
    entries = [e for e in ctx['legend'] if e.is_lithology]
    if not entries:
        return [], {}
    swatch_cols = np.stack([_swatch_rgb(e) for e in entries])
    # Lab distances are perceptually more stable than raw RGB distances.
    swatch_lab = cv2.cvtColor(swatch_cols.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    codes = _colour_rows(rgb)
    row_rgb = np.zeros((len(codes), 3), np.uint8)
    valid = codes >= 0
    row_rgb[valid, 0] = (((codes[valid] >> 8) & 15) * 16 + 8).astype(np.uint8)
    row_rgb[valid, 1] = (((codes[valid] >> 4) & 15) * 16 + 8).astype(np.uint8)
    row_rgb[valid, 2] = ((codes[valid] & 15) * 16 + 8).astype(np.uint8)
    row_lab = cv2.cvtColor(row_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    # Scanned grayscale paper often has a weak colour cast; require meaningful
    # saturation before treating per-row tint as a lithology fill colour.
    hsv_swatches = cv2.cvtColor(swatch_cols.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    chromatic = hsv_swatches[:, 1] > 35
    colour_entries = np.flatnonzero(chromatic)
    predicted = np.full(len(codes), -1, np.int32)
    if len(colour_entries):
        delta = row_lab[:, None, :] - swatch_lab[colour_entries][None, :, :]
        dist = np.sum(delta * delta, axis=2)
        best = dist.argmin(axis=1)
        # Ignore out-of-palette pixels and ink fluctuations.
        good = valid & (dist[np.arange(len(codes)), best] < 2600)
        predicted[good] = colour_entries[best[good]]
    # If a row has no usable fill colour, classify the local ink texture.
    need_texture = ~valid | (predicted < 0)
    if need_texture.any():
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ink = (gray < 210).astype(np.float32)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        # Average over a short vertical neighbourhood so sparse hatch marks
        # contribute their orientation even on rows between individual strokes.
        den = np.convolve(ink.mean(axis=1), np.ones(9, np.float32) / 9, mode='same')
        ex = np.convolve(gx.mean(axis=1), np.ones(9, np.float32) / 9, mode='same')
        ey = np.convolve(gy.mean(axis=1), np.ones(9, np.float32) / 9, mode='same')
        feats = np.stack([den, ex / (ex + ey + 1e-4), ey / (ex + ey + 1e-4)], axis=1)
        sigs = np.stack([_texture_signature(e.swatch) for e in entries])
        # Use texture only for monochrome legends or uncoloured rows.
        monochrome = np.flatnonzero(~chromatic)
        candidates = monochrome if len(monochrome) else np.arange(len(entries))
        delta = feats[:, None, :] - sigs[candidates][None, :, :]
        # Density matters most; orientation distinguishes hatch directions.
        dist = delta[:, :, 0] ** 2 * 2.0 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2
        texture_best = candidates[dist.argmin(axis=1)]
        if not len(colour_entries):
            predicted[:] = texture_best
            predicted[den < .012] = -1
        else:
            predicted[need_texture] = texture_best[need_texture]
            # A nearly blank row is a real gap, not a texture label.
            predicted[need_texture & (den < .015)] = -1
    width = rgb.shape[1]
    min_len = max(3, int(round(width * .008)))
    predicted = _runs(predicted, min_len)
    rows = []
    starts = np.r_[0, np.flatnonzero(predicted[1:] != predicted[:-1]) + 1]
    ends = np.r_[starts[1:], len(predicted)]
    for a, b in zip(starts, ends):
        idx = int(predicted[a])
        if idx < 0 or b <= a:
            continue
        entry = entries[idx]
        rows.append(dict(top_y=y0 + int(a), bottom_y=y0 + int(b), lithology=entry.label, cluster_id=int(entry.id)))
    # Compact pipeline stages: input, modal colour and final interval labels.
    thumb_h = min(500, len(predicted))
    step = max(1, len(predicted) // max(1, thumb_h))
    src = rgb[::step]
    stage1 = cv2.resize(src, (max(1, min(180, src.shape[1])), min(500, src.shape[0])))
    stage2 = np.full((min(500, len(codes)), 180, 3), 245, np.uint8)
    sample = np.arange(stage2.shape[0]) * max(1, len(codes) // stage2.shape[0])
    for k, yy in enumerate(sample):
        if valid[yy]:
            stage2[k, :, :] = row_rgb[yy]
        else:
            stage2[k, :, :] = (245, 245, 245)
    stage3 = stage2.copy()
    for row in rows:
        a = min(len(stage3) - 1, (row['top_y'] - y0) * len(stage3) // max(1, len(predicted)))
        b = min(len(stage3), max(a + 1, (row['bottom_y'] - y0) * len(stage3) // max(1, len(predicted))))
        e = next((q for q in entries if q.id == row['cluster_id']), entries[0])
        stage3[a:b] = np.asarray(e.mean_rgb, np.uint8)
    stages = {'1 column': stage1, '2 modal colour': stage2, '3 intervals': stage3}
    return rows, stages
