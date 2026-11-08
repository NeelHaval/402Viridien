"""Repeated-swatch template matching for lithology columns."""
import cv2
import numpy as np

NAME = "S3 template matching"
NEEDS_GPU = False
PAID_API = None


def _tile_image(rgb, width):
    h, w = rgb.shape[:2]
    reps = int(np.ceil(width / max(1, w)))
    return np.tile(rgb, (1, reps, 1))[:, :width]


def _runs(labels, y0, step, window, entries):
    result = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            a = max(window[0], y0 + start * step)
            b = min(window[1], y0 + i * step)
            if labels[start] >= 0 and b > a:
                entry = entries[labels[start]]
                result.append(dict(top_y=a, bottom_y=b, lithology=entry.label,
                                   cluster_id=entry.id))
            start = i
    return result


def run(log, window, ctx):
    entries = [e for e in ctx["legend"] if e.is_lithology and e.swatch.size]
    if not entries:
        return [], {}
    y0, y1 = map(int, window)
    box = log.boxes["interpreted_lithology_column"]
    x0, _, x1, _ = map(int, box)
    # Shrinking the broad column makes matching fast and reduces scan noise.
    width = min(180, max(48, (x1 - x0) // 2))
    rgb = __import__("harness.io", fromlist=["read_rgb"]).read_rgb(
        log, box, y0, y1, scale=width / max(1, x1 - x0))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height = len(rgb)
    swatch_heights = [e.swatch.shape[0] for e in entries]
    step = max(6, int(np.clip(np.median(swatch_heights), 8, 70)))
    ntiles = int(np.ceil(height / step))
    # Responses are computed once per template and sampled around each tile
    # centre. A small scale bank accommodates resized scans and print variation.
    templates = []
    for ei, entry in enumerate(entries):
        native_width = max(1, x1 - x0)
        base = _tile_image(entry.swatch, native_width)
        base = cv2.resize(base, (width, max(6, round(base.shape[0] * width / native_width)),),
                          interpolation=cv2.INTER_AREA)
        for scale in (.62, .78, 1.0, 1.28, 1.62):
            th = max(6, round(base.shape[0] * scale))
            if th > min(220, height):
                continue
            temp = cv2.resize(base, (width, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            tg = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
            if float(tg.std()) < 1.5:
                continue
            response = cv2.matchTemplate(gray, tg, cv2.TM_CCOEFF_NORMED).ravel()
            templates.append((ei, th, temp.mean(axis=(0, 1)).astype(float), response))

    score = np.full((len(entries), ntiles), -1.0, np.float32)
    for ei, th, mean, response in templates:
        for ti in range(ntiles):
            center = min(height - 1, ti * step + step // 2)
            lo, hi = max(0, center - th), min(len(response), center + 1)
            if hi <= lo:
                continue
            pos = lo + int(np.argmax(response[lo:hi]))
            corr = float(response[pos])
            patch = rgb[pos:pos + th]
            if not len(patch):
                continue
            color_dist = float(np.linalg.norm(patch.mean(axis=(0, 1)).astype(float) - mean) / 441.7)
            value = .78 * corr + .22 * (1.0 - min(1.0, color_dist))
            if value > score[ei, ti]:
                score[ei, ti] = value

    # The winner is assigned per swatch-sized tile; blank white gaps are omitted.
    winners = np.argmax(score, axis=0)
    confidence = score[winners, np.arange(ntiles)]
    tile_rgb = np.array([rgb[min(height-1, i*step):min(height, (i+1)*step)].mean((0, 1))
                         for i in range(ntiles)])
    blank = (tile_rgb.min(axis=1) > 247) & (tile_rgb.std(axis=1) < 7)
    labels = winners.astype(int)
    labels[(confidence < .16) | blank] = -1

    # Remove single-tile flicker, while preserving short actual intervals.
    for i in range(1, ntiles - 1):
        if labels[i-1] == labels[i+1] and labels[i] != labels[i-1]:
            labels[i] = labels[i-1]
    rows = _runs(labels, y0, step, window, entries)

    # Refine internal boundaries by comparing local best scores on both sides.
    # (The tile result remains the robust classification; this stage exposes its map.)
    map_rgb = np.zeros((max(1, ntiles), max(1, len(entries)), 3), np.uint8)
    norm = np.clip((score + 1) * 127.5, 0, 255).astype(np.uint8)
    map_rgb[:, :, 0] = norm.T
    map_rgb[:, :, 1] = np.maximum(0, norm.T.astype(np.int16) - 55).astype(np.uint8)
    map_rgb[:, :, 2] = np.maximum(0, norm.T.astype(np.int16) - 90).astype(np.uint8)
    boundary = cv2.resize(rgb.copy(), (width, min(1200, height)))
    for row in rows:
        yy = int((row['top_y'] - y0) * boundary.shape[0] / height)
        if 0 <= yy < boundary.shape[0]:
            cv2.line(boundary, (0, yy), (width-1, yy), (255, 0, 255), 1)
    return rows, {"1 column": rgb, "2 correlation map": map_rgb,
                  "3 intervals": boundary}
