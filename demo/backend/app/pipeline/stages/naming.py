"""Stage 5: name every bed with the closest legend swatch, splitting beds that mix rocks, then merge neighbours."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline.stages.legend import is_rock
from app.pipeline.types import ColumnImage, Legend

RUN_BINS = np.array([1, 2, 3, 4, 6, 9, 13, 21, 10**6])  # run-length bin edges (px)
W_COLOR, W_INK, W_TEXTURE, TEXTURE_FULL_PX = 1 / 10, 4.0, 1.0, 24
NB = len(RUN_BINS) - 1
Descriptor = dict
Prototype = tuple[int, str, Descriptor]


def srgb_to_lab(rgb) -> np.ndarray:
    """sRGB (0-255) to CIE Lab."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def run_hist(mask: np.ndarray) -> np.ndarray:
    """Length-weighted histogram of runs of True along each row."""
    m = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    lengths = np.where(d == -1)[1] - np.where(d == 1)[1]
    h = np.histogram(lengths, bins=RUN_BINS, weights=lengths)[0].astype(np.float64)
    return h / h.sum() if h.sum() else h


def describe(rgb: np.ndarray) -> Descriptor:
    """Colour + texture descriptor of an RGB patch (swatch interior or rows of the column)."""
    rgb = np.asarray(rgb)
    dark = rgb.mean(2) < 128
    light = rgb[~dark]
    return {"lab": srgb_to_lab(np.median(light, 0) if len(light) else np.zeros(3)), "ink": float(dark.mean()),
            "h_dark": run_hist(dark), "v_dark": run_hist(dark.T), "h_gap": run_hist(~dark), "v_gap": run_hist(~dark.T),
            "rows": rgb.shape[0]}


LIGHT_SUM = 384  # a pixel is dark when mean(rgb) < 128, i.e. sum(rgb) < 384 (exact in integers)
CHUNK_ROWS = 16384
_LAB_CACHE: dict[tuple, np.ndarray] = {}


def _lab_of(median_rgb: tuple) -> np.ndarray:
    """srgb_to_lab with a cache: pages have few distinct paper colours, so most windows hit it."""
    lab = _LAB_CACHE.get(median_rgb)
    if lab is None:
        lab = _LAB_CACHE[median_rgb] = srgb_to_lab(np.array(median_rgb, np.float64))
    return lab


def _hist_horizontal(mask: np.ndarray, row_win: np.ndarray, n: int) -> np.ndarray:
    """Per-window run-length histograms (same bins and weights as `run_hist`) of runs of True along each row of `mask`."""
    pad = np.zeros((mask.shape[0], mask.shape[1] + 2), bool)
    pad[:, 1:-1] = mask
    rows, starts = np.nonzero(pad[:, 1:] & ~pad[:, :-1])
    _, ends = np.nonzero(~pad[:, 1:] & pad[:, :-1])
    return _bin_runs(row_win[rows], ends - starts, n)


def _hist_vertical(mask: np.ndarray, first: np.ndarray, row_win: np.ndarray, n: int) -> np.ndarray:
    """Same for runs along the columns, broken at window boundaries (`first` marks the first row of each window)."""
    r = mask.shape[0]
    prev = np.zeros_like(mask)
    prev[1:] = mask[:-1]
    prev[first] = False
    start_mask = mask & ~prev
    # a run ends before row i when row i-1 is set and row i is unset, or starts a new window (row r is a sentinel)
    ext = np.zeros((r + 1, mask.shape[1]), bool)
    ext[:r] = mask
    new_window = np.append(first, True)
    end_mask = np.zeros_like(ext)
    end_mask[1:] = mask & (~ext[1:] | new_window[1:, None])
    xs, s_rows = np.nonzero(start_mask.T)
    _, e_rows = np.nonzero(end_mask.T)
    return _bin_runs(row_win[s_rows], e_rows - s_rows, n)


def _bin_runs(win: np.ndarray, lengths: np.ndarray, n: int) -> np.ndarray:
    """Length-weighted histogram over RUN_BINS per window, normalised like `run_hist` (all zeros when there are no runs)."""
    bins = np.searchsorted(RUN_BINS, lengths, side="right") - 1
    h = np.bincount(win * NB + bins, weights=lengths, minlength=n * NB).reshape(n, NB).astype(np.float64)
    total = h.sum(1)
    nonzero = total > 0
    h[nonzero] /= total[nonzero, None]
    return h


def describe_ranges(rgb: np.ndarray, ranges: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """`describe` for many row ranges of one column at once: arrays lab (n,3), ink (n), h_dark/v_dark/h_gap/v_gap (n,NB), rows (n).

    Identical arithmetic to calling `describe(rgb[a:b])` per range (verified by tests), without the per-range Python loop.
    """
    n = len(ranges)
    out = {"lab": np.zeros((n, 3)), "ink": np.zeros(n), "rows": np.zeros(n), "h_dark": np.zeros((n, NB)),
           "v_dark": np.zeros((n, NB)), "h_gap": np.zeros((n, NB)), "v_gap": np.zeros((n, NB))}
    start = 0
    while start < n:
        stop, rows = start, 0
        while stop < n and (stop == start or rows + ranges[stop][1] - ranges[stop][0] <= CHUNK_ROWS):
            rows += ranges[stop][1] - ranges[stop][0]
            stop += 1
        _describe_chunk(rgb, ranges[start:stop], out, start)
        start = stop
    return out


def _describe_chunk(rgb: np.ndarray, ranges: list[tuple[int, int]], out: dict, offset: int) -> None:
    """Fill `out[...][offset:offset+len(ranges)]` for a group of ranges that fits in memory."""
    m = len(ranges)
    lengths = np.array([b - a for a, b in ranges])
    block = np.concatenate([rgb[a:b] for a, b in ranges]) if m > 1 else rgb[ranges[0][0]:ranges[0][1]]
    row_win = np.repeat(np.arange(m), lengths)
    first = np.zeros(len(block), bool)
    first[np.cumsum(lengths) - lengths] = True
    dark = block.sum(2, dtype=np.int32) < LIGHT_SUM
    width = block.shape[1]
    dark_count = np.bincount(row_win, weights=dark.sum(1), minlength=m)
    light_count = lengths * width - dark_count
    sl = slice(offset, offset + m)
    out["rows"][sl] = lengths
    out["ink"][sl] = dark_count / (lengths * width)
    light = ~dark
    med = np.zeros((m, 3))
    for ch in range(3):
        key = (row_win[:, None] * 256 + block[..., ch])[light]
        counts = np.bincount(key, minlength=m * 256).reshape(m, 256).cumsum(1)
        lo, hi = (light_count - 1) // 2, light_count // 2
        med[:, ch] = ((counts > lo[:, None]).argmax(1) + (counts > hi[:, None]).argmax(1)) / 2.0
    med[light_count == 0] = 0.0
    out["lab"][sl] = [_lab_of(tuple(row)) for row in med]
    out["h_dark"][sl] = _hist_horizontal(dark, row_win, m)
    out["h_gap"][sl] = _hist_horizontal(light, row_win, m)
    out["v_dark"][sl] = _hist_vertical(dark, first, row_win, m)
    out["v_gap"][sl] = _hist_vertical(light, first, row_win, m)


def distances_to_prototypes(desc: dict[str, np.ndarray], protos: list[Prototype]) -> np.ndarray:
    """(n, len(protos)) matrix equal to `descriptor_distance(window_i, proto_j)` for every pair."""
    n = len(desc["ink"])
    labs = np.array([p["lab"] for _, _, p in protos])
    uniq, inverse = np.unique(desc["lab"], axis=0, return_inverse=True)
    lab_d = np.array([[np.linalg.norm(u - lab) for lab in labs] for u in uniq])[inverse.ravel()]
    tv = lambda k: 0.5 * np.abs(desc[k][:, None, :] - np.array([p[k] for _, _, p in protos])[None]).sum(2)  # noqa: E731
    v_weight = np.minimum(1.0, desc["rows"] / TEXTURE_FULL_PX)[:, None]
    d_tex = (tv("h_dark") + tv("h_gap") + v_weight * (tv("v_dark") + tv("v_gap"))) / 4
    ink = np.array([p["ink"] for _, _, p in protos])[None]
    return lab_d * W_COLOR + np.abs(desc["ink"][:, None] - ink) * W_INK + W_TEXTURE * d_tex if n else np.zeros((0, len(protos)))


def descriptor_distance(seg: Descriptor, sw: Descriptor) -> float:
    """Distance between a bed and a swatch; vertical runs count less in thin beds."""
    tv = lambda k: 0.5 * np.abs(seg[k] - sw[k]).sum()  # noqa: E731
    v_weight = min(1.0, seg["rows"] / TEXTURE_FULL_PX)
    d_tex = (tv("h_dark") + tv("h_gap") + v_weight * (tv("v_dark") + tv("v_gap"))) / 4
    return float(np.linalg.norm(seg["lab"] - sw["lab"]) * W_COLOR + abs(seg["ink"] - sw["ink"]) * W_INK + W_TEXTURE * d_tex)


BLANK: Descriptor = {"lab": srgb_to_lab([255, 255, 255]), "ink": 0.0, "h_dark": np.zeros(NB), "v_dark": np.zeros(NB),
                     "h_gap": np.eye(NB)[-1], "v_gap": np.eye(NB)[-1], "rows": 1}


def prototypes(legend: Legend) -> list[Prototype]:
    """Blank first, then every rock swatch of the legend."""
    return [(-1, "blank", BLANK)] + [(e.legend_id, e.lithology, describe(e.pixels)) for e in legend.entries if is_rock(e.lithology)]


def split_mixed(beds: pd.DataFrame, column: ColumnImage, protos: list[Prototype], win: int = 16, min_run: int = 3) -> pd.DataFrame:
    """Cut beds whose windows clearly match different swatches (boundaries the rule missed).

    All windows of all long-enough beds are described and matched in one batch; the cutting logic is unchanged.
    """
    tops = (beds.y_top - column.y_offset).to_numpy()
    counts = ((beds.y_bottom - beds.y_top) // win).to_numpy()
    windows = [(int(a) + i * win, int(a) + (i + 1) * win) for a, n in zip(tops, counts) if n >= 2 * min_run for i in range(n)]
    best = np.argmin(distances_to_prototypes(describe_ranges(column.rgb, windows), protos), axis=1) if windows else np.zeros(0, int)
    rows, used = [], 0
    for s, n in zip(beds.itertuples(), counts):
        if n < 2 * min_run:
            rows.append((s.y_top, s.y_bottom))
            continue
        labels = best[used:used + n].tolist()
        used += n
        runs: list[list[int]] = []  # [label, first window, n windows]
        for i, lab in enumerate(labels):
            if runs and runs[-1][0] == lab:
                runs[-1][2] += 1
            else:
                runs.append([lab, i, 1])
        kept = [r for r in runs if r[2] >= min_run] or [max(runs, key=lambda r: r[2])]
        starts = [r[1] for k, r in enumerate(kept) if k == 0 or r[0] != kept[k - 1][0]][1:]
        edges = [s.y_top] + [s.y_top + i * win for i in starts] + [s.y_bottom]
        rows += list(zip(edges[:-1], edges[1:]))
    return pd.DataFrame({"y_top": [r[0] for r in rows], "y_bottom": [r[1] for r in rows]})


def name_beds(beds: pd.DataFrame, column: ColumnImage, protos: list[Prototype], trim: int = 2) -> pd.DataFrame:
    """Name each bed with its closest prototype (all beds described and matched in one batch)."""
    ranges = []
    for s in beds.itertuples():
        a, b = s.y_top - column.y_offset, s.y_bottom - column.y_offset
        ranges.append((a + trim, b - trim) if b - a > 4 * trim else (a, b))
    best = np.argmin(distances_to_prototypes(describe_ranges(column.rgb, ranges), protos), axis=1) if ranges else np.zeros(0, int)
    out = [{"legend_id": protos[b][0], "lithology": protos[b][1]} for b in best]
    return pd.concat([beds[["y_top", "y_bottom"]].reset_index(drop=True), pd.DataFrame(out)], axis=1)


def merge_named(named: pd.DataFrame) -> pd.DataFrame:
    """Merge touching beds that carry the same legend entry."""
    rows: list[dict] = []
    for r in named.itertuples():
        if rows and rows[-1]["legend_id"] == r.legend_id and rows[-1]["y_bottom"] == r.y_top:
            rows[-1]["y_bottom"] = r.y_bottom
        else:
            rows.append({"y_top": r.y_top, "y_bottom": r.y_bottom, "legend_id": r.legend_id, "lithology": r.lithology})
    return pd.DataFrame(rows, columns=["y_top", "y_bottom", "legend_id", "lithology"])


def name_layers(beds: pd.DataFrame, column: ColumnImage, legend: Legend, *, split_window: int = 16, split_min_run: int = 3) -> pd.DataFrame:
    """Named lithology layers. With no legend every bed stays its own layer called `Unknown` (legend_id -2)."""
    if not legend.entries:
        return beds.assign(legend_id=-2, lithology="Unknown")[["y_top", "y_bottom", "legend_id", "lithology"]]
    protos = prototypes(legend)
    return merge_named(name_beds(split_mixed(beds, column, protos, split_window, split_min_run), column, protos))
