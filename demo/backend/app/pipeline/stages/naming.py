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
    """Cut beds whose windows clearly match different swatches (boundaries the rule missed)."""
    rows = []
    for s in beds.itertuples():
        a, n = s.y_top - column.y_offset, (s.y_bottom - s.y_top) // win
        if n < 2 * min_run:
            rows.append((s.y_top, s.y_bottom))
            continue
        labels = []
        for i in range(n):
            window = describe(column.rgb[a + i * win:a + (i + 1) * win])  # once per window, not once per prototype
            labels.append(int(np.argmin([descriptor_distance(window, p) for _, _, p in protos])))
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
    """Name each bed with its closest prototype."""
    out = []
    for s in beds.itertuples():
        a, b = s.y_top - column.y_offset, s.y_bottom - column.y_offset
        if b - a > 4 * trim:
            a, b = a + trim, b - trim
        bed = describe(column.rgb[a:b])
        dist = np.array([descriptor_distance(bed, p) for _, _, p in protos])
        best = int(np.argmin(dist))
        out.append({"legend_id": protos[best][0], "lithology": protos[best][1]})
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
