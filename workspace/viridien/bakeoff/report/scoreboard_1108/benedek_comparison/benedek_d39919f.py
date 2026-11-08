# %pip -q install torch scipy pandas pillow numpy matplotlib easyocr


# Local: install torch scipy pandas pillow numpy matplotlib easyocr in a venv.
# Colab: install dependencies in the notebook's setup cell before running.
import os
import sys

IN_COLAB = "google.colab" in sys.modules
# Google Drive setup (disabled while using local data).
# if IN_COLAB:
#     from google.colab import drive
#     drive.mount('/content/drive')
if not IN_COLAB:
    os.environ.setdefault("MPLBACKEND", "Agg")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

Image.MAX_IMAGE_PIXELS = None


LOCAL_REPO = (
    Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
)
# Support notebook kernels started from either the repository root or notebooks/.
if LOCAL_REPO.name == "notebooks":
    LOCAL_REPO = LOCAL_REPO.parent
ROOT = Path(os.environ.get("HTB_ROOT", str(LOCAL_REPO / "data")))
GT_DIR = Path(os.environ.get("HTB_LABEL_DIR", str(ROOT / "test")))
OUTPUT_DIR = Path(os.environ.get("HTB_OUTPUT_DIR", str(ROOT / "predictions")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(min(4, os.cpu_count() or 1))
STEM = "60808400140000_hadrian 2_masterlog_2in_10000-21962"
IMAGE_PATH = ROOT / "dataset" / "images" / f"{STEM}.tif"
ANNOTATION_PATH = ROOT / "dataset" / "bbox_annotation" / f"{STEM}.json"
LABEL_PATH = GT_DIR / f"{STEM}.json"

WINDOW = 3
MIN_GAP = 2
INK_K = 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Crop a region of a very tall 1-bit or colour scan as RGB without decoding the whole page
def crop_rgb(img, box, scale=1.0, strip=2000):
    x0, y0, x1, y1 = box
    out = Image.new("RGB", (round((x1 - x0) * scale), round((y1 - y0) * scale)))
    for ys in range(y0, y1, strip):
        ye = min(ys + strip, y1)
        part = img.crop((x0, ys, x1, ye)).convert("RGB")
        top, bottom = round((ys - y0) * scale), round((ye - y0) * scale)
        if scale != 1.0:
            part = part.resize((out.width, max(1, bottom - top)), Image.Resampling.BOX)
        out.paste(part, (0, top))
    return out


# Open the image and its annotation, and print the bounding boxes
image = Image.open(IMAGE_PATH)
annotation = json.loads(ANNOTATION_PATH.read_text())
boxes = annotation["bounding_boxes"]

assert (annotation["width"], annotation["height"]) == image.size
print("size:", image.size, "mode:", image.mode)
for name, (x0, y0, x1, y1) in boxes.items():
    print(f"{name:30s} x {x0:5d}-{x1:5d}   y {y0:6d}-{y1:6d}")


# Load the interpreted lithology column as an RGB array and remember its y offset in the full image
def load_column(image_path, annotation_path):
    ann = json.loads(Path(annotation_path).read_text())
    x0, y0, x1, y1 = ann["bounding_boxes"]["interpreted_lithology_column"]
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Empty lithology column in {annotation_path}")
    with Image.open(image_path) as img:
        return np.array(crop_rgb(img, (x0, y0, x1, y1))), y0


col, y0 = load_column(IMAGE_PATH, ANNOTATION_PATH)
print("column array:", col.shape, "y offset:", y0)


# Prediction files: one record per page in the label-file schema; save_record validates before writing
import datetime

SCHEMA_VERSION = 1
RUN_NAME = "main"
RUN_DIR = OUTPUT_DIR / RUN_NAME
TEST_STEMS = sorted(
    p.stem for p in GT_DIR.glob("*.json") if json.loads(p.read_text()).get("layers")
)
print(f"{len(TEST_STEMS)} labelled test pages in {GT_DIR}")


def record_path(stem):
    return RUN_DIR / f"{stem}.json"


# The page's record, or an empty one if no section has written it yet
def load_record(stem):
    if record_path(stem).exists():
        return json.loads(record_path(stem).read_text())
    ann = json.loads(
        (ROOT / "dataset" / "bbox_annotation" / f"{stem}.json").read_text()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stem": stem,
        "image_name": ann["image_name"],
        "written_by": {},
        "legend": [],
        "layers": [],
    }


# Layers from a table with y_top / y_bottom (+ legend_id / lithology once segments are named; legend_id < 0 = unnamed)
def layers_from_table(table):
    out = []
    for r in table.sort_values("y_top").itertuples():
        layer = {"top_px": int(r.y_top), "bottom_px": int(r.y_bottom)}
        lid = getattr(r, "legend_id", None)
        if lid is not None and not pd.isna(lid) and int(lid) >= 0:
            layer["composition"] = [
                {"legend_id": int(lid), "type": str(r.lithology), "proportion": 1.0}
            ]
        out.append(layer)
    return out


# Legend entries from a table with legend_id / lithology / box (box in full-image pixels)
def legend_from_table(table):
    return [
        {
            "id": int(r.legend_id),
            "box": [int(v) for v in r.box],
            "label_printed": str(r.lithology),
        }
        for r in table.itertuples()
    ]


# Every problem with a record, as readable strings (empty list = valid)
def record_errors(rec):
    errs = [
        f"missing key {k!r}"
        for k in (
            "schema_version",
            "stem",
            "image_name",
            "written_by",
            "legend",
            "layers",
        )
        if k not in rec
    ]
    if errs:
        return errs
    if rec["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {rec['schema_version']} != {SCHEMA_VERSION}")
    ids = set()
    for i, e in enumerate(rec["legend"]):
        box = e.get("box")
        if not isinstance(e.get("id"), int) or e["id"] in ids:
            errs.append(f"legend[{i}]: id must be a unique int, got {e.get('id')!r}")
        ids.add(e.get("id"))
        if not isinstance(e.get("label_printed"), str):
            errs.append(f"legend[{i}]: label_printed must be a string")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(v, int) for v in box)
            and box[0] < box[2]
            and box[1] < box[3]
        ):
            errs.append(
                f"legend[{i}]: box must be [x0, y0, x1, y1] ints with x0 < x1, y0 < y1, got {box!r}"
            )
    prev = None
    for i, l in enumerate(rec["layers"]):
        t, b = l.get("top_px"), l.get("bottom_px")
        if not (isinstance(t, int) and isinstance(b, int) and t < b):
            errs.append(f"layers[{i}]: needs int top_px < bottom_px, got {t!r}, {b!r}")
            continue
        if prev is not None and t < prev:
            errs.append(
                f"layers[{i}]: starts at {t}, before the previous layer ends ({prev}); layers must be sorted and not overlap"
            )
        prev = b
        for j, c in enumerate(l.get("composition", [])):
            if not (
                isinstance(c.get("legend_id"), int)
                and isinstance(c.get("type"), str)
                and isinstance(c.get("proportion"), (int, float))
            ):
                errs.append(
                    f"layers[{i}].composition[{j}]: needs int legend_id, str type, numeric proportion"
                )
            elif rec["legend"] and c["legend_id"] not in ids:
                errs.append(
                    f"layers[{i}].composition[{j}]: legend_id {c['legend_id']} is not in this record's legend"
                )
    return errs


# Validate and write a page's record; section names who wrote it and meta records the settings used
def save_record(rec, section, **meta):
    rec["written_by"][section] = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        **meta,
    }
    errs = record_errors(rec)
    if errs:
        raise ValueError(
            f"invalid prediction record for {rec.get('stem')} ({section}):\n  "
            + "\n  ".join(errs[:20])
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    record_path(rec["stem"]).write_text(json.dumps(rec, indent=1))


# Check every test page has a valid record (run before evaluating)
def check_run(stems=None):
    for stem in TEST_STEMS if stems is None else stems:
        if not record_path(stem).exists():
            raise FileNotFoundError(f"no prediction record for {stem} in {RUN_DIR}")
        errs = record_errors(json.loads(record_path(stem).read_text()))
        if errs:
            raise ValueError(
                f"invalid prediction record for {stem}:\n  " + "\n  ".join(errs[:20])
            )
    print(
        f"{len(TEST_STEMS if stems is None else stems)} valid prediction records in {RUN_DIR}"
    )


from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist

H = 8
TRAIN_PAGES = [
    (p, ROOT / "dataset" / "bbox_annotation" / f"{p.stem}.json")
    for ext in ("*.tif", "*.tiff", "*.png")
    for p in sorted((ROOT / "dataset" / "images").glob(ext))
    if (ROOT / "dataset" / "bbox_annotation" / f"{p.stem}.json").is_file()
]
print(
    f"{len(TRAIN_PAGES)} training pages:", *[p.stem for p, _ in TRAIN_PAGES], sep="\n  "
)


# Trim the thick border lines on either side of the column
def trim_borders(col):
    dark = col.mean(2, dtype=np.float32) < 100
    profile = dark.mean(0)
    idx = np.where(profile > 0.6)[0]
    mid = col.shape[1] // 2
    left, right = idx[idx < mid], idx[idx >= mid]
    x0 = left.max() + 3 if left.size else 0
    x1 = right.min() - 2 if right.size else col.shape[1]
    return col if x1 - x0 < 10 else col[:, x0:x1]


# Feature vector of one stripe: black-pixel density, component count, mean distance between components,
# mean component size, and the R, G, B of the original (non-black) background
STRIPE_FEATURE_NAMES = [
    "density",
    "n_components",
    "mean_dist",
    "mean_size",
    "bg_r",
    "bg_g",
    "bg_b",
]


def stripe_features(stripe, thresh=128, max_comp=2000):
    black = (
        stripe.mean(2, dtype=np.float32) < thresh
    )  # background -> white, ink -> black
    density = black.mean()
    bg = stripe[~black]
    bg_rgb = (
        bg.mean(0) / 255.0 if len(bg) else np.ones(3)
    )  # original background colour (white if no background)
    lab, n = label(black, structure=np.ones((3, 3)))  # 8-connected black components
    if n == 0:
        return np.array([density, 0, 0, 0, *bg_rgb], np.float32)
    ids = np.arange(1, n + 1)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    if n > 1:
        cen = np.array(center_of_mass(black, lab, ids))
        if n > max_comp:  # keep pairwise distances cheap on very noisy stripes
            cen = cen[np.random.default_rng(0).choice(n, max_comp, replace=False)]
        mean_dist = pdist(cen).mean()
    else:
        mean_dist = 0.0
    return np.array([density, n, mean_dist, sizes.mean(), *bg_rgb], np.float32)


FEATURE_NAMES = STRIPE_FEATURE_NAMES


# Turn the column into a list of feature vectors, one per stripe of h rows (the last stripe may be shorter)
def column_to_vectors(col, h=H):
    col = trim_borders(col)
    return [stripe_features(col[s : s + h]) for s in range(0, len(col), h)]


# Raw feature matrix (n_stripes x 7): the stripe vectors stacked
def raw_stripe_features(col, h=H):
    return np.stack(column_to_vectors(col, h)).astype(np.float32)


# Z-score each feature over the column
def zscore(R):
    scale = R.std(0)
    scale[scale < 1e-6] = 1.0
    return (R - R.mean(0)) / scale


# Same features, z-scored (input to both boundary detectors)
def column_features(col, h=H):
    return zscore(raw_stripe_features(col, h))


# Turn boundary stripes into a table of bed tops and bottoms in full-image coordinates (n_rows = rows in the column)
def to_table(peaks, n_rows, y_offset, h=H):
    edges = np.concatenate([[0], np.sort(peaks) * h, [n_rows]]).astype(int)
    return pd.DataFrame(
        {"y_top": edges[:-1] + y_offset, "y_bottom": edges[1:] + y_offset}
    )


# Compute the stripe features once for the column loaded in Section 1
F = column_features(col)
print(
    f"feature matrix: {F.shape} ({F.shape[0]} stripes of {H} px x {len(FEATURE_NAMES)} features: {FEATURE_NAMES})"
)
print("first vector:", raw_stripe_features(col[: 4 * H])[0])

# Visualise the stripe feature vectors of a column section: the column is rotated to horizontal (top of the
# column at the left), cut into H-px stripes, and each feature channel is plotted below, aligned with the stripes
import matplotlib.pyplot as plt

VIS_START = (
    13000  # first row of the column section to show (px, relative to the column)
)
VIS_ROWS = 3000  # number of rows to show (px)


def show_stripe_features(col, start=VIS_START, rows=VIS_ROWS, h=H):
    start = (
        min(max(0, start), max(0, len(col) - h)) // h
    ) * h  # align to a stripe boundary
    c = trim_borders(col)[start : start + rows]
    R = raw_stripe_features(col, h)[start // h : (start + rows + h - 1) // h]
    n = len(R)
    rot = np.rot90(c)  # column top -> left, so depth runs horizontally
    width = rot.shape[1]
    centers = (np.arange(n) + 0.5) * h
    colors = dict(bg_r="tab:red", bg_g="tab:green", bg_b="tab:blue")

    fig, axes = plt.subplots(
        1 + len(STRIPE_FEATURE_NAMES),
        1,
        sharex=True,
        figsize=(16, 2 + 0.9 * len(STRIPE_FEATURE_NAMES)),
        gridspec_kw={
            "height_ratios": [3] + [1] * len(STRIPE_FEATURE_NAMES),
            "hspace": 0.12,
        },
    )
    ax = axes[0]
    ax.imshow(
        rot, extent=(0, width, 0, rot.shape[0]), interpolation="nearest", aspect="auto"
    )
    for x in np.arange(0, width + 1, h):  # stripe boundaries
        for a in axes:
            a.axvline(x, color="k", lw=0.3, alpha=0.35)
    ax.set_yticks([])
    ax.set_title(
        f"Column rows {start}-{start + width} px, stripes of {h} px (top of column at left)"
    )

    for a, k, name in zip(axes[1:], range(R.shape[1]), STRIPE_FEATURE_NAMES):
        a.step(
            np.arange(n + 1) * h,
            np.r_[R[:, k], R[-1, k]],
            where="post",
            color=colors.get(name, "k"),
            lw=1,
        )
        a.plot(centers, R[:, k], ".", color=colors.get(name, "k"), ms=3)
        a.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        a.tick_params(labelsize=7)
    axes[-1].set_xlim(0, width)
    axes[-1].set_xlabel("row within column (px)")
    if IN_COLAB:
        plt.show()
    else:
        plt.close(fig)


show_stripe_features(col)


# Boundary rule: score each stripe by how much the average features change between the windows above and below it
def boundary_score(F, w=WINDOW):
    c = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, 0)])
    n = len(F)
    ys = np.arange(w, n - w)
    above = (c[ys] - c[ys - w]) / w
    below = (c[ys + w] - c[ys]) / w
    score = np.zeros(n)
    score[ys] = np.linalg.norm(below - above, axis=1)
    return score


# Robust peak threshold: median + k * MAD, never below 1
def robust_threshold(s, k):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return max(med + k * max(mad, 1e-6), 1.0)


# Boundary stripes from the rule
def ink_boundaries(F, w=WINDOW, k=INK_K, gap=MIN_GAP):
    s = boundary_score(F, w)
    peaks, _ = find_peaks(s, height=robust_threshold(s, k), distance=gap)
    return peaks


# Detect the boundaries of the Section 1 column
peaks_rule = ink_boundaries(F)
print(f"{len(peaks_rule)} boundaries from the rule")
beds_rule = to_table(peaks_rule, len(col), y0)
beds_rule

# Write the beds as the page's prediction record (label-file schema, 1.1): layers = [{top_px, bottom_px}], validated on save
rec = load_record(STEM)
rec["layers"] = layers_from_table(beds_rule[beds_rule.y_bottom > beds_rule.y_top])
save_record(
    rec,
    "2.2",
    method="windowed_change_rule",
    h=H,
    k=INK_K,
    window=WINDOW,
    min_gap=MIN_GAP,
)
check_run([STEM])
print(json.dumps({**rec, "layers": rec["layers"][:3]}, indent=1))


# Pages: locate, load (cached) and segment any page by stem
SAVE_METHOD = "rule"  # the Section 2.2 windowed-change rule
PAGES = {}  # stem -> {'col', 'y_offset', 'features'}


# Image and annotation of a page
def page_paths(stem):
    ann = ROOT / "dataset" / "bbox_annotation" / f"{stem}.json"
    for ext in (".tif", ".tiff", ".png"):
        img = ROOT / "dataset" / "images" / f"{stem}{ext}"
        if img.is_file() and ann.is_file():
            return img, ann
    raise FileNotFoundError(
        f"no image + bbox_annotation pair for {stem} under {ROOT / 'dataset'}"
    )


# Column, its y offset and its stripe features, decoded once per page
def load_page(stem):
    if stem not in PAGES:
        c, off = load_column(*page_paths(stem))
        PAGES[stem] = {"col": c, "y_offset": off, "features": column_features(c)}
    return PAGES[stem]


# Predicted beds of a page as a y_top / y_bottom table in full-image pixels
def predict_segments(stem, method=SAVE_METHOD):
    page = load_page(stem)
    if method != "rule":
        raise ValueError(f"method must be 'rule', got {method!r}")
    peaks = ink_boundaries(page["features"])
    beds = to_table(peaks, len(page["col"]), page["y_offset"])
    return beds[beds.y_bottom > beds.y_top].reset_index(drop=True)


# The Section 1 page is already decoded: seed the cache instead of reading the scan again
PAGES[STEM] = {"col": col, "y_offset": y0, "features": F}

# Segments of the Section 1 page, used by the Section 4 cells
segments = predict_segments(STEM)
print(f"{len(segments)} segments for {STEM} ({SAVE_METHOD}); features {F.shape[1]} ch")
segments.head()


from scipy import ndimage
import re

# Legend entries that are drilling / sampling symbols rather than rock types
NON_ROCK = re.compile(
    r"\b(core|dst|shoe|survey|mud (loss|gain)|lot|fit|rft|mdt|sidewall|casing|fossils?|"
    r"foraminifera|diatoms?|gas|oil|show|cuttings|sample|bit|trip)\b",
    re.I,
)


def is_rock(label):
    return not NON_ROCK.search(label or "")


# Overlap of two boxes (intersection over union)
def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union else 0.0


# Find swatch rectangles in the legend box: dark-framed boxes, plus solid or closed non-white rectangles
# (colour fills, light frames); flat bars up to 8:1 are allowed
def detect_swatches(
    img, legend_box, min_w=25, max_w=200, min_h=15, max_h=120, max_aspect=8
):
    lx0, ly0, _, _ = legend_box
    rgb = np.array(crop_rgb(img, legend_box)).astype(np.int16)

    # Swatch-sized blobs of a mask, after dropping table rules longer than any swatch (swatches drawn against them separate)
    def candidates(mask):
        lines = ndimage.binary_opening(
            mask, np.ones((1, max_w + 1))
        ) | ndimage.binary_opening(mask, np.ones((max_h + 1, 1)))
        lab, _ = ndimage.label(mask & ~lines)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            w, h = x1 - x0, y1 - y0
            if (
                min_w <= w <= max_w
                and min_h <= h <= max_h
                and 0.7 <= w / h <= max_aspect
            ):
                yield lab[y0:y1, x0:x1] == i, (x0 + lx0, y0 + ly0, x1 + lx0, y1 + ly0)

    framed = []
    for m, b in candidates(rgb.mean(2) < 110):
        sides = [
            max(m[k].mean() for k in (0, 1, 2)),
            max(m[-k].mean() for k in (1, 2, 3)),
            max(m[:, k].mean() for k in (0, 1, 2)),
            max(m[:, -k].mean() for k in (1, 2, 3)),
        ]
        if min(sides) >= 0.8:  # all four sides framed (allowing a stray pixel outside)
            framed.append(b)
    solid = [
        b
        for m, b in candidates((255 - rgb).max(2) > 40)
        if ndimage.binary_fill_holes(m).mean() >= 0.9
    ]
    boxes = framed + [b for b in solid if not any(box_iou(b, f) > 0.3 for f in framed)]
    boxes = [
        b
        for b in boxes
        if not any(
            o != b and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3]
            for o in boxes
        )
    ]
    return sorted(boxes, key=lambda b: (round(b[1] / 20), b[0]))


# OCR the printed name to the right of each swatch (needs easyocr; returns None names without it).
# The name is the text line that overlaps the swatch most (text often sits lower than the swatch), cropped and upscaled.
def read_swatch_labels(img, boxes, legend_box, text_px=64, pad=4, trace=None):
    try:
        import easyocr
    except ImportError:
        print(
            "easyocr not installed (pip install easyocr): swatches will be named legend_<id>"
        )
        return [None] * len(boxes)
    if not hasattr(read_swatch_labels, "reader"):
        read_swatch_labels.reader = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            verbose=False,
            model_storage_directory=str(OUTPUT_DIR / "ocr_models"),
            user_network_directory=str(OUTPUT_DIR / "ocr_networks"),
        )
    names = []
    for bi, (x0, y0, x1, y1) in enumerate(boxes):
        h = y1 - y0
        step = {
            "i": bi,
            "swatch": (x0, y0, x1, y1),
            "search": None,
            "line": None,
            "crop": None,
            "text": None,
        }
        if trace is not None:  # optional evidence for the Section 3.2 visual check
            trace.append(step)
        right = [b[0] for b in boxes if b[0] > x1 and abs(b[1] - y0) < h / 2]
        xe = min(min(right) - 4 if right else legend_box[2], x1 + 5 * (x1 - x0))
        step["search"] = (x1 + 2, y0 - h, xe, y1 + h)
        if xe <= x1 + 2:
            names.append(None)
            continue
        band = np.array(
            img.crop((x1 + 2, y0 - h, xe, y1 + h)).convert("L")
        )  # swatch rows are band rows h..2h
        ink = band < 160
        rules = np.zeros_like(ink)
        rules[:, ink.mean(0) > 0.9] = True  # table rules, not text
        rules[ink.mean(1) > 0.9, :] = True
        band = np.where(rules, 255, band)
        ink &= ~rules
        edges = np.flatnonzero(np.diff(np.r_[0, ink.any(1).astype(np.int8), 0]))
        runs = [
            r for r in zip(edges[::2], edges[1::2]) if min(r[1], 2 * h) > max(r[0], h)
        ]
        if not runs:
            names.append(None)
            continue
        a, b = max(runs, key=lambda r: min(r[1], 2 * h) - max(r[0], h))
        cols = np.flatnonzero(ink[a:b].any(0))
        crop = band[
            max(a - pad, 0) : b + pad, max(cols[0] - pad, 0) : cols[-1] + pad
        ].astype(np.uint8)
        step["line"] = (x1 + 2 + cols[0], y0 - h + a, x1 + 2 + cols[-1], y0 - h + b)
        step["crop"] = crop
        f = max(1.0, text_px / crop.shape[0])
        crop = Image.fromarray(crop).resize(
            (round(crop.shape[1] * f), round(crop.shape[0] * f)), Image.LANCZOS
        )
        crop = np.pad(np.array(crop), 12, constant_values=255)
        names.append(
            " ".join(
                read_swatch_labels.reader.readtext(crop, detail=0, paragraph=True)
            ).strip()
            or None
        )
        step["text"] = names[-1]
    return names


# Legend of a page: labelled boxes from ROOT/test when available, otherwise detected swatches (+ OCR names)
def load_legend(stem, source="auto", ocr=True):
    img_path, ann_path = page_paths(stem)
    label_path = GT_DIR / f"{stem}.json"
    if source in ("auto", "labels") and label_path.exists():
        entries = json.loads(label_path.read_text()).get("legend") or []
        if entries:
            return pd.DataFrame(
                [
                    {
                        "legend_id": e["id"],
                        "lithology": e["label_printed"],
                        "box": tuple(e["box"]),
                    }
                    for e in entries
                ]
            )
    if source == "labels":
        raise FileNotFoundError(f"no labelled legend for {stem}")
    img = Image.open(img_path)
    legend_box = json.loads(Path(ann_path).read_text())["bounding_boxes"][
        "lithology_legend"
    ]
    boxes = detect_swatches(img, legend_box)
    names = read_swatch_labels(img, boxes, legend_box) if ocr else [None] * len(boxes)
    return pd.DataFrame(
        [
            {"legend_id": i, "lithology": n or f"legend_{i}", "box": b}
            for i, (b, n) in enumerate(zip(boxes, names))
        ],
        columns=["legend_id", "lithology", "box"],
    )


# Swatch interior with the frame removed
def swatch_pixels(img, box, inset=3):
    x0, y0, x1, y1 = box
    return np.array(
        img.crop((x0 + inset, y0 + inset, x1 - inset, y1 - inset)).convert("RGB")
    )


# Legend of the Section 1 page, then every swatch with its id and name (greyed names are not rock candidates)
legend = load_legend(STEM)
legend_box = annotation["bounding_boxes"]["lithology_legend"]
detected = detect_swatches(image, legend_box)
print(len(legend), "legend entries,", len(detected), "swatches detected automatically")


def show_legend(legend, img):
    n = len(legend)
    if n == 0:
        print("No legend swatches detected")
        return
    cols = min(n, 9)
    fig, axes = plt.subplots(
        (n + cols - 1) // cols,
        cols,
        figsize=(1.6 * cols, 1.5 * ((n + cols - 1) // cols)),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, legend.itertuples()):
        ax.imshow(swatch_pixels(img, r.box), interpolation="nearest")
        ax.set_title(
            f"{r.legend_id}: {r.lithology}",
            fontsize=8,
            color="black" if is_rock(r.lithology) else "grey",
        )
    plt.tight_layout()
    if IN_COLAB:
        plt.show()
    else:
        plt.close(fig)


show_legend(legend, image)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
from IPython.display import display, Image as IPImage

CHECK_DIR = OUTPUT_DIR / "legend_check"
CHECK_DIR.mkdir(parents=True, exist_ok=True)


def _show(fig, name):
    path = CHECK_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    display(IPImage(filename=str(path)))
    return path


def _norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


# Run extraction + OCR with a trace, pair with the labelled legend by IoU, and draw the evidence
def check_legend(stem, iou_min=0.5):
    img_path, ann_path = page_paths(stem)
    lbox = json.loads(Path(ann_path).read_text())["bounding_boxes"]["lithology_legend"]
    lx0, ly0, lx1, ly1 = lbox
    with Image.open(img_path) as im:
        boxes = detect_swatches(im, lbox)
        trace = []
        names = read_swatch_labels(im, boxes, lbox, trace=trace)
        view = crop_rgb(im, lbox)
        crops = {t["i"]: swatch_pixels(im, t["swatch"]) for t in trace}
    has_gt = (GT_DIR / f"{stem}.json").exists()
    gt = load_legend(stem, "labels") if has_gt else None

    # Pairing: best IoU per detected swatch, each labelled box used once
    pairs, used = [], set()
    for i, b in enumerate(boxes):
        best, bi = 0.0, None
        if gt is not None:
            for r in gt.itertuples():
                v = box_iou(b, r.box)
                if v > best and r.legend_id not in used:
                    best, bi = v, r.legend_id
        if bi is not None and best >= iou_min:
            used.add(bi)
        else:
            bi = None
        gl = gt.loc[gt.legend_id == bi, "lithology"].iloc[0] if bi is not None else None
        sim = SequenceMatcher(None, _norm(names[i]), _norm(gl)).ratio() if gl else None
        pairs.append(
            {
                "det": i,
                "box": b,
                "ocr": names[i],
                "gt_id": bi,
                "gt_label": gl,
                "iou": best,
                "name_sim": sim,
            }
        )
    pairs = pd.DataFrame(pairs)
    missed = (
        [] if gt is None else [r for r in gt.itertuples() if r.legend_id not in used]
    )
    is_paired = lambda p: p.gt_id is not None and not pd.isna(p.gt_id)

    # Figure 1: boxes + OCR search/line regions over the legend crop
    fig, ax = plt.subplots(figsize=(16, 16 * view.height / view.width + 0.8))
    ax.imshow(view)
    rect = lambda b, **k: mpatches.Rectangle(
        (b[0] - lx0, b[1] - ly0), b[2] - b[0], b[3] - b[1], fill=False, **k
    )
    for r in missed:
        ax.add_patch(rect(r.box, ec="red", lw=2.5, ls="--"))
        ax.text(
            r.box[0] - lx0,
            r.box[1] - ly0 - 3,
            f"MISSED {r.legend_id}: {r.lithology}",
            color="red",
            fontsize=8,
            va="bottom",
        )
    for t in trace:
        ok = is_paired(pairs.iloc[t["i"]])
        ax.add_patch(rect(t["swatch"], ec="lime" if ok else "orange", lw=2))
        if t["search"]:
            ax.add_patch(rect(t["search"], ec="cyan", lw=0.8, ls=":"))
        if t["line"]:
            ax.add_patch(rect(t["line"], ec="magenta", lw=1.5))
        ax.text(
            t["swatch"][0] - lx0,
            t["swatch"][1] - ly0 - 3,
            f"#{t['i']}",
            color="blue",
            fontsize=9,
            weight="bold",
            va="bottom",
        )
    ax.axis("off")
    ax.set_title(
        f"{stem}\nlime = detected & paired, orange = detected but unlabelled, red dashed = labelled but missed, "
        "cyan dotted = OCR search band, magenta = chosen text line",
        fontsize=9,
    )
    _show(fig, "1_boxes.png")

    # Figure 2: one row per detected swatch: swatch | OCR input | OCR text vs label
    n = max(len(trace), 1)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12, 0.9 * n + 0.6),
        gridspec_kw={"width_ratios": [1, 4, 5]},
        squeeze=False,
    )
    for t in trace:
        i = t["i"]
        p = pairs.iloc[i]
        a0, a1, a2 = axes[i]
        a0.imshow(crops[i], interpolation="nearest")
        if t["crop"] is not None:
            a1.imshow(t["crop"], cmap="gray", vmin=0, vmax=255)
        else:
            a1.text(
                0.02, 0.5, "no text line found", transform=a1.transAxes, color="red"
            )
        if not is_paired(p):
            verdict, col = "NO LABELLED BOX", "orange"
        elif p.name_sim is not None and p.name_sim >= 0.85:
            verdict, col = "name OK", "green"
        else:
            verdict, col = "NAME MISMATCH", "red"
        a2.text(0, 0.7, f"OCR:   {t['text']!r}", fontsize=9, va="center")
        a2.text(
            0,
            0.3,
            f"label: {p.gt_label!r}  (IoU {p.iou:.2f})  -> {verdict}",
            fontsize=9,
            va="center",
            color=col,
        )
        for a in (a0, a1, a2):
            a.set_xticks([])
            a.set_yticks([])
        for a in (a0, a1):
            for s_ in a.spines.values():
                s_.set_color(col)
                s_.set_linewidth(2)
        a2.axis("off")
        a0.set_ylabel(f"#{i}", rotation=0, labelpad=14, fontsize=9)
    axes[0][0].set_title("swatch", fontsize=9)
    axes[0][1].set_title("OCR input", fontsize=9)
    axes[0][2].set_title("OCR vs label", fontsize=9)
    plt.tight_layout()
    _show(fig, "2_ocr_pairing.png")

    if gt is not None:
        paired = pairs.gt_id.notna()
        good = int((pairs.name_sim.fillna(0) >= 0.85).sum())
        print(
            f"labelled swatches: {len(gt)} | detected: {len(boxes)} | paired: {int(paired.sum())} | "
            f"missed: {len(missed)} | extra: {int((~paired).sum())} | names matching (>=0.85): {good}/{int(paired.sum())}"
        )
        for r in missed:
            print(f"  MISSED   id {r.legend_id:>2} {r.lithology!r} at {r.box}")
        for p in pairs[~paired].itertuples():
            print(f"  EXTRA    det #{p.det} {p.ocr!r} at {p.box}")
        for p in pairs[paired & (pairs.name_sim.fillna(0) < 0.85)].itertuples():
            print(f"  MISMATCH det #{p.det}: OCR {p.ocr!r} vs label {p.gt_label!r}")
    return pairs


legend_pairs = check_legend(STEM)
legend_pairs


RUN_BINS = np.array([1, 2, 3, 4, 6, 9, 13, 21, 10**6])  # run-length bin edges (px)
W_COLOR, W_INK, W_TEXTURE, TEXTURE_FULL_PX = 1 / 10, 4.0, 1.0, 24


def srgb_to_lab(rgb):
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = (
        c
        @ np.array(
            [
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505],
            ]
        ).T
    )
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


# Length-weighted histogram of runs of True along each row
def run_hist(mask):
    m = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    lengths = np.where(d == -1)[1] - np.where(d == 1)[1]
    h = np.histogram(lengths, bins=RUN_BINS, weights=lengths)[0].astype(np.float64)
    return h / h.sum() if h.sum() else h


# Colour + texture descriptor of an RGB patch (swatch interior or rows of the column)
def describe(rgb):
    rgb = np.asarray(rgb)
    dark = rgb.mean(2) < 128
    light = rgb[~dark]
    return {
        "lab": srgb_to_lab(np.median(light, 0) if len(light) else np.zeros(3)),
        "ink": float(dark.mean()),
        "h_dark": run_hist(dark),
        "v_dark": run_hist(dark.T),
        "h_gap": run_hist(~dark),
        "v_gap": run_hist(~dark.T),
        "rows": rgb.shape[0],
    }


# Distance between a segment and a swatch; vertical runs count less in thin segments (they are cut short)
def descriptor_distance(seg, sw):
    tv = lambda k: 0.5 * np.abs(seg[k] - sw[k]).sum()
    v_weight = min(1.0, seg["rows"] / TEXTURE_FULL_PX)
    d_tex = (tv("h_dark") + tv("h_gap") + v_weight * (tv("v_dark") + tv("v_gap"))) / 4
    return (
        np.linalg.norm(seg["lab"] - sw["lab"]) * W_COLOR
        + abs(seg["ink"] - sw["ink"]) * W_INK
        + W_TEXTURE * d_tex
    )


# Pseudo-swatch for empty white rows
NB = len(RUN_BINS) - 1
BLANK = {
    "lab": srgb_to_lab([255, 255, 255]),
    "ink": 0.0,
    "h_dark": np.zeros(NB),
    "v_dark": np.zeros(NB),
    "h_gap": np.eye(NB)[-1],
    "v_gap": np.eye(NB)[-1],
    "rows": 1,
}


# Candidate prototypes: blank first, then every rock swatch of the legend
def prototypes(legend, img, rocks_only=True):
    leg = (
        legend.loc[legend.lithology.map(is_rock).astype(bool)] if rocks_only else legend
    )
    return [(-1, "blank", BLANK)] + [
        (r.legend_id, r.lithology, describe(swatch_pixels(img, r.box)))
        for r in leg.itertuples()
    ]


SPLIT_WINDOW, SPLIT_MIN_RUN = (
    16,
    3,
)  # window height (px), min consecutive windows for a split-off lithology


# Split segments whose windows clearly match different swatches (boundaries Section 2 missed)
def split_mixed(
    segments, col, y_offset, protos, win=SPLIT_WINDOW, min_run=SPLIT_MIN_RUN
):
    cols = trim_borders(col)
    rows = []
    for s in segments.itertuples():
        a = s.y_top - y_offset
        n = (s.y_bottom - s.y_top) // win
        if n < 2 * min_run:
            rows.append((s.y_top, s.y_bottom))
            continue
        windows = [describe(cols[a + i * win : a + (i + 1) * win]) for i in range(n)]
        lab = [
            int(np.argmin([descriptor_distance(w, p) for _, _, p in protos]))
            for w in windows
        ]
        runs = []  # [label, first window, n windows]
        for i, l in enumerate(lab):
            if runs and runs[-1][0] == l:
                runs[-1][2] += 1
            else:
                runs.append([l, i, 1])
        kept = [r for r in runs if r[2] >= min_run] or [max(runs, key=lambda r: r[2])]
        starts = [r[1] for k, r in enumerate(kept) if k == 0 or r[0] != kept[k - 1][0]][
            1:
        ]
        edges = [s.y_top] + [s.y_top + i * win for i in starts] + [s.y_bottom]
        rows += list(zip(edges[:-1], edges[1:]))
    return pd.DataFrame(
        {
            "segment": range(len(rows)),
            "y_top": [r[0] for r in rows],
            "y_bottom": [r[1] for r in rows],
        }
    )


# Name every segment with its closest prototype; margin = gap to the runner-up (small = unsure)
def name_segments(segments, col, y_offset, protos, trim=2):
    cols = trim_borders(col)
    out = []
    for s in segments.itertuples():
        a, b = s.y_top - y_offset, s.y_bottom - y_offset
        if b - a > 4 * trim:
            a, b = a + trim, b - trim
        d = describe(cols[a:b])
        dist = np.array([descriptor_distance(d, p) for _, _, p in protos])
        order = np.argsort(dist)
        out.append(
            {
                "legend_id": protos[order[0]][0],
                "lithology": protos[order[0]][1],
                "distance": dist[order[0]],
                "margin": dist[order[1]] - dist[order[0]] if len(order) > 1 else 0.0,
            }
        )
    return pd.concat(
        [segments[["y_top", "y_bottom"]].reset_index(drop=True), pd.DataFrame(out)],
        axis=1,
    )


# Merge neighbouring segments with the same legend entry into lithology layers
def merge_named(named):
    rows = []
    for r in named.itertuples():
        if (
            rows
            and rows[-1]["legend_id"] == r.legend_id
            and rows[-1]["y_bottom"] == r.y_top
        ):
            rows[-1]["y_bottom"] = r.y_bottom
            rows[-1]["n_segments"] += 1
        else:
            rows.append(
                {
                    "y_top": r.y_top,
                    "y_bottom": r.y_bottom,
                    "legend_id": r.legend_id,
                    "lithology": r.lithology,
                    "n_segments": 1,
                }
            )
    return pd.DataFrame(rows)


# Section 2 segments of a column -> named lithology layers
def lithology_layers(segments, col, y_offset, legend, img, split=True):
    protos = prototypes(legend, img)
    if split:
        segments = split_mixed(segments, col, y_offset, protos)
    named = name_segments(segments, col, y_offset, protos)
    return named, merge_named(named)


# Name the Section 2 segments of the test page
named, layers = lithology_layers(segments, col, y0, legend, image)
print(
    len(segments),
    "segments ->",
    len(named),
    "after splitting ->",
    len(layers),
    "lithology layers",
)
print((layers.y_bottom - layers.y_top).groupby(layers.lithology).sum().rename("rows"))
layers


# Per-row legend id from a label file (-1 where there is no layer)
def gt_rows(label_path, y_offset, n):
    ids = np.full(n, -1)
    for l in json.loads(Path(label_path).read_text())["layers"]:
        main = max(l["composition"], key=lambda p: p["proportion"])["legend_id"]
        ids[max(l["top_px"] - y_offset, 0) : max(l["bottom_px"] - y_offset, 0)] = main
    return ids


def pred_rows(named, y_offset, n):
    ids = np.full(n, -1)
    for r in named.itertuples():
        ids[r.y_top - y_offset : r.y_bottom - y_offset] = r.legend_id
    return ids


def lithology_accuracy(named, label_path, y_offset, n):
    g, p = gt_rows(label_path, y_offset, n), pred_rows(named, y_offset, n)
    lab = g >= 0
    return {
        "row_accuracy": round(float((p[lab] == g[lab]).mean()), 3),
        "blank_kept_blank": round(float((p[~lab] == -1).mean()), 3)
        if (~lab).any()
        else None,
    }


# Rows of each true lithology (rows of the table) predicted as each lithology (columns)
def lithology_confusion(named, label_path, y_offset, n, legend):
    names = dict(zip(legend.legend_id, legend.lithology)) | {-1: "blank"}
    g, p = gt_rows(label_path, y_offset, n), pred_rows(named, y_offset, n)
    return pd.crosstab(
        pd.Series(g).map(names).rename("true"),
        pd.Series(p).map(names).rename("predicted"),
    )


# Row accuracy with the Section 2 segments, and with the true layer boundaries (naming only)
gt = pd.DataFrame(
    [
        (l["top_px"], l["bottom_px"])
        for l in json.loads(LABEL_PATH.read_text())["layers"]
    ],
    columns=["y_top", "y_bottom"],
)
print("Section 2 segments:", lithology_accuracy(named, LABEL_PATH, y0, len(col)))
named_gt, _ = lithology_layers(gt, col, y0, legend, image, split=False)
print("true segments     :", lithology_accuracy(named_gt, LABEL_PATH, y0, len(col)))
lithology_confusion(named, LABEL_PATH, y0, len(col), legend)


# Scan, predicted lithology and (if labelled) true lithology as bands, coloured like the legend swatches
def plot_lithology(
    col, named, y_offset, legend, img, start=0, length=None, label_path=None
):
    start = min(max(0, start), max(0, len(col) - 1))
    end = len(col) if length is None else min(start + length, len(col))
    colors = {
        r.legend_id: np.median(swatch_pixels(img, r.box).reshape(-1, 3), 0) / 255
        for r in legend.itertuples()
    }
    colors[-1] = np.ones(3)
    names = dict(zip(legend.legend_id, legend.lithology)) | {-1: "blank"}
    tracks = [("predicted", pred_rows(named, y_offset, len(col)))]
    if label_path is not None and Path(label_path).exists():
        tracks.append(("true", gt_rows(label_path, y_offset, len(col))))
    fig, axes = plt.subplots(
        1 + len(tracks),
        1,
        figsize=(18, 1.4 * (1 + len(tracks))),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    axes[0].imshow(
        col[start:end].transpose(1, 0, 2), aspect="auto", extent=(start, end, 0, 1)
    )
    axes[0].set_yticks([])
    axes[0].set_title(f"Lithology, rows {start}-{end}")
    used = set()
    for ax, (title, ids) in zip(axes[1:], tracks):
        ids = ids[start:end]
        ax.imshow(
            np.array([colors[i] for i in ids])[None],
            aspect="auto",
            extent=(start, end, 0, 1),
            interpolation="nearest",
        )
        ax.set_yticks([0.5], [title])
        used |= set(np.unique(ids))
    axes[-1].set_xlabel("row in column (top of log → bottom)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[i], ec="k", lw=0.5) for i in sorted(used)
    ]
    fig.legend(
        handles,
        [names[i] for i in sorted(used)],
        loc="lower center",
        ncol=min(len(used), 8),
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
    )
    from IPython.display import display

    display(fig)  # always show it; plt.close alone leaves the cell empty
    plt.close(fig)


plot_lithology(col, named, y0, legend, image, label_path=LABEL_PATH)
plot_lithology(
    col, named, y0, legend, image, start=30000, length=3000, label_path=LABEL_PATH
)


# 'detect' = read the legend from the page (swatches + OCR), as on an unlabelled page; 'labels' = copy it from
# test/, which makes the legend evaluation trivially perfect
LEGEND_SOURCE = "detect"


# Full pipeline for one page: segments (Section 2, SAVE_METHOD) -> named lithology layers (Section 4)
def predict_lithology(stem, legend, split=True):
    page = load_page(stem)
    segments = predict_segments(stem, SAVE_METHOD)
    _, layers = lithology_layers(
        segments,
        page["col"],
        page["y_offset"],
        legend,
        Image.open(page_paths(stem)[0]),
        split,
    )
    return layers


completed_stems = []
for stem in TEST_STEMS:
    try:
        legend_used = load_legend(stem, LEGEND_SOURCE)
        lith = predict_lithology(stem, legend_used)
        rec = load_record(stem)
        rec["legend"] = legend_from_table(legend_used)
        rec["layers"] = layers_from_table(lith[lith.legend_id >= 0])
        save_record(
            rec, "legend_naming", legend_source=LEGEND_SOURCE, segments=SAVE_METHOD
        )
        print(
            f"{len(rec['layers']):5d} lithology layers, {len(rec['legend']):3d} legend entries  {stem}"
        )
    except FileNotFoundError as exc:
        print(f"Skipping {stem}: {exc}")
        continue
    completed_stems.append(stem)
check_run(completed_stems)


# ===================== Layer segmentation evaluation (single Colab cell) =====================
# Set the config below and run. GT_PATH / PRED_PATH are either two .json files (single pair)
# or two directories (batch; files are paired by identical filename).
# Prediction format: {"layers": [...]} or a bare list; each item needs top_px / bottom_px.
#
# Metrics per pair:
#   1. Pixel-wise accuracy (segments are half-open [top_px, bottom_px); accuracy = agreement of the
#      covered/uncovered pixel mask between GT and prediction)
#   2. Segment-count difference (pred - gt), absolute and relative
#   3. Segment height (bottom_px - top_px): min / max / mean for GT and prediction
# Segments with bottom_px < top_px are invalid: a warning is printed and they are excluded.
# Batch adds an aggregate (micro/macro pixel accuracy, pooled height stats).
# After running, results are available in `RESULT` (single) or `RESULTS` / `AGGREGATE` (batch).

# ----------------------------------- CONFIG -----------------------------------
GT_PATH = str(GT_DIR)  # file or directory
PRED_PATH = str(RUN_DIR)  # file or directory (same kind as GT_PATH)
EVAL_RANGE = "union"  # "union" = GT+pred extent (default), "gt" = GT extent only
SAVE_JSON_TO = str(
    OUTPUT_DIR / "segmentation_eval.json"
)  # e.g. "/content/eval_results.json", or None to skip
# -------------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import numpy as np


def load_layers(path):
    """Return (valid segments sorted by top, invalid segments as (index, top, bottom))."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data["layers"] if isinstance(data, dict) else data
    valid, invalid = [], []
    for i, l in enumerate(layers):
        top, bot = l["top_px"], l["bottom_px"]
        if bot < top:
            invalid.append((i, top, bot))
        else:
            valid.append((int(round(top)), int(round(bot))))
    return sorted(valid), invalid


def count_overlaps(layers):
    return sum(b[0] < a[1] for a, b in zip(layers, layers[1:]))


def rasterise(layers, lo, hi):
    """Boolean mask over pixel rows [lo, hi): True where covered by a segment."""
    covered = np.zeros(hi - lo, dtype=bool)
    for top, bot in layers:
        a, b = max(top, lo) - lo, min(bot, hi) - lo
        if b > a:
            covered[a:b] = True
    return covered


def height_stats(layers):
    h = np.array([b - t for t, b in layers], dtype=float)
    if h.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": int(h.min()), "max": int(h.max()), "mean": float(h.mean())}


def evaluate(gt, pred, eval_range="union", n_invalid_pred=0):
    both = gt + pred
    if eval_range == "gt":
        lo, hi = min(t for t, _ in gt), max(b for _, b in gt)
    else:
        lo, hi = min(t for t, _ in both), max(b for _, b in both)

    g, p = rasterise(gt, lo, hi), rasterise(pred, lo, hi)
    n_gt, n_pr = len(gt), len(pred)
    return {
        "eval_range_px": [lo, hi],
        "n_pixels": hi - lo,
        "pixel_accuracy": float((g == p).mean()),
        "pixel_confusion": {
            "tp": int((g & p).sum()),
            "tn": int((~g & ~p).sum()),
            "fp": int((~g & p).sum()),
            "fn": int((g & ~p).sum()),
        },
        "n_gt_segments": n_gt,
        "n_pred_segments": n_pr,
        "n_invalid_pred_segments_excluded": n_invalid_pred,
        "segment_diff": n_pr - n_gt,
        "segment_abs_diff": abs(n_pr - n_gt),
        "segment_rel_diff": (n_pr - n_gt) / n_gt if n_gt else float("nan"),
        "segment_height_px": {"gt": height_stats(gt), "pred": height_stats(pred)},
    }


def format_report(r):
    c, h = r["pixel_confusion"], r["segment_height_px"]
    lo, hi = r["eval_range_px"]
    line = "=" * 52

    def hrow(name, s):
        if s["min"] is None:
            return f"  {name:<6}{'n/a':>10}{'n/a':>10}{'n/a':>12}"
        return f"  {name:<6}{s['min']:>10,}{s['max']:>10,}{s['mean']:>12,.1f}"

    return "\n".join(
        [
            line,
            " LAYER SEGMENTATION EVALUATION",
            line,
            f" Evaluated range : {lo:,} - {hi:,} px ({r['n_pixels']:,} px)",
            "",
            " 1. Pixel-wise accuracy",
            f"      Accuracy        : {r['pixel_accuracy']:.2%}",
            f"      Correct pixels  : {c['tp'] + c['tn']:,} / {r['n_pixels']:,}",
            f"      Segmented in both (TP)       : {c['tp']:,}",
            f"      Gap in both (TN)             : {c['tn']:,}",
            f"      Pred-only segmented (FP)     : {c['fp']:,}",
            f"      GT-only segmented (FN)       : {c['fn']:,}",
            "",
            " 2. Segment count",
            f"      GT              : {r['n_gt_segments']:,}",
            f"      Prediction      : {r['n_pred_segments']:,} (valid only)",
            f"      Invalid in pred : {r['n_invalid_pred_segments_excluded']:,} (bottom_px < top_px, excluded)",
            f"      Difference      : {r['segment_diff']:+,} (pred - gt)",
            f"      Abs. difference : {r['segment_abs_diff']:,}",
            f"      Relative diff.  : {r['segment_rel_diff']:+.2%}",
            "",
            " 3. Segment height (px)",
            f"  {'':<6}{'min':>10}{'max':>10}{'mean':>12}",
            hrow("GT", h["gt"]),
            hrow("Pred", h["pred"]),
            line,
        ]
    )


def warn_file(label, name, bad, layers):
    prefix = f"WARNING [{label}]" if label else "WARNING"
    if bad:
        shown = ", ".join(f"#{i} (top={t}, bottom={b})" for i, t, b in bad[:10])
        more = f" ... and {len(bad) - 10} more" if len(bad) > 10 else ""
        print(
            f"{prefix}: {name} has {len(bad)} invalid segment(s) with bottom_px < top_px; "
            f"excluded from evaluation: {shown}{more}",
            file=sys.stderr,
        )
    n = count_overlaps(layers)
    if n:
        print(
            f"{prefix}: {name} has {n} overlapping adjacent segments (overlaps merge in the mask).",
            file=sys.stderr,
        )


def eval_pair(gt_path, pred_path, eval_range, label=""):
    gt, gt_bad = load_layers(gt_path)
    pred, pred_bad = load_layers(pred_path)
    warn_file(label, "GT", gt_bad, gt)
    warn_file(label, "prediction", pred_bad, pred)
    if not gt:
        raise ValueError("GT has no valid segments")
    if not pred:
        raise ValueError("prediction has no valid segments")
    return evaluate(gt, pred, eval_range, len(pred_bad)), gt, pred


def aggregate(results, all_gt, all_pred):
    pix = sum(r["n_pixels"] for r in results)
    correct = sum(
        r["pixel_confusion"]["tp"] + r["pixel_confusion"]["tn"] for r in results
    )
    n_gt = sum(r["n_gt_segments"] for r in results)
    n_pr = sum(r["n_pred_segments"] for r in results)
    diffs = np.array([r["segment_diff"] for r in results], dtype=float)
    return {
        "n_files": len(results),
        "pixel_accuracy_micro": correct / pix,
        "pixel_accuracy_macro": float(np.mean([r["pixel_accuracy"] for r in results])),
        "n_gt_segments": n_gt,
        "n_pred_segments": n_pr,
        "n_invalid_pred_segments_excluded": sum(
            r["n_invalid_pred_segments_excluded"] for r in results
        ),
        "segment_diff_total": n_pr - n_gt,
        "segment_diff_mean_per_file": float(diffs.mean()),
        "segment_abs_diff_mean_per_file": float(np.abs(diffs).mean()),
        "segment_height_px": {
            "gt": height_stats(all_gt),
            "pred": height_stats(all_pred),
        },
    }


def format_batch(per_file, agg, unmatched_gt, unmatched_pred, failed):
    line = "=" * 92
    out = [
        line,
        " BATCH LAYER SEGMENTATION EVALUATION",
        line,
        f" {'File':<44}{'PixAcc':>9}{'GT n':>7}{'Pred n':>8}{'Diff':>7}{'Invalid':>9}",
        "-" * 92,
    ]
    for name, r in per_file.items():
        short = name if len(name) <= 43 else name[:40] + "..."
        out.append(
            f" {short:<44}{r['pixel_accuracy']:>9.2%}{r['n_gt_segments']:>7,}{r['n_pred_segments']:>8,}"
            f"{r['segment_diff']:>+7,}{r['n_invalid_pred_segments_excluded']:>9,}"
        )
    if agg:
        h = agg["segment_height_px"]

        def hrow(n, s):
            return f"  {n:<6}{s['min']:>10,}{s['max']:>10,}{s['mean']:>12,.1f}"

        out += [
            "-" * 92,
            f" AGGREGATE over {agg['n_files']} file(s)",
            "",
            f"   Pixel accuracy, micro (pooled pixels) : {agg['pixel_accuracy_micro']:.2%}",
            f"   Pixel accuracy, macro (mean of files) : {agg['pixel_accuracy_macro']:.2%}",
            "",
            f"   Segments  GT: {agg['n_gt_segments']:,}   Pred: {agg['n_pred_segments']:,} (valid only)   "
            f"Invalid in pred: {agg['n_invalid_pred_segments_excluded']:,}",
            f"   Count difference (pred - gt)  total: {agg['segment_diff_total']:+,}   "
            f"mean/file: {agg['segment_diff_mean_per_file']:+.1f}   "
            f"mean abs/file: {agg['segment_abs_diff_mean_per_file']:.1f}",
            "",
            "   Segment height (px), pooled over all segments",
            f"  {'':<6}{'min':>10}{'max':>10}{'mean':>12}",
            hrow("GT", h["gt"]),
            hrow("Pred", h["pred"]),
        ]
    if unmatched_gt:
        out += [
            "",
            f" GT files without a prediction ({len(unmatched_gt)}): "
            + ", ".join(unmatched_gt),
        ]
    if unmatched_pred:
        out += [
            "",
            f" Predictions without a GT file ({len(unmatched_pred)}): "
            + ", ".join(unmatched_pred),
        ]
    if failed:
        out += ["", f" Failed ({len(failed)}):"] + [
            f"   {n}: {e}" for n, e in failed.items()
        ]
    out.append(line)
    return "\n".join(out)


# Section 5 uses the same JSON-pair evaluator for both challenge and external cases.
# Re-discover files every time this evaluation cell runs; no hard-coded test stems.
import csv
from contextlib import contextmanager

_EVAL_ROOT = RUN_DIR / "evaluation_pairs"
_EVAL_GT = _EVAL_ROOT / "ground_truth"
_EVAL_PRED = _EVAL_ROOT / "predictions"
_EVAL_GT.mkdir(parents=True, exist_ok=True)
_EVAL_PRED.mkdir(parents=True, exist_ok=True)

# Clean stale pairs so deleted/renamed tests cannot contaminate subsequent aggregate scores.
for _folder in (_EVAL_GT, _EVAL_PRED):
    for _old in _folder.glob("*.json"):
        _old.unlink()


def _test_cases():
    cases = []
    seen = set()
    for folder in dict.fromkeys((GT_DIR, ROOT / "tests")):
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.json")):
            data = json.loads(file.read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("layers"), list)
                or not data["layers"]
            ):
                continue
            if file.stem in seen:
                raise ValueError(f"Duplicate JSON test filename: {file.stem}")
            seen.add(file.stem)
            cases.append((file.stem, file, None, None))
    external = ROOT / "test" / "external_shared"
    if not external.is_dir():
        external = ROOT / "external_shared"
    if external.is_dir():
        for window in json.loads(
            (external / "eval_set.json").read_text(encoding="utf-8")
        ):
            stem, a, b = window["log"], int(window["y0"]), int(window["y1"])
            if a >= b:
                raise ValueError(f"Invalid external window {window}")
            truth = external / "truth" / f"{stem}.csv"
            if not truth.is_file():
                raise FileNotFoundError(truth)
            key = f"{stem}__y{a}_{b}"
            if key in seen:
                raise ValueError(f"Duplicate test: {key}")
            seen.add(key)
            cases.append((key, truth, (a, b), external))
    return cases


@contextmanager
def _use_external_page(stem, external):
    # Redirect image/annotation lookup only. Keep ROOT, model weights and user config unchanged.
    global page_paths, PAGES
    old_paths, old_cache = page_paths, PAGES

    def _paths(s):
        if s != stem:
            return old_paths(s)
        ann = external / "bbox_annotation" / f"{s}.json"
        for ext in (".tif", ".tiff", ".png"):
            image = external / "images" / f"{s}{ext}"
            if image.is_file() and ann.is_file():
                return image, ann
        raise FileNotFoundError(f"Missing image or bbox for {s} in {external}")

    page_paths, PAGES = _paths, {}
    try:
        yield
    finally:
        page_paths, PAGES = old_paths, old_cache


def _clip(layers, window):
    if window is None:
        return layers
    a, b = window
    output = []
    for layer in layers:
        t = max(a, int(layer["top_px"]))
        z = min(b, int(layer["bottom_px"]))
        if t < z:
            output.append({**layer, "top_px": t, "bottom_px": z})
    return output


def _external_truth(file, window):
    layers = []
    with file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t, b = int(float(row["top_y"])), int(float(row["bottom_y"]))
            if t >= b:
                continue
            layers.append(
                {
                    "top_px": t,
                    "bottom_px": b,
                    "composition": [
                        {
                            "legend_id": int(float(row["cluster_id"])),
                            "type": row["lithology"],
                            "proportion": 1.0,
                        }
                    ],
                }
            )
    return {"layers": _clip(layers, window), "legend": []}


def _predict_case(stem, external):
    if external is None:
        legend = load_legend(stem, "detect")  # never expose test labels to prediction
        return legend, predict_lithology(stem, legend)
    with _use_external_page(stem, external):
        legend = load_legend(stem, "detect")
        return legend, predict_lithology(stem, legend)


# _EVAL_CASES = _test_cases()
_EVAL_CASES = _test_cases()
# _EVAL_CASES = _EVAL_CASES[0]
if not _EVAL_CASES:
    raise FileNotFoundError(
        "No JSON tests in test/ or tests/, and no external_shared windows"
    )
_EVAL_PRED_CACHE = {}
_EVAL_FAILURES = {}
for _key, _truth_file, _window, _external in _EVAL_CASES:
    _stem = _truth_file.stem if _external is None else _key.split("__y")[0]
    if _external is None:
        _truth_data = json.loads(_truth_file.read_text(encoding="utf-8"))
    else:
        _truth_data = _external_truth(_truth_file, _window)
    if not _truth_data["layers"]:
        _EVAL_FAILURES[_key] = "No labelled layers in evaluation window"
        continue
    (_EVAL_GT / f"{_key}.json").write_text(json.dumps(_truth_data), encoding="utf-8")
    _cache_key = (_stem, str(_external) if _external is not None else "")
    if _cache_key not in _EVAL_PRED_CACHE:
        try:
            _legend, _named = _predict_case(_stem, _external)
            _EVAL_PRED_CACHE[_cache_key] = {
                "legend": legend_from_table(_legend),
                "layers": layers_from_table(_named[_named.legend_id >= 0]),
            }
        except Exception as exc:
            _EVAL_PRED_CACHE[_cache_key] = None
            _EVAL_FAILURES[_key] = f"{type(exc).__name__}: {exc}"
    _record = _EVAL_PRED_CACHE[_cache_key]
    if _record is not None:
        _pred_data = {**_record, "layers": _clip(_record["layers"], _window)}
        (_EVAL_PRED / f"{_key}.json").write_text(
            json.dumps(_pred_data), encoding="utf-8"
        )
    else:
        _EVAL_FAILURES[_key] = _EVAL_FAILURES.get(
            _key, "Prediction failed for this page"
        )

# These original evaluators still receive exactly the same JSON schema.
GT_PATH = str(_EVAL_GT)
PRED_PATH = str(_EVAL_PRED)
print(
    f"Section 5: {len(_EVAL_CASES)} test cases ({sum(c[3] is not None for c in _EVAL_CASES)} external); "
    f"{len(_EVAL_FAILURES)} prediction/label failures"
)
for _name, _err in sorted(_EVAL_FAILURES.items()):
    print(f"WARNING [{_name}]: {_err}", file=sys.stderr)


# ----------------------------------- RUN -----------------------------------
assert EVAL_RANGE in ("union", "gt"), 'EVAL_RANGE must be "union" or "gt"'
_gt_p, _pred_p = Path(GT_PATH), Path(PRED_PATH)
if not _gt_p.exists():
    raise FileNotFoundError(f"GT_PATH not found: {GT_PATH}")
if not _pred_p.exists():
    raise FileNotFoundError(f"PRED_PATH not found: {PRED_PATH}")
if _gt_p.is_dir() != _pred_p.is_dir():
    raise ValueError("GT_PATH and PRED_PATH must both be files or both be directories")

RESULT = RESULTS = AGGREGATE = None

if not _gt_p.is_dir():  # single pair
    RESULT, _, _ = eval_pair(_gt_p, _pred_p, EVAL_RANGE)
    print(format_report(RESULT))
    _to_save = RESULT
else:  # batch
    _gt_files = {f.name: f for f in sorted(_gt_p.glob("*.json"))}
    _pred_files = {f.name: f for f in sorted(_pred_p.glob("*.json"))}
    if not _gt_files:
        raise FileNotFoundError(f"no .json files in {GT_PATH}")
    _unmatched_gt = [n for n in _gt_files if n not in _pred_files]
    _unmatched_pred = [n for n in _pred_files if n not in _gt_files]
    for n in _unmatched_gt:
        print(f"WARNING: no prediction for GT file {n}", file=sys.stderr)
    for n in _unmatched_pred:
        print(f"WARNING: no GT file for prediction {n}", file=sys.stderr)

    RESULTS, _failed, _all_gt, _all_pred = {}, {}, [], []
    for name in _gt_files:
        if name not in _pred_files:
            continue
        try:
            r, g, p = eval_pair(_gt_files[name], _pred_files[name], EVAL_RANGE, name)
        except Exception as e:  # keep going; reported at the end
            _failed[name] = f"{type(e).__name__}: {e}"
            print(f"ERROR [{name}]: {_failed[name]}", file=sys.stderr)
            continue
        RESULTS[name] = r
        _all_gt += g
        _all_pred += p

    AGGREGATE = (
        aggregate(list(RESULTS.values()), _all_gt, _all_pred) if RESULTS else None
    )
    print(format_batch(RESULTS, AGGREGATE, _unmatched_gt, _unmatched_pred, _failed))
    _to_save = {
        "per_file": RESULTS,
        "aggregate": AGGREGATE,
        "gt_without_prediction": _unmatched_gt,
        "prediction_without_gt": _unmatched_pred,
        "failed": _failed,
    }

if SAVE_JSON_TO:
    Path(SAVE_JSON_TO).write_text(json.dumps(_to_save, indent=2), encoding="utf-8")
    print(f"Saved metrics to {SAVE_JSON_TO}")


# Visual check of the segmentation: original column next to the predicted and true layers, in windows spread over the
# labelled rows. Layers alternate light/dark so every boundary is visible; red lines mark predicted boundaries and
# green lines true ones. One figure per evaluation case (the pairs scored above, external windows included)
def segmentation_panels(key, stem, window=None, n=4, length=1500):
    page = load_page(stem)
    col, y_offset = page["col"], page["y_offset"]
    pred = json.loads((_EVAL_PRED / f"{key}.json").read_text(encoding="utf-8"))[
        "layers"
    ]
    label = (
        json.loads((_EVAL_GT / f"{key}.json").read_text(encoding="utf-8"))["layers"]
        or None
    )

    # Rows of the column covered by layers, as spans in full-image pixels
    def spans(layers):
        return sorted((l["top_px"], l["bottom_px"]) for l in layers or [])

    pred_spans, true_spans = spans(pred), spans(label)
    ref = true_spans or pred_spans
    lo, hi = ref[0][0] - y_offset, ref[-1][1] - y_offset
    if window is not None:
        lo, hi = max(lo, window[0] - y_offset), min(hi, window[1] - y_offset)
    lo, hi = max(lo, 0), min(hi, len(col))
    length = min(length, hi - lo)
    starts = [lo] if hi - lo <= length else np.linspace(lo, hi - length, n).astype(int)

    tracks = [("original", None), ("predicted", pred_spans)] + (
        [("true", true_spans)] if label else []
    )
    fig, axes = plt.subplots(
        1,
        len(starts) * len(tracks),
        figsize=(1.5 * len(starts) * len(tracks), 10),
        gridspec_kw={"wspace": 0.25},
        squeeze=False,
    )
    for j, start in enumerate(starts):
        end = min(start + length, len(col))
        y0_, y1_ = start + y_offset, end + y_offset
        for k, (title, sp) in enumerate(tracks):
            ax = axes[0, j * len(tracks) + k]
            ax.imshow(
                col[start:end],
                aspect="auto",
                interpolation="nearest",
                extent=(0, col.shape[1], y1_, y0_),
            )
            if sp is not None:
                for i, (top, bot) in enumerate(sp):
                    if bot > y0_ and top < y1_:
                        ax.axhspan(
                            max(top, y0_),
                            min(bot, y1_),
                            color="white" if i % 2 else "black",
                            alpha=0.85,
                            lw=0,
                        )
                for top, bot in sp:
                    for y in (top, bot):
                        if y0_ <= y <= y1_:
                            ax.axhline(
                                y,
                                color="red" if title == "predicted" else "green",
                                lw=0.8,
                            )
            ax.set_xlim(0, col.shape[1])
            ax.set_ylim(y1_, y0_)
            ax.set_xticks([])
            ax.set_title(title, fontsize=9)
            if k:
                ax.set_yticks([])
    fig.suptitle(
        f"{key}: {len(pred_spans)} predicted / {len(true_spans)} true layers",
        fontsize=11,
    )
    from IPython.display import display

    display(fig)
    plt.close(fig)


for key, _, window, external in _EVAL_CASES:
    stem = key if external is None else key.split("__y")[0]
    if (
        not (_EVAL_GT / f"{key}.json").exists()
        or not (_EVAL_PRED / f"{key}.json").exists()
    ):
        print(f"Skipping {key}: {_EVAL_FAILURES.get(key, 'no prediction')}")
        continue
    try:
        if external is None:
            segmentation_panels(key, stem)
        else:
            with _use_external_page(stem, external):
                segmentation_panels(key, stem, window)
    except FileNotFoundError as exc:
        print(f"Skipping segmentation visual check for {key}: {exc}")


# ================= Segment classification evaluation (single Colab cell) =================
# Set the config below and run. GT_PATH / PRED_PATH are either two .json files (single pair)
# or two directories (batch; files are paired by identical filename).
# GT: layers[].composition[] -> class = entry with the largest proportion ("type").
# Prediction: {"layers": [...]} or a bare list; each item needs top_px / bottom_px and a class under
# "type" / "class" / "label" / "rock_name" / "lithology" (or a GT-style "composition").
# Class names are compared ignoring case/whitespace.
#
# Segments are half-open [top_px, bottom_px), rasterised to one class per pixel row; uncovered rows
# are "background". Overlaps: the later segment (by top) wins. Invalid segments (bottom_px < top_px
# or no class) are excluded with a warning.
#
# Metrics (multi-class, pixel-wise): pixel accuracy, mean class accuracy, mIoU, frequency-weighted
# IoU, macro precision/recall/F1, per-class support/precision/recall/F1/IoU, confusion matrix, and
# segment-level accuracy (dominant predicted class over each GT segment).
# Background is in the confusion matrix and pixel accuracy but not the macro means unless
# INCLUDE_BACKGROUND = True. Batch adds pooled (micro) and per-file-mean (macro) aggregates.
# After running, results are available in `RESULT` (single) or `RESULTS` / `AGGREGATE` (batch).

# ----------------------------------- CONFIG -----------------------------------
GT_PATH = str(_EVAL_GT)  # file or directory
PRED_PATH = str(_EVAL_PRED)  # file or directory (same kind as GT_PATH)
EVAL_RANGE = "union"  # "union" = GT+pred extent (default), "gt" = GT extent only
INCLUDE_BACKGROUND = False  # True = count background as a class in the macro means
SAVE_JSON_TO = None  # e.g. "/content/eval_results.json", or None to skip
# -------------------------------------------------------------------------------
if "_EVAL_GT" in globals() and "_EVAL_PRED" in globals():
    GT_PATH, PRED_PATH = str(_EVAL_GT), str(_EVAL_PRED)


import json
import re
import sys
from pathlib import Path

import numpy as np

BG = "background"
CLASS_KEYS = ("type", "class", "label", "rock_name", "lithology")


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def seg_class(item):
    """Class label of a layer item, or None."""
    for k in CLASS_KEYS:
        if item.get(k) not in (None, ""):
            return norm(item[k])
    comp = item.get("composition")
    if comp:
        best = max(comp, key=lambda c: c.get("proportion", 0) or 0)
        t = best.get("type", best.get("label"))
        if t not in (None, ""):
            return norm(t)
    return None


def load_segments(path):
    """Return (valid [(top, bottom, class)] sorted by top, invalid [(index, reason)])."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data["layers"] if isinstance(data, dict) else data
    valid, invalid = [], []
    for i, l in enumerate(layers):
        top, bot, cls = l["top_px"], l["bottom_px"], seg_class(l)
        if bot < top:
            invalid.append((i, f"bottom_px {bot} < top_px {top}"))
        elif cls is None:
            invalid.append((i, "no class"))
        else:
            valid.append((int(round(top)), int(round(bot)), cls))
    return sorted(valid), invalid


def rasterise(segs, lo, hi):
    """Object array of class names over pixel rows [lo, hi); background where uncovered."""
    out = np.full(hi - lo, BG, dtype=object)
    for top, bot, cls in segs:
        a, b = max(top, lo) - lo, min(bot, hi) - lo
        if b > a:
            out[a:b] = cls
    return out


def confusion(g, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    gi = np.fromiter((idx[c] for c in g), dtype=np.int64, count=len(g))
    pi = np.fromiter((idx[c] for c in p), dtype=np.int64, count=len(p))
    n = len(classes)
    return np.bincount(gi * n + pi, minlength=n * n).reshape(n, n)


def div(a, b):
    return float(a) / float(b) if b else float("nan")


def class_metrics(cm, classes, include_bg=False):
    """Per-class + overall metrics from a confusion matrix (rows GT, cols pred)."""
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    gt_n, pr_n = cm.sum(1), cm.sum(0)
    per = {}
    for i, c in enumerate(classes):
        if gt_n[i] == 0 and pr_n[i] == 0:
            continue  # class not present anywhere
        union = gt_n[i] + pr_n[i] - tp[i]
        prec, rec = div(tp[i], pr_n[i]), div(tp[i], gt_n[i])
        per[c] = {
            "support_px": int(gt_n[i]),
            "pred_px": int(pr_n[i]),
            "tp_px": int(tp[i]),
            "precision": prec,
            "recall": rec,
            "f1": div(2 * tp[i], gt_n[i] + pr_n[i]),
            "iou": div(tp[i], union),
        }
    scored = [c for c in per if include_bg or c != BG]

    def macro(key):
        v = [per[c][key] for c in scored if per[c][key] == per[c][key]]
        return float(np.mean(v)) if v else float("nan")

    fw_iou = float("nan")
    if total:
        w = gt_n / total
        ious = np.array([per.get(c, {}).get("iou", 0.0) for c in classes])
        ious = np.nan_to_num(ious)
        fw_iou = float((w * ious).sum())
    return {
        "pixel_accuracy": div(tp.sum(), total),
        "mean_class_accuracy": macro("recall"),
        "miou": macro("iou"),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "freq_weighted_iou": fw_iou,
        "per_class": per,
        "n_classes_scored": len(scored),
    }


def segment_level(gt, pred_rows, lo):
    """GT segments whose dominant predicted class (over their rows, background excluded) == GT class."""
    ok = n = 0
    for top, bot, cls in gt:
        a, b = top - lo, bot - lo
        if b <= a:
            continue
        n += 1
        vals, cnt = np.unique(pred_rows[a:b], return_counts=True)
        if cnt.size:
            ok += vals[cnt.argmax()] == cls
    return {"segments": n, "segments_correct": int(ok), "segment_accuracy": div(ok, n)}


def evaluate(gt, pred, classes, eval_range="union", include_bg=False):
    both = gt + pred
    if eval_range == "gt":
        lo, hi = min(t for t, _, _ in gt), max(b for _, b, _ in gt)
    else:
        lo, hi = min(t for t, _, _ in both), max(b for _, b, _ in both)
    g, p = rasterise(gt, lo, hi), rasterise(pred, lo, hi)
    cm = confusion(g, p, classes)
    r = class_metrics(cm, classes, include_bg)
    r.update(
        {
            "eval_range_px": [lo, hi],
            "n_pixels": hi - lo,
            "n_gt_segments": len(gt),
            "n_pred_segments": len(pred),
            "segment_level": segment_level(gt, p, lo),
            "confusion": cm,  # ndarray, converted for JSON by the caller
        }
    )
    return r


def warn(label, name, bad):
    if bad:
        prefix = f"WARNING [{label}]" if label else "WARNING"
        shown = ", ".join(f"#{i} ({why})" for i, why in bad[:10])
        more = f" ... and {len(bad) - 10} more" if len(bad) > 10 else ""
        print(
            f"{prefix}: {name} has {len(bad)} invalid segment(s), excluded: {shown}{more}",
            file=sys.stderr,
        )


def load_pair(gt_path, pred_path, label=""):
    gt, gt_bad = load_segments(gt_path)
    pred, pred_bad = load_segments(pred_path)
    warn(label, "GT", gt_bad)
    warn(label, "prediction", pred_bad)
    if not gt:
        raise ValueError("GT has no valid classified segments")
    if not pred:
        raise ValueError("prediction has no valid classified segments")
    return gt, pred


def class_list(*segment_sets):
    names = {c for segs in segment_sets for _, _, c in segs}
    return [BG] + sorted(names - {BG})


def pct(x):
    return "n/a" if x != x else f"{x:.2%}"


def num(x):
    return "n/a" if x != x else f"{x:.3f}"


def per_class_table(per, include_bg):
    rows = [
        f"  {'Class':<22}{'GT px':>11}{'Pred px':>11}{'Prec':>8}{'Recall':>8}{'F1':>8}{'IoU':>8}"
    ]
    for c, m in sorted(per.items(), key=lambda kv: -kv[1]["support_px"]):
        mark = "" if include_bg or c != BG else " *"
        name = (c if len(c) <= 20 else c[:17] + "...") + mark
        rows.append(
            f"  {name:<22}{m['support_px']:>11,}{m['pred_px']:>11,}{pct(m['precision']):>8}"
            f"{pct(m['recall']):>8}{pct(m['f1']):>8}{pct(m['iou']):>8}"
        )
    if not include_bg and BG in per:
        rows.append("  (* background: shown but excluded from macro means)")
    return rows


def confusion_table(cm, classes, max_cols=12):
    keep = [i for i, c in enumerate(classes) if cm[i].sum() or cm[:, i].sum()]
    if len(keep) > max_cols:  # keep the biggest classes only
        keep = sorted(
            sorted(keep, key=lambda i: -(cm[i].sum() + cm[:, i].sum()))[:max_cols]
        )
    short = lambda c: c if len(c) <= 9 else c[:8] + "."
    rows = [
        "  " + f"{'GT \\ Pred':<14}" + "".join(f"{short(classes[j]):>10}" for j in keep)
    ]
    for i in keep:
        rows.append(
            "  "
            + f"{short(classes[i]):<14}"
            + "".join(f"{cm[i, j]:>10,}" for j in keep)
        )
    if len(keep) < sum(1 for i in range(len(classes)) if cm[i].sum() or cm[:, i].sum()):
        rows.append(
            f"  (showing the {len(keep)} largest classes; full matrix in --json)"
        )
    return rows


def format_report(r, classes, include_bg):
    line = "=" * 78
    lo, hi = r["eval_range_px"]
    s = r["segment_level"]
    return "\n".join(
        [
            line,
            " SEGMENT CLASSIFICATION EVALUATION",
            line,
            f" Evaluated range : {lo:,} - {hi:,} px ({r['n_pixels']:,} px)",
            f" Segments        : GT {r['n_gt_segments']:,}   Pred {r['n_pred_segments']:,}   "
            f"Classes scored: {r['n_classes_scored']}",
            "",
            " 1. Overall (pixel-wise)",
            f"      Pixel accuracy        : {pct(r['pixel_accuracy'])}",
            f"      Mean class accuracy   : {pct(r['mean_class_accuracy'])}",
            f"      mIoU (macro)          : {pct(r['miou'])}",
            f"      Freq.-weighted IoU    : {pct(r['freq_weighted_iou'])}",
            f"      Macro P / R / F1      : {pct(r['macro_precision'])} / {pct(r['macro_recall'])} / {pct(r['macro_f1'])}",
            "",
            " 2. Per class",
            *per_class_table(r["per_class"], include_bg),
            "",
            " 3. Confusion matrix (pixels; rows = GT)",
            *confusion_table(r["confusion"], classes),
            "",
            " 4. Segment level (dominant predicted class over each GT segment)",
            f"      Correct         : {pct(s['segment_accuracy'])} ({s['segments_correct']:,} / {s['segments']:,})",
            line,
        ]
    )


def aggregate(results, cm_total, classes, include_bg):
    agg = class_metrics(cm_total, classes, include_bg)
    seg_n = sum(r["segment_level"]["segments"] for r in results)
    seg_ok = sum(r["segment_level"]["segments_correct"] for r in results)

    def macro(key):
        v = [r[key] for r in results if r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    agg.update(
        {
            "n_files": len(results),
            "n_pixels": sum(r["n_pixels"] for r in results),
            "n_gt_segments": sum(r["n_gt_segments"] for r in results),
            "n_pred_segments": sum(r["n_pred_segments"] for r in results),
            "segment_level": {
                "segments": seg_n,
                "segments_correct": seg_ok,
                "segment_accuracy": div(seg_ok, seg_n),
            },
            "per_file_mean": {
                k: macro(k)
                for k in ("pixel_accuracy", "miou", "macro_f1", "freq_weighted_iou")
            },
            "confusion": cm_total,
        }
    )
    return agg


def format_batch(
    per_file, agg, classes, include_bg, unmatched_gt, unmatched_pred, failed
):
    line = "=" * 96
    out = [
        line,
        " BATCH SEGMENT CLASSIFICATION EVALUATION",
        line,
        f" {'File':<42}{'PixAcc':>9}{'mIoU':>9}{'mF1':>9}{'fwIoU':>9}{'GT n':>7}{'Pred n':>8}{'SegAcc':>8}",
        "-" * 96,
    ]
    for name, r in per_file.items():
        short = name if len(name) <= 41 else name[:38] + "..."
        out.append(
            f" {short:<42}{pct(r['pixel_accuracy']):>9}{pct(r['miou']):>9}{pct(r['macro_f1']):>9}"
            f"{pct(r['freq_weighted_iou']):>9}{r['n_gt_segments']:>7,}{r['n_pred_segments']:>8,}"
            f"{pct(r['segment_level']['segment_accuracy']):>8}"
        )
    if agg:
        m, s = agg["per_file_mean"], agg["segment_level"]
        out += [
            "-" * 96,
            f" AGGREGATE over {agg['n_files']} file(s), {agg['n_pixels']:,} px pooled",
            "",
            "   Pooled (micro over all pixels)",
            f"     Pixel accuracy      : {pct(agg['pixel_accuracy'])}",
            f"     Mean class accuracy : {pct(agg['mean_class_accuracy'])}",
            f"     mIoU (macro/class)  : {pct(agg['miou'])}",
            f"     Freq.-weighted IoU  : {pct(agg['freq_weighted_iou'])}",
            f"     Macro P / R / F1    : {pct(agg['macro_precision'])} / {pct(agg['macro_recall'])} / {pct(agg['macro_f1'])}",
            f"     Segment accuracy    : {pct(s['segment_accuracy'])} ({s['segments_correct']:,} / {s['segments']:,})",
            "",
            "   Mean of per-file values (macro over files)",
            f"     Pixel acc {pct(m['pixel_accuracy'])}   mIoU {pct(m['miou'])}   "
            f"macro-F1 {pct(m['macro_f1'])}   fwIoU {pct(m['freq_weighted_iou'])}",
            "",
            "   Per class (pooled)",
            *per_class_table(agg["per_class"], include_bg),
            "",
            "   Confusion matrix (pooled pixels; rows = GT)",
            *confusion_table(agg["confusion"], classes),
        ]
    if unmatched_gt:
        out += [
            "",
            f" GT files without a prediction ({len(unmatched_gt)}): "
            + ", ".join(unmatched_gt),
        ]
    if unmatched_pred:
        out += [
            "",
            f" Predictions without a GT file ({len(unmatched_pred)}): "
            + ", ".join(unmatched_pred),
        ]
    if failed:
        out += ["", f" Failed ({len(failed)}):"] + [
            f"   {n}: {e}" for n, e in failed.items()
        ]
    out.append(line)
    return "\n".join(out)


def jsonable(r, classes):
    """Copy of a result with the confusion ndarray as a labelled nested dict."""
    r = dict(r)
    cm = r.pop("confusion")
    r["confusion_matrix"] = {"classes": classes, "rows_gt_cols_pred": cm.tolist()}
    return r


# ----------------------------------- RUN -----------------------------------
assert EVAL_RANGE in ("union", "gt"), 'EVAL_RANGE must be "union" or "gt"'
_gt_p, _pred_p = Path(GT_PATH), Path(PRED_PATH)
if not _gt_p.exists():
    raise FileNotFoundError(f"GT_PATH not found: {GT_PATH}")
if not _pred_p.exists():
    raise FileNotFoundError(f"PRED_PATH not found: {PRED_PATH}")
if _gt_p.is_dir() != _pred_p.is_dir():
    raise ValueError("GT_PATH and PRED_PATH must both be files or both be directories")

RESULT = RESULTS = AGGREGATE = None

if not _gt_p.is_dir():  # single pair
    _gt, _pred = load_pair(_gt_p, _pred_p)
    CLASSES = class_list(_gt, _pred)
    RESULT = evaluate(_gt, _pred, CLASSES, EVAL_RANGE, INCLUDE_BACKGROUND)
    print(format_report(RESULT, CLASSES, INCLUDE_BACKGROUND))
    _to_save = jsonable(RESULT, CLASSES)
else:  # batch
    _gt_files = {f.name: f for f in sorted(_gt_p.glob("*.json"))}
    _pred_files = {f.name: f for f in sorted(_pred_p.glob("*.json"))}
    if not _gt_files:
        raise FileNotFoundError(f"no .json files in {GT_PATH}")
    _unmatched_gt = [n for n in _gt_files if n not in _pred_files]
    _unmatched_pred = [n for n in _pred_files if n not in _gt_files]
    for n in _unmatched_gt:
        print(f"WARNING: no prediction for GT file {n}", file=sys.stderr)
    for n in _unmatched_pred:
        print(f"WARNING: no GT file for prediction {n}", file=sys.stderr)

    # Load everything first so all files share one class list (needed to pool confusion matrices).
    _pairs, _failed = {}, {}
    for name in _gt_files:
        if name not in _pred_files:
            continue
        try:
            _pairs[name] = load_pair(_gt_files[name], _pred_files[name], name)
        except Exception as e:  # keep going; reported at the end
            _failed[name] = f"{type(e).__name__}: {e}"
            print(f"ERROR [{name}]: {_failed[name]}", file=sys.stderr)
    CLASSES = class_list(*(s for g, p in _pairs.values() for s in (g, p)))

    RESULTS, _cm_total = {}, np.zeros((len(CLASSES),) * 2, dtype=np.int64)
    for name, (g, p) in _pairs.items():
        RESULTS[name] = evaluate(g, p, CLASSES, EVAL_RANGE, INCLUDE_BACKGROUND)
        _cm_total += RESULTS[name]["confusion"]

    AGGREGATE = (
        aggregate(list(RESULTS.values()), _cm_total, CLASSES, INCLUDE_BACKGROUND)
        if RESULTS
        else None
    )
    print(
        format_batch(
            RESULTS,
            AGGREGATE,
            CLASSES,
            INCLUDE_BACKGROUND,
            _unmatched_gt,
            _unmatched_pred,
            _failed,
        )
    )
    _to_save = {
        "per_file": {n: jsonable(r, CLASSES) for n, r in RESULTS.items()},
        "aggregate": jsonable(AGGREGATE, CLASSES) if AGGREGATE else None,
        "gt_without_prediction": _unmatched_gt,
        "prediction_without_gt": _unmatched_pred,
        "failed": _failed,
    }

if SAVE_JSON_TO:
    Path(SAVE_JSON_TO).write_text(json.dumps(_to_save, indent=2), encoding="utf-8")
    print(f"Saved metrics to {SAVE_JSON_TO}")


# ============================ Legend evaluation (Colab cell) ============================
# Compares predicted legend entries (rock_name + bbox) with GT legend entries (label_printed + box).
# Metrics: 1) number of patterns detected, 2) text accuracy, 3) bbox precision, 4) pairing success.
# GT / PRED may each be a single .json file or a directory of .json files (batch; paired by filename).
# Predicted boxes are relative to the legend crop; the crop origin is read from
# <ANNOTATIONS_DIR>/<GT stem>.json -> bounding_boxes.lithology_legend and added before matching.

# ------------------------------------- CONFIG -------------------------------------
GT_PATH = str(GT_DIR)  # GT file or directory
PRED_PATH = str(RUN_DIR)  # prediction file or directory
IOU_THRESHOLD = 0.5  # IoU needed for a predicted box to match a GT box
ANNOTATIONS_DIR = str(
    ROOT / "dataset" / "bbox_annotation"
)  # legend-crop origin lives here (None = <GT dir>/../bbox_annotation)
OFFSET = None  # or (x, y): crop origin given directly, overrides ANNOTATIONS_DIR
IMAGE_COORDS = True  # True if predictions are already in full-image pixels (no shift)
PRINT_JSON = False  # True to print the metrics dict as JSON instead of the report
# -----------------------------------------------------------------------------------
if "_EVAL_GT" in globals() and "_EVAL_PRED" in globals():
    GT_PATH, PRED_PATH = str(_EVAL_GT), str(_EVAL_PRED)


import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PRED_SUFFIX = "_swatches"


def norm_text(s):
    """Case-, whitespace- and edge-punctuation-insensitive form of a label."""
    return re.sub(r"\s+", " ", str(s or "")).strip().strip(".:;,").lower()


def _box(item):
    if "bbox" in item or "box" in item:
        x0, y0, x1, y1 = item.get("bbox", item.get("box"))
    else:
        x0, y0, x1, y1 = item["x0"], item["y0"], item["x1"], item["y1"]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def load_items(path, kind, offset=(0, 0)):
    """Return list of (box, label), boxes shifted by offset=(dx, dy). kind is 'gt' or 'pred'."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        key = "legend" if kind == "gt" else "swatches"
        if key in data:
            items = data[key]
        else:  # be lenient: accept whichever list key the other format uses
            items = data.get("swatches", data.get("legend"))
            if items is None:
                raise KeyError(f"no '{key}' list in {Path(path).name}")
    else:
        items = data
    out = []
    for it in items:
        label = next(
            (it[k] for k in ("label_printed", "rock_name", "label") if k in it), ""
        )
        x0, y0, x1, y1 = _box(it)
        out.append(
            (
                (x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]),
                label if label is not None else "",
            )
        )
    return out


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt, pred, thr):
    """Greedy one-to-one matching by descending IoU. Returns [(gt_idx, pred_idx, iou)]."""
    cand = sorted(
        (
            (iou(g[0], p[0]), gi, pi)
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
        ),
        reverse=True,
    )
    used_g, used_p, pairs = set(), set(), []
    for v, gi, pi in cand:
        if v < thr or v <= 0:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, v))
    return pairs


def text_scores(gt_labels, pred_labels):
    """Position-independent label agreement: (n exact multiset matches, mean best-match similarity)."""
    g = [norm_text(t) for t in gt_labels]
    p = [norm_text(t) for t in pred_labels]
    exact = sum((Counter(g) & Counter(p)).values())
    # fuzzy: greedy best-similarity assignment, one predicted label per GT label
    cand = sorted(
        (
            (difflib.SequenceMatcher(None, a, b).ratio(), i, j)
            for i, a in enumerate(g)
            for j, b in enumerate(p)
        ),
        reverse=True,
    )
    used_g, used_p, total = set(), set(), 0.0
    for s, i, j in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += s
    return exact, (total / len(g) if g else float("nan"))


def rate(num, den):
    return num / den if den else float("nan")


def evaluate(gt, pred, thr=0.5):
    n_gt, n_pr = len(gt), len(pred)
    pairs = match_boxes(gt, pred, thr)
    n_tp = len(pairs)
    correct = [
        (gi, pi)
        for gi, pi, _ in pairs
        if norm_text(gt[gi][1]) == norm_text(pred[pi][1])
    ]
    exact, fuzzy = text_scores([g[1] for g in gt], [p[1] for p in pred])
    return {
        "iou_threshold": thr,
        "n_gt_patterns": n_gt,
        "n_pred_patterns": n_pr,
        "pattern_diff": n_pr - n_gt,
        "pattern_abs_diff": abs(n_pr - n_gt),
        "pattern_rel_diff": (n_pr - n_gt) / n_gt if n_gt else float("nan"),
        "text_exact_matches": exact,
        "text_accuracy": rate(exact, n_gt),
        "text_similarity_mean": fuzzy,
        "bbox_tp": n_tp,
        "bbox_fp": n_pr - n_tp,
        "bbox_fn": n_gt - n_tp,
        "bbox_precision": rate(n_tp, n_pr),
        "bbox_recall": rate(n_tp, n_gt),
        "bbox_mean_iou": float(np.mean([v for _, _, v in pairs]))
        if pairs
        else float("nan"),
        "pairing_correct": len(correct),
        "pairing_success": rate(len(correct), n_gt),
        "pairing_success_given_bbox": rate(len(correct), n_tp),
    }


def pct(x):
    return "n/a" if x != x else f"{x:.2%}"


def num(x, f=".3f"):
    return "n/a" if x != x else format(x, f)


def format_report(r):
    line = "=" * 56
    return "\n".join(
        [
            line,
            " LEGEND EVALUATION",
            line,
            f" Pred box offset : +{r['pred_offset_px'][0]}, +{r['pred_offset_px'][1]} px (legend crop origin)",
            "",
            " 1. Patterns detected",
            f"      GT              : {r['n_gt_patterns']:,}",
            f"      Prediction      : {r['n_pred_patterns']:,}",
            f"      Difference      : {r['pattern_diff']:+,} (pred - gt)",
            f"      Relative diff.  : {r['pattern_rel_diff']:+.2%}"
            if r["n_gt_patterns"]
            else "      Relative diff.  : n/a",
            "",
            " 2. Text accuracy (position-independent)",
            f"      Exact           : {pct(r['text_accuracy'])} ({r['text_exact_matches']:,} / {r['n_gt_patterns']:,} GT labels found)",
            f"      Mean similarity : {num(r['text_similarity_mean'])} (best-match difflib ratio per GT label)",
            "",
            f" 3. Bbox precision (IoU >= {r['iou_threshold']:g})",
            f"      Precision       : {pct(r['bbox_precision'])} ({r['bbox_tp']:,} / {r['n_pred_patterns']:,} predicted boxes)",
            f"      Recall          : {pct(r['bbox_recall'])} ({r['bbox_tp']:,} / {r['n_gt_patterns']:,} GT boxes)",
            f"      Mean IoU (TP)   : {num(r['bbox_mean_iou'])}",
            f"      FP / FN         : {r['bbox_fp']:,} / {r['bbox_fn']:,}",
            "",
            " 4. Pairing success (right text on the right box)",
            f"      Of all GT       : {pct(r['pairing_success'])} ({r['pairing_correct']:,} / {r['n_gt_patterns']:,})",
            f"      Of found boxes  : {pct(r['pairing_success_given_bbox'])} ({r['pairing_correct']:,} / {r['bbox_tp']:,})",
            line,
        ]
    )


def crop_offset(gt_path, ann_dir):
    """Top-left (x, y) of the legend crop in full-image px, from bbox_annotation/<GT stem>.json."""
    f = Path(ann_dir) / (Path(gt_path).stem + ".json")
    if not f.is_file():
        raise FileNotFoundError(
            f"legend crop origin: {f} not found (set OFFSET or IMAGE_COORDS in the config)"
        )
    box = json.loads(f.read_text(encoding="utf-8"))["bounding_boxes"][
        "lithology_legend"
    ]
    return box[0], box[1]


def eval_pair(gt_path, pred_path, thr, offset=None, ann_dir=None):
    """offset: fixed (x, y) crop origin, or None to read it from ann_dir (None ann_dir -> no shift)."""
    if offset is None:
        offset = crop_offset(gt_path, ann_dir) if ann_dir is not None else (0, 0)
    gt = load_items(gt_path, "gt")
    pred = load_items(pred_path, "pred", offset)
    if not gt:
        raise ValueError("GT has no legend entries")
    r = evaluate(gt, pred, thr)
    r["pred_offset_px"] = list(offset)
    return r


SUM_KEYS = [
    "n_gt_patterns",
    "n_pred_patterns",
    "text_exact_matches",
    "bbox_tp",
    "bbox_fp",
    "bbox_fn",
    "pairing_correct",
]


def aggregate(results):
    s = {k: sum(r[k] for r in results) for k in SUM_KEYS}
    diffs = np.array([r["pattern_diff"] for r in results], dtype=float)

    def macro(key):
        v = [r[key] for r in results if r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    ious = [r["bbox_mean_iou"] * r["bbox_tp"] for r in results if r["bbox_tp"]]
    return {
        "n_files": len(results),
        **s,
        "pattern_diff_total": s["n_pred_patterns"] - s["n_gt_patterns"],
        "pattern_diff_mean_per_file": float(diffs.mean()),
        "pattern_abs_diff_mean_per_file": float(np.abs(diffs).mean()),
        "text_accuracy_micro": rate(s["text_exact_matches"], s["n_gt_patterns"]),
        "text_accuracy_macro": macro("text_accuracy"),
        "text_similarity_macro": macro("text_similarity_mean"),
        "bbox_precision_micro": rate(s["bbox_tp"], s["n_pred_patterns"]),
        "bbox_precision_macro": macro("bbox_precision"),
        "bbox_recall_micro": rate(s["bbox_tp"], s["n_gt_patterns"]),
        "bbox_recall_macro": macro("bbox_recall"),
        "bbox_mean_iou_micro": sum(ious) / s["bbox_tp"]
        if s["bbox_tp"]
        else float("nan"),
        "pairing_success_micro": rate(s["pairing_correct"], s["n_gt_patterns"]),
        "pairing_success_macro": macro("pairing_success"),
        "pairing_success_given_bbox_micro": rate(s["pairing_correct"], s["bbox_tp"]),
    }


def format_batch(per_file, agg, unmatched_gt, unmatched_pred, failed):
    line = "=" * 100
    out = [
        line,
        " BATCH LEGEND EVALUATION",
        line,
        f" {'File':<40}{'GT n':>6}{'Pred n':>8}{'Diff':>6}{'Text':>8}{'BoxP':>8}{'BoxR':>8}{'Pair':>8}{'Pair|box':>10}",
        "-" * 100,
    ]
    for name, r in per_file.items():
        short = name if len(name) <= 39 else name[:36] + "..."
        out.append(
            f" {short:<40}{r['n_gt_patterns']:>6,}{r['n_pred_patterns']:>8,}{r['pattern_diff']:>+6,}"
            f"{pct(r['text_accuracy']):>8}{pct(r['bbox_precision']):>8}{pct(r['bbox_recall']):>8}"
            f"{pct(r['pairing_success']):>8}{pct(r['pairing_success_given_bbox']):>10}"
        )
    if agg:
        out += [
            "-" * 100,
            f" AGGREGATE over {agg['n_files']} file(s)",
            "",
            f"   Patterns  GT: {agg['n_gt_patterns']:,}   Pred: {agg['n_pred_patterns']:,}   "
            f"diff total: {agg['pattern_diff_total']:+,}   mean/file: {agg['pattern_diff_mean_per_file']:+.1f}   "
            f"mean abs/file: {agg['pattern_abs_diff_mean_per_file']:.1f}",
            "",
            f"   Text accuracy      micro: {pct(agg['text_accuracy_micro'])}   macro: {pct(agg['text_accuracy_macro'])}"
            f"   (mean similarity, macro: {num(agg['text_similarity_macro'])})",
            f"   Bbox precision     micro: {pct(agg['bbox_precision_micro'])}   macro: {pct(agg['bbox_precision_macro'])}",
            f"   Bbox recall        micro: {pct(agg['bbox_recall_micro'])}   macro: {pct(agg['bbox_recall_macro'])}"
            f"   (mean IoU of TP, micro: {num(agg['bbox_mean_iou_micro'])})",
            f"   Pairing success    micro: {pct(agg['pairing_success_micro'])}   macro: {pct(agg['pairing_success_macro'])}"
            f"   (given box found, micro: {pct(agg['pairing_success_given_bbox_micro'])})",
        ]
    if unmatched_gt:
        out += [
            "",
            f" GT files without a prediction ({len(unmatched_gt)}): "
            + ", ".join(unmatched_gt),
        ]
    if unmatched_pred:
        out += [
            "",
            f" Predictions without a GT file ({len(unmatched_pred)}): "
            + ", ".join(unmatched_pred),
        ]
    if failed:
        out += ["", f" Failed ({len(failed)}):"] + [
            f"   {n}: {e}" for n, e in failed.items()
        ]
    out.append(line)
    return "\n".join(out)


def pair_key(path, strip_suffix=False):
    """Pairing key: file stem with whitespace -> '_', and (for predictions) no trailing _swatches."""
    stem = re.sub(r"\s+", "_", path.stem)
    return (
        stem[: -len(PRED_SUFFIX)]
        if strip_suffix and stem.endswith(PRED_SUFFIX)
        else stem
    )


def run_legend_evaluation(
    gt_path,
    pred_path,
    iou_threshold=0.5,
    annotations_dir=None,
    offset=None,
    image_coords=False,
    print_json=False,
):
    """Evaluate a single GT/pred pair of files, or two directories (batch). Returns the metrics dict."""
    gt_p, pred_p = Path(gt_path), Path(pred_path)
    if gt_p.is_dir() != pred_p.is_dir():
        raise ValueError(
            "GT_PATH and PRED_PATH must both be files or both be directories"
        )

    gt_dir = gt_p if gt_p.is_dir() else gt_p.parent
    ann_dir = (
        Path(annotations_dir) if annotations_dir else gt_dir.parent / "bbox_annotation"
    )
    off = (0, 0) if image_coords else (tuple(offset) if offset else None)
    if off is None and not ann_dir.is_dir():
        raise FileNotFoundError(
            f"annotation dir {ann_dir} not found (set ANNOTATIONS_DIR, OFFSET or IMAGE_COORDS)"
        )

    if not gt_p.is_dir():  # single pair
        r = eval_pair(gt_p, pred_p, iou_threshold, off, ann_dir)
        print(json.dumps(r, indent=2) if print_json else format_report(r))
        return r

    gt_files = {pair_key(f): f for f in sorted(gt_p.glob("*.json"))}
    pred_files = {pair_key(f, True): f for f in sorted(pred_p.glob("*.json"))}
    if not gt_files:
        raise FileNotFoundError(f"no .json files in {gt_p}")
    unmatched_gt = [gt_files[k].name for k in gt_files if k not in pred_files]
    unmatched_pred = [pred_files[k].name for k in pred_files if k not in gt_files]
    for n in unmatched_gt:
        print(f"WARNING: no prediction for GT file {n}", file=sys.stderr)
    for n in unmatched_pred:
        print(f"WARNING: no GT file for prediction {n}", file=sys.stderr)

    per_file, failed = {}, {}
    for key, gt_f in gt_files.items():
        if key not in pred_files:
            continue
        try:
            per_file[gt_f.name] = eval_pair(
                gt_f, pred_files[key], iou_threshold, off, ann_dir
            )
        except Exception as e:  # keep going; reported at the end
            failed[gt_f.name] = f"{type(e).__name__}: {e}"
            print(f"ERROR [{gt_f.name}]: {failed[gt_f.name]}", file=sys.stderr)

    agg = aggregate(list(per_file.values())) if per_file else None
    out = {
        "per_file": per_file,
        "aggregate": agg,
        "gt_without_prediction": unmatched_gt,
        "prediction_without_gt": unmatched_pred,
        "failed": failed,
    }
    print(
        json.dumps(out, indent=2)
        if print_json
        else format_batch(per_file, agg, unmatched_gt, unmatched_pred, failed)
    )
    return out


# External truth has rock intervals but no GT legend swatch boxes; legend IoU is not measurable.
# Include every newly added JSON test with a real annotated legend, without making up GT boxes.
_legend_gt = _EVAL_ROOT / "legend_ground_truth"
_legend_pred = _EVAL_ROOT / "legend_predictions"
_legend_gt.mkdir(parents=True, exist_ok=True)
_legend_pred.mkdir(parents=True, exist_ok=True)
for _folder in (_legend_gt, _legend_pred):
    for _old in _folder.glob("*.json"):
        _old.unlink()
for _key, _file, _window, _external in _EVAL_CASES:
    if _external is not None:
        continue
    _gt = json.loads(_file.read_text(encoding="utf-8"))
    if not _gt.get("legend"):
        continue
    (_legend_gt / f"{_key}.json").write_text(json.dumps(_gt), encoding="utf-8")
    _pred_file = _EVAL_PRED / f"{_key}.json"
    if _pred_file.exists():
        (_legend_pred / f"{_key}.json").write_text(
            _pred_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
GT_PATH, PRED_PATH = str(_legend_gt), str(_legend_pred)
print(
    "External legend evaluations: not applicable (truth CSVs contain no legend bounding boxes)."
)
legend_results = run_legend_evaluation(
    GT_PATH, PRED_PATH, IOU_THRESHOLD, ANNOTATIONS_DIR, OFFSET, IMAGE_COORDS, PRINT_JSON
)


# Original column next to the predicted and true lithology in batches of windows spread over the labelled rows.
# Reads the prediction files (1.1); lithologies are matched by name (as the classification evaluator does) and
# coloured like their legend swatch
def lithology_panels(stem, starts=None, n=4, length=1500):
    page = load_page(stem)
    col, y_offset = page["col"], page["y_offset"]
    img = Image.open(page_paths(stem)[0])
    rec = load_record(stem)
    label = (
        json.loads((GT_DIR / f"{stem}.json").read_text())
        if (GT_DIR / f"{stem}.json").exists()
        else None
    )
    key = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()

    colors = {"": np.ones(3), "unclassified": np.full(3, 0.5)}
    for entries in ([label["legend"]] if label else []) + [rec["legend"]]:
        for e in entries:
            if not e.get(
                "box"
            ):  # label legends without swatch boxes: colour comes from the predicted legend
                continue
            colors.setdefault(
                key(e["label_printed"]),
                np.median(swatch_pixels(img, tuple(e["box"])).reshape(-1, 3), 0) / 255,
            )

    # Lithology name of every row ('' = no layer)
    def row_names(layers):
        out = np.full(len(col), "", dtype=object)
        for l in layers:
            comp = l.get("composition") or []
            name = (
                key(max(comp, key=lambda p: p["proportion"])["type"])
                if comp
                else "unclassified"
            )
            out[max(l["top_px"] - y_offset, 0) : max(l["bottom_px"] - y_offset, 0)] = (
                name
            )
        return out

    pred_names = row_names(rec["layers"])
    true_names = row_names(label["layers"]) if label else None
    if starts is None:
        rows = (
            np.where(true_names != "")[0]
            if true_names is not None and (true_names != "").any()
            else np.arange(len(col))
        )
        lo, hi = rows.min(), rows.max() + 1
        starts = (
            [max(0, (lo + hi - length) // 2)]
            if hi - lo <= length
            else np.linspace(lo, hi - length, n).astype(int)
        )

    tracks = [("original", None), ("predicted", pred_names)] + (
        [("true", true_names)] if label else []
    )
    fig, axes = plt.subplots(
        1,
        len(starts) * len(tracks),
        figsize=(1.5 * len(starts) * len(tracks), 10),
        gridspec_kw={"wspace": 0.25},
        squeeze=False,
    )
    used = set()
    for j, start in enumerate(starts):
        end = min(start + length, len(col))
        extent = (0, col.shape[1], end + y_offset, start + y_offset)
        for k, (title, names) in enumerate(tracks):
            ax = axes[0, j * len(tracks) + k]
            if names is None:
                ax.imshow(
                    col[start:end],
                    aspect="auto",
                    interpolation="nearest",
                    extent=extent,
                )
            else:
                window = names[start:end]
                used |= set(window) - {""}
                ax.imshow(
                    np.array([colors.get(s, np.full(3, 0.5)) for s in window])[:, None],
                    aspect="auto",
                    interpolation="nearest",
                    extent=extent,
                )
            ax.set_xticks([])
            ax.set_title(title, fontsize=9)
            if k:
                ax.set_yticks([])
    used = sorted(used)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors.get(s, np.full(3, 0.5)), ec="k", lw=0.5)
        for s in used
    ]
    fig.legend(
        handles,
        used,
        loc="lower center",
        ncol=min(max(len(used), 1), 6),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    fig.suptitle(stem, fontsize=11)
    from IPython.display import display

    display(fig)  # always show it; plt.close alone leaves the cell empty
    plt.close(fig)


for stem in TEST_STEMS:
    try:
        lithology_panels(stem)
    except FileNotFoundError as exc:
        print(f"Skipping visual check for {stem}: {exc}")

_eval_errors = {
    key: message
    for key, message in _EVAL_FAILURES.items()
    if not str(message).startswith("FileNotFoundError:")
}
if _eval_errors:
    raise RuntimeError(f"Pipeline evaluation failed: {_eval_errors}")


