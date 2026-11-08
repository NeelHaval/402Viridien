"""Inventory, sanity checks and well grouping for the lithology dataset."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".tif", ".tiff"}
BOX_KEYS = ("depth_column", "interpreted_lithology_column", "lithology_legend")
EXPECTED_TOP_KEYS = {"image_name", "width", "height", "bounding_boxes"}
API_WELL_DIGITS = 10  # state(2)+county(3)+well(5); further digits = sidetrack/bypass
MIN_API_DIGITS = 10


def group_id(stem: str) -> str:
    """Well-level group id derived from the filename.

    - Leading digit run of >= 10 digits that is not all-zero padding -> US API
      number: first 10 digits (sidetracks/bypasses collapse to one well).
    - Anything else (zero-padded ids like "0000000009_d-582_...", or names
      without an API number) -> prefix up to the 2nd "_" (whole stem if it has
      fewer underscores).
    """
    m = re.match(r"\d+", stem)
    if m and len(m.group()) >= MIN_API_DIGITS and not m.group().startswith("0000"):
        return m.group()[:API_WELL_DIGITS]
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else stem


# Field-level merges (reviewed by hand 2026-11-07, see splits/report.md). Wells
# on the same block/pad/lease share legend, symbols and geology -> one group.
# Keys are group ids produced by group_id(); values are the merged group id.
DEFAULT_GROUP_OVERRIDES: dict[str, str] = {
    # BP Atlantis, Green Canyon 743: three different API wells
    "6081140410": "field_GC743", "6081140730": "field_GC743", "6081140757": "field_GC743",
    # Mississippi Canyon 776 wells #5 and #6
    "6081741170": "field_MC776", "6081741284": "field_MC776",
    # Critter Creek 201-1807H / 202-1807H: same pad (Sec 18 T11N R63W, Hereford field, Weld Co)
    "Critter Creek 201-1807H_Mud": "field_CritterCreek_Hereford",
    "Critter Creek 202-1807H_Mud": "field_CritterCreek_Hereford",
    # Rio_*: Endeavor Energy, Spraberry, Midland Co; same Reservoir Group template + legend
    "Rio_Lavaca": "field_Rio_Endeavor_Spraberry",
    "Rio_Nueces": "field_Rio_Endeavor_Spraberry",
    "Rio_San": "field_Rio_Endeavor_Spraberry",
}

# API numbers read off the log headers of pages whose filename has none.
# None of them matches an API-named group, so no cross-merge was needed.
HEADER_API = {
    "0000000009_d-582": "608114024601 (Conoco OCS-G-11075 GC562 #1 ST#1)",
    "MC765_P011": "608174143300",
    "Shell_08852": "608174139501 (MC764 KE001)",
    "Shell_AC0336": "608054009701",
    "Callaway 3-33 mudlog to TD.PDF": "05039066800000",
    "Critter Creek 201-1807H_Mud": "05-123-44933",
    "Critter Creek 202-1807H_Mud": "051234493500",
    "Mudlog-Trail Mtn1Palisades": "27-011-05309",
    "Mudlog_MD": "05-055-06309 (Freeman 3-24 ST)",
    "Rio_Lavaca": "42-329-42079",
    "Rio_Nueces": "42-329-41851 (#212; #222 not checked)",
    "Rio_San": "42-329-42511",
}


def load_overrides(path: str | Path | None) -> dict[str, str]:
    """DEFAULT_GROUP_OVERRIDES plus an optional JSON {stem_or_group_id: group_id}."""
    ov = dict(DEFAULT_GROUP_OVERRIDES)
    if path and Path(path).exists():
        ov.update(json.loads(Path(path).read_text()))
    return ov


def _check_box(name, box, w, h, issues, stem):
    if box is None:
        issues.append(f"{stem}: missing box '{name}'")
        return None
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        issues.append(f"{stem}: malformed box '{name}': {box!r}")
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        issues.append(f"{stem}: empty/inverted box '{name}': {box}")
        return None
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        issues.append(f"{stem}: box '{name}' {box} outside image bounds {w}x{h}")
    return [int(v) for v in box]


def load_inventory(dataset_dir: str | Path) -> tuple[list[dict], list[str]]:
    """Pair images with annotations and sanity-check them.

    Returns (pages, issues). Only paired pages are returned; every anomaly is
    reported as a human-readable string in issues.
    """
    ds = Path(dataset_dir)
    img_dir, ann_dir = ds / "images", ds / "bbox_annotation"
    images = {p.stem: p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    other = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() not in IMAGE_EXTS)
    anns = {p.stem: p for p in sorted(ann_dir.glob("*.json"))}
    issues: list[str] = []
    for name in other:
        issues.append(f"non-image file in images/: {name} (ignored)")
    for stem in sorted(set(images) - set(anns)):
        issues.append(f"image without annotation: {images[stem].name} (excluded)")
    for stem in sorted(set(anns) - set(images)):
        issues.append(f"annotation without image: {anns[stem].name} (excluded)")

    pages = []
    for stem in sorted(set(images) & set(anns)):
        d = json.loads(anns[stem].read_text())
        with Image.open(images[stem]) as im:  # header only, no pixel decode
            real_w, real_h = im.size
            mode = im.mode
        w, h = d.get("width"), d.get("height")
        if (w, h) != (real_w, real_h):
            issues.append(f"{stem}: JSON size {w}x{h} != image size {real_w}x{real_h} (using image size)")
        if d.get("image_name") != images[stem].name:
            issues.append(f"{stem}: image_name {d.get('image_name')!r} != file {images[stem].name!r}")
        extra = sorted(set(d) - EXPECTED_TOP_KEYS)
        bb = d.get("bounding_boxes") or {}
        extra_bb = sorted(set(bb) - set(BOX_KEYS))
        boxes = {k: _check_box(k, bb.get(k), real_w, real_h, issues, stem) for k in BOX_KEYS}
        pages.append({
            "stem": stem,
            "image_path": str(images[stem]),
            "json_path": str(anns[stem]),
            "width": real_w,
            "height": real_h,
            "image_mode": mode,
            "boxes": boxes,
            "extra_keys": extra + [f"bounding_boxes.{k}" for k in extra_bb],
            "extra_values": {k: d[k] for k in extra},
        })
    return pages, issues


def build_groups(pages: list[dict], overrides: dict[str, str] | None = None) -> dict[str, list[str]]:
    overrides = overrides or {}
    groups: dict[str, list[str]] = defaultdict(list)
    for p in pages:
        gid = group_id(p["stem"])
        gid = overrides.get(p["stem"], overrides.get(gid, gid))
        groups[gid].append(p["stem"])
    return {g: sorted(s) for g, s in sorted(groups.items())}


def grouping_warnings(groups: dict[str, list[str]]) -> list[str]:
    """Heuristics that flag groupings a human should look at."""
    warns = []
    non_api = [g for g in groups if not re.fullmatch(r"\d{10}", g)]
    for g in non_api:
        warns.append(f"non-API group (fallback rule): {g!r} <- {groups[g]}")
    # Same lease/block/field tokens shared across different groups -> possible same field/pad.
    tok_re = re.compile(r"(?i)(?<![a-z])(?:(?:gc|mc|at|ac|gb|ke)[ _]?\d{3,4}|ocs-?g[ _-]?\d{4,5}|g\d{5})(?!\d)")
    by_tok = defaultdict(set)
    for g, stems in groups.items():
        for s in stems:
            for t in tok_re.findall(s):
                by_tok[re.sub(r"[ _-]", "", t.lower())].add(g)
    for t, gs in sorted(by_tok.items()):
        if len(gs) > 1:
            warns.append(f"token {t!r} shared by groups {sorted(gs)} (same block/lease, different wells?)")
    # Name prefixes shared across groups (e.g. 'Critter Creek 201' vs '202').
    first_word = defaultdict(set)
    for g in non_api:
        first_word[re.split(r"[ _-]", g)[0].lower()].add(g)
    for w, gs in sorted(first_word.items()):
        if len(gs) > 1:
            warns.append(f"non-API groups share leading word {w!r}: {sorted(gs)}")
    return warns
