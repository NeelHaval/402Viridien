"""Extract visible, clipped WellSight image tiles; never infer rocks from pixels.

Run with python -I -X utf8 external/kgs_extract.py --help from bakeoff.
PDFs are untrusted data. No PDF actions, attachments, or scripts are executed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from io import BytesIO
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

import fitz
import numpy as np
from PIL import Image, ImageDraw
from pypdf.generic import (ContentStream, DecodedStreamObject, DictionaryObject,
                           IndirectObject, NameObject, read_object)

# Only reopen our own bounded renders. Keep Pillow's limit finite while
# allowing the explicitly supported 160-megapixel page size.
Image.MAX_IMAGE_PIXELS = 160_000_000

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "external_data" / "kgs"
SAMPLE = ROOT / "external_untrusted" / "kgs" / "386825.pdf"
FIELDS = ["top_depth", "bottom_depth", "lithology", "top_y", "bottom_y", "cluster_id"]
EPS = 0.002  # MuPDF float32 geometry noise, in points (not a bed-size cutoff).


class Skip(ValueError):
    """A page cannot supply exact image-identity truth."""


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"Outputs must be under {OUTPUT_ROOT}")
    return path


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def default_resolution():
    path = ROOT / "analysis" / "volume_logs.csv"
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            values = [float(r["ft_per_px"]) for r in csv.DictReader(f)
                      if "texas" in r.get("family", "").lower() and r.get("ft_per_px")
                      and math.isfinite(float(r["ft_per_px"])) and float(r["ft_per_px"]) > 0]
        if values:
            return statistics.median(values), str(path)
    return 0.05, "fallback: 0.05 ft/px (no Texas measurements)"


def axis_aligned(matrix):
    return abs(matrix.b) < 1e-5 and abs(matrix.c) < 1e-5


def pdf_object(doc, value):
    """Resolve a resource using MuPDF; pypdf only reads the returned bytes."""
    seen = set()
    while isinstance(value, IndirectObject):
        if value.idnum in seen:
            raise Skip("cyclic PDF resource")
        seen.add(value.idnum)
        value = read_object(BytesIO(doc.xref_object(value.idnum).encode()), doc)
    return value


def page_resources(page):
    doc, ref, seen = page.parent, page.xref, set()
    while ref and ref not in seen:
        seen.add(ref)
        obj = read_object(BytesIO(doc.xref_object(ref).encode()), doc)
        if "/Resources" in obj:
            return pdf_object(doc, obj.raw_get("/Resources"))
        parent = obj.get("/Parent")
        ref = parent.idnum if isinstance(parent, IndirectObject) else 0
    return DictionaryObject()


def inline_image(doc, args, resources, sources, cache):
    """Decode inline pixels with MuPDF, retaining exact identity and pixels.

    A scratch image object handles PDF filters, bit depths and color spaces;
    it is never rendered or matched by similarity. No source PDF is modified.
    """
    aliases = {"/W": "/Width", "/H": "/Height", "/BPC": "/BitsPerComponent",
               "/CS": "/ColorSpace", "/F": "/Filter", "/DP": "/DecodeParms",
               "/D": "/Decode", "/IM": "/ImageMask", "/I": "/Interpolate",
               "/RGB": "/DeviceRGB", "/G": "/DeviceGray", "/CMYK": "/DeviceCMYK",
               "/Fl": "/FlateDecode", "/AHx": "/ASCIIHexDecode",
               "/A85": "/ASCII85Decode", "/LZW": "/LZWDecode",
               "/RL": "/RunLengthDecode", "/CCF": "/CCITTFaxDecode",
               "/DCT": "/DCTDecode"}

    def expand(value):
        if isinstance(value, IndirectObject) and doc.xref_is_stream(value.idnum):
            raise Skip("inline color-space resource streams are unsupported")
        value = pdf_object(doc, value)
        if isinstance(value, NameObject):
            return NameObject(aliases.get(str(value), str(value)))
        if isinstance(value, DictionaryObject):
            return DictionaryObject({NameObject(aliases.get(str(k), str(k))): expand(v)
                                     for k, v in value.items()})
        if isinstance(value, list):
            return type(value)([expand(v) for v in value])
        return value

    settings = expand(args["settings"])
    cs = settings.get("/ColorSpace")
    spaces = pdf_object(doc, resources.get("/ColorSpace", DictionaryObject()))
    if isinstance(cs, NameObject) and cs in spaces:
        settings[NameObject("/ColorSpace")] = expand(spaces.raw_get(cs))
    mask = settings.get("/ImageMask")
    if getattr(mask, "value", mask):
        raise Skip("inline image masks are unsupported")
    buf = BytesIO()
    settings.write_to_stream(buf)
    dictionary = buf.getvalue()
    key = (dictionary, args["data"])
    if key not in cache:
        with fitz.open() as scratch:
            ref = scratch.get_new_xref()
            scratch.update_object(ref, "<<>>")
            scratch.update_stream(ref, args["data"], compress=False)
            scratch.update_object(ref, (b"<< /Type /XObject /Subtype /Image /Length "
                + str(len(args["data"])).encode() + b" " + dictionary[2:-2] + b">>").decode("latin1"))
            pix = fitz.Pixmap(scratch, ref)
            header = f"{pix.width},{pix.height},{pix.n},{pix.alpha},{pix.colorspace.name}".encode()
            identity = -int(hashlib.sha256(header + b"\0" + pix.samples).hexdigest(), 16)
            if pix.colorspace.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            sources.setdefault(identity, pix.tobytes("png"))
            cache[key] = identity
    return cache[key]


def placed_images(page, inline_sources=None):
    """Read graphics state and RECTANGULAR clips, not untrimmed image bboxes.

    pypdf only tokenizes bytes supplied by fitz; it never opens the PDF. Reject
    unsupported geometry on an image instead of manufacturing exact truth.
    """
    doc = page.parent
    sources = inline_sources if inline_sources is not None else {}
    images, cache, objects_cache = [], {}, {}

    def resolve(value):
        if not isinstance(value, IndirectObject):
            return value
        if value.idnum not in objects_cache:
            objects_cache[value.idnum] = pdf_object(doc, value)
        return objects_cache[value.idnum]

    def walk(data, resources, matrix, clip, uncertain=False, ancestors=()):
        if len(ancestors) > 32:
            raise Skip("Form nesting exceeds 32 levels")
        stack, path, pending, polygon = [], [], False, []
        nonrect = False
        decoded = DecodedStreamObject()
        decoded.set_data(data)
        stream = ContentStream(decoded, None)
        for args, operator in stream.operations:
            if operator == b"q":
                stack.append((fitz.Matrix(matrix), fitz.Rect(clip), uncertain))
            elif operator == b"Q":
                if not stack:
                    raise Skip("unbalanced PDF graphics state")
                matrix, clip, uncertain = stack.pop()
            elif operator == b"cm":
                matrix = fitz.Matrix(*map(float, args)) * matrix
            elif operator == b"re":
                x, y, w, h = map(float, args)
                rect = fitz.Rect(min(x, x+w), min(y, y+h), max(x, x+w), max(y, y+h))
                path.append(rect * matrix)
                nonrect |= not axis_aligned(matrix)
            elif operator in (b"m", b"l"):
                if operator == b"m" and polygon:
                    nonrect = True
                polygon.append(fitz.Point(*map(float, args)) * matrix)
            elif operator in (b"c", b"v", b"y"):
                nonrect = True
            elif operator in (b"W", b"W*"):
                pending = True
            elif operator in (b"n", b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"):
                if pending:
                    if polygon:
                        points = polygon[:-1] if len(polygon) == 5 and polygon[0] == polygon[-1] else polygon
                        xs, ys = {p.x for p in points}, {p.y for p in points}
                        rectangular = (len(points) == 4 and len(xs) == len(ys) == 2
                            and len({(p.x, p.y) for p in points}) == 4
                            and all(a.x == b.x or a.y == b.y
                                    for a, b in zip(points, points[1:]+points[:1])))
                        if rectangular:
                            path.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
                        else:
                            nonrect = True
                    if nonrect or len(path) != 1:
                        uncertain = True
                    else:
                        clip &= path[0]
                path, pending, nonrect, polygon = [], False, False, []
            elif operator in (b"Do", b"INLINE IMAGE"):
                if operator == b"INLINE IMAGE":
                    ref = inline_image(doc, args, resources, sources, cache)
                else:
                    objects = resolve(resources.get("/XObject", DictionaryObject()))
                    target = objects.raw_get(str(args[0]))
                    if not isinstance(target, IndirectObject):
                        raise Skip("XObject has no indirect identity")
                    ref = target.idnum
                    obj = resolve(target)
                    if obj.get("/Subtype") == "/Form":
                        if ref in ancestors:
                            raise Skip("cyclic Form XObject")
                        fm = fitz.Matrix(*map(float, obj.get("/Matrix", [1, 0, 0, 1, 0, 0]))) * matrix
                        form_clip = fitz.Rect(list(map(float, obj["/BBox"]))) * fm
                        local = resolve(obj.raw_get("/Resources")) if "/Resources" in obj else resources
                        walk(doc.xref_stream(ref), local, fm, clip & form_clip,
                             uncertain or not axis_aligned(fm), ancestors + (ref,))
                        continue
                    if obj.get("/Subtype") != "/Image":
                        raise Skip("unsupported XObject subtype")
                if uncertain or not axis_aligned(matrix):
                    raise Skip("nonrectangular clip or rotated/sheared image")
                bbox = fitz.Rect(0, 0, 1, 1) * matrix
                visible = bbox & clip
                images.append({"xref": ref, "bbox": list(bbox),
                               "visible_bbox": list(visible), "order": len(images),
                               "flip_y": matrix.d > 0})
        if stack:
            raise Skip("unbalanced PDF graphics state")

    walk(page.read_contents(), page_resources(page), fitz.Matrix(page.transformation_matrix),
         fitz.Rect(page.rect))
    # Annotation appearances are outside page contents and can add images
    # (e.g. a PTS signature stamp). Compare the same painting domain.
    textpage = fitz.TextPage(page.get_displaylist(annots=False).get_textpage(flags=fitz.TEXT_PRESERVE_IMAGES))
    if len(images) != len(textpage.extractIMGINFO()):
        raise Skip("placed-image count differs from MuPDF (page contents)")
    return images


def border_lines(page):
    result = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:3]
                if abs(a.x-b.x) < EPS and abs(a.y-b.y) > 80:
                    result.append((float(a.x), float(min(a.y, b.y)), float(max(a.y, b.y))))
            elif item[0] == "re":
                r = item[1]
                if r.height > 80:
                    result.extend([(r.x0, r.y0, r.y1), (r.x1, r.y0, r.y1)])
    return sorted(set(result))


def union_boxes(boxes):
    rect = fitz.Rect(boxes[0])
    for box in boxes[1:]:
        rect |= fitz.Rect(box)
    return list(rect)


def page_words(page):
    """Exclude invisible OCR text; only printed PDF glyphs can label truth."""
    blocks = page.get_text("rawdict", flags=fitz.TEXTFLAGS_WORDS)["blocks"]
    def visible(span):
        return span.get("alpha", 255) > 0 and bool(span.get("char_flags", 24) & 24)
    if all(visible(s) for b in blocks for l in b.get("lines", []) for s in l["spans"]):
        return page.get_text("words")
    words = []
    for block in blocks:
        for line_index, line in enumerate(block.get("lines", [])):
            chars, word_index = [], 0
            for char in [c for s in line["spans"] if visible(s) for c in s["chars"]] + [{"c": " "}]:
                if char["c"].isspace():
                    if chars:
                        words.append((*union_boxes([c["bbox"] for c in chars]),
                            "".join(c["c"] for c in chars), block["number"], line_index, word_index))
                        chars, word_index = [], word_index+1
                else:
                    chars.append(char)
    return words


def parse_legend(page, images, borders):
    words = page_words(page)
    heads = []
    for word in words:
        if word[4].upper() == "ROCK":
            following = [w for w in words if w[4].upper() == "TYPES"
                         and abs(w[1]-word[1]) < 3 and 0 <= w[0]-word[2] < 25]
            if following:
                heads.append(union_boxes([word[:4], min(following, key=lambda w: w[0])[:4]]))
    accessory_heads = [list(w[:4]) for w in words if w[4].upper() == "ACCESSORIES"]
    sections, entries = [], []
    for head in sorted(heads, key=lambda b: (b[1], b[0])):
        stops = [b[1] for b in heads + accessory_heads if b[1] > head[3]]
        stops += [b[1] for b in borders if b[1] > head[3]]
        end = min(stops, default=page.rect.y1)
        candidates = [im for im in images if head[3] <= im["bbox"][1] < end]
        section_entries = []
        for im in candidates:
            label = nearest_label(words, im["bbox"])
            if label:
                item = {"label": label[4], "kind": "rock", "xref": im["xref"],
                        "page": page.number+1, "bbox_pt": im["bbox"],
                        "label_bbox_pt": list(label[:4]), "section": len(sections)}
                section_entries.append(item)
        if section_entries:
            box = union_boxes([head] + [e[k] for e in section_entries
                                      for k in ("bbox_pt", "label_bbox_pt")])
            sections.append({"kind": "rock", "bbox_pt": box})
            entries.extend(section_entries)
    for head in accessory_heads:
        ends = [b[1] for b in heads + accessory_heads if b[1] > head[3]]
        ends += [b[1] for b in borders if b[1] > head[3]]
        end = min(ends, default=page.rect.y1)
        candidates = [im for im in images if head[3] <= im["bbox"][1] < end]
        columns = sorted({round(im["bbox"][0], 1) for im in candidates})
        headings = [w for w in words if head[3] <= w[1] < end
                    and w[4].upper() in ("FOSSIL", "MINERAL", "TEXTURE", "STRINGER")]
        def key(x, y):
            return (min(range(len(columns)), key=lambda i: abs(columns[i]-x)), y)
        for im in candidates:
            label = nearest_label(words, im["bbox"])
            if not label:
                continue
            preceding = [w for w in headings if key(w[0], w[1]) <= key(im["bbox"][0], im["bbox"][1]+1)]
            kind = max(preceding, key=lambda w: key(w[0], w[1]))[4].lower() if preceding else "accessory"
            entries.append({"label": label[4], "kind": kind, "xref": im["xref"],
                            "page": page.number+1, "bbox_pt": im["bbox"],
                            "label_bbox_pt": list(label[:4])})
    return {"entries": entries, "sections": sections}


def nearest_label(words, bbox):
    x0, y0, x1, y1 = bbox
    choices = [w for w in words if 0 <= w[0]-x1 < 100
               and abs(w[1]-y0) <= max(3, (y1-y0)*0.6)
               and w[4].upper() not in ("TYPES", "ACCESSORIES")]
    first = min(choices, key=lambda w: math.hypot(w[0]-x1, 3*(w[1]-y0)), default=None)
    if first is None:
        return None
    line = sorted([w for w in words if w[5:7] == first[5:7] and w[0] >= first[0]], key=lambda w: w[0])
    return (*union_boxes([w[:4] for w in line]), " ".join(w[4] for w in line), *first[5:])


def legend_map(entries, kind="rock"):
    mapping = {}
    for entry in entries:
        if (entry["kind"] == "rock") != (kind == "rock"):
            continue
        ref = entry["xref"]
        if ref in mapping and mapping[ref]["label"] != entry["label"]:
            # Identical pixels cannot distinguish aliases printed in a legend.
            # Preserve the complete set instead of guessing one rock name.
            labels = sorted(set(mapping[ref].get("label_aliases", [mapping[ref]["label"]])) | {entry["label"]})
            mapping[ref] = dict(entry, label=" / ".join(labels), label_aliases=labels)
        else:
            mapping[ref] = entry
    return mapping


def find_track(page, images, rocks, borders, sections, track_x_hint=None):
    """Choose the long border pair with greatest integrated rock-tile coverage."""
    candidates = []
    for im in images:
        r = fitz.Rect(im["visible_bbox"])
        if im["xref"] in rocks and not r.is_empty and not any(
                fitz.Rect(s["bbox_pt"]).contains(r) for s in sections):
            candidates.append(im)
    if not candidates:
        raise Skip("raster_only_or_no_placed_rock_tiles_in_track")
    # The densest band is found from placements first. Borders then establish
    # its actual width; the untrimmed image width is deliberately not used.
    edges = sorted({round(im["visible_bbox"][k], 3) for im in candidates for k in (0, 2)})
    bands = []
    for left, right in zip(edges, edges[1:]):
        if right-left < EPS:
            continue
        mid = (left+right)/2
        selected = [im for im in candidates if im["visible_bbox"][0] <= mid < im["visible_bbox"][2]]
        if not selected:
            continue
        spans = sorted((im["visible_bbox"][1], im["visible_bbox"][3]) for im in selected)
        score, end = 0, -math.inf
        for top, bottom in spans:
            score += max(0, bottom-max(top, end))
            end = max(end, bottom)
        bands.append((score, left, right, selected))
    if not bands:
        raise Skip("no vertical band of rock tiles")
    def rank(band):
        score, left, right, _ = band
        distance = 0 if track_x_hint is None else max(left-track_x_hint, track_x_hint-right, 0)
        return (-distance, round(score/EPS), right-left, -left)
    _, left, right, selected = max(bands, key=rank)
    mid = (left+right)/2
    ymin = min(im["visible_bbox"][1] for im in selected)
    ymax = max(im["visible_bbox"][3] for im in selected)
    long = [b for b in borders if b[2] >= ymax-EPS and (b[1] <= ymin+EPS or (
        ymin <= page.rect.y0+EPS and max(0, ymax-b[1]) >= 0.99*(ymax-ymin)))]
    lefts = [b for b in long if b[0] <= mid+EPS]
    rights = [b for b in long if b[0] > mid+EPS]
    if not lefts or not rights:
        raise Skip("rock band has no enclosing drawn track borders")
    a, b = max(lefts, key=lambda x: x[0]), min(rights, key=lambda x: x[0])
    bounds = [a[0], max(a[1], b[1], page.rect.y0), b[0], min(a[2], b[2], page.rect.y1)]
    track = fitz.Rect(bounds)
    tiles = []
    accessories = []
    for im in images:
        visible = fitz.Rect(im["visible_bbox"]) & track
        if visible.width <= EPS or visible.height <= EPS:
            continue
        item = dict(im, visible_bbox=list(visible))
        (tiles if im["xref"] in rocks else accessories).append(item)
    if not tiles:
        raise Skip("no rock tiles inside detected track")
    return bounds, tiles, accessories


def fit_depth(page, bounds, borders, depth_unit="ft"):
    # Cluster numeric words by x position, then prefer a consistent linear
    # depth sequence left of the lithology column. No fixed x/depth constants.
    groups = defaultdict(list)
    for word in page_words(page):
        pattern = r"\d{1,6}(?:\.\d+)?" if depth_unit == "m" else r"\d{3,6}(?:\.\d+)?"
        if re.fullmatch(pattern, word[4]) and word[2] <= bounds[0]:
            groups[round((word[0]+word[2])/2/3)].append(word)
    fits = []
    for words in groups.values():
        cx = statistics.median((w[0]+w[2])/2 for w in words)
        enclosing = [b for b in borders if b[1] <= bounds[1]+EPS and b[2] >= bounds[3]-EPS]
        lefts, rights = [b for b in enclosing if b[0] < cx], [b for b in enclosing if b[0] > cx]
        low, high = bounds[1], bounds[3]
        if lefts and rights:
            left = max(lefts, key=lambda b: (b[0], b[2]-b[1]))
            right = min(rights, key=lambda b: (b[0], b[1]-b[2]))
            low, high = max(left[1], right[1], page.rect.y0), min(left[2], right[2], page.rect.y1)
        # At page breaks a clipped rotated "230" can be extracted as "30".
        # Only complete glyph boxes safely inside the depth track may calibrate.
        words = [w for w in words if low+0.5 < w[1] and w[3] < high-0.5]
        words.sort(key=lambda w: w[1])
        if len(words) >= 4:
            tail, previous = words[-1], words[-2]
            if (float(tail[4]) <= float(previous[4]) and len(tail[4]) < len(previous[4])
                    and high-tail[3] < 2*(previous[3]-previous[1])):
                words.pop()  # Partial rotated suffix inside a PDF clip at the page foot.
        if len(words) < 3:
            continue
        y = np.array([(w[1]+w[3])/2 for w in words])
        depths = np.array([float(w[4]) for w in words]) * (1/0.3048 if depth_unit == "m" else 1)
        order = np.argsort(y)
        y, depths = y[order], depths[order]
        if not np.all(np.diff(depths) > 0) or np.ptp(y) < (bounds[3]-bounds[1])*0.4:
            continue
        slope, intercept = np.polyfit(y, depths, 1)
        residual = depths-(slope*y+intercept)
        rmse = float(np.sqrt(np.mean(residual**2)))
        if slope > 0 and rmse < 2:
            fits.append((len(words), -rmse, words, float(slope), float(intercept), rmse))
    if not fits:
        raise Skip("no consistent depth-label regression (>=3 labels, RMSE <2 ft)")
    _, _, words, slope, intercept, rmse = max(fits, key=lambda f: (f[0], f[1]))
    cx = statistics.median((w[0]+w[2])/2 for w in words)
    enclosing = [b for b in borders if b[1] <= bounds[1]+EPS and b[2] >= bounds[3]-EPS]
    left = max((b[0] for b in enclosing if b[0] < cx), default=min(w[0] for w in words))
    right = min((b[0] for b in enclosing if b[0] > cx), default=max(w[2] for w in words))
    return {"slope_ft_per_pt": slope, "intercept_ft": intercept, "rmse_ft": rmse,
            "source_unit": depth_unit,
            "labels": [{"depth_ft": float(w[4]) * (1/0.3048 if depth_unit == "m" else 1), "y_pt": (w[1]+w[3])/2,
                        "bbox_pt": list(w[:4])} for w in sorted(words, key=lambda w: w[1])],
            "depth_track_bounds_pt": [left, bounds[1], right, bounds[3]],
            "position_convention": "word bounding-box vertical midpoint; OLS, no depth snapping"}


def canonical_edges(values):
    values = sorted(set(values))
    groups = []
    for value in values:
        if groups and value-groups[-1][-1] <= EPS:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [statistics.mean(g) for g in groups]


def truth_intervals(bounds, tiles, rocks):
    """Sweep visible rectangles, resolving overlaps in PDF painting order."""
    ys = canonical_edges([bounds[1], bounds[3]] + [t["visible_bbox"][k] for t in tiles for k in (1, 3)])
    ys[0], ys[-1] = bounds[1], bounds[3]
    result = []
    for top, bottom in zip(ys, ys[1:]):
        mid = (top+bottom)/2
        active = [t for t in tiles if t["visible_bbox"][1]-EPS/2 <= mid < t["visible_bbox"][3]-EPS/2]
        xs = canonical_edges([bounds[0], bounds[2]] + [t["visible_bbox"][k] for t in active for k in (0, 2)])
        shares = defaultdict(float)
        for left, right in zip(xs, xs[1:]):
            center = (left+right)/2
            covering = [t for t in active if t["visible_bbox"][0] <= center < t["visible_bbox"][2]]
            ref = max(covering, key=lambda t: t["order"])["xref"] if covering else 0
            shares[ref] += (right-left)/(bounds[2]-bounds[0])
        mix = [{"xref": ref or None, "lithology": rocks[ref]["label"] if ref else "(blank)",
                "width_share": share} for ref, share in sorted(shares.items()) if share > 1e-8]
        dominant = min(mix, key=lambda m: (-round(m["width_share"], 10), m["lithology"], m["xref"] or 0))
        signature = [(m["xref"], round(m["width_share"], 7)) for m in mix]
        if result and result[-1]["signature"] == signature:
            result[-1]["bottom_pt"] = bottom
        else:
            result.append({"top_pt": top, "bottom_pt": bottom, "lithology": dominant["lithology"],
                           "xref": dominant["xref"], "components": mix, "signature": signature})
    for row in result:
        del row["signature"]
    return result


def depth_at(fit, y):
    return fit["slope_ft_per_pt"]*y + fit["intercept_ft"]


def image_rows(intervals, scale, fit, rocks, *, subpixel=False):
    clusters = {name: i for i, name in enumerate(sorted({v["label"] for v in rocks.values()}))}
    rows = []
    for interval in intervals:
        top, bottom = interval["top_pt"]*scale, interval["bottom_pt"]*scale
        if not subpixel:
            top, bottom = round(top), round(bottom)
        if bottom <= top:
            raise Skip("target resolution collapses a truth interval to zero image rows")
        rows.append({"top_depth": depth_at(fit, interval["top_pt"]),
                     "bottom_depth": depth_at(fit, interval["bottom_pt"]),
                     "lithology": interval["lithology"], "top_y": top, "bottom_y": bottom,
                     "cluster_id": clusters.get(interval["lithology"], -1)})
    return rows


def pixel_box(box, scale):
    return [round(v*scale) for v in box]


def image_pixmap(doc, ref, inline_sources):
    return fitz.Pixmap(inline_sources[ref]) if ref < 0 else fitz.Pixmap(doc, ref)


def expand_cached_tiles(doc, images, rocks, inline_sources):
    """Split raster caches ONLY into byte-identical complete legend images.

    Some WellSight writers concatenate many rock swatches into one taller
    XObject. Every source row must be accounted for by exact decoded bytes;
    a partial match, resampling or approximate pattern match is forbidden.
    """
    candidates = defaultdict(list)
    for ref in rocks:
        if ref < 0:
            continue  # Inline identities already match their decoded source.
        pix = image_pixmap(doc, ref, inline_sources)
        key = (pix.width, pix.n, pix.alpha, pix.colorspace.name)
        candidates[key].append((ref, pix.height, pix.samples))
    plans, proofs, expanded = {}, [], []
    for image in images:
        source = image["xref"]
        if source not in plans:
            plans[source] = None
            if source > 0 and source not in rocks and all(
                    doc.xref_get_key(source, key)[0] == "null" for key in ("SMask", "Mask")):
                pix = fitz.Pixmap(doc, source)
                if pix.colorspace is not None:
                    key = (pix.width, pix.n, pix.alpha, pix.colorspace.name)
                    choices = candidates.get(key, [])
                    stride, data = pix.stride, pix.samples
                    # Dynamic programming accepts only a complete exact tiling.
                    endings = {pix.height: []}
                    for top in range(pix.height-1, -1, -1):
                        matches = [(ref, height) for ref, height, raw in choices
                                   if top+height in endings and data[top*stride:(top+height)*stride] == raw]
                        if len({rocks[ref]["label"] for ref, _ in matches}) > 1:
                            raise Skip("cached image has ambiguous exact legend matches")
                        if matches:
                            ref, height = min(matches)
                            endings[top] = [(ref, top, top+height)] + endings[top+height]
                    if 0 in endings:
                        plans[source] = (pix.height, endings[0])
                        proofs.append({"source_xref": source, "height_px": pix.height,
                            "decoded_sha256": hashlib.sha256(data).hexdigest(),
                            "segments": [{"xref": ref, "top_row": top, "bottom_row": bottom}
                                         for ref, top, bottom in endings[0]],
                            "match": "complete decoded byte equality; no resampling"})
        if plans[source] is None:
            expanded.append(image)
            continue
        height, segments = plans[source]
        box, clip = fitz.Rect(image["bbox"]), fitz.Rect(image["visible_bbox"])
        for ref, top, bottom in segments:
            y0, y1 = top/height, bottom/height
            if image.get("flip_y"):
                y0, y1 = 1-y1, 1-y0
            part = fitz.Rect(box.x0, box.y0+y0*box.height, box.x1, box.y0+y1*box.height)
            visible = part & clip
            if not visible.is_empty:
                expanded.append(dict(image, xref=ref, source_xref=source,
                                     bbox=list(part), visible_bbox=list(visible)))
    return [dict(im, order=index) for index, im in enumerate(expanded)], proofs


def render_swatch(doc, entry, scale, out, inline_sources=None):
    # Use only the embedded pixels. A temporary in-memory page gives exactly
    # the PDF renderer's resampling at the original placement size and scale.
    source = image_pixmap(doc, entry["xref"], inline_sources or {})
    if source.colorspace is None:
        raise Skip("image-mask legend swatches are unsupported")
    if source.colorspace.n != 3:
        source = fitz.Pixmap(fitz.csRGB, source)
    r = fitz.Rect(entry["bbox_pt"])
    with fitz.open() as tile_doc:
        tile_page = tile_doc.new_page(width=r.width, height=r.height)
        tile_page.insert_image(tile_page.rect, pixmap=source, keep_proportion=False)
        rendered = tile_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        rendered.set_dpi(max(1, round(scale*72)), max(1, round(scale*72)))
        rendered.save(str(out))


def save_overlay(doc, page, xref, out, inline_sources=None):
    pix = image_pixmap(doc, xref, inline_sources or {})
    smask = next((im[1] for im in page.get_images(full=True) if im[0] == xref), 0)
    if pix.colorspace is not None and pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if smask:
        pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask))
    pix.save(str(out))
    return smask


def palette(label):
    fixed = {"(blank)": "#ffffff", "Lmst": "#4590d0", "Shale": "#7b607f",
             "Ss": "#e5b94a", "Sltst": "#69a775"}
    return fixed.get(label, "#" + hashlib.sha256(label.encode()).hexdigest()[:6])


def contact_panel(out, rows, bounds, scale, title):
    """Split the column into readable vertical panels; never shrink away beds."""
    image = Image.open(out / "image.png")
    box = pixel_box(bounds, scale)
    nonblank = [r for r in rows if r["lithology"] != "(blank)"]
    if nonblank:
        # Keep the complete truth in CSV/boxes, but spend review space on beds.
        box[1] = max(box[1], nonblank[0]["top_y"]-20)
        box[3] = min(box[3], nonblank[-1]["bottom_y"]+20)
    column = image.crop(box).convert("RGB")
    image.close()
    width = 130
    display_scale = width / column.width
    height = max(1, round(column.height*display_scale))
    column = column.resize((width, height), Image.Resampling.LANCZOS)
    chunk_height, panel_width = 1600, 260
    count = math.ceil(height/chunk_height)
    labels = sorted({r["lithology"] for r in rows})
    sheet = Image.new("RGB", (count*panel_width, 1675+15*len(labels)), "#f1f1f1")
    draw = ImageDraw.Draw(sheet)
    for i in range(count):
        x, start = i*panel_width, i*chunk_height
        draw.text((x+6, 6), f"{title}  part {i+1}/{count}", fill="black")
        draw.text((x+6, 24), f"PDF / truth; from image row {box[1]}", fill="black")
        crop = column.crop((0, start, width, min(height, start+chunk_height)))
        sheet.paste(crop, (x+6, 48))
        for row in rows:
            y0 = (row["top_y"]-box[1])*display_scale-start
            y1 = (row["bottom_y"]-box[1])*display_scale-start
            if y1 <= 0 or y0 >= chunk_height:
                continue
            a, b = 48+max(0, round(y0)), 48+min(chunk_height, round(y1))
            draw.rectangle((x+148, a, x+178, max(a, b-1)), fill=palette(row["lithology"]))
            if 0 <= y0 < chunk_height and row["lithology"] != "(blank)":
                draw.text((x+182, a), f"{row['top_depth']:.1f}", fill="black")
        for j, label in enumerate(labels):
            yy = 1655+j*15
            draw.rectangle((x+6, yy, x+18, yy+11), fill=palette(label), outline="black")
            draw.text((x+24, yy), label, fill="black")
    sheet.save(out / "contact.png")
    return sheet


def extract_page(doc, page, out, images, borders, local_legend, all_entries, ft_per_px,
                 resolution_source, pdf_hash, inline_sources=None, track_x_hint=None, depth_unit="ft"):
    rocks = legend_map(local_legend["entries"]) or legend_map(all_entries)
    if not rocks:
        raise Skip("no ROCK TYPES legend on this PDF")
    images, exact_expansions = expand_cached_tiles(doc, images, rocks, inline_sources or {})
    sections = local_legend["sections"]
    bounds, tiles, unknown = find_track(page, images, rocks, borders, sections, track_x_hint)
    # An unmatched opaque cache across the column cannot be called blank or
    # dismissed as a small accessory. Require exact decomposition or skip.
    rock_sizes = [image_pixmap(doc, ref, inline_sources or {}) for ref in rocks if ref > 0]
    checked = set()
    for image in unknown:
        ref = image["xref"]
        if ref <= 0 or ref in checked or fitz.Rect(image["visible_bbox"]).width < 0.9*(bounds[2]-bounds[0]):
            continue
        checked.add(ref)
        if any(doc.xref_get_key(ref, key)[0] != "null" for key in ("SMask", "Mask")):
            continue
        pix = fitz.Pixmap(doc, ref)
        if any(pix.width == swatch.width and pix.height > swatch.height for swatch in rock_sizes):
            raise Skip("unmatched cached rock image in lithology track")
    fit = fit_depth(page, bounds, borders, depth_unit)
    scale = fit["slope_ft_per_pt"] / ft_per_px
    if page.rect.width*page.rect.height*scale**2 > 160_000_000:
        raise Skip("requested render exceeds 160 million pixels")
    intervals = truth_intervals(bounds, tiles, rocks)
    subpixel = any(round(r["top_pt"]*scale) == round(r["bottom_pt"]*scale) for r in intervals)
    rows = image_rows(intervals, scale, fit, rocks, subpixel=subpixel)
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    swatches = out / "legend_swatches"
    swatches.mkdir(exist_ok=True)
    entries = [dict(e) for e in (local_legend["entries"] or all_entries)]
    if not any(e["kind"] == "rock" for e in entries):
        entries.extend(dict(e) for e in rocks.values())
    saved = {}
    for entry in entries:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", entry["label"])
        if entry["kind"] != "rock":
            name = entry["kind"] + "_" + name
        identity = (name, entry["xref"])
        if identity not in saved:
            filename = name + ".png"
            if filename in saved.values():
                filename = f"{name}_xref{entry['xref']}.png"
            render_swatch(doc, entry, scale, swatches / filename, inline_sources)
            saved[identity] = filename
        entry["swatch_png"] = "legend_swatches/" + saved[identity]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    pix.set_dpi(max(1, round(scale*72)), max(1, round(scale*72)))
    pix.save(str(out / "image.png"))
    boxes = {"depth_column": pixel_box(fit["depth_track_bounds_pt"], scale),
             "interpreted_lithology_column": pixel_box(bounds, scale)}
    if sections:
        boxes["lithology_legend"] = pixel_box(union_boxes([s["bbox_pt"] for s in sections]), scale)
    write_json(out / "boxes.json", {"image_name": "image.png", "width": pix.width,
                                   "height": pix.height, "bounding_boxes": boxes})
    with (out / "truth.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    accessories_by_ref = legend_map(all_entries, "accessory")
    accessories = []
    overlay_dir = out / "accessories"
    unknown_orders = {u["order"] for u in unknown}
    overlay_files = {}
    for im in images:
        r = fitz.Rect(im["visible_bbox"])
        if r.is_empty or r.y1 <= bounds[1] or r.y0 >= bounds[3]:
            continue
        if im["xref"] not in accessories_by_ref and im["order"] not in unknown_orders:
            continue
        if im["xref"] in rocks:
            continue
        entry = accessories_by_ref.get(im["xref"], {})
        overlap = r & fitz.Rect(bounds)
        if im["xref"] not in overlay_files:
            overlay_dir.mkdir(exist_ok=True)
            filename = f"xref_{im['xref']}.png"
            smask = save_overlay(doc, page, im["xref"], overlay_dir / filename, inline_sources)
            overlay_files[im["xref"]] = ("accessories/" + filename, smask)
        overlay_file, smask = overlay_files[im["xref"]]
        accessories.append({"xref": im["xref"], "label": entry.get("label", "(unclassified overlay)"),
                            "kind": entry.get("kind", "unclassified_overlay"), "bbox_pt": list(r),
                            "bbox_px": pixel_box(r, scale), "over_column": not overlap.is_empty,
                            "image_png": overlay_file, "smask_xref": smask,
                            "top_depth": depth_at(fit, r.y0), "bottom_depth": depth_at(fit, r.y1)})
    nonblank = [r for r in intervals if any(c["xref"] is not None for c in r["components"])]
    meta = {"status": "ok", "pdf_sha256": pdf_hash, "page": page.number+1,
            "page_size_pt": list(page.rect), "render": {"dpi": scale*72, "px_per_pt": scale,
                "ft_per_px": ft_per_px, "resolution_source": resolution_source},
            "legend_entries": entries, "legend_sections": sections,
            "legend_inherited": not bool(local_legend["sections"]),
            "track_bounds_pt": bounds, "track_bounds_px": boxes["interpreted_lithology_column"],
            "lithology_header_x_pt": track_x_hint,
            "depth_fit": fit, "intervals_pt": intervals, "mixed_lithology_rows": [
                dict(interval, top_y=row["top_y"], bottom_y=row["bottom_y"],
                     top_depth=row["top_depth"], bottom_depth=row["bottom_depth"])
                for interval, row in zip(intervals, rows) if len(interval["components"]) > 1],
            "accessories": accessories, "rock_tile_count": len(tiles),
            "rock_xrefs_in_column": {str(ref): rocks[ref]["label"] for ref in sorted({t["xref"] for t in tiles})},
            "legend_label_aliases": {str(ref): rock["label_aliases"] for ref, rock in rocks.items()
                                     if "label_aliases" in rock},
            "exact_tile_expansions": exact_expansions,
            "pattern_depth_range_ft": [depth_at(fit, nonblank[0]["top_pt"]),
                                       depth_at(fit, nonblank[-1]["bottom_pt"])],
            "truth_depth_range_ft": [rows[0]["top_depth"], rows[-1]["bottom_depth"]],
            "known_limits": (["Unclassified raster overlays are preserved with alpha; their individual symbols "
                              "are not decoded. Rock truth describes the underlying vector tiles."]
                             if any(a["kind"] == "unclassified_overlay" for a in accessories) else []),
            "truth_conventions": {"y": ("absolute fractional pixel boundaries; [top_y, bottom_y); subpixel intervals retained"
                if subpixel else "absolute image rows; [top_y, bottom_y), nearest pixel boundary"),
                "depth": "fit at original point boundaries; no rounding or nominal-depth snapping",
                "blank": "all uncovered width, including before/after logged patterns, is (blank)",
                "dominant_ties": "alphabetical label, then xref; blank participates in width comparison",
                "clipping": "rectangular PDF clips and painting order; coordinate tolerance 0.002 pt"}}
    write_json(out / "meta.json", meta)
    contact_panel(out, rows, bounds, scale, f"{doc.name and Path(doc.name).stem} p{page.number+1}")
    return meta


def extract_pdf(pdf, out, ft_per_px=None, resolution_source=None):
    out = output_path(out)
    if ft_per_px is None:
        ft_per_px, resolution_source = default_resolution()
    if not math.isfinite(ft_per_px) or ft_per_px <= 0:
        raise ValueError("ft_per_px must be finite and positive")
    resolution_source = resolution_source or "explicit --ft-per-px"
    pdf = Path(pdf)
    with pdf.open("rb") as f:
        pdf_hash = hashlib.file_digest(f, "sha256").hexdigest()
    pages = []
    with fitz.open(pdf) as doc:
        prepared, all_entries, inline_sources = [], [], {}
        texts = [page.get_text() for page in doc]
        has_legend = any(re.search(r"ROCK\s+TYPES", text, re.I) for text in texts)
        header_x = defaultdict(list)
        for index, text in enumerate(texts):
            if re.search(r"\bLithology\b", text, re.I):
                page = doc[index]
                words = page_words(page)
                labels = [w for w in words if w[4].lower() == "lithology"
                          and {a[4].lower() for a in words if a[5:7] == w[5:7]}
                          <= {"lithology", "%", "interpreted"}]
                if labels:
                    first = min(labels, key=lambda w: w[0])
                    header_x[(page.rect.width, page.rect.height)].append((first[0]+first[2])/2)
        for page in doc:
            try:
                if not has_legend:
                    raise Skip("no ROCK TYPES legend on this PDF")
                if not re.search(r"LITHOLOGY\s+STRIP\s+LOG|WellSight", texts[page.number], re.I):
                    if not re.search(r"ROCK\s+TYPES", texts[page.number], re.I):
                        # A continuation must actually reuse legend images;
                        # this avoids parsing graphics on unrelated reports.
                        refs = {e["xref"] for e in all_entries if e["kind"] == "rock"}
                        reusable = any(im[0] in refs for im in page.get_images(full=True))
                        inline = inline_sources and len(page.get_image_info()) >= 4
                        if not reusable and not inline:
                            raise Skip("not a strip log or no legend tiles")
                if page.rotation:
                    raise Skip("rotated pages are unsupported")
                images = placed_images(page, inline_sources)
                borders = border_lines(page)
                legend = parse_legend(page, images, borders)
                all_entries.extend(legend["entries"])
                prepared.append((images, borders, legend))
            except Exception as exc:
                prepared.append({"status": "skip", "page": page.number+1,
                                 "reason": f"{type(exc).__name__}: {exc}"})
        inherited_entries = []
        depth_unit = "ft"
        for index, data in enumerate(prepared):
            if re.search(r"Scale[^\n]*Metric|Logged Interval\s*\(m\)", texts[index], re.I):
                depth_unit = "m"
            elif re.search(r"Scale[^\n]*Imperial|Logged Interval\s*\(ft\)", texts[index], re.I):
                depth_unit = "ft"
            page_out = out if len(doc) == 1 else out / f"page_{index+1:03d}"
            if isinstance(data, dict):
                pages.append(data)
                continue
            try:
                if legend_map(data[2]["entries"]):inherited_entries = data[2]["entries"]
                meta = extract_page(doc, doc[index], page_out, *data, inherited_entries or all_entries,
                                    ft_per_px, resolution_source, pdf_hash, inline_sources,
                                    statistics.median(header_x[(doc[index].rect.width, doc[index].rect.height)])
                                    if header_x[(doc[index].rect.width, doc[index].rect.height)] else None,
                                    depth_unit)
                pages.append({"status": "ok", "page": index+1, "out": str(page_out), "meta": meta})
            except Exception as exc:
                pages.append({"status": "skip", "page": index+1,
                              "reason": f"{type(exc).__name__}: {exc}"})
    manifest = {"pdf": str(pdf.resolve()), "pdf_sha256": pdf_hash,
                "pages": [{k: v for k, v in p.items() if k != "meta"} for p in pages]}
    write_json(out / "extraction.json", manifest)
    return pages


def batch(pdf_dir, out, ft_per_px=None, limit=None):
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted((p for p in Path(pdf_dir).iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
                  key=lambda p: p.name.lower())
    if limit is not None:
        pdfs = pdfs[:limit]
    summaries, contacts = [], []
    seen = Counter(p.stem.lower() for p in pdfs)
    for pdf in pdfs:
        name = pdf.stem if seen[pdf.stem.lower()] == 1 else pdf.name.replace(".", "_")
        try:
            pages = extract_pdf(pdf, out / name, ft_per_px)
            good = [p for p in pages if p["status"] == "ok"]
            counts = Counter(p["reason"] for p in pages if p["status"] == "skip")
            reasons = "; ".join(f"{reason} ({count})" for reason, count in list(counts.items())[:3])
            if len(counts) > 3:
                reasons += f"; +{len(counts)-3} other reasons ({sum(list(counts.values())[3:])})"
            rock_types = sorted({label for p in good for label in p["meta"]["rock_xrefs_in_column"].values()})
            summaries.append({"pdf": pdf.name, "status": "ok" if good else "skip", "reason": reasons,
                "pages_ok": len(good), "pages_skipped": len(pages)-len(good), "rock_types": "|".join(rock_types),
                "legend_rock_type_count": len({e["label"] for p in good for e in p["meta"]["legend_entries"]
                                               if e["kind"] == "rock"}),
                "intervals": sum(len(p["meta"]["intervals_pt"]) for p in good),
                "logged_intervals": sum(sum(any(c["xref"] is not None for c in r["components"])
                                            for r in p["meta"]["intervals_pt"]) for p in good),
                "top_depth": min((p["meta"]["truth_depth_range_ft"][0] for p in good), default=""),
                "bottom_depth": max((p["meta"]["truth_depth_range_ft"][1] for p in good), default=""),
                "pattern_top_depth": min((p["meta"]["pattern_depth_range_ft"][0] for p in good), default=""),
                "pattern_bottom_depth": max((p["meta"]["pattern_depth_range_ft"][1] for p in good), default=""),
                "rmse_ft": max((p["meta"]["depth_fit"]["rmse_ft"] for p in good), default="")})
            contacts.extend(Path(p["out"]) / "contact.png" for p in good)
        except Exception as exc:
            summaries.append({"pdf": pdf.name, "status": "skip", "reason": f"{type(exc).__name__}: {exc}"})
        print(f"{pdf.name}: {summaries[-1]['status']} {summaries[-1].get('reason', '')}", flush=True)
    fields = ["pdf", "status", "reason", "pages_ok", "pages_skipped", "rock_types", "legend_rock_type_count",
              "intervals", "logged_intervals",
              "top_depth", "bottom_depth", "pattern_top_depth", "pattern_bottom_depth", "rmse_ft"]
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    # Stack existing contact panels without ever decoding all full log pages.
    dimensions = []
    for path in contacts:
        with Image.open(path) as im:
            ratio = min(1, 1560/im.width)
            dimensions.append((round(im.width*ratio), max(1, round(im.height*ratio))))
    width = max((w for w, h in dimensions), default=640)
    height = sum(h for w, h in dimensions) or 100
    if width*height > 160_000_000:
        raise ValueError("contact sheet exceeds 160 million pixels; use a smaller --limit")
    sheet = Image.new("RGB", (width, height), "#f1f1f1")
    offset = 0
    for path, size in zip(contacts, dimensions):
        with Image.open(path) as im:
            if im.size != size:
                im = im.resize(size, Image.Resampling.LANCZOS)
            sheet.paste(im, (0, offset))
            offset += im.height
    if not contacts:
        ImageDraw.Draw(sheet).text((15, 15), "No extractable vector logs. See summary.csv.", fill="black")
    sheet.save(out / "contact_sheet.png")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ft-per-px", type=float, help="default: Texas family median, otherwise 0.05")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.ft_per_px is not None and (not math.isfinite(args.ft_per_px) or args.ft_per_px <= 0):
        parser.error("--ft-per-px must be finite and positive")
    try:
        batch(args.pdf_dir, args.out, args.ft_per_px, args.limit)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
