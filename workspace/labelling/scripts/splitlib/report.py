"""Markdown report + PIL contact sheet for a chosen train/test split."""

from __future__ import annotations

import io
import json
import math
import os
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .search import (
    NUMERIC_FEATURES,
    TERM_NAMES,
    WEIGHTS,
    group_strata,
    score_split,
    stratum_label,
    to_bool,
    to_float,
)

REPORT_FEATURES = tuple(NUMERIC_FEATURES) + tuple(
    f for f in ("mean_sat", "px_per_ft", "legend_area_frac", "depth_range_ft") if f not in NUMERIC_FEATURES
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "NaN"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.{nd}f}"


def _md_cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _vals(rows_by_stem, stems, feat) -> np.ndarray:
    v = np.array([to_float(rows_by_stem[s].get(feat)) for s in stems], dtype=float)
    return v[~np.isnan(v)]


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    dof = len(a) + len(b) - 2
    ss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
    sd = math.sqrt(ss / dof) if dof > 0 else 0.0
    if sd <= 1e-12:
        allv = np.concatenate([a, b])
        sd = float(allv.std(ddof=1)) if len(allv) > 1 else 0.0
    if sd <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / sd)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_report(path, rows: Sequence[Mapping], groups: Mapping[str, Sequence[str]], split: dict, meta: dict) -> None:
    rows_by_stem = {str(r["stem"]): r for r in rows}
    test_set = set(split["test"])
    test_gids = set(split.get("test_groups") or {g for g, st in groups.items() if set(st) & test_set})
    strata = group_strata(rows, groups)
    test_frac = float(split.get("test_frac", meta.get("test_frac", 0.2)))
    score = split.get("score") or score_split(test_gids, rows, groups, test_frac, strata)

    gids = sorted(groups)
    train_stems = sorted(s for g in gids if g not in test_gids for s in groups[g])
    test_stems = sorted(s for g in gids if g in test_gids for s in groups[g])
    N = len(train_stems) + len(test_stems)

    L: list[str] = []
    L.append("# Train / test split report\n")
    L.append(f"- pages: **{N}** total — train **{len(train_stems)}**, test **{len(test_stems)}** "
             f"({len(test_stems) / max(N, 1):.1%})")
    L.append(f"- groups: **{len(gids)}** total — train **{len(gids) - len(test_gids)}**, test **{len(test_gids)}**")
    L.append(f"- score total: **{_fmt(score['total'])}** (lower is better)\n")

    L.append("## Run metadata\n")
    L.append("| key | value |")
    L.append("|---|---|")
    meta_all = dict(meta)
    for k in ("seed", "n_iter", "test_frac", "n_unique_candidates", "force_test", "exclude_test"):
        if k in split and k not in meta_all:
            meta_all[k] = split[k]
    for k, v in meta_all.items():
        if not isinstance(v, str):
            v = json.dumps(v, default=_json_default)
        L.append(f"| {_md_cell(k)} | `{_md_cell(v)}` |")
    L.append("")

    # ---- groups table
    L.append("## Groups\n")
    L.append("| group_id | n_pages | split | strata | stems |")
    L.append("|---|---:|---|---|---|")
    for g in sorted(gids, key=lambda g: (g not in test_gids, g)):
        st = sorted(groups[g])
        page_strata = sorted({stratum_label((to_bool(rows_by_stem[s].get("is_colour")),
                                             str(rows_by_stem[s].get("log_type") or "other"))) for s in st})
        strat = stratum_label(strata[g])
        if len(page_strata) > 1:
            strat += f" (pages: {', '.join(page_strata)})"
        L.append(f"| {_md_cell(g)} | {len(st)} | {'**test**' if g in test_gids else 'train'} | "
                 f"{_md_cell(strat)} | {_md_cell('<br>'.join(st))} |")
    L.append("")

    # ---- per-stratum counts
    L.append("## Strata (group-majority `(is_colour, log_type)`)\n")
    agg = defaultdict(lambda: {"train_p": 0, "test_p": 0, "train_g": 0, "test_g": 0})
    for g in gids:
        a = agg[strata[g]]
        side = "test" if g in test_gids else "train"
        a[f"{side}_p"] += len(groups[g])
        a[f"{side}_g"] += 1
    L.append("| stratum | train pages | test pages | test page % | train groups | test groups |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k in sorted(agg):
        a = agg[k]
        tot = a["train_p"] + a["test_p"]
        L.append(f"| {stratum_label(k)} | {a['train_p']} | {a['test_p']} | {a['test_p'] / tot:.0%} | "
                 f"{a['train_g']} | {a['test_g']} |")
    L.append("")
    # Per-page (raw) strata too, since group majority can hide mixed wells.
    raw = defaultdict(lambda: [0, 0])
    for s in train_stems + test_stems:
        r = rows_by_stem[s]
        raw[(to_bool(r.get("is_colour")), str(r.get("log_type") or "other"))][int(s in test_set)] += 1
    if any(raw[k] != [agg[k]["train_p"], agg[k]["test_p"]] for k in set(raw) | set(agg)):
        L.append("Per-page (own label, not group majority):\n")
        L.append("| stratum | train pages | test pages |")
        L.append("|---|---:|---:|")
        for k in sorted(raw):
            L.append(f"| {stratum_label(k)} | {raw[k][0]} | {raw[k][1]} |")
        L.append("")

    # ---- feature means
    L.append("## Feature means\n")
    L.append("SMD = (test − train) / pooled SD, NaNs ignored. Features marked * are used in the score.\n")
    L.append("| feature | train mean | test mean | SMD | train n | test n |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for f in REPORT_FEATURES:
        a = _vals(rows_by_stem, train_stems, f)
        b = _vals(rows_by_stem, test_stems, f)
        L.append(f"| {f}{'*' if f in NUMERIC_FEATURES else ''} | {_fmt(a.mean() if len(a) else math.nan)} | "
                 f"{_fmt(b.mean() if len(b) else math.nan)} | {_fmt(_smd(b, a), 2)} | {len(a)} | {len(b)} |")
    a = np.array([to_bool(rows_by_stem[s].get("is_colour")) for s in train_stems], float)
    b = np.array([to_bool(rows_by_stem[s].get("is_colour")) for s in test_stems], float)
    L.append(f"| is_colour (fraction) | {_fmt(a.mean() if len(a) else math.nan)} | "
             f"{_fmt(b.mean() if len(b) else math.nan)} | — | {len(a)} | {len(b)} |")
    L.append("")

    # ---- score breakdown
    L.append("## Score breakdown\n")
    smd_pf = score.get("smd_per_feature", {})
    raw_desc = {
        "size": f"|{score.get('n_test_pages')} − {score.get('target')}| = {_fmt(score.get('size_dev'), 0)} pages",
        "strata_tv": f"TV = {_fmt(score.get('tv'))}",
        "smd": f"mean |SMD| = {_fmt(score.get('smd_mean'))} ("
               + ", ".join(f"{k} {_fmt(v, 2)}" for k, v in smd_pf.items()) + ")",
        "empty_stratum": f"{score.get('n_empty_strata')} empty multi-group strata"
                         + (f": {', '.join(score.get('empty_strata', []))}" if score.get("empty_strata") else ""),
        "dominant_group": f"max group share = {_fmt(score.get('max_group_share'))} "
                          f"({score.get('dominant_gid')}; limit {score.get('max_group_share_limit', '')})",
    }
    weights = score.get("weights", WEIGHTS)
    L.append("| term | raw value | weight | weighted |")
    L.append("|---|---|---:|---:|")
    for t in TERM_NAMES:
        L.append(f"| {t} | {_md_cell(raw_desc[t])} | {_fmt(weights.get(t), 2)} | {_fmt(score.get(t))} |")
    L.append(f"| **total** | | | **{_fmt(score['total'])}** |")
    L.append("")

    # ---- labelling load
    L.append("## Estimated labelling load (test set)\n")
    known_ppf = _vals(rows_by_stem, list(rows_by_stem), "px_per_ft")
    known_ppf = known_ppf[known_ppf > 0]
    med_ppf = float(np.median(known_ppf)) if len(known_ppf) else math.nan
    px_total = 0.0
    px_unknown = 0
    ft_known = 0.0
    ft_est = 0.0
    n_unknown_depth = 0
    n_unestimable = 0
    for s in test_stems:
        r = rows_by_stem[s]
        h = to_float(r.get("column_height_px"))
        if math.isnan(h):
            px_unknown += 1
        else:
            px_total += h
        d = to_float(r.get("depth_range_ft"))
        if not math.isnan(d):
            ft_known += d
        else:
            n_unknown_depth += 1
            if not math.isnan(h) and not math.isnan(med_ppf):
                ft_est += h / med_ppf
            else:
                n_unestimable += 1
    L.append(f"- test lithology column length: **{px_total:,.0f} px** across {len(test_stems)} pages"
             + (f" ({px_unknown} page(s) with unknown column height excluded)" if px_unknown else ""))
    L.append(f"- depth covered: **{ft_known + ft_est:,.0f} ft** "
             f"= {ft_known:,.0f} ft known + {ft_est:,.0f} ft estimated")
    L.append(f"- pages with unknown depth_range_ft: **{n_unknown_depth}**"
             + (f" — estimated as column_height_px / median px_per_ft ({_fmt(med_ppf, 2)} px/ft, "
                f"median over {len(known_ppf)} pages with known px_per_ft)" if n_unknown_depth else ""))
    if n_unestimable:
        L.append(f"- {n_unestimable} page(s) could not be estimated (no column height or no known px_per_ft)")
    L.append("")

    # ---- warnings
    L.append("## Feature warnings\n")
    warn = [(s, str(rows_by_stem[s].get("feature_warnings") or "").strip()) for s in sorted(rows_by_stem)]
    warn = [(s, w) for s, w in warn if w and w.lower() != "nan"]
    if warn:
        L.append("| stem | split | warnings |")
        L.append("|---|---|---|")
        for s, w in warn:
            L.append(f"| {_md_cell(s)} | {'test' if s in test_set else 'train'} | {_md_cell(w)} |")
    else:
        L.append("None.")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------
TILE_COL_H = 900
COL_W_MIN, COL_W_MAX = 24, 220
LEGEND_BOX = 350
TILE_GAP = 12
CAPTION_FONT = 20
TITLE_FONT = 40
TILES_PER_ROW = 6
CAPTION_LINES = 5
MARGIN = 30
MAX_PNG_BYTES = 20 * 1024 * 1024


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


_GLYPH_FALLBACKS = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}


def _safe_text(text: str) -> str:
    """Pillow's bundled default font lacks some punctuation (e.g. em dash)."""
    return "".join(_GLYPH_FALLBACKS.get(ch, ch) for ch in str(text))


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    text = _safe_text(text)
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else f"{cur} {word}"
            if draw.textlength(cand, font=font) <= width:
                cur = cand
                continue
            if cur:
                lines.append(cur)
            # hard-break words longer than the line
            while draw.textlength(word, font=font) > width and len(word) > 1:
                k = len(word)
                while k > 1 and draw.textlength(word[:k], font=font) > width:
                    k -= 1
                lines.append(word[:k])
                word = word[k:]
            cur = word
        lines.append(cur)
    return lines


def _placeholder(w: int, h: int, text: str = "missing") -> Image.Image:
    im = Image.new("RGB", (w, h), (190, 190, 190))
    d = ImageDraw.Draw(im)
    f = _font(18)
    tw = d.textlength(text, font=f)
    d.text(((w - tw) / 2, h / 2 - 10), text, fill=(90, 90, 90), font=f)
    return im


def _load(path) -> Image.Image | None:
    if not path:
        return None
    p = str(path)
    if p.lower() == "nan" or not os.path.isfile(p):
        return None
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except Exception:
        return None


def _column_tile(path) -> Image.Image:
    im = _load(path)
    if im is None:
        return _placeholder(100, TILE_COL_H)
    w = round(im.width * TILE_COL_H / max(im.height, 1))
    w = max(COL_W_MIN, min(COL_W_MAX, w))
    return im.resize((w, TILE_COL_H), Image.LANCZOS)


def _legend_tile(path) -> Image.Image:
    im = _load(path)
    if im is None:
        return _placeholder(LEGEND_BOX, LEGEND_BOX)
    s = min(LEGEND_BOX / im.width, LEGEND_BOX / im.height)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)


def contact_sheet(rows: Sequence[Mapping], stems: Sequence[str], out_path, title: str,
                  groups: Mapping[str, Sequence[str]] | None = None) -> None:
    """Render a PNG grid of (column strip, legend) tiles for ``stems``.

    The caption's group id comes from ``groups`` if given, else from a
    ``group_id``/``group`` key on the row, else "?".
    """
    rows_by_stem = {str(r["stem"]): r for r in rows}
    stem_gid = {s: g for g, st in (groups or {}).items() for s in st}
    stems = list(stems)
    cap_font = _font(CAPTION_FONT)
    title_font = _font(TITLE_FONT)
    line_h = CAPTION_FONT + 6
    caption_h = CAPTION_LINES * line_h + 6
    tile_w = COL_W_MAX + TILE_GAP + LEGEND_BOX
    tile_h = caption_h + TILE_COL_H
    n_cols = max(1, min(TILES_PER_ROW, len(stems)))
    n_rows = max(1, math.ceil(len(stems) / n_cols))
    title_h = TITLE_FONT + 30
    W = MARGIN * 2 + n_cols * tile_w + (n_cols - 1) * TILE_GAP * 3
    H = MARGIN * 2 + title_h + n_rows * tile_h + (n_rows - 1) * TILE_GAP * 3

    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for i, line in enumerate(_wrap(d, title, title_font, W - 2 * MARGIN)[:1]):
        d.text((MARGIN, MARGIN), line, fill="black", font=title_font)
    d.text((W - MARGIN - d.textlength(f"{len(stems)} pages", font=cap_font), MARGIN + 12),
           f"{len(stems)} pages", fill=(80, 80, 80), font=cap_font)

    for k, s in enumerate(stems):
        r = rows_by_stem.get(s, {})
        x0 = MARGIN + (k % n_cols) * (tile_w + TILE_GAP * 3)
        y0 = MARGIN + title_h + (k // n_cols) * (tile_h + TILE_GAP * 3)
        colour = "colour" if to_bool(r.get("is_colour")) else "BW"
        info = f"{colour} · {r.get('log_type', '?')} · {stem_gid.get(s) or r.get('group_id') or r.get('group') or '?'}"
        stem_lines = _wrap(d, s, cap_font, tile_w)
        if len(stem_lines) > CAPTION_LINES - 1:
            stem_lines = stem_lines[: CAPTION_LINES - 1]
            stem_lines[-1] = stem_lines[-1][:-1] + "…"
        y = y0
        for line in stem_lines:
            d.text((x0, y), line, fill="black", font=cap_font)
            y += line_h
        d.text((x0, y), info, fill=(0, 70, 160), font=cap_font)

        col = _column_tile(r.get("col_thumb"))
        leg = _legend_tile(r.get("legend_thumb"))
        yi = y0 + caption_h
        sheet.paste(col, (x0, yi))
        d.rectangle([x0 - 1, yi - 1, x0 + col.width, yi + col.height], outline=(160, 160, 160))
        lx = x0 + COL_W_MAX + TILE_GAP
        sheet.paste(leg, (lx, yi))
        d.rectangle([lx - 1, yi - 1, lx + leg.width, yi + leg.height], outline=(160, 160, 160))

    _save_capped(sheet, out_path)


def _save_capped(im: Image.Image, out_path) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    scale = 1.0
    for _ in range(8):
        cur = im if scale == 1.0 else im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                                                Image.LANCZOS)
        buf = io.BytesIO()
        cur.save(buf, format="PNG", optimize=True)
        if buf.tell() <= MAX_PNG_BYTES or scale < 0.2:
            with open(out_path, "wb") as fh:
                fh.write(buf.getvalue())
            return
        scale *= 0.75
