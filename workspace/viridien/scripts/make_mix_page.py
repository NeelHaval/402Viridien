"""analysis/{difficulty_by_family,label_shares_dataset}.csv + mix_manifest.json + img/mix_*.png
-> analysis/label_mix_examples.html (standalone, images embedded) and label_mix_examples_preview.html (absolute paths)."""
import os, json, base64, io, html, pandas as pd
from PIL import Image
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
F = pd.read_csv(os.path.join(A, "difficulty_by_family.csv"))
T = pd.read_csv(os.path.join(A, "label_shares_dataset.csv")).head(10)
M = json.load(open(os.path.join(A, "mix_manifest.json"), encoding="utf8"))
NOTE = {
 "d582": ("Hardest family (49). 1-bit scan: no colour at all, so every filled row is hatch-only and colour matching names nothing. "
          "This is log06, the only log with a true answer (right strip): Shale, Siltstone and Sandstone differ only in pattern. "
          "In that strip green = Shale, amber = Siltstone, blue = Sandstone (not the key below). The 'blank' share is the white gaps between hatch lines, not empty rock."),
 "geo": ("Colour works, but two swatches share it: the green column fits both 'Calcareous clay-shale' and 'Clay Shale' (95% ambiguous). "
         "Telling them apart needs the hatch pattern or the legend text."),
 "mc761": ("Mixed legend: some swatches coloured, some hatch-only. About a third is named Sandstone by colour; the hatch-only rows still need pattern matching."),
 "mc765a": ("Colour grid with 23 rock types and many near-identical colours: every coloured swatch clashes with another, so 90% of rows are ambiguous (Clay-S vs Clay-SACS)."),
 "mc765b": ("Same well as the 1-inch log, scanned at 5 inch. In this window every row is ambiguous; over the whole log 92% of rows match no swatch, "
            "which looks like a scan colour shift rather than new rocks."),
 "gulf_ok": ("Known estimator gap: this column is grey-filled with black hatch, but the estimator ignores unsaturated grey, so it reports blank/hatch "
             "where a person would read Clay-shale. Grey swatches are under-counted across the dataset."),
 "gulf_fail": ("Before the 7 Nov fix the parser found no rock swatches in this 3-panel layout (10 of 18 Gulf logs, a third of all column length). "
               "It now reads 15 rock entries. The Sol+Terra labellers call this pink window Halite."),
 "oneoff": ("One-off layout. Two-thirds ambiguous (Claystone vs Marl share a colour); 29% of rows match no swatch at all."),
 "texas": ("Easiest family (20). Distinct colours per rock: 90% of this window is named Shale with high confidence."),
}
ORDER = ["d582", "gulf_fail", "gulf_ok", "geo", "mc765a", "mc765b", "mc761", "oneoff", "texas"]
CMAP = {"(legend parser found no rock swatches)": "#616161", "(blank)": "#bdbdbd", "(ambiguous by colour)": "#f9a825",
        "(colour not in legend)": "#d32f2f", "(hatch only: colour can't tell)": "#7b1fa2"}


def img(key, embed):
    p = os.path.join(A, "img", f"mix_{key}.png")
    if not embed:
        return os.path.abspath(p).replace("\\", "/")
    im = Image.open(p).convert("RGB"); b = io.BytesIO(); im.save(b, "WEBP", quality=82)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


import re, cv2, numpy as np
R = json.load(open(os.path.join(A, "label_mix.json"), encoding="utf8"))
GOLDKEY = "log06"


def family(n):
    if n.startswith("0000000009"): return "D-582 1-bit grid"
    if n.startswith("MC765"): return "MC765 colour grid"
    if "mc761" in n.lower(): return "MC761 mineral/lithology"
    if any(k in n for k in ("Rio_", "Critter", "Callaway")): return "Texas 'Rock Types' box"
    if any(k in n for k in ("bp_gc743", "gb506", "hadrian", "gc743_dc141")): return "Geoservices grid"
    if any(k in n for k in ("Allegheny", "Trail Mtn", "FREEMAN", "Shell_")): return "One-off layout"
    return "Gulf 3-panel"


def short(n):
    m = re.search(r"(log\d\d)", n)
    if n.startswith("0000000009") and m: return "D-582 " + m.group(1)
    n = re.sub(r"^\d{10,}[ _-]*", "", n).replace("_", " ")
    return n[:46]


def lab2hex(l):
    a = np.array([[[l[0] * 255 / 100, l[1] + 128, l[2] + 128]]], np.uint8)
    r, g, b = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def tier(fit):
    """Confidence of a colour-named label in one log, from the median colour distance to its swatch (d1) and the gap to the next swatch (margin)."""
    if fit is None: return "medium"
    if fit["d1"] <= 12 and fit["margin"] >= 15: return "high"
    if fit["d1"] <= 20 and fit["margin"] >= 8: return "medium"
    return "low"


SWJ = json.load(open(os.path.join(A, "swatches.json"), encoding="utf8"))["logs"] if os.path.exists(os.path.join(A, "swatches.json")) else {}
USED = {}


def norm(l): return re.sub(r"[^a-z]", "", l.lower()).replace("sandandsandstone", "sandstone")


def swfile(log, label):
    """Swatch tile file for a label of one log, '' if none."""
    want = norm(label)
    for r in SWJ.get(log, []):
        if norm(r["label"]) == want:
            USED[r["file"]] = r; return r["file"]
    return ""


def swcss(embed):
    out = []
    for f, r in USED.items():
        p = os.path.join(A, "swatches", f)
        src = ("data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()) if embed else os.path.abspath(p).replace("\\", "/")
        out.append(f".p{f[:12]}{{background:url({src}) left center/auto 100% repeat !important;image-rendering:pixelated}}")
    return "\n".join(out)


TIERC = {"certain": "#1b5e20", "high": "#2e7d32", "medium": "#9e9d24", "low": "#e65100", "none": "#757575"}


def legend_rows():
    """Per log: segments of the filled column (label, share of filled, colour, kind, confidence)."""
    out = []
    for r in R:
        if r.get("error"): continue
        sh = {k: v for k, v in r["share"].items() if k != "(blank)"}; tot = sum(sh.values()) or 1
        fit, sw = r.get("fit", {}), r.get("swatch_lab", {})
        segs = []
        for k, v in sorted(sh.items(), key=lambda kv: -kv[1]):
            p = v / tot
            if r["n_rock"] == 0: segs.append(("legend not parsed", p, "#616161", "cat", "none", "")); continue
            if k.endswith("(true answer)"):
                lbl = k.replace(" (true answer)", "")
                segs.append((lbl, p, {"Shale": "#2e7d32", "Siltstone": "#f9a825", "Sandstone": "#1565c0"}.get(lbl, "#888"), "rock", "certain", swfile(r["log"], lbl))); continue
            if k.startswith("(ambiguous) "): segs.append(("either " + k[12:].replace(" | ", " or "), p, "#f9a825", "amb", "low", "")); continue
            if k.startswith("(hatch"): segs.append(("hatch only (pattern unread)", p, "#7b1fa2", "cat", "none", "")); continue
            if k.startswith("(colour"): segs.append(("colour not in legend", p, "#d32f2f", "cat", "none", "")); continue
            segs.append((k, p, lab2hex(sw[k]) if k in sw else "#888", "rock", tier(fit.get(k)), swfile(r["log"], k)))
        # merge parse-failed rows into one segment
        if r["n_rock"] == 0: segs = [("legend not parsed", 1.0, "#616161", "cat", "none", "")]
        out.append(dict(log=r["log"], short=short(r["log"]), fam=family(r["log"]), filled=1 - r["share"].get("(blank)", 0), col_h=r["col_h"], segs=segs))
    return out


def legend_section():
    L = legend_rows(); h = ""
    order = ["D-582 1-bit grid", "Geoservices grid", "MC761 mineral/lithology", "MC765 colour grid", "One-off layout", "Gulf 3-panel", "Texas 'Rock Types' box"]
    for fam in order:
        rows = [x for x in L if x["fam"] == fam]
        if not rows: continue
        h += f"<h3 class=fam>{html.escape(fam)} <span class=mut>({len(rows)} legends)</span></h3>"
        for x in sorted(rows, key=lambda x: x["short"]):
            bar_ = "".join(
                f'<span class="seg {k} {("p" + f[:12]) if f else ""}" style="flex:{p:.4f};background:{c}" title="{html.escape(l)}: {100 * p:.0f}% ({t})"></span>' for l, p, c, k, t, f in x["segs"] if p >= 0.004)
            top = " &middot; ".join(
                f'<b>{html.escape(l)}</b> {100 * p:.0f}% <i class="t t-{t}">{t}</i>' for l, p, c, k, t, f in x["segs"][:4] if p >= 0.03)
            h += (f'<div class=lrow><div class=lname title="{html.escape(x["log"])}">{html.escape(x["short"])}<span class=mut> &middot; {100 * x["filled"]:.0f}% filled</span></div>'
                  f'<div class=lbar>{bar_}</div><div class=ltop>{top}</div></div>')
    return h, L


FAMS = ["D-582 1-bit grid", "Geoservices grid", "MC761 mineral/lithology", "MC765 colour grid", "One-off layout", "Gulf 3-panel", "Texas 'Rock Types' box"]
ROCKS = [("Shale", r"shale"), ("Claystone / clay", r"clay|mudst|mud ?stone"), ("Siltstone", r"silt"), ("Sandstone", r"sand"), ("Limestone", r"lime"),
         ("Dolomite", r"dolo"), ("Marl", r"marl"), ("Halite / salt", r"halite|salt"), ("Anhydrite", r"anhy"), ("Gypsum", r"gyps"), ("Coal / lignite", r"coal|lign"),
         ("Chert", r"chert"), ("Tuff / volcanic", r"tuff|volcan|ash|bentonite"), ("Conglomerate", r"congl|gravel")]


def rock_of(label):
    l = label.lower()
    if re.search(r"silty sand|silty_sand", l): return "Sandstone"
    for name, pat in ROCKS:
        if re.search(pat, l): return name
    return None


def tile(f, title=""):
    return f'<i class="sw p{f[:12]}" title="{html.escape(title)}"></i>' if f else '<i class="sw none"></i>'


def legend_types(L):
    """Group legends whose rock entries are identical, across the whole dataset."""
    byfam = {}
    for x in L:
        ents = SWJ.get(x["log"], [])
        sig = tuple(sorted({norm(e["label"]) for e in ents}))
        byfam.setdefault(x["fam"], {}).setdefault(sig, []).append(x)
    h = ""
    for fam in FAMS:
        groups = sorted(byfam.get(fam, {}).items(), key=lambda kv: -len(kv[1]))
        if not groups: continue
        h += f"<h3 class=fam>{html.escape(fam)} <span class=mut>&middot; {sum(len(g) for _, g in groups)} logs, {len(groups)} distinct legend{'s' if len(groups) > 1 else ''}</span></h3>"
        for sig, xs in groups:
            ents = SWJ.get(xs[0]["log"], [])
            for e in ents: USED[e["file"]] = e
            tiles = "".join(f'<span class=ent>{tile(e["file"], e["label"])}<span>{html.escape(e["label"])}</span></span>' for e in ents) or \
                '<span class=mut>Parser found no rock swatches in these legends.</span>'
            agg = {}
            for x in xs:
                w = x["col_h"] * x["filled"]
                for l, p, c, k, t, f in x["segs"]:
                    a = agg.setdefault(l, [0, t, f, k]); a[0] += w * p
            tot = sum(a[0] for a in agg.values()) or 1
            top = " &middot; ".join(f'{tile(a[2]) if a[3] == "rock" else ""}<b>{html.escape(l)}</b> {100 * a[0] / tot:.0f}% <i class="t t-{a[1]}">{a[1]}</i>'
                                    for l, a in sorted(agg.items(), key=lambda kv: -kv[1][0])[:6] if a[0] / tot >= 0.02)
            names = ", ".join(html.escape(x["short"]) for x in sorted(xs, key=lambda x: x["short"]))
            h += (f'<div class=ltype><div class=mut><b>{len(xs)} log{"s" if len(xs) > 1 else ""}</b>: {names}</div>'
                  f'<div class=ents>{tiles}</div><div class=ltop2>Across these logs: {top}</div></div>')
    return h


def rock_matrix(L):
    """How each legend type draws the same rock, and how many legends carry it."""
    cell = {}
    for x in L:
        for e in SWJ.get(x["log"], []):
            r = rock_of(e["label"])
            if not r: continue
            c = cell.setdefault((r, x["fam"]), {"f": e["file"], "label": e["label"], "logs": set()})
            c["logs"].add(x["log"]); USED.setdefault(c["f"], e if e["file"] == c["f"] else USED.get(c["f"], e))
    head = "".join(f"<th>{html.escape(f.split(' ')[0] if f != 'One-off layout' else 'One-off')}</th>" for f in FAMS)
    rows = ""
    for rock, _ in ROCKS:
        tds = ""
        for fam in FAMS:
            c = cell.get((rock, fam))
            tds += (f'<td class=mx>{tile(c["f"], c["label"])}<div class=mut>{len(c["logs"])} log{"s" if len(c["logs"]) > 1 else ""}</div></td>' if c else "<td class=mx><span class=mut>-</span></td>")
        rows += f"<tr><td><b>{rock}</b></td>{tds}</tr>"
    return f"<div class=scroll><table class=matrix><tr><th>Rock</th>{head}</tr>{rows}</table></div>"


def budget_section():
    q = os.path.join(A, "volume_budget.csv")
    if not os.path.exists(q): return ""
    B = pd.read_csv(q)
    mx = B.filled_rows.max()
    rows = ""
    for r in B.itertuples():
        thin = r.bed_px < r.patch_rows
        rows += (f"<tr><td><b>{html.escape(r.family)}</b><div class=mut>{r.logs} logs &middot; {r.beds_measured} beds measured</div></td>"
                 f"<td class=n>{r.filled_rows / 1e3:,.0f}k<div class=mut>{100 * r.share:.1f}%</div>{bar(r.filled_rows, mx, '#1565c0')}</td>"
                 f"<td class=n>{r.col_w_min}-{r.col_w_max}</td><td class=n>{r.ft_per_px_min:g}-{r.ft_per_px_max:g}</td>"
                 f"<td class=n>{r.period_med:.0f} / {r.patch_rows:.0f}</td><td class=n>{r.bed_px:.0f}{' <i class=\"t t-low\">thin</i>' if thin else ''}</td>"
                 f"<td class=n>{r.stride:.0f}</td><td class=n>{r.intervals:,.0f}</td>"
                 f"<td class=n><b>{r.analyse_pct:.0f}%</b><div class=mut>{r.analyse_rows / 1e3:,.0f}k rows</div></td>"
                 f"<td class=n><b>{r.analyse_pct_cluster:.0f}%</b><div class=mut>{r.vlm_calls_cluster:,} calls</div></td>"
                 f"<td class=n><b>{100 * r.required_acc:.1f}%</b><div class=mut>&le;{r.rows_wrong_allowed / 1e3:,.0f}k rows wrong</div></td>"
                 f"<td class=n>{r.tol_ft_min:g}-{r.tol_ft_max:g} ft</td></tr>")
    tot = B.filled_rows.sum()
    return f"""<h2>How much of each legend type to analyse, and how accurate it must be</h2>
<p class=mut>Measured on all 51 columns. Rows are native image rows. A <b>pattern repeat</b> is the vertical period of the hatch (median across logs); a <b>naming patch</b> must show about two repeats of the largest pattern (90th percentile), or 8 rows for a flat colour.
<b>Bed</b> = median interval height from the Sol+Terra labels (log06's true answer for D-582). <b>Stride</b> = the step that still finds every boundary to within the scorer's 20-row tolerance.</p>
<div class=scroll><table class=budget><tr><th>Legend type</th><th class=n>Filled rows</th><th class=n>Width px</th><th class=n>ft/px</th>
<th class=n>Repeat / patch</th><th class=n>Bed</th><th class=n>Stride</th><th class=n>Intervals</th>
<th class=n>Read, per interval</th><th class=n>Read, per cluster</th><th class=n>Needed accuracy</th><th class=n>Tolerance</th></tr>{rows}</table></div>
<p class=mut>Rows to read = one row signature every stride (finds boundaries) + 2 x 20 full-resolution rows around each boundary + naming patches. Per-interval naming reads one patch per interval and needs one vision call per interval.
Per-cluster naming groups similar intervals and names each group once from 3 patches (12 groups per log), so its vision-call count is fixed by the number of logs. Total filled rows: {tot / 1e6:.2f} million.</p>
<p><b>Reading the table.</b> The Gulf 3-panel type is {100 * B.share.iloc[0]:.0f}% of all filled rows, so it alone sets most of the score: it needs about 90% row accuracy, and half of its column is ambiguous by colour alone.
Texas is {100 * B.share.iloc[1]:.0f}% and easy, so it is held to the strictest target. D-582 and MC761 are tiny (under 2%), so even 82-88% there costs little overall.
Where beds are thinner than a naming patch (marked thin), every row must be read at full resolution and patches overlap neighbouring beds: these are the logs where accuracy will be hardest to buy.</p>
<p class=mut>Accuracy targets split a 10% overall error budget across types in proportion to their difficulty score, weighted by volume. They are a planning tool, not a promise. Bed heights come from only 1-52 labelled beds per type (Geoservices: 1), so treat those columns as rough.</p>"""


def confident_table(L):
    items = []
    for x in L:
        for l, p, c, k, t, f in x["segs"]:
            if k == "rock" and t in ("certain", "high", "medium") and p >= 0.03:
                items.append((["certain", "high", "medium"].index(t), -p, x, l, p, c, t, f))
    items.sort(key=lambda z: (z[0], z[1]))
    why = {"certain": "organisers' true answer", "high": "colour matches its swatch closely and no other swatch is near",
           "medium": "colour fits its swatch, another swatch is fairly close"}
    rows = "".join(
        f'<tr><td><i class="sw {("p" + f[:12]) if f else ""}" style="background:{c}"></i>{html.escape(l)}</td><td>{html.escape(x["short"])}</td><td class=mut>{html.escape(x["fam"])}</td>'
        f'<td class=n>{100 * p:.0f}%</td><td><i class="t t-{t}">{t}</i></td><td class=mut>{why[t]}</td></tr>'
        for _, _, x, l, p, c, t, f in items[:30])
    # dataset-wide, weighted by filled column length, split by confidence
    agg = {}
    for x in L:
        w = x["col_h"] * x["filled"]
        for l, p, c, k, t, f in x["segs"]:
            key = l if k == "rock" else ("ambiguous colour" if k == "amb" else l)
            key = re.sub(r"\s+", " ", key.strip()).title() if k == "rock" else key
            a = agg.setdefault(key, {"certain": 0, "high": 0, "medium": 0, "low": 0, "none": 0, "c": c, "k": k})
            a[t] += w * p
    tot = sum(sum(v[t] for t in TIERC) for v in agg.values())
    ds = sorted(agg.items(), key=lambda kv: -sum(kv[1][t] for t in TIERC))[:16]
    drows = ""
    for name, v in ds:
        s = sum(v[t] for t in TIERC)
        segs = "".join(f'<span style="flex:{v[t] / s:.4f};background:{TIERC[t]}" title="{t}: {100 * v[t] / tot:.1f}%"></span>' for t in TIERC if v[t] > 0)
        conf = 100 * (v["certain"] + v["high"]) / tot
        drows += (f'<tr><td>{html.escape(name)}</td><td class=n>{100 * s / tot:.1f}%</td><td class=n>{conf:.1f}%</td>'
                  f'<td><div class="lbar small" style="width:{min(100, 100 * s / tot / 30 * 100):.0f}%">{segs}</div></td></tr>')
    return rows, drows


def bar(v, mx, colour):
    w = 0 if pd.isna(v) else min(100, 100 * v / mx)
    return f'<div class="bar"><span style="width:{w:.0f}%;background:{colour}"></span></div>'


CSS = """
:root{--fg:#1b1b1b;--mut:#5f6368;--line:#e0e0e0;--bg:#fff;--soft:#f1f2f4}
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg)}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 8px;border-bottom:2px solid var(--fg);padding-bottom:4px} h3{margin:0 0 2px;font-size:17px}
p{margin:6px 0} .mut,.log{color:var(--mut);font-size:13px} .log{overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:14px} td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:12px;text-transform:uppercase;color:var(--mut)} .n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{background:var(--soft);height:12px;min-width:90px} .bar span{display:block;height:100%}
.scroll{overflow-x:auto} .card{border:1px solid var(--line);padding:14px;margin:18px 0} .card img{width:100%;height:auto;display:block;margin-top:10px}
.key{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;margin:8px 0} .key i{display:inline-block;width:12px;height:12px;margin-right:5px;vertical-align:-1px}
.warn{border-left:4px solid #f9a825;background:#fff8e1;padding:10px 12px;margin:14px 0}
.lrow{display:grid;grid-template-columns:230px 1fr;gap:2px 12px;padding:5px 0;border-bottom:1px solid var(--line);align-items:center}
.lname{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap} .ltop{grid-column:2;font-size:12.5px;color:var(--mut)} .ltop b{color:var(--fg);font-weight:600}
.lbar{display:flex;height:18px;border:1px solid var(--line)} .lbar.small{height:12px} .lbar span{display:block;height:100%}
.seg.amb{background-image:repeating-linear-gradient(45deg,rgba(0,0,0,.18) 0 3px,transparent 3px 7px)!important}
.t{font-style:normal;font-size:11px;padding:0 5px;border-radius:3px;color:#fff;white-space:nowrap} .t-certain{background:#1b5e20}.t-high{background:#2e7d32}.t-medium{background:#9e9d24}.t-low{background:#e65100}.t-none{background:#757575}
.sw{display:inline-block;width:34px;height:16px;margin-right:6px;vertical-align:-1px;border:1px solid rgba(0,0,0,.2)}
h3.fam{margin:22px 0 4px}
.ltype{border:1px solid var(--line);padding:10px 12px;margin:8px 0} .ents{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0}
.ent{display:inline-flex;align-items:center;gap:6px;font-size:12.5px} .ent .sw{width:44px;height:22px;margin:0}
.ltop2{font-size:12.5px;color:var(--mut)} .ltop2 b{color:var(--fg)} .ltop2 .sw{width:26px;height:13px}
.sw.none{background:repeating-linear-gradient(45deg,#ddd 0 2px,#fff 2px 5px)}
table.budget{font-size:12.5px} table.budget td,table.budget th{padding:5px 5px} table.budget th{white-space:normal}
table.matrix td.mx{text-align:center} table.matrix .sw{width:48px;height:24px;margin:0}
@media(max-width:700px){.lrow{grid-template-columns:1fr}.ltop{grid-column:1}}
.two{display:grid;grid-template-columns:1fr 1fr;gap:24px} @media(max-width:800px){.two{grid-template-columns:1fr}}
"""


def page(embed):
    fam = "".join(
        f"<tr><td>{html.escape(r.family)}</td><td class=n>{r.logs}</td><td class=n>{int(r.parse_failed)}</td>"
        f"<td class=n><b>{r.difficulty:.0f}</b></td><td>{bar(r.difficulty, 60, '#c62828')}</td>"
        f"<td class=n>{r.named_pct:.0f}%</td><td class=n>{r.ambiguous_pct:.0f}%</td><td class=n>{r.hatch_only_pct:.0f}%</td>"
        f"<td class=n>{r.colour_clash_pct:.0f}%</td></tr>" for r in F.itertuples())
    ds = "".join(f"<tr><td>{html.escape(r.bucket)}</td><td class=n>{r.col_rows:.1f}%</td><td>{bar(r.col_rows, 35, CMAP.get(r.bucket, '#2e7d32'))}</td></tr>"
                 for r in T.itertuples())
    leg, L = legend_section(); conf, ds2 = confident_table(L); types = legend_types(L); matrix = rock_matrix(L); budget = budget_section()
    cards = ""
    for k in ORDER:
        m = M[k]
        cards += (f'<section class=card><h3>{html.escape(m["family"])}</h3>'
                  f'<p class=log>{html.escape(m["log"])} &middot; rows {m["rows"][0]:,}-{m["rows"][1]:,} &middot; {m["n_rock"]} rock swatches parsed</p>'
                  f'<p>{html.escape(NOTE[k])}</p><img loading=lazy src="{img(k, embed)}" alt="{html.escape(m["family"])} example"></section>')
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Label mix and difficulty per legend</title><style>{CSS}
{swcss(embed)}</style></head><body><main>
<h1>What the 51 lithology columns are made of, and how hard each legend is</h1>
<p class=mut>Each row of each column has its fill colour matched to that log's own legend swatches. These are estimates only: the one true answer is log06 (the organisers' example CSV).</p>
<div class=warn><b>Read with care.</b> The shares come from a colour estimator, not from labels. It cannot read hatch patterns, it ignores grey fills (see the parsed Gulf example),
and its legend reading is automatic (the parser now reads all 51 legends; 11 were fixed on 7 Nov). Use it to rank difficulty, not as accuracy.</div>
<h2>Difficulty per legend family</h2>
<p class=mut>Score 0-100 = 40% rows only a pattern can name (all of them on 1-bit scans) + 20% ambiguous colour + 10% colour not in legend + 10% many rock types + 8% busy column + 6% tiny swatches + 6% narrow column. Higher is harder.</p>
<div class=scroll><table><tr><th>Family</th><th class=n>Logs</th><th class=n>Parse failed</th><th class=n>Difficulty</th><th></th><th class=n>Named</th><th class=n>Ambiguous</th><th class=n>Hatch only</th><th class=n>Swatch colour clash</th></tr>{fam}</table></div>
<p class=mut>Named, ambiguous and hatch-only are % of filled rows, averaged over logs. All 18 Gulf legends are now parsed and included.</p>
<h2>Dataset-wide: share of all column length</h2>
<div class=two><div class=scroll><table><tr><th>Label</th><th class=n>Share</th><th></th></tr>{ds}</table></div>
<div><p>With every legend parsed, the biggest bucket is colour that fits two or more swatches: a colour-only reader cannot name over a third of the column, and a fifth matches no swatch. Pattern reading (or a vision model) is needed for both.</p>
<p>Shale is the largest named rock at about 10%, and every other named rock is under 3%. So a strategy that only gets Shale right can still look decent on row accuracy.</p></div></div>
{budget}
<h2>Legend types across the dataset</h2>
<p class=mut>Logs whose legends list exactly the same rock entries are grouped, so each distinct legend appears once with its real swatch patterns (as drawn in the legend, not stretched).
The line under each legend gives the estimated label shares pooled over its logs.</p>
{types}
<h2>The same rock, drawn by each legend type</h2>
<p class=mut>One example swatch per legend family, and how many logs' legends contain that rock. This is why one colour or pattern matcher cannot serve every log: each family draws the same rock differently.</p>
{matrix}
<h2>Each legend: what its column is made of</h2>
<p class=mut>One bar per log (= per legend), covering the filled part of its lithology column. Rock segments use that legend's own swatch colour; striped amber = colour fits two rocks; purple = hatch-only; red = colour not in the legend; grey = legend not parsed.
Hover a segment for its share and confidence. Below each bar: the biggest labels with my confidence in them.</p>
<div class=key><i class="t t-certain">certain</i> organisers' answer <i class="t t-high">high</i> close colour match, no rival swatch <i class="t t-medium">medium</i> fits, a rival swatch is fairly close <i class="t t-low">low</i> two swatches fit equally <i class="t t-none">none</i> can't be named by colour</div>
{leg}
<h2>The labels I'm most confident in</h2>
<p class=mut>Named rock labels covering at least 3% of a log's filled column, ranked by confidence, then share. Confidence is per label per log (median over its rows), not per row. It measures colour fit only: a close match can still be the wrong rock if two rocks share a colour that the legend separates by pattern. Halite at 23-50% on mc822 and mc776 fits its swatch well and is plausible for Gulf of Mexico salt wells, but nobody has checked it.</p>
<div class=scroll><table><tr><th>Label</th><th>Log</th><th>Family</th><th class=n>Share</th><th>Confidence</th><th>Why</th></tr>{conf}</table></div>
<h2>Dataset-wide, by label and confidence</h2>
<p class=mut>% of all filled column length (blank rows removed), weighted by column length. Bar colour shows how much of each label is certain/high (dark green) versus medium, low or unnamed.</p>
<div class=scroll><table><tr><th>Label</th><th class=n>Share</th><th class=n>Certain or high</th><th></th></tr>{ds2}</table></div>
<h2>One example per family</h2>
<p class=mut>Each figure shows the parsed legend, a 1,600-row window of the lithology column at its true shape, a same-scale zoom (red box), the estimator's reading of each row, and that window's shares.</p>
<div class=key><span><i style="background:#2e7d32"></i>named (drawn in the swatch's own colour)</span><span><i style="background:#f9a825"></i>ambiguous</span><span><i style="background:#7b1fa2"></i>hatch only</span><span><i style="background:#d32f2f"></i>colour not in legend</span><span><i style="background:#616161"></i>legend not parsed</span><span><i style="background:#eee;border:1px solid #ccc"></i>blank</span></div>
{cards}
<h2>What this means for the pipeline</h2>
<p>1. Done: the Gulf 3-panel legend parser is fixed (harness/legend_v2.py). 2. Add pattern (hatch) matching for 1-bit and mixed legends. 3. Break colour ties with the hatch or the legend text, or with one JEV call per cluster. 4. Treat grey as a real fill colour.</p>
</main></body></html>"""


for fn, emb in (("label_mix_examples.html", True), ("label_mix_examples_preview.html", False)):
    open(os.path.join(A, fn), "w", encoding="utf8").write(page(emb))
    print(fn, os.path.getsize(os.path.join(A, fn)))
