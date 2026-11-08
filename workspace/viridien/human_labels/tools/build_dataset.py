"""Build viridien/human_labels/ (PR-49 layout) from labelling-tool JSON files.

python build_dataset.py LABELS_DIR CHALLENGE_DATASET OUT_DIR [--fixes fixes.json] [--anchors stem=anchors.json ...]
"""
import argparse, csv, hashlib, json, sys, datetime
from pathlib import Path
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
LITHO_DIR = None

ap = argparse.ArgumentParser()
ap.add_argument("labels"); ap.add_argument("dataset"); ap.add_argument("out")
ap.add_argument("--labelling", required=True, help="repo labelling/ dir (for labelslib)")
ap.add_argument("--fixes"); ap.add_argument("--anchors", nargs="*", default=[])
ap.add_argument("--stems", nargs="+", required=True)
ap.add_argument("--snapshot", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.labelling) / "scripts"))
import labelslib as L  # noqa

out = Path(a.out); [ (out / d).mkdir(parents=True, exist_ok=True) for d in ("truth", "bbox_annotation", "labels", "preview") ]
fixes = json.load(open(a.fixes)) if a.fixes else {}
anchor_override = {k: json.load(open(v)) for k, v in (x.split("=", 1) for x in a.anchors)}
manifest, evalset = [], []
for stem in a.stems:
    lab = json.loads((Path(a.labels) / f"{stem}.json").read_text())
    notes = []
    if stem in anchor_override:
        lab["depth_calibration"]["anchors"] = anchor_override[stem]; notes.append(f"{len(anchor_override[stem])} depth anchors added from detected 100 ft ticks")
    names = {e["id"]: e.get("label_printed") for e in lab["legend"]}
    byname = {v: k for k, v in names.items()}
    beds = sorted(lab["layers"], key=lambda b: b["top_px"])
    f = fixes.get(stem, {})
    for e in f.get("legend_rename", []):
        for le in lab["legend"]:
            if le["id"] == e["id"]: le["label_printed"] = e["name"]
        notes.append(f"legend {e['id']} renamed to {e['name']}")
    names = {e["id"]: e.get("label_printed") for e in lab["legend"]}; byname = {v: k for k, v in names.items()}
    import copy
    nrel = 0
    for e in f.get("relabel", []):
        b = beds[e["bed"]]; assert b["top_px"] == e["top_px"], (stem, e)
        c = b["composition"][0]; c["legend_id"] = byname[e["to"]]; c["type"] = e["to"]; c["proportion"] = 1.0; nrel += 1
    drop = set()
    for e in f.get("delete", []):
        i = e["bed"]; assert beds[i]["top_px"] == e["top_px"], (stem, e); drop.add(i)
        if e["fill"] == "above" and i > 0: beds[i - 1]["bottom_px"] = max(beds[i - 1]["bottom_px"], beds[i]["bottom_px"])
        if e["fill"] == "below" and i + 1 < len(beds): beds[i + 1]["top_px"] = min(beds[i + 1]["top_px"], beds[i]["top_px"])
    beds = [b for i, b in enumerate(beds) if i not in drop]
    for e in f.get("insert", []):
        A, B = e["top_y"], e["bottom_y"]; nb = []
        for b in beds:
            if b["bottom_px"] <= A or b["top_px"] >= B: nb.append(b); continue
            if b["top_px"] < A: up = copy.deepcopy(b); up["bottom_px"] = A; nb.append(up)
            if b["bottom_px"] > B: dn = copy.deepcopy(b); dn["top_px"] = B; nb.append(dn)
        tmpl = copy.deepcopy(next(b for b in beds if b["top_px"] < B and b["bottom_px"] > A))
        tmpl.update(top_px=A, bottom_px=B); tmpl["composition"] = [dict(tmpl["composition"][0], legend_id=byname[e["rock"]], type=e["rock"], proportion=1.0)]
        nb.append(tmpl); beds = sorted(nb, key=lambda b: b["top_px"])
    if drop: notes.append(f"{len(drop)} non-rock or overlapping beds removed")
    if nrel: notes.append(f"{nrel} beds relabelled")
    if f.get("insert"): notes.append(f"{len(f['insert'])} missing thin beds inserted")
    lab["layers"] = beds
    lab = L.recompute_depths(L.sync_legend_labels(lab))
    v = L.validate(lab, stem)
    if v["errors"]: raise SystemExit(f"{stem}: {v['errors'][:3]}")
    (out / "labels" / f"{stem}.json").write_text(L.dumps(lab))
    cls, cnames = L.class_of(lab["legend"]), L.class_names(lab["legend"])
    rows = []
    with open(out / "truth" / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["top_y", "bottom_y", "lithology", "cluster_id", "top_depth", "bottom_depth"])
        for b in sorted(lab["layers"], key=lambda b: b["top_px"]):
            lid = b["composition"][0]["legend_id"]
            w.writerow([b["top_px"], b["bottom_px"], cnames[lid], cls[lid], b["top_ft"], b["bottom_ft"]]); rows.append(b)
    bb = json.loads((Path(a.dataset) / "bbox_annotation" / f"{stem}.json").read_text())
    (out / "bbox_annotation" / f"{stem}.json").write_text(json.dumps(bb, indent=2))
    top, bot = rows[0]["top_px"], rows[-1]["bottom_px"]
    y = top
    while y < bot:
        y1 = min(y + 6000, bot)
        if y1 - y >= 1500: evalset.append(dict(log=stem, y0=y, y1=y1, tier="human"))
        y = y1
    fit = v.get("fit") or {}
    # preview: page thumbnail + labelled column strip
    im = Image.open(Path(a.dataset) / "images" / f"{stem}.tif").convert("RGB")
    x0, _, x1, _ = bb["bounding_boxes"]["interpreted_lithology_column"]
    H = 1600; s = H / im.height
    thumb = im.resize((max(1, int(im.width * s)), H))
    strip = Image.new("RGB", (60, H), "white"); d = ImageDraw.Draw(strip)
    pal = {}
    import colorsys
    for b in rows:
        lid = b["composition"][0]["legend_id"]
        if lid not in pal:
            h = (len(pal) * 0.618) % 1; r, g, bl = colorsys.hsv_to_rgb(h, .65, .9); pal[lid] = (int(r * 255), int(g * 255), int(bl * 255))
        d.rectangle([0, b["top_px"] * s, 59, max(b["top_px"] * s + 1, b["bottom_px"] * s)], fill=pal[lid])
    pv = Image.new("RGB", (thumb.width + 70, H), "white"); pv.paste(thumb, (0, 0)); pv.paste(strip, (thumb.width + 10, 0))
    pv.save(out / "preview" / f"{stem}.png")
    counts = {}
    for b in rows: counts[cnames[b["composition"][0]["legend_id"]]] = counts.get(cnames[b["composition"][0]["legend_id"]], 0) + 1
    manifest.append(dict(id=stem, labeller="Neel Haval (NeelHaval)", snapshot=a.snapshot, beds=len(rows), labelled_y=f"{top}-{bot}",
        labelled_depth_ft=f"{rows[0]['top_ft']}-{rows[-1]['bottom_ft']}", px_per_ft=round(1 / fit["slope"], 4) if fit.get("slope") else "",
        depth_anchors=len(lab["depth_calibration"]["anchors"]), max_anchor_residual_ft=fit.get("max_residual_ft", ""),
        rock_counts="; ".join(f"{k} {n}" for k, n in sorted(counts.items(), key=lambda x: -x[1])),
        validation=f"0 errors, {len(v['warnings'])} warnings", notes="; ".join(notes)))
with open(out / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
(out / "eval_set.json").write_text(json.dumps(evalset, indent=1))
files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "HUMAN_TAG.json")
tag = dict(tag="HUMAN-LABELS-NEEL", built_on=datetime.date.today().isoformat(), labeller="NeelHaval", snapshot=a.snapshot,
    pages=len(manifest), beds=sum(m["beds"] for m in manifest), windows=len(evalset),
    sha256={str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(out / "HUMAN_TAG.json").write_text(json.dumps(tag, indent=1))
print(json.dumps({k: v for k, v in tag.items() if k != "sha256"})); [print(m) for m in manifest]
