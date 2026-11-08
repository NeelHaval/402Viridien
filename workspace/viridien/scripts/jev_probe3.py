import json, re, sys, time, base64, io, glob, urllib.request
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
key = next(m.group(1).strip().strip('"\'') for l in open(sys.argv[1], encoding="utf8") if (m := re.match(r"\s*JEV_API_KEY\s*=\s*(.*)", l)))
DS = sys.argv[2]
def b64(im, fmt="PNG"):
    b = io.BytesIO(); im.convert("RGB").save(b, fmt); return base64.b64encode(b.getvalue()).decode()
def call(body, model="jev-latest"):
    body["model"] = model
    req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", json.dumps(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r: out = json.loads(r.read())
    except urllib.error.HTTPError as e: out = {"error": e.code, "body": e.read().decode()[:400]}
    return out, time.time() - t
# 1) swatch classification, three swatches
leg = Image.open("peek/col_legend.png")
Q = {"lith": {"type": "choice", "instructions": "Which legend entry does the swatch image show?",
     "criteria": {"Coarse sandstone": "yellow with large black dots", "Bituminous limestone": "cyan with brick lines and black blobs", "Clay Shale": "green with black dashes"}}}
for name, box in [("Clay Shale", (10, 40, 75, 95)), ("Bituminous limestone", (10, 165, 75, 215)), ("Coarse sandstone", (510, 165, 570, 215))]:
    out, dt = call({"state": {"swatch": "img1"}, "questions": Q, "assets": {"img1": {"type": "image", "format": "png", "data_base64": b64(leg.crop(box).resize((512, 512), Image.NEAREST))}}})
    a = out.get("answers", {}).get("lith") or out; print(str(out)[:300])
    print(f"choice truth={name:22s} got={a.get('choice')} conf={a.get('confidence')} {dt:.1f}s")
# 2) boxes on page thumbnails (bonus task): compare with supplied boxes
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy; return ix*iy/u if u else 0
BQ = {"depth_column": "the narrow vertical column of depth numbers (e.g. 25400, 25420) running down the log",
      "lithology_column": "the narrow vertical interpreted lithology column: a strip filled with rock colours or hatch patterns running down the log",
      "legend": "the lithology legend: the panel of small rock-type swatches with their names"}
for n in ["0000000009_d-582_0000020435-0001_log06_fe_5in_md", "60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872", "60807402860000_gb506#2_md_mudlog"]:
    gt = json.load(open(f"{DS}/bbox_annotation/{n}.json"))["bounding_boxes"]
    im = Image.open(glob.glob(f"{DS}/images/{glob.escape(n)}.tif*")[0])
    H = min(im.height, max(gt["interpreted_lithology_column"][1], gt["lithology_legend"][3]) + 3 * im.width)  # page top: header + a slice of log
    page = im.crop((0, 0, im.width, H)).convert("RGB"); page.thumbnail((1024, 1024))
    canvas = Image.new("RGB", (max(512, page.width), max(512, page.height)), "white"); canvas.paste(page, (0, 0))
    k = im.width / page.width; sx, sy = canvas.width * k, canvas.height * k; page = canvas
    for model in ["jev-latest", "jev-preview"]:
        out, dt = call({"state": {"page": "pg"}, "questions": {k: {"type": "box", "instructions": v} for k, v in BQ.items()},
                        "assets": {"pg": {"type": "image", "format": "jpeg", "data_base64": b64(page, "JPEG")}}}, model)
        if "answers" not in out: print(n[:30], model, out); continue
        res = []
        for k, gk in [("depth_column", "depth_column"), ("lithology_column", "interpreted_lithology_column"), ("legend", "lithology_legend")]:
            bx = out["answers"][k]["box"]; p = [bx["left"]*sx, bx["top"]*sy, bx["right"]*sx, bx["bottom"]*sy]
            g = list(gt[gk]); g[3] = min(g[3], H)
            xo = max(0, min(p[2], g[2]) - max(p[0], g[0])) / max(1, g[2]-g[0])
            res.append(f"{k}: IoU={iou(p, g):.2f} x-overlap={xo:.2f}")
        print(f"{n[:30]:30s} {model:11s} {dt:.1f}s | " + " | ".join(res))
