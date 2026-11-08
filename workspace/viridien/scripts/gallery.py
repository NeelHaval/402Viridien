"""Dataset overview figures: legend contact sheet, column contact sheet, page-with-boxes overviews, stats CSV."""
import json, sys, os, glob, csv
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
DS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
def font(s):
    for n in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(n, s)
        except OSError: pass
    return ImageFont.load_default()
F = font(14); FS = font(11)
items = []
for a in sorted(glob.glob(f"{DS}/bbox_annotation/*.json")):
    ann = json.load(open(a)); n = os.path.splitext(os.path.basename(a))[0]
    p = [x for x in glob.glob(f"{DS}/images/{glob.escape(n)}.tif*")]
    if not p: continue
    items.append((n, p[0], ann["bounding_boxes"]))
stats = []
legs, cols, pages = [], [], []
for n, p, b in items:
    im = Image.open(p)
    W, H = im.size
    c, d, L = b["interpreted_lithology_column"], b["depth_column"], b["lithology_legend"]
    stats.append(dict(name=n, mode=im.mode, width=W, height=H, col_w=c[2]-c[0], col_h=c[3]-c[1], legend_w=L[2]-L[0], legend_h=L[3]-L[1]))
    lg = im.crop(L).convert("RGB"); lg.thumbnail((360, 300)); legs.append((n, lg))
    # column strip: 1500 px tall sample from 40% down, depth + litho side by side
    y0 = c[1] + int((c[3]-c[1])*0.4); y1 = min(c[3], y0 + 3000)
    dc = im.crop((d[0], y0, d[2], y1)).convert("RGB"); lc = im.crop((c[0], y0, c[2], y1)).convert("RGB")
    s = 600 / (y1 - y0)
    dc = dc.resize((max(1, int(dc.width*s*1.5)), 600)); lc = lc.resize((max(8, min(120, int(lc.width*s*1.5))), 600))
    strip = Image.new("RGB", (dc.width + lc.width + 6, 600), "white"); strip.paste(dc, (0, 0)); strip.paste(lc, (dc.width + 6, 0))
    cols.append((n, strip))
    # page overview: top 1 page-width tall region scaled, with boxes
    top = max(0, min(L[1], c[1]) - 200); bot = min(H, max(L[3], c[1]) + int(W*1.2))
    pg = im.crop((0, top, W, bot)).convert("RGB"); sc = 500 / pg.width
    pg = pg.resize((500, int(pg.height*sc))); dr = ImageDraw.Draw(pg)
    for key, colr in (("depth_column", (230, 40, 40)), ("interpreted_lithology_column", (0, 120, 230)), ("lithology_legend", (240, 130, 0))):
        x0, yy0, x1, yy1 = b[key]; dr.rectangle([x0*sc, (yy0-top)*sc, x1*sc, (yy1-top)*sc], outline=colr, width=3)
    pages.append((n, pg))
with open(f"{OUT}/dataset_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
def sheet(tiles, ncol, cw, ch, path, title):
    rows = (len(tiles) + ncol - 1)//ncol
    S = Image.new("RGB", (ncol*cw, 40 + rows*(ch+22)), "white"); dr = ImageDraw.Draw(S)
    dr.text((10, 10), title, fill="black", font=font(20))
    for i, (n, t) in enumerate(tiles):
        x, y = (i % ncol)*cw, 40 + (i//ncol)*(ch+22)
        dr.text((x+4, y+3), n[:int(cw/6.5)], fill=(60, 60, 60), font=FS)
        t2 = t.copy(); t2.thumbnail((cw-8, ch)); S.paste(t2, (x+4, y+20))
    S.save(path); print(path, S.size)
sheet(legs, 6, 370, 310, f"{OUT}/01_all_legends.png", f"All {len(legs)} lithology legends (cropped from the supplied boxes)")
for k in range(0, len(cols), 26):
    sheet(cols[k:k+26], 13, 150, 600, f"{OUT}/02_columns_{k//26+1}.png", "Depth column (left) + interpreted lithology column (right), 3000 px sample from 40% down each log")
for k in range(0, len(pages), 12):
    sheet(pages[k:k+12], 6, 510, 700, f"{OUT}/03_pages_{k//12+1}.png", "Page tops with supplied boxes: red = depth, blue = lithology column, orange = legend")
