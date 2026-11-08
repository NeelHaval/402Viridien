import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(r"C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_challenge\Viridien_HTB_challenge\hack27\project\dataset")
HERE = Path(__file__).resolve().parent
OUT = HERE / "views"
OUT.mkdir(parents=True, exist_ok=True)
ev = json.loads((HERE.parents[1] / "eval_set.json").read_text(encoding="utf8"))
font = ImageFont.truetype("arial.ttf", 16)

for e in ev:
    name = e["log"]
    ann = json.loads((ROOT / "bbox_annotation" / f"{name}.json").read_text(encoding="utf8"))
    boxes = ann["bounding_boxes"]
    x0, by0, x1, by1 = boxes["interpreted_lithology_column"]
    lx0, ly0, lx1, ly1 = boxes["lithology_legend"]
    y0, y1 = e["y0"], e["y1"]
    im = Image.open(ROOT / "images" / ann["image_name"])
    legend = im.crop((lx0, ly0, lx1, ly1)).convert("RGB")
    # Preserve readable legend text while fitting a consistent side panel.
    legend.thumbnail((1050, 900))
    for start in range(y0, y1, 1500):
        end = min(start + 1500, y1)
        col = im.crop((x0, start, x1, end)).convert("RGB")
        col = col.resize((col.width * 4, col.height * 3))
        ruler = Image.new("RGB", (100, col.height), "white")
        d = ImageDraw.Draw(ruler)
        for yy in range(start, end + 1, 100):
            ry = (yy - start) * 3
            d.line((72, ry, 100, ry), fill="black", width=2)
            d.text((0, ry - 10), str(yy), fill="black", font=font)
        panel = Image.new("RGB", (1100, max(col.height, legend.height + 50)), "white")
        panel.paste(legend, (20, 40))
        ImageDraw.Draw(panel).text((20, 10), "LEGEND", fill="black", font=font)
        sheet = Image.new("RGB", (ruler.width + col.width + panel.width, col.height), "white")
        sheet.paste(ruler, (0, 0))
        sheet.paste(col, (ruler.width, 0))
        sheet.paste(panel.crop((0, 0, panel.width, col.height)), (ruler.width + col.width, 0))
        tag = f"{name[:28]}_{start}-{end}".replace("/", "_")
        sheet.save(OUT / f"{tag}.png")
    im.close()
print(OUT)
