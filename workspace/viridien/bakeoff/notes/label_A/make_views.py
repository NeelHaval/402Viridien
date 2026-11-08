import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_challenge\Viridien_HTB_challenge\hack27\project\dataset")
OUT = ROOT / "notes" / "label_A" / "views"
OUT.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.load_default()


def source(name):
    for ext in (".tif", ".tiff"):
        p = DATA / "images" / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    entries = json.loads((ROOT / "eval_set.json").read_text())
    for e in entries:
        name, y0, y1 = e["log"], e["y0"], e["y1"]
        boxes = json.loads((DATA / "bbox_annotation" / (name + ".json")).read_text())["bounding_boxes"]
        lith = boxes["interpreted_lithology_column"]
        legend = boxes["lithology_legend"]
        stem = name.replace("/", "_")
        with Image.open(source(name)) as im:
            # Legend: reduced only enough to fit an ordinary image viewer.
            leg = im.crop(tuple(legend)).convert("RGB")
            leg.thumbnail((1200, 900))
            leg.save(OUT / f"{stem}__legend.png")
            for start in range(y0, y1, 1500):
                end = min(y1, start + 1500)
                col = im.crop((lith[0], start, lith[2], end)).convert("RGB")
                # Preserve every vertical pixel; widen only so texture is obvious.
                col = col.resize((col.width * 4, col.height), Image.Resampling.NEAREST)
                ruler_w = 92
                canvas = Image.new("RGB", (ruler_w + col.width, col.height), "white")
                canvas.paste(col, (ruler_w, 0))
                d = ImageDraw.Draw(canvas)
                for yy in range(0, end - start + 1, 100):
                    d.line((ruler_w - 9, yy, ruler_w - 1, yy), fill="black", width=1)
                    d.text((1, max(0, yy - 6)), str(start + yy), fill="black", font=FONT)
                for yy in range(0, end - start + 1, 20):
                    d.line((ruler_w - 4, yy, ruler_w - 1, yy), fill="gray", width=1)
                canvas.save(OUT / f"{stem}__{start}-{end}.png")


if __name__ == "__main__":
    main()
