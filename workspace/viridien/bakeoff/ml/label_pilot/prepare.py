"""Make source crops for the non-eval vision pilot. No truth CSV is opened."""
from pathlib import Path
import argparse
import hashlib
import json
import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "3"

from PIL import Image, ImageDraw

HOME = Path(__file__).resolve().parent
REPO = HOME.parents[2]
BAKEOFF = REPO / "bakeoff"
Image.MAX_IMAGE_PIXELS = None
LOGS = {
    "callaway": "Callaway 3-33 mudlog to TD.PDF",
    "freeman": "Mudlog_MD_FREEMAN 3 24 ST",
}


def eval_names():
    return {r["log"] for r in json.loads((BAKEOFF / "eval_set.json").read_text())}


def require_non_eval(name):
    if name in eval_names():
        raise ValueError("Eval log is blocked")
    if name not in LOGS.values():
        raise ValueError("Log is outside the two-log pilot")


def audit_references():
    blocked = eval_names()
    # Use names only. Do not open any reference CSV.
    paths = sorted((BAKEOFF / "gt" / "silver").glob("*/*.csv"))
    eligible = [p.relative_to(REPO).as_posix() for p in paths if p.stem not in blocked]
    result = dict(files_checked=len(paths), eval_files_excluded=sum(p.stem in blocked for p in paths),
                  eligible_references=eligible, truth_csv_contents_read=False,
                  agreement_available=False,
                  reason="No non-eval silver reference exists." if not eligible else "A reference needs review.")
    (HOME / "reference_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(dataset):
    audit_references()
    records = []
    for key, name in LOGS.items():
        require_non_eval(name)
        annotation = dataset / "bbox_annotation" / (name + ".json")
        data = json.loads(annotation.read_text())
        source = dataset / "images" / data["image_name"]
        folder = HOME / "evidence" / key
        folder.mkdir(parents=True, exist_ok=True)
        boxes = data["bounding_boxes"]
        x0, start, x1, end = boxes["interpreted_lithology_column"]
        stop = min(start + 1800, end)
        with Image.open(source) as image:
            # These two pages are small enough to decode below the RAM limit.
            # Convert each crop only. Never convert a whole page to RGB.
            legend = image.crop(tuple(boxes["lithology_legend"])).convert("RGB")
            legend.save(folder / "legend.png")
            for top in range(start, stop, 600):
                bottom = min(top + 600, stop)
                col = image.crop((x0, top, x1, bottom)).convert("RGB")
                dx0, _, dx1, _ = boxes["depth_column"]
                dep = image.crop((dx0, top, dx1, bottom)).convert("RGB")
                # Keep original y scale. Enlarge narrow tracks in x only.
                col = col.resize((180, bottom-top), Image.Resampling.NEAREST)
                dep = dep.resize((150, bottom-top), Image.Resampling.NEAREST)
                canvas = Image.new("RGB", (490, bottom-top+40), "white")
                canvas.paste(dep, (70, 30))
                canvas.paste(col, (280, 30))
                draw = ImageDraw.Draw(canvas)
                draw.text((70, 8), "Depth track", fill="black")
                draw.text((280, 8), "Lithology track", fill="black")
                for y in range(top, bottom, 50):
                    yy = y-top+30
                    draw.text((4, yy), str(y), fill="black")
                    draw.line((262, yy, 278, yy), fill="#777777")
                canvas.save(folder / f"rows_{top}_{bottom}.png")
            # Save unscaled source tracks for row and depth checks.
            image.crop((x0, start, x1, stop)).convert("RGB").save(folder / "track_native.png")
            image.crop((dx0, start, dx1, stop)).convert("RGB").save(folder / "depth_native.png")
        records.append(dict(key=key, log=name, window=[start, stop], boxes=boxes,
                            image_name=source.name, image_sha256=sha256(source),
                            annotation_sha256=sha256(annotation),
                            source_paths_relative_to_dataset=True))
    (HOME / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
    # Use only saved source crops for these review views.
    for key, box, factor in (("callaway", (0, 224, 60, 384), 4),
                              ("freeman", (0, 140, 29, 440), 5)):
        folder = HOME / "evidence" / key
        with Image.open(folder / "track_native.png") as image:
            crop = image.crop(box)
            crop.resize((crop.width*factor, crop.height*factor),
                        Image.Resampling.NEAREST).save(folder / "pattern_detail.png")
        with Image.open(folder / "legend.png") as image:
            image.resize((image.width*2, image.height*2),
                         Image.Resampling.NEAREST).save(folder / "legend_large.png")
    for key, origin, tops, height in (("freeman", 1833, (1953, 2193), 240),
                                      ("callaway", 5096, (5296,), 60)):
        folder = HOME / "evidence" / key
        with Image.open(folder / "track_native.png") as image:
            for top in tops:
                crop = image.crop((0, top-origin, image.width, top-origin+height))
                crop = crop.resize((image.width*4, height*4), Image.Resampling.NEAREST)
                canvas = Image.new("RGB", (image.width*4+85, height*4), "white")
                canvas.paste(crop, (85, 0))
                draw = ImageDraw.Draw(canvas)
                for y in range(top, top+height, 5):
                    yy = (y-top)*4
                    draw.text((0, yy), str(y), fill="black")
                    draw.line((48, yy, 83, yy), fill="#999999")
                canvas.save(folder / f"boundary_{top}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.environ.get("LITHOLOGY_DATASET", REPO.parent / "dataset")))
    args = parser.parse_args()
    prepare(args.dataset.expanduser())
