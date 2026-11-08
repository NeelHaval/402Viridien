"""Make a local contact sheet from all supplied legend entries. No approval is set."""

import argparse
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw

from generator import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "validation")
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    entries = annotation["legend"]
    cell_width, cell_height, columns = 166, 194, 6
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#eef1f2")
    draw = ImageDraw.Draw(sheet)
    results = []
    started = time.perf_counter()
    with Image.open(args.image) as source:
        for index, entry in enumerate(entries):
            generated, metadata = generate(source, entry["box"], seed=0)
            filename = f"legend-{entry['id']:02d}.png"
            generated.save(args.output / filename)
            x, y = index % columns * cell_width, index // columns * cell_height
            draw.text((x + 4, y + 3), str(entry["id"]) + " " + entry["label_printed"][:21], fill="#172b31")
            draw.text((x + 4, y + 20), "Source / Generated", fill="#455e66")
            crop = source.crop(entry["box"]).convert("RGB")
            sheet.paste(crop, (x + 3, y + 40))
            sheet.paste(generated.crop((0, 0, 90, 125)), (x + 72, y + 40))
            draw.text((x + 4, y + 172), str(metadata["estimated_period"]), fill="#455e66")
            results.append(dict(legend_id=entry["id"], label=entry["label_printed"], file=filename,
                                provenance=dict(image=str(args.image), annotation=str(args.annotation)),
                                generation=metadata))
    sheet.save(args.output / "legend-contact.png")
    report = dict(source_example_count=len(results), approval="Not set. These are previews.",
                  elapsed_seconds=round(time.perf_counter() - started, 3), examples=results)
    (args.output / "source-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for result in results:
        meta = result["generation"]
        print(f"{result['legend_id']:2d} {result['label']:<18} {meta['method']:<18} period={meta['estimated_period']}")
    print(f"Source check: {len(results)} previews; contact sheet and provenance report written.")


if __name__ == "__main__":
    main()
