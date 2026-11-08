# Pattern generator and lab

This directory has the generator, dataset export, review controls, and tests.
It uses local source pixels. It does not use a remote API.

## Python contract

```python
from pathlib import Path
from PIL import Image
from viridien_work.pattern_lab.generator import generate, create_dataset

# If server.py runs from pattern_lab, `from generator import ...` also works.
with Image.open(source_path) as source:
    sample, metadata = generate(
        source, [532, 1738, 582, 1838],
        width=180, height=480, seed=0, variation=0.0,
    )
    # Each record must be an explicitly approved source example.
    manifest = create_dataset(source, approved_examples, Path(export_path),
                              count_per_example=3, seed=0)
```

The runtime needs Pillow and NumPy. There is no dependency on the bake-off code.

`generate` returns `(PIL.Image.Image, dict)`. The image is RGB. Its size is
exactly `(width, height)`. Coordinates are integer xyxy. The right and bottom
edges are exclusive. A box must have a positive area inside the source image.
Invalid input raises `ValueError`. A source of the wrong type raises `TypeError`.
The output limit is 16,777,216 pixels and 16,384 pixels on either axis.
Seeds must be non-negative integers. Variation must be a finite number from
0 to 1. The generator does not change the source image.

The generator removes thin dark rails at the crop edge. It measures an integer
repeat on each axis. At variation 0, it copies a supported repeat. It uses
patches and a minimum-error seam if it cannot find a reliable repeat.
Variation above 0 uses patches even if a repeat is available. It increases
the choice of patches that have close seam matches. It does not change size,
colour, orientation, or symbol spacing by image scaling. All output colours
come from source pixels. No interpolation or colour blending is used.

Metadata has:

- `source_box`: the exact input coordinates.
- `texture_box` and `frame_trim`: the coordinates after dark rails are removed.
- `method`, `seed`, `variation`, and `scale: [1.0, 1.0]`.
- `estimated_period: {x: int | null, y: int | null}` in native source pixels.
- `repeat_estimation`: confidence, observed cycles, and constant-axis flags.
- `passes`: crop offsets, tile origin and phase, or patch placements and seams.
- `source_crop_sha256`, source and output dimensions, and `limitations`.

For a quilting pass, its source offset is relative to the image that entered
that pass. The vertical pass uses the texture box. The horizontal pass uses
the vertical result. The metadata keeps the order of both passes.

`generate` makes a preview. It does not set or require source approval.
Small crops can have partial symbols. Seams can cut symbols. A measured repeat
with less than two full source cycles is recorded as a limitation. Samples
repeat the supplied example; they do not add new geological evidence.

## Dataset contract

Each source example must contain:

```json
{
  "id": "segment-7",
  "box": [532, 1800, 582, 1900],
  "legend_ids": [8],
  "document_id": "hadrian",
  "provenance": {"annotation_file": "labels.json", "layer_index": 7},
  "approved": true
}
```

`provenance` must be a non-empty string or object. Extra JSON fields are kept.
The function checks the full input list before it creates the output directory.
**If one example has `approved` other than the boolean `true`, the full request
raises `ValueError`. It writes no files.** Missing approval also fails.
An empty list fails. Example ids must be unique within each document.

Use a new or empty output directory for each export. Existing exports are not
overwritten. The function writes 180 × 480 PNG files and `manifest.json`. It
returns the same manifest object. Samples are staged before files are published.
The manifest is published last.

The manifest has `examples` with the exact source records, and `samples` with
file names, PNG hashes, source document and example ids, exact `source_box`,
`legend_ids`, `provenance`, `output_box`, `annotations`, and generation metadata.
Sample seeds are `seed + example_index * count_per_example + sample_index`.
File names use an internal index and hash. A source id cannot create a path.

An output annotation covers `[0, 0, 180, 480]` with the example's full label set.
A crop with several legend ids remains a multi-label crop. The exporter does
not invent component coordinates or proportions. Original composition data
can be kept in the source record or provenance. Use approved component boxes
as separate examples when exact component labels are needed.

`source_approved: true` means that the real example has approval.
`generation_review: null` means that the exported sample has not received a
separate review. Dataset creation does not approve generated samples.

## Browser contract for the parent

Load `generator/lab.css` and then `generator/lab.js`. The module sets
`window.PatternLab`. It does not start a server or load a framework.

```javascript
const panel = window.PatternLab.mount(container, {
  document: { id: documentId, legend: [{ id: 8, label_printed: "Halite", box: legendBox }] },
  segment: { id: segmentId, box: sourceBox, legend_ids: [8], status: "approved" },
  onApprove(reviewPayload, savedResponse) { /* Update the parent editor. */ }
});
// Optional: await panel.refresh();
// On panel close: panel.unmount();
```

`mount` replaces only the contents of its supplied container. A second mount
in that container stops the prior instance. `unmount` stops requests and frees
image URLs. `onApprove` runs after a successful positive review save. It does
not run for rejection. The controls block review until the real crop and the
current generated sample have loaded. A settings change blocks approval until
the displayed parameters match. Older responses cannot replace newer samples.
Labels, server errors, and comments use DOM text. Dataset links must have the
same origin as the editor.

| Request | Expected response |
| --- | --- |
| `GET /api/crop?doc=ID&box=x0,y0,x1,y1&width=SOURCE_WIDTH` | PNG of the source crop. |
| `GET /api/generate?doc=ID&segment=ID&width=180&height=480&seed=0&variation=0` | PNG from `generate`. |
| `POST /api/generation-review` | JSON object or HTTP 204. |
| `POST /api/dataset` | JSON object. It can include `download_url` or `manifest_url`. |

The review body contains `document_id`, `segment_id`, `seed`, `approved`, and
`comment`. **It also sends `width`, `height`, and `variation`** so that approval
identifies the exact displayed sample. The parent should keep these fields.
The dataset body is `{document_id, count_per_example, seed}`. The parent must
load the source image, select explicitly approved examples, and give each
export a separate output directory. An error response can use
`{"error": "Message"}` with a non-2xx status.

The crop request uses width 180 as specified. The lab restores the source
crop's native display width for boxes up to 180 pixels across. This makes
symbol size easier to compare with the generated image. Both image areas
scroll. The supplied lithology boxes are 43 to 57 pixels across.

## Checks and source examples

Run from the repository root:

```text
python -X utf8 -m unittest discover -s viridien_work/pattern_lab/generator -v
node --test viridien_work/pattern_lab/generator/test_lab.cjs
node --check viridien_work/pattern_lab/generator/lab.js
```

The Python checks cover determinism, dimensions, exact native repeats, source
colours, seam paths, frame coordinates, input limits, source provenance,
dataset determinism, and rejection of unapproved examples before any writes.
The JS checks use a small DOM model. They cover request contracts, safe label
text, review state, source load failure, stale response order, and dataset
export. A live browser check was not run. Endpoint integration is owned by the
parent.

`validate_source.py` can rebuild the legend previews:

```text
python -X utf8 -m viridien_work.pattern_lab.generator.validate_source --image IMAGE_PATH --annotation JSON_PATH
```

`validation/legend-contact.png` compares all 18 source legend entries with
generated pixels at native scale. `validation/layer-contact.png` compares five
real layer crops. The JSON reports keep coordinates and generation metadata.
These files are previews. They do not set source or sample approval.
