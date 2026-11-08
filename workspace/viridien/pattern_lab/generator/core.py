"""Generate real-pattern samples and approved datasets with local pixels only.

Boxes are integer xyxy source coordinates. The right and bottom edges are
exclusive, as in PIL.Image.crop. No scale, rotation, colour change, or remote
service is used. The only runtime dependencies are Pillow and NumPy.
"""

from copy import deepcopy
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from ._texture import estimate_period, extend_axis, trim_frame

VERSION = "1.0"
MAX_OUTPUT_PIXELS = 16_777_216


def _integer(value, name, low=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < low:
        raise ValueError(f"{name} must be an integer of {low} or more.")
    return int(value)


def _box(image, box):
    if not isinstance(image, Image.Image):
        raise TypeError("source_image must be a PIL image.")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must have four integer xyxy coordinates.")
    result = [_integer(value, "Box coordinate") for value in box]
    x0, y0, x1, y1 = result
    if not (x0 < x1 <= image.width and y0 < y1 <= image.height):
        raise ValueError("box must have a positive area inside the source image.")
    return result


def generate(source_image: Image.Image, box: list[int], width: int = 180,
             height: int = 480, seed: int = 0, variation: float = 0.0) -> tuple[Image.Image, dict]:
    """Copy a source pattern at native scale into an RGB output image.

    ``variation`` is 0..1. At zero, a supported repeat is tiled. Above zero,
    real patches are quilted, with greater choice among close seam matches.
    Seed controls crop phase and patch choice; it does not change symbol size.
    This function makes a preview. Approval is enforced by create_dataset.
    """
    source_box = _box(source_image, box)
    width, height = _integer(width, "width", 1), _integer(height, "height", 1)
    if width * height > MAX_OUTPUT_PIXELS or max(width, height) > 16384:
        raise ValueError("The output exceeds the image size limit.")
    seed = _integer(seed, "seed")
    if (isinstance(variation, bool) or not isinstance(variation, Real)
            or not np.isfinite(variation) or not 0 <= variation <= 1):
        raise ValueError("variation must be a finite number from 0 to 1.")
    variation = float(variation)
    # Crop before RGB conversion. Do not convert the full 50,000-row page.
    crop = source_image.crop(source_box).convert("RGB")
    source = np.asarray(crop).copy()
    texture, trims = trim_frame(source)
    x0, y0, x1, y1 = source_box
    left, top, right, bottom = trims
    texture_box = [x0 + left, y0 + top, x1 - right, y1 - bottom]
    repeats = {"x": estimate_period(texture, 1), "y": estimate_period(texture, 0)}
    rng = np.random.default_rng(seed)
    # The vertical pass copies full source rows; horizontal seams then match
    # those same rows. Neither pass interpolates or blends colour values.
    result, vertical = extend_axis(texture, height, 0, repeats["y"], rng, variation)
    result, horizontal = extend_axis(result, width, 1, repeats["x"], rng, variation)
    methods = {horizontal["method"], vertical["method"]}
    method = ("exemplar-quilting" if "minimum-error-quilting" in methods else
              "short-source-tile" if "short-source-tile" in methods else
              "native-repeat" if "native-repeat" in methods else "native-crop")
    limitations = []
    if any(trims):
        limitations.append("Thin dark lines at the crop edge were removed. See texture_box.")
    for axis, dimension in (("x", texture.shape[1]), ("y", texture.shape[0])):
        repeat = repeats[axis]
        if repeat["pixels"] is None:
            limitations.append(f"No reliable {axis} repeat was found in the {dimension}-pixel source extent.")
        elif not repeat["constant_axis"] and repeat["observed_cycles"] < 2:
            limitations.append(f"The {axis} repeat has less than two full cycles in the source crop.")
        if dimension < 8:
            limitations.append(f"The {axis} source extent is less than eight pixels; symbol shape may be incomplete.")
    if method == "exemplar-quilting":
        limitations.append("Seams can cut symbols. Review the generated sample against the source crop.")
    limitations.append("The sample repeats one source example. It does not add new geological evidence.")
    metadata = dict(generator_version=VERSION, source_box=source_box, texture_box=texture_box,
                    source_size=list(source_image.size), source_crop_size=list(crop.size),
                    output_size=[width, height], coordinate_system="xyxy; right and bottom exclusive",
                    method=method, seed=seed, variation=variation, scale=[1.0, 1.0],
                    estimated_period={axis: repeats[axis]["pixels"] for axis in ("x", "y")},
                    repeat_estimation=repeats, frame_trim=trims,
                    passes={"vertical": vertical, "horizontal": horizontal},
                    source_crop_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                    limitations=limitations)
    return Image.fromarray(np.ascontiguousarray(result)), metadata


def _examples(image, examples):
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list of approved source examples.")
    clean = []
    seen = set()
    required = ("id", "box", "legend_ids", "document_id", "provenance", "approved")
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or any(key not in example for key in required):
            raise ValueError(f"Example {index} must have: {', '.join(required)}.")
        if example["approved"] is not True:
            raise ValueError(f"Example {index} is not explicitly approved. No dataset was written.")
        if (isinstance(example["id"], bool) or not isinstance(example["id"], (str, int))
                or str(example["id"]).strip() == ""):
            raise ValueError(f"Example {index} needs a source id.")
        if not isinstance(example["document_id"], str) or not example["document_id"].strip():
            raise ValueError(f"Example {index} needs a document_id string.")
        ids = example["legend_ids"]
        if (not isinstance(ids, list) or not ids
                or any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in ids)):
            raise ValueError(f"Example {index} needs a non-empty legend_ids list.")
        if not example["provenance"] or not isinstance(example["provenance"], (str, dict)):
            raise ValueError(f"Example {index} needs source provenance.")
        key = (example["document_id"], str(example["id"]))
        if key in seen:
            raise ValueError("Source example ids must be unique within each document.")
        seen.add(key)
        item = deepcopy(example)
        item["box"] = _box(image, example["box"])
        try:
            json.dumps(item, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Example {index} must contain finite JSON values.") from error
        clean.append(item)
    return clean


def create_dataset(image: Image.Image, examples: list[dict], output_dir: Path,
                   count_per_example: int = 3, seed: int = 0) -> dict:
    """Write approved 180 x 480 PNG samples and manifest.json.

    All examples are checked before the output directory is created. One
    unapproved example rejects the whole request. The source record is copied
    into the manifest without changing its ids, coordinates, or provenance.
    Full output bounds carry the example's label set; component positions or
    proportions are not inferred. A mixed crop remains a multi-label example.
    The returned dict is the exact object saved in manifest.json.
    """
    count_per_example = _integer(count_per_example, "count_per_example", 1)
    seed = _integer(seed, "seed")
    approved = _examples(image, examples)
    output_dir = Path(output_dir)
    # A server should give each export its own directory. Never replace a prior
    # dataset or unrelated source files in a reused directory.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be new or empty. Use a separate directory for each export.")
    manifest = dict(schema_version=1, generator_version=VERSION, seed=seed,
                    count_per_example=count_per_example, sample_count=len(approved) * count_per_example,
                    output_size=[180, 480], source_image_size=list(image.size),
                    coordinate_system="xyxy; right and bottom exclusive",
                    label_scope="The full output has the source example label set; component bounds are not inferred.",
                    examples=approved, samples=[])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage all images before publishing files. manifest.json is written last.
    with tempfile.TemporaryDirectory(prefix=".patterns-", dir=output_dir) as stage_name:
        stage = Path(stage_name)
        for example_index, example in enumerate(approved):
            token = hashlib.sha256((example["document_id"] + ":" + str(example["id"])).encode("utf-8")).hexdigest()[:12]
            for sample_index in range(count_per_example):
                sample_seed = seed + example_index * count_per_example + sample_index
                generated, metadata = generate(image, example["box"], seed=sample_seed)
                filename = f"example-{example_index:04d}-{token}-{sample_index:03d}.png"
                generated.save(stage / filename, format="PNG")
                output_box = [0, 0, generated.width, generated.height]
                manifest["samples"].append(dict(id=f"{example_index}:{sample_index}", file=filename,
                    example_id=example["id"], document_id=example["document_id"], source_approved=True,
                    generation_review=None,
                    source_box=list(example["box"]), legend_ids=deepcopy(example["legend_ids"]),
                    provenance=deepcopy(example["provenance"]), output_box=output_box,
                    annotations=[dict(box=output_box, legend_ids=deepcopy(example["legend_ids"]))],
                    png_sha256=hashlib.sha256((stage / filename).read_bytes()).hexdigest(), generation=metadata))
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + "\n", encoding="utf-8")
        for sample in manifest["samples"]:
            (stage / sample["file"]).replace(output_dir / sample["file"])
        (stage / "manifest.json").replace(output_dir / "manifest.json")
    return manifest
