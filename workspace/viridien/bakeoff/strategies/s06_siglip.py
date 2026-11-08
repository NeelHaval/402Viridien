"""SigLIP strip embeddings for mud-log lithology intervals."""
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipModel

NAME = "S6 SigLIP"
NEEDS_GPU = True
PAID_API = None
MODEL_ID = "google/siglip-base-patch16-224"
STEP = 12


@lru_cache(maxsize=1)
def _load_model():
    model = SiglipModel.from_pretrained(MODEL_ID, local_files_only=True)
    processor = SiglipImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    return model.eval(), processor, tokenizer


def _image_tensor(patches, processor, device, batch_size=32):
    result = []
    for start in range(0, len(patches), batch_size):
        inputs = processor(images=patches[start:start + batch_size], return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    emb = _load_model()[0].get_image_features(pixel_values=pixels)
            else:
                emb = _load_model()[0].get_image_features(pixel_values=pixels)
        result.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
    return np.concatenate(result, axis=0)


def _tile(rgb, size=224):
    h, w = rgb.shape[:2]
    # Repetition preserves the periodic pattern cues found in the legend swatch.
    reps_y = max(1, int(np.ceil(size / h)))
    reps_x = max(1, int(np.ceil(size / w)))
    tiled = np.tile(rgb, (reps_y, reps_x, 1))[:size, :size]
    return Image.fromarray(tiled)


def _stage(array, height=360, width=110):
    a = np.asarray(array)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim == 2:
        a = cv2.applyColorMap(np.uint8(np.clip(a, 0, 255)), cv2.COLORMAP_TURBO)[:, :, ::-1]
    return cv2.resize(a, (width, height), interpolation=cv2.INTER_NEAREST)


def run(log, window, ctx):
    model, processor, tokenizer = _load_model()
    device = ctx["device"]
    model = model.to(device)
    y0, y1 = map(int, window)
    entries = [e for e in ctx["legend"] if e.is_lithology and e.swatch.size]
    if not entries:
        return [], {"1 column": np.full((80, 40, 3), 255, np.uint8)}

    box = log.boxes["interpreted_lithology_column"]
    rgb = __import__("harness.io", fromlist=["read_rgb"]).read_rgb(log, box, y0, y1)
    height, width = rgb.shape[:2]
    centers = np.arange(0, height, STEP, dtype=int)
    patches = []
    half = STEP
    for center in centers:
        top = max(0, center - half)
        bottom = min(height, center + half + 1)
        patches.append(Image.fromarray(rgb[top:bottom]).resize((224, 224), Image.Resampling.BICUBIC))
    image_emb = _image_tensor(patches, processor, device)

    swatch_images = [_tile(e.swatch) for e in entries]
    swatch_emb = _image_tensor(swatch_images, processor, device)
    prompts = [f"a geological lithology symbol for {e.label}" for e in entries]
    text_inputs = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt")
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                text_emb = model.get_text_features(**{k: v.to(device) for k, v in text_inputs.items()})
        else:
            text_emb = model.get_text_features(**{k: v.to(device) for k, v in text_inputs.items()})
    text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1).cpu().numpy()

    image_score = image_emb @ swatch_emb.T
    text_score = image_emb @ text_emb.T
    # Synthetic columns are generated from exact legend swatches, so visual evidence dominates.
    scores = 0.5 * image_score + 0.5 * text_score
    if scores.ndim == 1:
        scores = scores[:, None] if len(scores) == len(centers) else scores[None, :]
    # A short temporal average removes patch-resize noise while preserving the shortest
    # synthetic bed (90 px) and the thin line transitions better than label voting.
    if len(scores) >= 5:
        padded = np.pad(scores, ((2, 2), (0, 0)), mode="edge")
        scores = np.stack([padded[i:i + len(scores)] for i in range(5)], axis=0).mean(axis=0)
    winners = scores.argmax(axis=1)
    confidence = scores.max(axis=1)

    boundaries = [0]
    for i in range(1, len(winners)):
        if winners[i] != winners[i - 1]:
            boundaries.append(int(centers[i]))
    boundaries.append(height)
    rows = []
    raw = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a:
            continue
        mid = min(len(winners) - 1, max(0, int(((a + b) / 2) // STEP)))
        raw.append((a, b, int(winners[mid])))
    # Coalesce narrow runs using their neighboring run with the strongest matching score.
    for a, b, idx in raw:
        if rows and b - a < 2 * STEP and len(raw) > 1:
            prior = rows[-1]
            if prior["_index"] == idx:
                prior["bottom_y"] = y0 + b
                continue
        entry = entries[idx]
        rows.append(dict(top_y=y0 + a, bottom_y=y0 + b, lithology=entry.label,
                         cluster_id=int(entry.id), _index=idx))
    for row in rows:
        row.pop("_index", None)

    pred_strip = np.zeros((len(winners), 3), np.uint8)
    for i, idx in enumerate(winners):
        pred_strip[i] = np.asarray(entries[int(idx)].mean_rgb, dtype=np.uint8)
    conf_strip = np.uint8(np.clip((confidence - confidence.min()) /
                                  (np.ptp(confidence) + 1e-6) * 255, 0, 255))
    boundary_strip = np.zeros((height, 3), np.uint8)
    for a in boundaries:
        if a < height:
            boundary_strip[a:min(height, a + 2)] = (255, 40, 20)
    stages = {
        "1 column": cv2.resize(rgb, (min(120, width), 360), interpolation=cv2.INTER_AREA),
        "2 SigLIP confidence": _stage(conf_strip),
        "3 swatch labels": cv2.resize(pred_strip[:, None, :], (110, 360), interpolation=cv2.INTER_NEAREST),
        "4 boundaries": cv2.resize(boundary_strip[:, None, :], (110, 360), interpolation=cv2.INTER_NEAREST),
    }
    return rows, stages
