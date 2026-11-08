"""Frozen v4 beds with the native-scale DINOv2 and hand-distance fusion probe.

The column crop is never resized. Swatches alone are tested at three nearest-
neighbour scales, as in dino_probe_scales.py. No truth or training data is read.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

# Set before importing transformers/huggingface_hub. Refuse implicit network IO.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from PIL import Image
from strategies import scoreboard_v4 as adapter

NAME = 'S21 v4 beds + frozen DINOv2-small fusion'
NEEDS_GPU = True
PAID_API = None
MODEL = 'facebook/dinov2-small'
SCALES = (0.5, 1.0, 2.0)
BATCH = 32


def canvas(array, size=224):
    """Tile native pixels into one model canvas; crop oversized inputs at top left."""
    a = np.ascontiguousarray(array[..., :3])
    if a.ndim != 3 or not a.shape[0] or not a.shape[1] or a.shape[2] != 3:
        raise ValueError('DINO input must be a nonempty RGB image')
    a = a[:, :size]
    return np.tile(a, ((size + len(a) - 1) // len(a),
                       (size + a.shape[1] - 1) // a.shape[1], 1))[:size, :size]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModel
    from transformers.utils.hub import cached_file
    if not torch.cuda.is_available():
        raise RuntimeError('S21 requires CUDA for the frozen DINOv2 model')
    # Both config and model are resolved from the existing HF cache only.
    config = cached_file(MODEL, 'config.json', local_files_only=True)
    weights = cached_file(MODEL, 'model.safetensors', local_files_only=True,
                          _raise_exceptions_for_missing_entries=False)
    if weights is None:
        weights = cached_file(MODEL, 'pytorch_model.bin', local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).to('cuda').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    return model, mean, std, {'model_id': MODEL, 'config_sha256': _sha256(config),
                              'weights_file': Path(weights).name,
                              'weights_sha256': _sha256(weights),
                              'device': torch.cuda.get_device_name(0)}


def embed(arrays):
    """Encode one or more RGB images on CUDA, synchronously, in bounded batches."""
    import torch
    if not arrays:
        return np.empty((0, 768), dtype=np.float32)
    model, mean, std, _ = _model()
    result = []
    with torch.inference_mode():
        for start in range(0, len(arrays), BATCH):
            batch = np.stack([canvas(a) for a in arrays[start:start + BATCH]])
            x = torch.from_numpy(batch).to('cuda').permute(0, 3, 1, 2).float() / 255
            hidden = model(pixel_values=(x - mean) / std).last_hidden_state
            vectors = torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], 1)
            result.append(torch.nn.functional.normalize(vectors, dim=1).cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(result)


def _swatches(classifier, log):
    labels, images = [], []
    for label, a in classifier._native().get(log, []):
        if classifier.NOT_ROCK.search(label):
            continue
        t = 2 if min(a.shape[:2]) > 12 else 0
        a = classifier.strip_borders(a[t:a.shape[0] - t or None,
                                       t:a.shape[1] - t or None])
        if not a.size or min(a.shape[:2]) < 3:
            continue
        for scale in SCALES:
            b = a if scale == 1 else np.asarray(
                Image.fromarray(np.ascontiguousarray(a).astype(np.uint8)).resize(
                    (max(3, round(a.shape[1] * scale)), max(3, round(a.shape[0] * scale))),
                    Image.Resampling.NEAREST))
            labels.append(label)
            images.append(b)
    return labels, images


def _state(classifier, image, check):
    width = image.shape[1]
    a = classifier.strip_borders(image[..., :3])
    if a.shape[1] < 0.6 * width:
        return a, 'blank'
    if check:
        return a, classifier.window_state(a)
    ai = a.astype(int)
    fill = ((ai.max(2) >= classifier.P['fill_dark']) & (ai.min(2) <= 235)).mean()
    return a, 'blank' if (ai.max(2) < 90).mean() < 0.012 and fill < 0.05 else 'ok'


def _z(values):
    return (values - values.mean()) / (values.std() + 1e-6)


def _classify(classifier, image, log, vector, swatch_labels, swatch_vectors,
              alpha=None, unknown=None, check=True):
    a, state = _state(classifier, image, check)
    if state != 'ok':
        return dict(state=state, label=None)
    features = classifier.features(a)
    twins = classifier.twins(log)
    hand = {}
    for label, swatch in classifier.legend(log, classifier.log_scale(log)):
        d = classifier.distance(features, swatch, alpha) + classifier.P['gamma'] * (
            1 - classifier.ncc(features['mask'], swatch['mask']))
        name = twins.get(label, label)
        hand[name] = min(d, hand.get(name, float('inf')))
    if not hand:
        return dict(state='unknown', label=None)
    similarities = swatch_vectors @ vector
    dino = {}
    for label, similarity in zip(swatch_labels, similarities):
        name = twins.get(label, label)
        dino[name] = max(float(similarity), dino.get(name, -float('inf')))
    names = list(hand)
    if any(name not in dino for name in names):
        raise RuntimeError('DINO and hand legend labels do not match')
    distances = np.array([hand[name] for name in names])
    sims = np.array([dino[name] for name in names])
    fused = _z(distances) - _z(sims)
    order = np.argsort(fused, kind='stable')
    first = int(order[0])
    second = int(order[1]) if len(order) > 1 else None
    # A bounded confidence gap keeps v4's row-weighted voting stable.
    margin = 1.0 if second is None else float(1 - np.exp(-(fused[second] - fused[first])))
    # Keep v4's hand-distance rejection gate independent of the new ranking.
    threshold = classifier.P['unknown'] if unknown is None else unknown
    state = 'unknown' if min(hand.values()) > threshold else 'ok'
    return dict(state=state, label=names[first],
                second=names[second] if second is not None else None,
                d1=round(float(distances[first]), 3),
                d2=round(float(distances[second]), 3) if second is not None else None,
                margin=round(margin, 3))


def run(log, window, ctx):
    classifier = adapter.make_classifier(log, ctx)
    swatch_labels, swatch_images = _swatches(classifier, log.name)
    chunks = []
    def collect(image, _log, alpha=None, unknown=None, check=True):
        a, state = _state(classifier, image, check)
        if state == 'ok' and swatch_images:
            chunks.append(a.copy())
        return dict(state=state if state != 'ok' else 'unknown', label=None)
    classifier.classify = collect
    adapter.segment(log, ctx, classifier=classifier, window=window)
    vectors = embed(swatch_images + chunks) if chunks else None
    legend_vectors = vectors[:len(swatch_images)] if vectors is not None else None
    chunk_vectors = iter(vectors[len(swatch_images):]) if vectors is not None else iter(())
    used = 0
    def classify(image, name, alpha=None, unknown=None, check=True):
        nonlocal used
        _, state = _state(classifier, image, check)
        if state != 'ok':
            return dict(state=state, label=None)
        if not swatch_images:
            return dict(state='unknown', label=None)
        used += 1
        return _classify(classifier, image, name, next(chunk_vectors), swatch_labels,
                         legend_vectors, alpha=alpha, unknown=unknown, check=check)
    classifier.classify = classify
    result = adapter.segment(log, ctx, classifier=classifier, window=window)
    if used != len(chunks):
        raise RuntimeError('Bed chunks changed between collection and classification')
    run.last_diagnostics = adapter.diagnostics(log, classifier, result)
    run.last_diagnostics.update(dino_swatch_vectors=len(swatch_images), dino_chunk_vectors=used,
                                model=_model()[3] if vectors is not None else None,
                                fusion='z(hand distance) - z(DINO similarity)',
                                column_resize=False, swatch_scales=list(SCALES),
                                implementation_sha256=_sha256(__file__))
    return adapter.to_rows(result, window, ctx['calib'])
