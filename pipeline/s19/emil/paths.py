"""Resolve portable datasets and paths from archived Windows results."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]


def mapped_path(value):
    value = str(value).replace('\\', '/')
    for marker, root in [('viridien_work/', WORK), ('project/dataset', WORK.parent/'dataset')]:
        if marker in value:
            suffix = value.split(marker, 1)[1].lstrip('/')
            return root/suffix
    return Path(value).expanduser()


def dataset_path():
    env = os.environ.get('LITHOLOGY_DATASET')
    if env:
        return mapped_path(env).resolve()
    config = HERE/'dataset_paths.json'
    if config.exists():
        candidate = mapped_path(json.loads(config.read_text(encoding='utf-8'))['viridien'])
        if candidate.is_dir():
            return candidate.resolve()
    candidates = [WORK.parent/'dataset', WORK.parent/'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset']
    return next((p for p in candidates if p.is_dir()), candidates[0]).resolve()


def ocr_reader():
    import easyocr
    import torch
    cache = HERE/'cache/easyocr'
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False,
                          model_storage_directory=str(cache/'model'),
                          user_network_directory=str(cache/'user_network'))


def cached_swatch_names(img, boxes, legend_box):
    """Reuse only exact legend-pixel and detector-box matches, never label files."""
    import hashlib
    import numpy as np
    from emil.pipeline import _crop_image
    rgb = np.asarray(_crop_image(img, legend_box).convert('RGB'))
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    for folder in [HERE/'cache', HERE.parent/'out/_cache/s12_emil']:
        path = folder/f'legend_{digest}.json'
        if not path.exists(): continue
        entries = json.loads(path.read_text(encoding='utf-8'))
        absolute = [[e['box'][0]+legend_box[0], e['box'][1]+legend_box[1],
                     e['box'][2]+legend_box[0], e['box'][3]+legend_box[1]] for e in entries]
        if absolute == [list(b) for b in boxes]:
            return [e['lithology'] for e in entries]
    return None
