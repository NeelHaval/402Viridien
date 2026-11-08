"""Compare the two native generators against every real crop, without rails."""
from __future__ import annotations

import copy
import numpy as np

from . import learn, render


def crop_scores(data, crops, doc, examples):
    from .candidates import pixel_match
    if data is None:
        return None
    if data['kind'] == 'tile':
        probe = data
    else:
        # A reproducible rendered quilt window is the candidate being judged.
        width = data['strips'][0].shape[1]
        rgb = render.layer(data, 0, width, 0, 240, np.random.default_rng(0))
        probe = dict(tile=rgb, bg=learn.background(rgb))
    values = []
    for e in examples:
        rgb, _ = learn.clean(crops.get(doc, e['box']))
        if rgb.size:
            values.append(pixel_match(rgb, probe))
    if not values:
        return None
    return dict(median=round(float(np.median(values)), 5), worst=round(min(values), 5),
                crops=len(values), scores=[round(v, 5) for v in values],
                evaluation='all real crops, best cyclic pixel match, rails excluded')


def choose(crops, doc, examples, model, data, swatch_data=None, seam_risk=True):
    """A matching safe swatch wins ties, quilts, and models with <2 voters."""
    column = crop_scores(data, crops, doc, examples)
    swatch = crop_scores(swatch_data, crops, doc, examples) if swatch_data is not None else None
    cs, ss = (column or {}).get('median', 0), (swatch or {}).get('median', 0)
    take = (not seam_risk and ss >= .85 and
            (model['kind'] == 'quilt' or model.get('voters', 0) < 2 or ss >= cs - .01))
    model = copy.deepcopy(model)
    model.update(tile_source='legend swatch' if take else 'column',
                 tile_scores=dict(column=column, legend_swatch=swatch))
    if take:
        data = copy.deepcopy(swatch_data)
        h, w = data['tile'].shape[:2]
        model.update(kind='tile', tile_size=[w, h], period=dict(y=h, x=w),
                     anchored=False, phase=[0, 0], reason='Native legend repeat matches the real crops.')
    return model, data
