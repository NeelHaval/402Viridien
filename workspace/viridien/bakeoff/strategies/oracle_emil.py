"""Explicit oracle control. Intentionally reads true legend boxes and names only."""
import json
import os
import pandas as pd
from harness import io
from emil import pipeline as p
from strategies.s12_emil import predict, map_name

NAME = 'S12 Emil CNN + oracle legend (leaked control)'
NEEDS_GPU = True
PAID_API = None


def run(log,window,ctx):
    folder = p.ROOT/'external_labels' if io.DATASET.resolve() == (p.WORK/'external_tests').resolve() else p.WORK/'emil/labels'
    path = folder/(log.name+'.json')
    entries = json.loads(path.read_text()).get('legend',[]) if path.exists() else []
    # For the gold log06, Emil supplied no true legend; report unavailable explicitly.
    if not entries: raise ValueError('Oracle legend unavailable for this page')
    legend = pd.DataFrame([dict(legend_id=e['id'],lithology=e['label_printed'],box=e['box']) for e in entries])
    layers,_,_,_ = predict(log,window,legend)
    return [dict(top_y=int(r.y_top),bottom_y=int(r.y_bottom),lithology=r.lithology,cluster_id=int(r.legend_id)) for r in layers.itertuples() if r.legend_id>=0]
