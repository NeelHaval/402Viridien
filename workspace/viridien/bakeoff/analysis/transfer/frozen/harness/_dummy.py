"""Deliberately trivial baseline: one majority-legend-colour interval."""
import numpy as np

NAME='Dummy majority legend colour'
NEEDS_GPU=False
PAID_API=None

def run(log,window,ctx):
    entries=[e for e in ctx['legend'] if e.is_lithology]
    if not entries: return []
    # Most frequent quantized mean colour among swatches, then first representative.
    colours=[tuple((np.asarray(e.mean_rgb)//32).astype(int)) for e in entries]
    common=max(dict.fromkeys(colours),key=colours.count)
    e=entries[colours.index(common)]
    return [dict(top_y=int(window[0]),bottom_y=int(window[1]),lithology=e.label,cluster_id=e.id)]
