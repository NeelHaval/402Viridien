"""Dev-CV-selected boundary/name integration, with interval confidence."""
import cv2
from harness import io
from ml.boundary.model import load_model
from strategies.s13_boundary_net import infer
from strategies.s14_learned_match import proposals
from ml.integrate.core import configuration, entries_only, evidence, boundary_points, decode, add_confidence

NAME = 'S18 dev-selected joint boundary and legend pipeline'
NEEDS_GPU = False
PAID_API = None


def run(log, window, ctx):
    cv2.setNumThreads(1)
    y0,y1=map(int,window)
    rgb=io.read_rgb(log,'interpreted_lithology_column',y0,y1)
    entries=entries_only(ctx['legend'])
    if not entries:return []
    net,bcfg=load_model();_,prob=infer(rgb,net,bcfg)
    s11,rules=proposals(log,window,ctx,rgb)
    cfg=configuration()
    points=boundary_points(prob,s11,cfg['mode'],cfg['threshold'],cfg['gap'])
    ev=evidence(rgb,points,entries)
    rows=add_confidence(decode(ev,entries,prob,rules,cfg))
    for row in rows:
        row['top_y']+=y0;row['bottom_y']+=y0
    return rows
