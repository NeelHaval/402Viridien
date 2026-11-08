"""S13 learned boundaries with S14's dev-selected visual pair matcher."""
import cv2
import numpy as np
from harness import io
from ml.boundary.model import load_model
from strategies.s13_boundary_net import infer
from strategies.s14_learned_match import name_intervals

NAME = 'S15 learned boundaries + learned legend matching'
NEEDS_GPU = False
PAID_API = None


def run(log,window,ctx):
    cv2.setNumThreads(1)
    rgb=io.read_rgb(log,'interpreted_lithology_column',*map(int,window))
    net,config=load_model()
    proposed,prob=infer(rgb,net,config)
    points=np.r_[0,proposed,len(rgb)]
    preserve=set(proposed.tolist()) if config.get('preserve',True) else set()
    return name_intervals(rgb,int(window[0]),points,ctx['legend'],preserve)
