"""Style-matched synthetic boundary and legend-pair models, selected on dev."""
import cv2
import numpy as np
from harness import io
from ml.synth_gulf.runtime import load
from strategies.s13_boundary_net import infer
from strategies.s14_learned_match import name_intervals

NAME = 'S17 style-matched synthetic boundaries and legend matching'
NEEDS_GPU = False
PAID_API = None


def run(log, window, ctx):
    cv2.setNumThreads(1)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *map(int, window))
    net, matcher, config = load()
    proposed, _ = infer(rgb, net, config['boundary'])
    points = np.r_[0, proposed, len(rgb)]
    preserve = set(proposed.tolist()) if config['boundary'].get('preserve',False) else set()
    return name_intervals(rgb, int(window[0]), points, ctx['legend'], preserve,
                          matcher=(matcher, config['matcher']))
