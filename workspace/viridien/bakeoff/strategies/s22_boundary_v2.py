"""S22: clean boundary training and S11 naming, selected by dev-well CV."""
import cv2
import numpy as np
from harness import io
from ml.boundary_v2.runtime import load, propose
from strategies.s13_boundary_net import name_intervals

NAME = 'S22 clean boundary CNN + S11 legend'
NEEDS_GPU = False
PAID_API = None


def run(log, window, ctx):
    cv2.setNumThreads(1)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *map(int, window))
    net, config = load()
    points, _ = propose(rgb, net, config)
    preserve = set(points.tolist()) if config['preserve'] else set()
    return name_intervals(rgb, int(window[0]), np.r_[0, points, len(rgb)],
                          ctx['legend'], preserve)
