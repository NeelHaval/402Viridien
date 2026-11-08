"""S21: clean synthetic and current-dev boundary and legend models."""
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from harness import io
from ml.synth_v2.runtime import load
from strategies.s13_boundary_net import infer
from strategies.s14_learned_match import name_intervals

NAME = 'S21 extended synthetic boundaries and legend matching'
NEEDS_GPU = False
PAID_API = None


@threadpool_limits.wrap(limits=3)
def run(log, window, ctx):
    cv2.setNumThreads(1)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *map(int, window))
    net, matcher, config = load()
    proposed, _ = infer(rgb, net, config['boundary'])
    return name_intervals(rgb, int(window[0]), np.r_[0, proposed, len(rgb)],
                          ctx['legend'], matcher=(matcher, config['matcher']))
