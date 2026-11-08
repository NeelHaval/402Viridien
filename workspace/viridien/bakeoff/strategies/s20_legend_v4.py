"""S18 with the opt-in legend v4 parser.

S18 model weights and decoder settings are inherited. See the lane report for
historical old-dev exposure. Inference reads no annotation truth.
"""
from harness import legend_v4
from strategies.s18_best import run as s18

NAME = 'S20 S18 with legend v4'
NEEDS_GPU = False
PAID_API = None


def run(log, window, ctx):
    return s18(log, window, dict(ctx, legend=legend_v4.parse(log)))
