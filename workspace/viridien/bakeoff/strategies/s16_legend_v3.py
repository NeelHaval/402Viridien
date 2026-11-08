"""S15 with opt-in complete-cell OCR legend names; shared parser unchanged."""
from harness import legend_v3
from strategies.s15_learned_match import run as s15

NAME = 'S16 learned boundaries + learned matching + legend v3'
NEEDS_GPU = False
PAID_API = None


def run(log, window, ctx):
    return s15(log, window, dict(ctx, legend=legend_v3.parse(log)))
