"""Whole-legend nearest-swatch classifier and v4 bed segmentation, frozen as-is."""
from strategies import scoreboard_v4 as adapter

NAME = 'S20 v4 whole-legend classifier + beds (frozen)'
NEEDS_GPU = False
PAID_API = None


def run(log, window, ctx):
    classifier = adapter.make_classifier(log, ctx)
    result = adapter.segment(log, ctx, classifier=classifier, window=window)
    # Diagnostics are attached for runners that preserve them; stages stay empty
    # because harness.viz expects all non-private stage values to be images.
    run.last_diagnostics = adapter.diagnostics(log, classifier, result)
    return adapter.to_rows(result, window, ctx['calib'])
