"""S21 DINO fusion with separately printed legend labels kept distinct.

This is a new evaluation candidate. The original S21 and its frozen v4
adapter remain unchanged. Repeated swatches for the same caption are still
aggregated; distinct captions are never joined by visual twin detection.
"""
from strategies import scoreboard_v4 as adapter
from strategies import s21_v4dino as dino

NAME = 'S23 v4 beds + DINOv2 with distinct legend identities'
NEEDS_GPU = True
PAID_API = None


def preserve_caption_identity(classifier):
    """Retain the text of each legend caption as its matching identity."""
    classifier.twins = lambda _log: {}
    return classifier


def run(log, window, ctx):
    classifier = preserve_caption_identity(adapter.make_classifier(log, ctx))
    swatch_labels, swatch_images = dino._swatches(classifier, log.name)
    chunks = []

    def collect(image, _log, alpha=None, unknown=None, check=True):
        a, state = dino._state(classifier, image, check)
        if state == 'ok' and swatch_images:
            chunks.append(a.copy())
        return dict(state=state if state != 'ok' else 'unknown', label=None)

    classifier.classify = collect
    adapter.segment(log, ctx, classifier=classifier, window=window)
    vectors = dino.embed(swatch_images + chunks) if chunks else None
    legend_vectors = vectors[:len(swatch_images)] if vectors is not None else None
    chunk_vectors = iter(vectors[len(swatch_images):]) if vectors is not None else iter(())
    used = 0

    def classify(image, name, alpha=None, unknown=None, check=True):
        nonlocal used
        _, state = dino._state(classifier, image, check)
        if state != 'ok':
            return dict(state=state, label=None)
        if not swatch_images:
            return dict(state='unknown', label=None)
        used += 1
        return dino._classify(classifier, image, name, next(chunk_vectors), swatch_labels,
                              legend_vectors, alpha=alpha, unknown=unknown, check=check)

    classifier.classify = classify
    result = adapter.segment(log, ctx, classifier=classifier, window=window)
    if used != len(chunks):
        raise RuntimeError('Bed chunks changed between collection and classification')
    run.last_diagnostics = adapter.diagnostics(log, classifier, result)
    run.last_diagnostics.update(
        dino_swatch_vectors=len(swatch_images), dino_chunk_vectors=used,
        model=dino._model()[3] if vectors is not None else None,
        fusion='z(hand distance) - z(DINO similarity)', column_resize=False,
        swatch_scales=list(dino.SCALES), legend_identity='exact printed caption',
        implementation_sha256=dino._sha256(__file__),
        dino_helper_sha256=dino._sha256(dino.__file__))
    return adapter.to_rows(result, window, ctx['calib'])
