"""Check output classes, including the explicitly approved Clay/Shale family."""
from strategies.grading_policy import canonical_grading_label


def validate_prediction_labels(rows, captions):
    """Accept printed captions and approved equivalents; reject other new groups.

    This function validates only. It does not change rows or infer synonyms.
    A correct class identity can still be a wrong classification.
    """
    allowed = set(captions)
    if any(not isinstance(label, str) or not label for label in allowed):
        raise ValueError('Legend captions must be nonempty strings')
    allowed = {canonical_grading_label(label) for label in allowed}
    allowed.add('Unknown')
    for index, row in enumerate(rows):
        label = row.get('lithology')
        if canonical_grading_label(label) not in allowed:
            raise ValueError(
                f'Prediction row {index} has label {label!r}, which is not a legend caption. '
                'Use a printed caption, an approved grading equivalent, or Unknown.'
            )
