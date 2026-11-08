"""Prevent the Clay/Shale group-label regression in new pipeline code."""
import pytest
from strategies.legend_identity import validate_prediction_labels


def test_unapproved_visual_group_cannot_replace_distinct_classes():
    rows = [dict(lithology='Mudstone / Wackstone')]
    with pytest.raises(ValueError, match='not a legend caption'):
        validate_prediction_labels(rows, ['Mudstone', 'Wackstone'])
    assert rows == [dict(lithology='Mudstone / Wackstone')]


def test_approved_clay_shale_family_is_accepted():
    validate_prediction_labels([dict(lithology='Clay / Shale')], ['Clay', 'Shale'])


def test_genuine_compound_caption_and_unknown_stay_valid():
    validate_prediction_labels(
        [dict(lithology='Clay / Shale'), dict(lithology='Unknown')],
        ['Clay / Shale'],
    )


def test_separate_predictions_keep_their_exact_captions():
    validate_prediction_labels([dict(lithology='Clay'), dict(lithology='Shale')], ['Clay', 'Shale'])
    with pytest.raises(ValueError, match='not a legend caption'):
        validate_prediction_labels([dict(lithology='Sandstone')], ['Sand and Sandstone'])
