"""Owner-approved Clay/Shale grading equivalence; raw predictions stay intact."""
import re

POLICY_ID = 'clay_shale_equivalent_v1'
EQUIVALENT_LABELS = ('Clay', 'Shale', 'Clay/Shale', 'Clay-Shale')
_KEYS = {'clay', 'shale', 'clay/shale', 'clay-shale'}


def canonical_grading_label(label):
    """Treat the approved names and compound spelling as one grading class."""
    key = re.sub(r'\s+', '', str(label)).casefold()
    return 'Clay/Shale' if key in _KEYS else label


def grading_rows(rows):
    """Copy labels for scoring; preserve every interval and all other fields."""
    return [dict(row, lithology=canonical_grading_label(row.get('lithology', 'Unknown')))
            for row in rows]


def score_equivalent(pred_rows, gt_rows, window, min_layer_px=0):
    """Use the unchanged shared scorer with symmetric approved class equality."""
    from harness.score import score
    return score(grading_rows(pred_rows), grading_rows(gt_rows), window,
                 min_layer_px=min_layer_px)
