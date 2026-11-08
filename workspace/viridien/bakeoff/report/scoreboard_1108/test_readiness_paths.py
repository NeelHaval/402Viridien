"""Check that readiness cannot read arbitrary paths via a portable CSV receipt."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from scan_readiness import validated_prediction_path


def test_local_and_remote_prediction_paths_are_allowed(tmp_path):
    local=tmp_path/'out'/'scoreboard_1108';remote=tmp_path/'report'/'remote'
    roots=(local,remote)
    assert validated_prediction_path(local/'candidate'/'prediction.csv',roots)==(local/'candidate'/'prediction.csv').resolve()
    assert validated_prediction_path(remote/'receipt'/'prediction.csv',roots)==(remote/'receipt'/'prediction.csv').resolve()


def test_traversal_absolute_and_prefix_collision_paths_are_rejected(tmp_path):
    local=tmp_path/'out'/'scoreboard_1108';remote=tmp_path/'report'/'remote';roots=(local,remote)
    for attempted in (local/'..'/'outside.csv',remote/'..'/'outside.csv',tmp_path/'private.csv',local.with_name('scoreboard_1108_other')/'prediction.csv'):
        with pytest.raises(ValueError,match='outside the approved'):validated_prediction_path(attempted,roots)
