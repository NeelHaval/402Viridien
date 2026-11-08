"""Run the scoped identity and Jev checks inside the private CPU environment."""
from pathlib import Path
import sys
import worker

worker.init()
import pytest
raise SystemExit(pytest.main(['-q',str(Path(__file__).resolve().parent/'source/strategies/test_s30_jev_sourcecards.py'),str(Path(__file__).resolve().parent/'source/strategies/test_legend_identity.py'),str(Path(__file__).resolve().parent/'source/strategies/test_grading_policy.py')]))
