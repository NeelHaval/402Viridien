"""Isolated-mode entrypoint for the shared harness (trusted package path only)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emil.bootstrap import installed_arrow
installed_arrow()
from harness.run import main
if __name__ == '__main__': main()
