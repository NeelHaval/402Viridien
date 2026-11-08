"""One bounded local API pass, explicitly refusing any credential-file fallback."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BAKE = HERE.parents[2]
LAB = BAKE.parent / 'pattern_lab'


def main():
    if len(sys.argv) != 2:
        raise ValueError('Expected one frozen case ID')
    if not os.environ.get('JEV_API_KEY'):
        raise ValueError('Environment-only Jev credential is unavailable')
    sys.path.insert(0, str(LAB))
    sys.path.insert(0, str(BAKE))
    from strategies import _jev
    _jev._key = lambda: os.environ.get('JEV_API_KEY')
    import jev_boat_api
    jev_boat_api.run(sys.argv[1])


if __name__ == '__main__':
    main()
