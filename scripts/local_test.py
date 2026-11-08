"""Create an isolated environment and run the local test workflow."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info<(3,10):
        raise RuntimeError('Python 3.10 or later is required. Python 3.12 is the tested version.')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--no-bootstrap', action='store_true')
    parser.add_argument('--software-tests', action='store_true')
    args, forwarded = parser.parse_known_args()
    if '--help' in forwarded or '-h' in forwarded:
        sys.path.insert(0, str(ROOT/'src'))
        from viridian402.local_test import main as run
        return run(forwarded)
    python = Path(sys.executable)
    if not args.no_bootstrap:
        environment = ROOT/'.venv-local-test'
        python = environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not python.is_file():
            print('Creating .venv-local-test', flush=True)
            venv.create(environment, with_pip=True)
        stamp = hashlib.sha256((ROOT/'requirements.lock').read_bytes() +
                               (ROOT/'pyproject.toml').read_bytes() + str(args.software_tests).encode()).hexdigest()
        marker = environment/'.local-test-install'
        if not marker.is_file() or marker.read_text(encoding='utf-8') != stamp:
            target = '.[test]' if args.software_tests else '.'
            subprocess.run([str(python),'-m','pip','install','-c','requirements.lock','-e',target], cwd=ROOT, check=True)
            marker.write_text(stamp, encoding='utf-8')
    if args.software_tests:
        return subprocess.run([str(python),'stress_testing/run_stress.py','--profile','quick','--include-regressions'], cwd=ROOT).returncode
    env = {**os.environ, 'PYTHONPATH':str(ROOT/'src')}
    return subprocess.run([str(python),'-X','utf8','-m','viridian402.local_test',*forwarded], cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Local test setup failed: {error}', file=sys.stderr)
        raise SystemExit(1)
