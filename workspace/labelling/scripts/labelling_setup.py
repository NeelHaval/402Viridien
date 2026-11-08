"""Shared paths and first-run checks for a teammate's local checkout."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(os.environ.get('LITHOLOGY_DATASET', ROOT.parent / 'viridien' / 'dataset')).resolve()
LEGACY_DATASET = 'untitled folder/dataset'


def require_dependencies():
    missing = []
    for module, package in [('numpy', 'numpy'), ('PIL', 'Pillow'), ('cv2', 'opencv-python-headless')]:
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            missing.append(f'{package}: {exc}')
    if missing:
        raise SystemExit(
            'Labelling dependencies are unavailable:\n  ' + '\n  '.join(missing)
            + f'\nPython in use: {sys.executable}\n'
            'From the labelling directory, run:\n'
            '  source .venv/bin/activate\n'
            '  python -m pip install -r requirements.txt\n'
            'Do not continue to the server if installation or prep fails.'
        )


def missing_pages(dataset, stems):
    images = {p.stem for p in (dataset / 'images').glob('*')
              if p.suffix.lower() in ('.tif', '.tiff') and p.is_file()}
    return [s for s in stems if s not in images or not (dataset / 'bbox_annotation' / f'{s}.json').is_file()]


def ensure_dataset(dataset, stems, repo=None):
    """Recover missing inputs from the already-published legacy snapshot, without overwriting files."""
    repo = Path(repo) if repo is not None else ROOT.parent
    missing = missing_pages(dataset, stems)
    if not missing:
        return
    try:
        revision = subprocess.check_output(
            ['git', '-C', str(repo), 'log', '-1', '--format=%H', '--diff-filter=A',
             '--', f'{LEGACY_DATASET}/images'], text=True, stderr=subprocess.DEVNULL).strip()
        if not revision:
            raise ValueError('dataset snapshot is absent from local Git history')
        paths = subprocess.check_output(
            ['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', '-z', revision,
             '--', LEGACY_DATASET]).decode().split('\0')
        wanted = set(missing)
        inputs = [p for p in paths if p and Path(p).stem in wanted
                  and (Path(p).parent.as_posix() == f'{LEGACY_DATASET}/bbox_annotation'
                       or (Path(p).parent.as_posix() == f'{LEGACY_DATASET}/images'
                           and Path(p).suffix.lower() in ('.tif', '.tiff')))]
        if not inputs:
            raise ValueError('no matching inputs in the dataset snapshot')
        print(f'Restoring dataset inputs for {len(missing)} pages from local Git history to {dataset}.', flush=True)
        with subprocess.Popen(['git', '-C', str(repo), 'archive', revision, '--', *inputs],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
            allowed = set(inputs)
            with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                for member in archive:
                    if member.name not in allowed or not member.isfile():
                        continue
                    relative = Path(member.name).relative_to(LEGACY_DATASET)
                    target = dataset / relative
                    if not target.resolve().is_relative_to(dataset.resolve()):
                        raise ValueError('dataset path escapes its directory')
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(target.name + '.importing')
                    try:
                        with archive.extractfile(member) as source, temp.open('wb') as out:
                            shutil.copyfileobj(source, out)
                        temp.replace(target)
                    finally:
                        temp.unlink(missing_ok=True)
            if proc.wait():
                raise ValueError('git archive could not restore the dataset')
        missing = missing_pages(dataset, stems)
        if missing:
            raise ValueError(f'{len(missing)} pages still have missing inputs')
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        raise SystemExit(
            f'Dataset setup failed: {exc}\n'
            f'Put the organiser dataset (images/ and bbox_annotation/) in {dataset},\n'
            'or set LITHOLOGY_DATASET to an existing dataset directory.\n'
            'For a shallow clone, run git fetch --unshallow, then rerun prep.'
        ) from exc
