"""First-run behavior using an independent miniature Git checkout."""
import importlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import labelling_setup as setup


def test_missing_dependencies_explain_the_interpreter_and_repair(monkeypatch):
    real_import = importlib.import_module

    def missing_cv2(name):
        if name == 'cv2':
            raise ModuleNotFoundError("No module named 'cv2'")
        return real_import(name)

    monkeypatch.setattr(setup.importlib, 'import_module', missing_cv2)
    with pytest.raises(SystemExit) as error:
        setup.require_dependencies()
    assert 'opencv-python-headless' in str(error.value)
    assert sys.executable in str(error.value)
    assert 'python -m pip install -r requirements.txt' in str(error.value)


def test_fresh_checkout_recovers_inputs_without_overwriting_existing_work(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True,
                              capture_output=True, text=True)

    git('init', '--quiet')
    source = tmp_path / setup.LEGACY_DATASET
    (source / 'images').mkdir(parents=True)
    (source / 'bbox_annotation').mkdir()
    (source / 'images' / 'page with spaces.tiff').write_bytes(b'original scan')
    (source / 'bbox_annotation' / 'page with spaces.json').write_text('{"original":true}')
    git('add', '.')
    git('-c', 'user.name=Setup test', '-c', 'user.email=setup@example.test',
        'commit', '--quiet', '-m', 'legacy snapshot')
    shutil.rmtree(tmp_path / 'untitled folder')
    destination = tmp_path / 'viridien' / 'dataset'
    (destination / 'bbox_annotation').mkdir(parents=True)
    annotation = destination / 'bbox_annotation' / 'page with spaces.json'
    annotation.write_text('{"local":true}')
    setup.ensure_dataset(destination, ['page with spaces'], repo=tmp_path)
    assert annotation.read_text() == '{"local":true}'
    assert (destination / 'images' / 'page with spaces.tiff').read_bytes() == b'original scan'
    assert not setup.missing_pages(destination, ['page with spaces'])
    # Once restored, startup works even with no Git executable/history available.
    setup.ensure_dataset(destination, ['page with spaces'], repo=tmp_path / 'absent')
    assert not list(destination.rglob('*.importing'))


def test_missing_dataset_without_history_has_actionable_error(tmp_path):
    with pytest.raises(SystemExit, match='LITHOLOGY_DATASET'):
        setup.ensure_dataset(tmp_path / 'dataset', ['missing'], repo=tmp_path)
