import importlib.util
from pathlib import Path
import stat
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("datasets", Path(__file__).parents[1] / "scripts/datasets.py")
datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datasets)


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "..\\outside"])
def test_archive_rejects_paths_before_writing(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("first.txt", "safe")
        entry = zipfile.ZipInfo("placeholder")
        entry.filename = name
        bundle.writestr(entry, "bad")
    with pytest.raises(ValueError):
        datasets.safe_extract(archive, tmp_path / "result")
    assert not (tmp_path / "result" / "first.txt").exists()


def test_archive_rejects_symlink_and_oversize(tmp_path):
    archive = tmp_path / "bad.zip"
    entry = zipfile.ZipInfo("link")
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(entry, "destination")
    with pytest.raises(ValueError):
        datasets.safe_extract(archive, tmp_path / "result")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("large", "12345")
    with pytest.raises(ValueError):
        datasets.safe_extract(archive, tmp_path / "result", max_bytes=4)
