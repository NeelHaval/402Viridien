"""Contributor-facing regressions found during independent shipping review."""
import importlib.util
import json
from pathlib import Path
import struct
import zipfile

import pytest

from viridian402.cli import doctor, main


@pytest.mark.parametrize('command', ['evaluate', 'classify'])
def test_doctor_and_execution_accept_same_bom_config(tmp_path, monkeypatch, command):
    config = tmp_path / 'windows-config.json'
    config.write_text('{"seed": 2026}', encoding='utf-8-sig')
    assert doctor(config=config)['ready']
    if command == 'evaluate':
        from viridian402 import evaluate as module
        function, arguments = 'evaluate', ['--manifest', 'unused.json']
    else:
        from viridian402 import crops as module
        function, arguments = 'classify', ['--data-root', 'unused']
    received = []
    def run(*args, **kwargs):
        received.append(kwargs['config'])
        return tmp_path, {'overall': {}, 'quality_gates': {'passed': True}}
    monkeypatch.setattr(module, function, run)
    assert main([command, *arguments, '--config', str(config)]) == 0
    assert received == [{'seed': 2026}]


@pytest.fixture
def bundle_tools():
    spec = importlib.util.spec_from_file_location(
        'independent_shipping_bundles', Path(__file__).parents[1] / 'scripts/datasets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrupt_archive_crc_preserves_existing_and_new_files(tmp_path, bundle_tools):
    archive = tmp_path / 'crc.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr('new.txt', b'new')
        bundle.writestr('existing.txt', b'keep')
        bundle.writestr('corrupt.txt', b'payload')
    with zipfile.ZipFile(archive) as bundle:
        entry = bundle.getinfo('corrupt.txt')
    content = bytearray(archive.read_bytes())
    name_len, extra_len = struct.unpack_from('<HH', content, entry.header_offset + 26)
    offset = entry.header_offset + 30 + name_len + extra_len
    content[offset] ^= 1
    archive.write_bytes(content)
    destination = tmp_path / 'data'
    destination.mkdir()
    (destination / 'existing.txt').write_bytes(b'keep')
    with pytest.raises(zipfile.BadZipFile, match='CRC'):
        bundle_tools.safe_extract(archive, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == {'existing.txt': b'keep'}


def test_conflicting_download_preserves_local_edits_before_committing(tmp_path, bundle_tools):
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('new.txt', b'new')
        bundle.writestr('existing.txt', b'upstream')
    destination = tmp_path / 'data'
    destination.mkdir()
    (destination / 'existing.txt').write_bytes(b'local edit')
    with pytest.raises(ValueError, match='Existing dataset file differs'):
        bundle_tools.safe_extract(archive, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == {'existing.txt': b'local edit'}


def test_pack_cannot_destroy_its_own_source_manifest(tmp_path, bundle_tools):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'records': []}))
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        bundle_tools.pack([manifest], manifest)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize('value', [None, [], 1])
def test_recursive_runs_skips_malformed_crop_summaries(tmp_path, capsys, value):
    directory = tmp_path / 'nested' / 'broken-run'
    directory.mkdir(parents=True)
    (directory / 'summary.json').write_text(json.dumps(value))
    assert main(['runs', '--output', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == []
