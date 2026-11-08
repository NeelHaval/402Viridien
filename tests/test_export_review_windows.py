"""Check the review boundary and the existing manifest/pack contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest
from PIL import Image

from viridian402.data import load_intervals, load_manifest, sha256


def module(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


exporter = module('export_review_windows')
bundles = module('datasets')


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def exchange(tmp_path):
    document = {'id': 'doc1', 'image_name': 'page.tif', 'width': 100, 'height': 300,
                'revision': 7, 'boxes': {'interpreted_lithology_column': [10, 50, 30, 250]},
                'legend': [{'id': 2, 'label_printed': 'Shale'}],
                'segments': [dict(id='s1', box=[10, 90, 30, 120], legend_ids=[2],
                                  approved=False, status='suggested', tier='low',
                                  reason='Boundary is uncertain', comments=['Check the edge'])]}
    write(tmp_path / 'document.json', document)
    Image.new('RGB', (20, 40), '#cccccc').save(tmp_path / 'window.png')
    item = {'id': 'doc1-y100-140', 'document': 'document.json', 'document_id': 'doc1',
            'image': 'window.png', 'box': [10, 100, 30, 140],
            'document_sha256': sha256(tmp_path / 'document.json'),
            'window_image_sha256': sha256(tmp_path / 'window.png'),
            'parent': {'id': 'parent1', 'well': 'Well 1', 'group': 'well:one', 'split': 'dev',
                       'exposure_history': [{'kind': 'visible_development', 'note': 'Seen by the model'}]}}
    path = tmp_path / 'windows.json'
    write(path, {'schema_version': 1, 'windows': [item]})
    return path, item, document


def update(exchange, *, accept=False, review=False):
    path, item, document = exchange
    if accept:
        document['segments'][0].update(status='approved', approved=True, _reviewer='Reviewer A')
    write(path.parent / 'document.json', document)
    item['document_sha256'] = sha256(path.parent / 'document.json')
    if review:
        item['review'] = {'kind': 'human', 'reviewer': 'Reviewer A',
                          'reviewed_at': '2026-11-08T12:00:00Z',
                          'document_sha256': item['document_sha256']}
    write(path, {'schema_version': 1, 'windows': [item]})


def test_suggestion_stays_unlabelled_and_pack_keeps_provenance(exchange, tmp_path):
    path, item, _ = exchange
    manifest = exporter.export_windows(path, tmp_path / 'export')
    record = load_manifest(manifest)['records'][0]
    assert record['truth'] is None and record['source'] == 'external_unlabelled'
    assert record['original_coordinates']['box'] == item['box']
    assert record['lineage']['parent_split'] == 'dev'
    assert record['fresh_holdout'] is False
    provenance = exporter.read_json(manifest.parent / record['review_provenance'])
    assert provenance['segments'][0]['segment']['comments'] == ['Check the edge']
    assert provenance['segments'][0]['segment']['tier'] == 'low'
    archive = tmp_path / 'bundle.zip'
    bundles.pack([manifest], archive)
    with zipfile.ZipFile(archive) as bundle:
        assert 'source-snapshot.json' in bundle.namelist()
        assert record['review_provenance'] in bundle.namelist()
        bundle.extractall(tmp_path / 'unpacked')
    assert load_manifest(tmp_path / 'unpacked/all.json')['records'][0]['truth'] is None


def test_approved_flag_alone_is_not_human_attestation(exchange, tmp_path):
    update(exchange, accept=True)
    record = load_manifest(exporter.export_windows(exchange[0], tmp_path / 'export'))['records'][0]
    assert record['truth'] is None


def test_only_accepted_rows_enter_csv_with_page_offsets(exchange, tmp_path):
    path, item, document = exchange
    document['segments'].append(dict(id='s2', box=[10, 120, 30, 140], legend_ids=[2],
                                     status='suggested', approved=False))
    update(exchange, accept=True, review=True)
    manifest = exporter.export_windows(path, tmp_path / 'export')
    record = load_manifest(manifest)['records'][0]
    rows = load_intervals(record['truth'])
    assert rows == [{'top_y': 0, 'bottom_y': 20, 'lithology': 'Shale', 'cluster_id': 2}]
    assert record['truth_window'] == [0, 40] and record['source'] == 'human'
    assert record['label_provenance']['reviewer'] == 'Reviewer A'
    assert rows[0]['top_y'] + record['original_coordinates']['box'][1] == 100
    provenance = exporter.read_json(manifest.parent / record['review_provenance'])
    assert provenance['segments'][0]['segment']['box'] == [10, 90, 30, 120]
    assert provenance['segments'][1]['truth_exported'] is False
    archive = tmp_path / 'human.zip'
    bundles.pack([manifest], archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(tmp_path / 'human-relocated')
    relocated = load_manifest(tmp_path / 'human-relocated/all.json')['records'][0]
    assert load_intervals(relocated['truth']) == rows
    assert relocated['original_coordinates'] == record['original_coordinates']


@pytest.mark.parametrize('change', [
    {'flags': ['uncertain']}, {'unknown': True}, {'uncertain': True},
    {'legend_ids': [2, 3]}, {'box': [10, 90, 20, 120]},
    {'status': 'rejected', 'approved': False},
])
def test_uncertain_rejected_and_lateral_beds_have_no_truth(exchange, tmp_path, change):
    update(exchange, accept=True)
    exchange[2]['segments'][0].update(change)
    update(exchange, review=True)
    record = load_manifest(exporter.export_windows(exchange[0], tmp_path / 'export'))['records'][0]
    assert record['truth'] is None


@pytest.mark.parametrize('split', ['test', 'verify'])
def test_holdout_parent_is_rejected_before_document_read(exchange, tmp_path, split):
    path, item, _ = exchange
    item['parent']['split'] = split
    item['document'] = 'must-not-open.json'
    write(path, {'schema_version': 1, 'windows': [item]})
    with pytest.raises(ValueError, match='train/dev parent split'):
        exporter.export_windows(path, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_changed_saved_document_cannot_use_old_review(exchange, tmp_path):
    update(exchange, accept=True, review=True)
    exchange[2]['revision'] += 1
    update(exchange)
    with pytest.raises(ValueError, match='saved document hash'):
        exporter.export_windows(exchange[0], tmp_path / 'export')


def test_image_checksum_and_native_crop_dimensions(exchange, tmp_path):
    Image.new('RGB', (10, 20)).save(tmp_path / 'window.png')
    with pytest.raises(ValueError, match='native window crop'):
        exporter.export_windows(exchange[0], tmp_path / 'export')


def test_overlapping_accepted_beds_fail(exchange, tmp_path):
    update(exchange, accept=True)
    exchange[2]['segments'].append({**exchange[2]['segments'][0], 'id': 'duplicate'})
    update(exchange, review=True)
    with pytest.raises(ValueError, match='non-overlapping'):
        exporter.export_windows(exchange[0], tmp_path / 'export')


def test_existing_combined_catalog_blocks_cross_split_group(exchange, tmp_path):
    path, item, _ = exchange
    page=Image.new('RGB', (100, 300))
    with Image.open(tmp_path/'window.png') as crop:page.paste(crop,(10,100))
    page.save(tmp_path / 'page.tif')
    write(tmp_path / 'annotation.json', {'width': 100, 'height': 300,
          'bounding_boxes': {'interpreted_lithology_column': [10, 50, 30, 250]}})
    parent = {'id': 'parent1', 'image': 'page.tif', 'annotation': 'annotation.json', 'truth': None,
              'source': 'external_unlabelled', 'split': 'dev', 'group': 'well:one', 'window': [50, 250]}
    catalog = tmp_path / 'catalog.json'
    write(catalog, {'schema_version': 1, 'records': [parent]})
    manifest = exporter.export_windows(path, tmp_path / 'valid', catalog)
    snapshot = exporter.read_json(manifest.parent / 'source-snapshot.json')
    assert snapshot['catalog_validation'] == 'passed'
    item['parent']['split'] = 'train'
    write(path, {'schema_version': 1, 'windows': [item]})
    with pytest.raises(ValueError, match='group/split differs'):
        exporter.export_windows(path, tmp_path / 'invalid', catalog)
    # A sealed catalog must fail before any asset or CSV is read.
    parent.update(split='verify', truth='sealed-must-not-open.csv')
    write(catalog, {'schema_version': 1, 'records': [parent]})
    with pytest.raises(ValueError, match='truth must stay closed'):
        exporter.export_windows(path, tmp_path / 'sealed', catalog)


def test_shared_validator_blocks_group_split_leakage_between_windows(exchange, tmp_path):
    path, item, _ = exchange
    other = {**item, 'id': 'other', 'parent': {**item['parent'], 'split': 'train'}}
    write(path, {'schema_version': 1, 'windows': [item, other]})
    with pytest.raises(ValueError, match='leakage across splits'):
        exporter.export_windows(path, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_changed_image_hash_and_missing_history_fail(exchange, tmp_path):
    path, item, _ = exchange
    Image.new('RGB', (20, 40), 'red').save(tmp_path / 'window.png')
    with pytest.raises(ValueError, match='changed window_image_sha256'):
        exporter.export_windows(path, tmp_path / 'changed')
    item['parent']['exposure_history'] = []
    write(path, {'schema_version': 1, 'windows': [item]})
    with pytest.raises(ValueError, match='exposure_history'):
        exporter.export_windows(path, tmp_path / 'missing-history')


def test_existing_export_is_not_overwritten(exchange, tmp_path):
    output = tmp_path / 'export'
    exporter.export_windows(exchange[0], output)
    digest = sha256(output / 'all.json')
    with pytest.raises(ValueError, match='new directory'):
        exporter.export_windows(exchange[0], output)
    assert sha256(output / 'all.json') == digest
