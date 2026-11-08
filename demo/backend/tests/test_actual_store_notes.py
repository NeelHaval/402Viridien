import csv
import pytest
from app.descriptions import build_facts, FactError
from tests.test_descriptions import saved, make_client


@pytest.mark.parametrize('bottom', ['bottom_depth', 'base_depth'])
@pytest.mark.parametrize('numeric', [False, True])
def test_partial_actual_store_shape(tmp_path, bottom, numeric):
    root = tmp_path / 'generated'
    root.mkdir()
    source = 'a' * 64
    with (root / 'abc123.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['top_depth', bottom, 'lithology', 'top_y', 'bottom_y', 'source_image_sha256'])
        writer.writerow([100 if numeric else '', 200 if numeric else '', 'Sandstone', 0, 60, source])
        writer.writerow([200 if numeric else '', 300 if numeric else '', 'Shale', 60, 100, source])
    row = {'id': 'abc123', 'status': 'partial', 'csv_path': None,
           'file_hash': 'job-identity-is-not-source-image', 'warnings': [], 'validation': {}}
    facts, digest = build_facts(row, root)
    assert facts['source_sha256'] == source
    assert facts['depth_range'] is None and facts['depth_units'] == 'unavailable'
    assert facts['pixel_lithology_shares'][0]['percent'] == 60
    assert facts['interval_count'] == 2 and len(digest) == 64
    assert 'depth_unknown' in facts['choices']['highlights']
    assert 'range' not in facts['choices']['highlights']


def test_partial_fallback_path_cannot_escape(tmp_path):
    outside = tmp_path / 'outside.csv'
    outside.write_text('top_depth,bottom_depth,lithology\n,,Shale\n', encoding='utf-8')
    root = tmp_path / 'generated'
    root.mkdir()
    with pytest.raises(FactError, match='unavailable'):
        build_facts({'id': '../outside', 'status': 'partial', 'csv_path': None}, root)


def test_mixed_source_rows_rejected(tmp_path):
    path = tmp_path / 'abc.csv'
    path.write_text('top_depth,bottom_depth,lithology,source_image_sha256\n,,Shale,a\n,,Sandstone,b\n', encoding='utf-8')
    with pytest.raises(FactError, match='mixes source'):
        build_facts({'id': 'abc', 'status': 'partial', 'csv_path': None}, tmp_path)


def test_actual_partial_store_well_and_area_routes(saved, tmp_path, monkeypatch):
    row, root, old = saved
    row['status'] = 'partial'
    row['csv_path'] = None
    generated = root / (row['id'] + '.csv')
    generated.write_text('top_depth,bottom_depth,lithology,top_y,bottom_y,source_image_sha256\n,,Sandstone,0,60,'+'b'*64+'\n,,Shale,60,100,'+'b'*64+'\n', encoding='utf-8')
    client, calls = make_client(saved, tmp_path, monkeypatch, signed=False,
                                output='{"highlight_id":"dominant","review_id":"depth_check"}')
    with client.app.state.db.connect() as conn:
        conn.execute("UPDATE wells SET lat=0,lon=0,status='partial',csv_path=NULL")
    first = client.post('/api/wells/well-test/description')
    assert first.status_code == 200, first.text
    assert not first.json()['fallback'] and 'depth is unavailable' in first.json()['note']
    assert client.post('/api/wells/well-test/description').json()['cached']
    # Area choices use a pooled-pixel ID rather than the per-well dominant ID.
    monkeypatch.setattr('app.providers.ask_gemini', lambda *a, **k: None)
    from app.area_descriptions import area_facts
    facts = area_facts(client.app.state.db, root, [(-1,-1),(1,-1),(1,1),(-1,1)])
    assert facts['well_count'] == 1 and facts['wells_with_depth_unavailable'] == 1
    assert facts['members'][0]['source_sha256'] == 'b'*64
    assert facts['pooled_pixel_shares'][0]['percent'] == 60
