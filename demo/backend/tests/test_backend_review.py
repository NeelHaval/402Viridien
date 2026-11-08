import io
import json

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline.model import ModelState
from app.services.result_exports import save_result
from app.store import wells as repo


@pytest.fixture
def review_client(tmp_path, monkeypatch):
    monkeypatch.setattr('app.main.load_model', lambda: ModelState(True, engine='review'))
    cfg = Settings(data_dir=tmp_path / 'private', frontend_dist=tmp_path)
    with TestClient(create_app(cfg), raise_server_exceptions=False) as client:
        yield client, cfg


@pytest.mark.parametrize('metadata', ['NaN', 'Infinity', '1e999'])
def test_nonfinite_annotation_is_rejected_and_cleaned(review_client, metadata):
    client, cfg = review_client
    image = io.BytesIO()
    Image.new('RGB', (12, 12)).save(image, format='PNG')
    response = client.post('/api/upload', files={
        'file': ('test.png', image.getvalue(), 'image/png'),
        'bbox_json': ('test.json', '{"metadata": {"value": ' + metadata + '}}', 'application/json'),
    })
    assert response.status_code == 400, response.text
    assert response.json()['error']['code'] == 'invalid_bbox'
    assert not list(cfg.scans_dir.glob('.incoming-*'))


@pytest.mark.parametrize('polygon', [
    [[0, 0], [0, 0], [0, 0]],
    [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, 1], [0, 1], [1, 0]],
])
def test_zero_area_report_rejected(review_client, polygon):
    response = review_client[0].post('/api/report', json={'polygon': polygon})
    assert response.status_code == 422


def test_ready_native_rows_can_be_saved_without_pixel_coordinates(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    cfg.ensure_dirs()
    source = tmp_path / 'scan.png'
    Image.new('RGB', (12, 12)).save(source)
    save_result(cfg, 'native', source, [{'top_depth': 100., 'base_depth': 120., 'lithology': 'Shale'}], 'native', 'available')
    result = json.loads((cfg.csv_dir / 'native.result.json').read_text(encoding='utf-8'))
    assert result['rows'][0]['top_depth'] == 100.
    assert (cfg.csv_dir / 'native.annotation.png').is_file()


def test_partial_detail_uses_app_storage(review_client):
    client, cfg = review_client
    repo.insert_queued(client.app.state.db, well_id='partial', file_hash='hash', filename='a.png', name='a', scan_path='a.png')
    with client.app.state.db.connect() as conn:
        conn.execute("UPDATE wells SET status='partial' WHERE id='partial'")
    result = {'well_id': 'partial', 'depth_status': 'unavailable', 'rows': [
        {'top_y': 10, 'bottom_y': 20, 'lithology': 'Shale', 'cluster_id': 1}]}
    (cfg.csv_dir / 'partial.result.json').write_text(json.dumps(result), encoding='utf-8')
    response = client.get('/api/wells/partial')
    assert response.status_code == 200
    assert len(response.json()['pixel_column']) == 1
    assert response.json()['result']['depth_status'] == 'unavailable'
    assert response.json()['total_depth_ft'] is None
