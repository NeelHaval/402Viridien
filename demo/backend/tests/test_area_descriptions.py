"""Area-specific cache and grounding checks. All rows are generated fixtures."""
import json

import pytest

from app.area_descriptions import AreaRequest, area_facts, canonical_polygon
from tests.test_descriptions import saved, make_client

POLYGON = [[-1, -1], [1, -1], [1, 1], [-1, 1]]


def test_polygon_canonicalization():
    points = [tuple(p) for p in POLYGON]
    assert canonical_polygon(points) == canonical_polygon(points[1:] + points[:1])
    assert canonical_polygon(points) == canonical_polygon(list(reversed(points)))
    assert canonical_polygon(points) == canonical_polygon(points + points[:1])
    assert canonical_polygon(points) != canonical_polygon([(0, 0), (1, 0), (0, 1)])


@pytest.mark.parametrize('polygon', [[[0, 0]] * 3, [[0, 0], [181, 0], [1, 1]], [[0, 0], [float('inf'), 0], [1, 1]], POLYGON * 65])
def test_polygon_bounds(polygon):
    with pytest.raises(ValueError):
        AreaRequest(polygon=polygon)


def test_empty_area_is_deterministic_and_free(saved, tmp_path, monkeypatch):
    client, calls = make_client(saved, tmp_path, monkeypatch)
    response = client.post('/api/report/description', json={'polygon': POLYGON})
    assert response.status_code == 200 and response.json()['empty'] is True
    assert calls == []


def test_area_cache_and_polygon_variation(saved, tmp_path, monkeypatch):
    client, calls = make_client(saved, tmp_path, monkeypatch)
    with client.app.state.db.connect() as conn:
        conn.execute("UPDATE wells SET lat=0,lon=0")
    first = client.post('/api/report/description', json={'polygon': POLYGON})
    assert first.status_code == 200 and first.json()['well_count'] == 1
    assert 'selected area' in first.json()['note']
    assert first.json()['cached'] is False
    rotated = POLYGON[1:] + POLYGON[:1]
    assert client.post('/api/report/description', json={'polygon': rotated}).json()['cached'] is True
    assert len(calls) == 1
    wider = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    assert client.post('/api/report/description', json={'polygon': wider}).json()['cached'] is False
    assert len(calls) == 2
    saved[2].write_text('top_depth,base_depth,lithology\n100,200,Sandstone\n', encoding='utf-8')
    assert client.post('/api/report/description', json={'polygon': POLYGON}).json()['cached'] is False
    assert len(calls) == 3


def test_included_result_facts_and_auth(saved, tmp_path, monkeypatch):
    client, calls = make_client(saved, tmp_path, monkeypatch, signed=False)
    assert client.post('/api/report/description', json={'polygon': POLYGON}).status_code == 200
    with client.app.state.db.connect() as conn:
        conn.execute("UPDATE wells SET lat=0,lon=0")
    facts = area_facts(client.app.state.db, saved[1], [tuple(p) for p in POLYGON])
    assert facts['members'][0]['source_sha256'] == 'a' * 64
    assert len(facts['members'][0]['result_hash']) == 64
    assert facts['depth_range_ft'] == [100, 200]
    assert facts['lithology_shares'][0]['percent'] == 60
    assert str(saved[1]) not in json.dumps(facts)
    assert not calls
