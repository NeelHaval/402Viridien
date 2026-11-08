"""Grounded note tests: generated CSV fixtures only; no live provider calls."""
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app import providers
from app.descriptions import FactError, build_facts, validate_selection
from app.store.db import Database
from tests.test_providers import config, cookie


@pytest.fixture
def saved(tmp_path):
    folder = tmp_path / 'csv'
    folder.mkdir()
    path = folder / 'result.csv'
    path.write_text('top_depth,base_depth,lithology\n100,160,Sandstone\n160,200,Shale grey\n', encoding='utf-8')
    row = dict(id='well-test', status='ready', csv_path=str(path), file_hash='a' * 64,
               warnings=[], validation={}, location_source='header')
    return row, folder, path


def test_facts_from_saved_intervals(saved):
    row, folder, _ = saved
    facts, result_hash = build_facts(row, folder)
    assert facts['depth_range'] == [100, 200] and facts['depth_units'] == 'ft'
    assert facts['lithology_shares'][0] == {'lithology': 'Sandstone', 'percent': 60.0}
    assert facts['standout_transition'] == {'depth_ft': 160.0, 'from': 'Sandstone', 'to': 'Shale grey'}
    assert facts['missing_coverage_ft'] == 0 and len(result_hash) == 64
    assert 'csv_path' not in facts and 'scan_path' not in facts


@pytest.mark.parametrize('status,code', [('processing', 'well_not_ready'), ('queued', 'well_not_ready'), ('failed', 'well_failed')])
def test_incomplete_results_never_read(saved, status, code):
    row, folder, path = saved
    row['status'] = status
    path.unlink()
    with pytest.raises(FactError) as error:
        build_facts(row, folder)
    assert error.value.code == code


def test_depth_unknown_and_coverage(saved):
    row, folder, path = saved
    path.write_text('top_depth,base_depth,lithology\n,,Unknown\n', encoding='utf-8')
    facts, _ = build_facts(row, folder)
    assert facts['depth_range'] is None and facts['depth_units'] == 'unavailable'
    assert not facts['lithology_shares']
    assert 'depth is unavailable' in facts['choices']['highlights']['depth_unknown']
    path.write_text('top_depth,base_depth,lithology\n100,120,Unknown\n130,200,Shale grey\n', encoding='utf-8')
    facts, _ = build_facts(row, folder)
    assert facts['missing_coverage_ft'] == 10
    assert facts['unknown_or_blank_thickness_ft'] == 20


def test_overlaps_do_not_produce_shares(saved):
    row, folder, path = saved
    path.write_text('top_depth,base_depth,lithology\n100,180,Sandstone\n160,200,Shale grey\n', encoding='utf-8')
    facts, _ = build_facts(row, folder)
    assert not facts['lithology_shares']
    assert 'overlap' in facts['choices']['reviews']


@pytest.mark.parametrize('text', ['Invented confidence 99%', '{}', '{"highlight_id":"invented","review_id":"scan_check"}',
                                  '{"highlight_id":"range","review_id":"scan_check","text":"invented"}'])
def test_unproved_output_is_rejected(saved, text):
    row, folder, _ = saved
    facts, _ = build_facts(row, folder)
    with pytest.raises(FactError, match='unsupported note'):
        validate_selection(text, facts)


def test_safe_path_and_changed_hash(saved, tmp_path):
    row, folder, path = saved
    _, first = build_facts(row, folder)
    path.write_text('top_depth,base_depth,lithology\n100,200,Unknown\n', encoding='utf-8')
    _, second = build_facts(row, folder)
    assert first != second
    outside = tmp_path / 'private.csv'
    outside.write_text('never read', encoding='utf-8')
    row['csv_path'] = str(outside)
    with pytest.raises(FactError) as error:
        build_facts(row, folder)
    assert error.value.code == 'result_unavailable'


def make_client(saved, tmp_path, monkeypatch, *, signed=True, key='test-key', output=None):
    row, folder, path = saved
    db = Database(tmp_path / 'notes.db')
    db.migrate()
    with db.connect() as conn:
        conn.execute('INSERT INTO wells (id,status,created_at,file_hash,csv_path,location_source) VALUES (?,?,?,?,?,?)',
                     (row['id'], row['status'], 'public-test', row['file_hash'], str(path), 'header'))
    calls = []
    async def fake(conf, prompt, *, system=None):
        calls.append(prompt)
        assert str(folder) not in prompt and 'scan_path' not in prompt
        return {'answer': output or '{"highlight_id":"dominant","review_id":"scan_check"}'}
    monkeypatch.setattr(providers, 'ask_gemini', fake)
    app = FastAPI()
    app.state.db = db
    app.state.settings = SimpleNamespace(csv_dir=folder)
    providers.install(app, config(gemini_key=key, required=False))
    client = TestClient(app, base_url='https://app.example.test')
    if signed:
        client.cookies.set('landmarkr_session', cookie())
    return client, calls


def test_note_endpoint_and_persistent_cache(saved, tmp_path, monkeypatch):
    client, calls = make_client(saved, tmp_path, monkeypatch)
    first = client.post('/api/wells/well-test/description')
    assert first.status_code == 200
    assert ('Sandstone accounts for 60.0% of the saved interval thickness. '
                                    'Check the saved lithology labels and interval boundaries against the scan.') in first.json()['note']
    assert first.json()['cached'] is False
    assert client.post('/api/wells/well-test/description').json()['cached'] is True
    assert len(calls) == 1
    saved[2].write_text('top_depth,base_depth,lithology\n100,200,Sandstone\n', encoding='utf-8')
    assert client.post('/api/wells/well-test/description').json()['cached'] is False
    assert len(calls) == 2


@pytest.mark.parametrize('signed,key,status,code', [(True, '', 503, 'gemini_not_configured')])
def test_disabled_states_do_not_generate(saved, tmp_path, monkeypatch, signed, key, status, code):
    client, calls = make_client(saved, tmp_path, monkeypatch, signed=signed, key=key)
    result = client.post('/api/wells/well-test/description')
    assert result.status_code == status and result.json()['error']['code'] == code
    assert calls == []


def test_failed_result_and_invalid_output_do_not_cache(saved, tmp_path, monkeypatch):
    client, calls = make_client(saved, tmp_path, monkeypatch, output='{"highlight_id":"madeup","review_id":"scan_check"}')
    assert client.post('/api/wells/well-test/description').json()['fallback'] is True
    assert client.post('/api/wells/well-test/description').json()['fallback'] is True
    assert len(calls) == 2
    with client.app.state.db.connect() as conn:
        conn.execute("UPDATE wells SET status='failed'")
    response = client.post('/api/wells/well-test/description')
    assert response.status_code == 409 and response.json()['error']['code'] == 'well_failed'
    assert len(calls) == 2


def test_note_rate_limit_and_retired_chat(saved, tmp_path, monkeypatch):
    client, calls = make_client(saved, tmp_path, monkeypatch)
    for i in range(10):
        saved[2].write_text(f'top_depth,base_depth,lithology\n100,{200+i},Sandstone\n', encoding='utf-8')
        assert client.post('/api/wells/well-test/description').status_code == 200
    saved[2].write_text('top_depth,base_depth,lithology\n100,300,Sandstone\n', encoding='utf-8')
    assert client.post('/api/wells/well-test/description').status_code == 429
    assert len(calls) == 10
    assert client.post('/api/assistant', json={'question': 'test'}).status_code == 404


def test_oversized_model_output_is_rejected(saved):
    facts, _ = build_facts(saved[0], saved[1])
    with pytest.raises(FactError):
        validate_selection(' ' * 1001, facts)
