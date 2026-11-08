import pytest

from app.store import wells as repo
from tests.test_backend_review import review_client


@pytest.mark.parametrize('status', ['queued', 'processing', 'failed'])
def test_unfinished_result_file_is_not_read(review_client, status):
    client, cfg = review_client
    repo.insert_queued(client.app.state.db, well_id='writing', file_hash='writing', filename='a.png', name='a', scan_path='a.png')
    with client.app.state.db.connect() as conn:
        conn.execute('UPDATE wells SET status=? WHERE id=?', (status, 'writing'))
    # Model the visible bytes during save_result.write_text, or after a failed write.
    (cfg.csv_dir / 'writing.result.json').write_text('{"rows": [', encoding='utf-8')
    response = client.get('/api/wells/writing')
    assert response.status_code == 200, response.text
    assert response.json()['status'] == status
    assert response.json()['result'] is None
