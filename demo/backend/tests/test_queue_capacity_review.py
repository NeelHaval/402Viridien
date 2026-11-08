import io

from PIL import Image

from app.store import wells as repo
from app.tools.location import Location
from tests.test_backend_review import review_client


def test_retry_cannot_bypass_full_queue(review_client, monkeypatch):
    client, cfg = review_client
    ingest = client.app.state.ingest
    monkeypatch.setattr(ingest.jobs, 'enqueue', lambda well_id: None)
    monkeypatch.setattr(ingest, '_locate', lambda *args: Location())
    image = io.BytesIO()
    Image.new('RGB', (12, 12)).save(image, format='PNG')
    files = {'file': ('retry.png', image.getvalue(), 'image/png')}
    first = client.post('/api/upload', files=files)
    assert first.status_code == 200
    repo.mark_failed(client.app.state.db, first.json()['id'], 'previous failure')
    for index in range(16):
        repo.insert_queued(client.app.state.db, well_id=f'queued-{index}', file_hash=f'hash-{index}',
                           filename='other.png', name='other', scan_path='unused')
    retry = client.post('/api/upload', files=files)
    assert retry.status_code == 429, retry.text
    assert retry.json()['error']['code'] == 'queue_full'
    assert repo.get(client.app.state.db, first.json()['id'])['status'] == 'failed'
    assert not list(cfg.scans_dir.glob('.incoming-*'))
