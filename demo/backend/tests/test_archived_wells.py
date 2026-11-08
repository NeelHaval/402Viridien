from app.store.db import Database
from app.store import wells
from app.descriptions import FactError,build_facts
import pytest


def test_archive_hides_without_deleting_source_record(tmp_path):
    db=Database(tmp_path/'app.db');db.migrate()
    for well_id in ('test','real'):
        wells.insert_queued(db,well_id=well_id,file_hash=well_id,filename=well_id,scan_path='/private/'+well_id,name=well_id)
    with db.connect() as conn:
        conn.execute("UPDATE wells SET archived_at='archive-receipt' WHERE id='test'")
    assert [w['id'] for w in wells.list_wells(db)]==['real']
    assert [w['id'] for w in wells.list_wells(db,'queued')]==['real']
    retained=wells.get(db,'test')
    assert retained['scan_path']=='/private/test' and retained['archived_at']
    with pytest.raises(FactError,match='No saved well'):
        build_facts(retained,tmp_path)
