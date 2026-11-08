from types import SimpleNamespace
import json
from app.store.db import Database
from app.store import wells
from app.api.catalog import completed_jobs,box_key


def test_catalog_links_by_source_and_all_boxes_not_name(tmp_path):
    db=Database(tmp_path/'app.db');db.migrate()
    boxes={'interpreted_lithology_column':[1,2,3,4],'depth_column':[5,6,7,8]}
    for well_id,actual in [('correct',boxes),('wrong-box',{**boxes,'depth_column':[6,6,7,8]})]:
        wells.insert_queued(db,well_id=well_id,file_hash=well_id,filename='same-name.tif',name='same-name',scan_path='/private')
        with db.connect() as conn:conn.execute("UPDATE wells SET status='partial' WHERE id=?",(well_id,))
        (tmp_path/(well_id+'.result.json')).write_text(json.dumps({'engine':'S19','source_image_sha256':'a'*64,'boxes':{'bounding_boxes':actual,'image_name':'metadata-only'}}),encoding='utf-8')
    matches=completed_jobs(db,SimpleNamespace(csv_dir=tmp_path))
    assert matches[('a'*64,box_key(boxes))]['id']=='correct'
    assert len(matches)==2
