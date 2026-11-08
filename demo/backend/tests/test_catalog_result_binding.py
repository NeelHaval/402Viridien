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
    matches=completed_jobs(db,SimpleNamespace(csv_dir=tmp_path,scans_dir=tmp_path/'scans'))
    assert matches[('a'*64,box_key(boxes))]['id']=='correct'
    assert len(matches)==2


def test_actual_store_null_boxes_uses_versioned_sidecar(tmp_path):
    db=Database(tmp_path/'app.db');db.migrate();scans=tmp_path/'scans';scans.mkdir()
    wells.insert_queued(db,well_id='real',file_hash='identity',filename='unused-name.tif',name='unused-name',scan_path='/private')
    with db.connect() as conn:conn.execute("UPDATE wells SET status='partial' WHERE id='real'")
    (tmp_path/'real.result.json').write_text(json.dumps({'engine':'S19','source_image_sha256':'b'*64,'boxes':None}),encoding='utf-8')
    sidecar=scans/'real.bbox.json';first={'interpreted_lithology_column':[1,2,3,4]};second={'interpreted_lithology_column':[2,2,3,4]}
    sidecar.write_text(json.dumps({'bounding_boxes':first,'image_name':'metadata-only'}),encoding='utf-8')
    settings=SimpleNamespace(csv_dir=tmp_path,scans_dir=scans)
    assert completed_jobs(db,settings)[('b'*64,box_key(first))]['id']=='real'
    sidecar.write_text(json.dumps({'bounding_boxes':second}),encoding='utf-8')
    updated=completed_jobs(db,settings)
    assert ('b'*64,box_key(first)) not in updated
    assert updated[('b'*64,box_key(second))]['id']=='real'


def test_bbox_sidecar_symlink_is_not_read(tmp_path):
    db=Database(tmp_path/'app.db');db.migrate();scans=tmp_path/'scans';scans.mkdir()
    wells.insert_queued(db,well_id='real',file_hash='identity',filename='unused.tif',name='unused',scan_path='/private')
    with db.connect() as conn:conn.execute("UPDATE wells SET status='partial' WHERE id='real'")
    (tmp_path/'real.result.json').write_text(json.dumps({'engine':'S19','source_image_sha256':'b'*64,'boxes':None}),encoding='utf-8')
    outside=tmp_path/'private.json';outside.write_text('{"column":[1,2,3,4]}',encoding='utf-8')
    (scans/'real.bbox.json').symlink_to(outside)
    assert completed_jobs(db,SimpleNamespace(csv_dir=tmp_path,scans_dir=scans))=={}
