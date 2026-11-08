import json
from app.api import wells


def test_partial_column_keeps_real_pixels_and_no_invented_depths(tmp_path, monkeypatch):
    rows=[{'top_y':100+i*10,'bottom_y':110+i*10,'lithology':'Sandstone' if i%2 else 'Shale grey','cluster_id':i%2} for i in range(145)]
    path=tmp_path/'result.json'
    path.write_text(json.dumps({'rows':rows,'depth_status':'unavailable'}),encoding='utf-8')
    monkeypatch.setattr(wells,'generated_path',lambda row,suffix:path)
    result=wells.pixel_column({'id':'test','status':'partial'})
    assert len(result)==145
    assert [(r['top_y'],r['bottom_y'],r['lithology'],r['cluster_id']) for r in result]==[(r['top_y'],r['bottom_y'],r['lithology'],r['cluster_id']) for r in rows]
    assert all(set(r)=={'top_y','bottom_y','lithology','cluster_id','color'} for r in result)
    assert all(r['color'].startswith('#') for r in result)
