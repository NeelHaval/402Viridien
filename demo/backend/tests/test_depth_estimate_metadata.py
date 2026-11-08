import json,csv,io
from PIL import Image
from app.config import Settings
from app.services.result_exports import save_result


def test_assumed_fit_is_metadata_not_verified_csv_depth(tmp_path):
    cfg=Settings(data_dir=tmp_path);cfg.ensure_dirs();source=tmp_path/'source.png';Image.new('RGB',(20,60),'white').save(source)
    rows=[{'top_y':10,'bottom_y':50,'lithology':'Shale','cluster_id':0}]
    details={'depth':{'output_unit':'ft (assumed, OCR)','unit_association_verified':False,'estimate':{'a':.5,'b':100,'n_points':4,'rmse_px':1}}}
    save_result(cfg,'estimate',source,rows,'S19','unavailable',details=details)
    result=json.loads((cfg.csv_dir/'estimate.result.json').read_text(encoding='utf-8'))
    assert result['depth_estimate']['unit']=='ft (assumed, OCR)'
    assert result['depth_estimate']['source_image_sha256']==result['source_image_sha256']
    assert result['depth_status']=='unavailable' and result['rows']==rows
    record=next(csv.DictReader(io.StringIO((cfg.csv_dir/'estimate.csv').read_text(encoding='utf-8'))))
    assert record['top_depth']=='' and record['bottom_depth']==''
