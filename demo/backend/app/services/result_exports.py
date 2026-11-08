"""Generated result exports. Unknown depths remain empty, never zero."""
from pathlib import Path
import csv, hashlib, io, json, html, math
from PIL import Image, ImageDraw

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_result(settings, well_id, source, rows, engine, depth_status, bbox=None, details=None):
    root=settings.csv_dir
    result={"well_id":well_id,"engine":engine,"source_image_sha256":sha(source),"depth_status":depth_status,
            "source_path":str(source),"rows":rows,"coordinate_system":"full-image pixels; bottom exclusive",
            "boxes":bbox,"model_source_pins":(details or {}).get('pins',{}),"automatic_fallback":False}
    result['model_identity_sha256']=hashlib.sha256(json.dumps(result['model_source_pins'],sort_keys=True).encode()).hexdigest()
    diagnostic=(details or {}).get('depth',{})
    fit=diagnostic.get('estimate',{})
    if (diagnostic.get('output_unit')=='ft (assumed, OCR)' and diagnostic.get('unit_association_verified') is False
        and all(isinstance(fit.get(k),(int,float)) and math.isfinite(fit[k]) for k in ('a','b','n_points','rmse_px'))
        and fit['a']>0 and fit['n_points']>=3 and 0<=fit['rmse_px']<=4):
        result['depth_estimate']={'unit':'ft (assumed, OCR)','unit_association_verified':False,
                                  'fit':fit,'source_image_sha256':result['source_image_sha256'],
                                  'calibration_sha256':hashlib.sha256(json.dumps(fit,sort_keys=True,allow_nan=False).encode()).hexdigest()}
    encoded=json.dumps(result,sort_keys=True,allow_nan=False).encode()
    result['result_sha256']=hashlib.sha256(encoded).hexdigest()
    (root/f'{well_id}.result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if depth_status=='unavailable':
        (root/f'{well_id}.csv').write_text(export_csv(result,'notebook'),encoding='utf-8')
    # The existing TIFF reader decodes bounded bands. Never decode a huge full
    # page into RGB merely to make a one-megapixel preview.
    from app.tools.read_upload import read_upload
    scan=read_upload(str(source))
    try:
        original_height=scan.height
        scale=min(1.0,700/scan.width,1400/scan.height)
        array=scan.reader.crop(0,scan.height,0,scan.width,scale=scale)
        if array.ndim==3 and array.shape[2]==1:array=array[:,:,0]
        im=Image.fromarray(array).convert('RGB')
    finally:
        scan.close()
    try:
        draw=ImageDraw.Draw(im)
        scale=im.height/original_height
        for row in rows:
            # Native depth rows do not carry pixel coordinates. Do not invent them.
            if row.get('top_y') is None or row.get('bottom_y') is None:
                continue
            y0,y1=int(row['top_y']*scale),int(row['bottom_y']*scale)
            draw.rectangle((0,y0,im.width-1,y1),outline='#e24c28',width=2)
            text=f"{row['top_y']}–{row['bottom_y']} px  {row['lithology']}"
            draw.rectangle((2,y0, min(im.width, len(text)*7+8),y0+16),fill='white')
            draw.text((4,y0+2),text,fill='black')
        im.save(root/f'{well_id}.annotation.png')
    finally:
        im.close()

def export_csv(result, mode):
    bottom='bottom_depth' if mode=='notebook' else 'base_depth'
    fields=['top_depth',bottom,'lithology','top_y','bottom_y','cluster_id','depth_status','engine','source_image_sha256','result_sha256','model_identity_sha256']
    out=io.StringIO(); writer=csv.DictWriter(out,fields);writer.writeheader()
    for row in result['rows']:
        depth=result['depth_status']=='available'
        data={k:row.get(k,'') for k in ['lithology','top_y','bottom_y','cluster_id']}
        # Spreadsheet-safe text, while preserving original labels in result JSON.
        if str(data['lithology']).startswith(('=','+','-','@')):data['lithology']="'"+data['lithology']
        data.update(top_depth=row.get('top_depth','') if depth else '',**{bottom:row.get('bottom_depth',row.get('base_depth','')) if depth else ''},depth_status=result['depth_status'],engine=result['engine'],source_image_sha256=result['source_image_sha256'],result_sha256=result['result_sha256'],model_identity_sha256=result.get('model_identity_sha256',''))
        writer.writerow(data)
    return out.getvalue()
