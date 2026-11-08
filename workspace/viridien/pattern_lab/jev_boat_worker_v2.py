"""Add source-only CPU legend/depth preparation for the remaining cases."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import worker

ROOT=Path(__file__).resolve().parent
original_init=worker.init


def init_with_preprocessing():
    io,adapter,api,env=original_init()
    if sys.argv[1]!='export':return io,adapter,api,env
    from PIL import Image
    from harness import depth,legend
    case_id=sys.argv[2];case=worker.read(ROOT/'cases'/f'{case_id}.json');start=time.monotonic()
    for p,h in [(case['remote_image'],case['image_sha256']),(case['remote_annotation'],case['annotation_sha256'])]:
        if worker.sha(Path(p))!=h:raise ValueError('Preprocessing input hash mismatch')
    # The private OCR model files must match the reviewed real-job hashes.
    model_dir=Path(importlib.util.find_spec('rapidocr_onnxruntime').origin).parent/'models'
    models={}
    for e in worker.read(ROOT/'ocr-assets.json'):
        path=model_dir/e['name'];actual=worker.sha(path)
        if actual!=e['sha256']:raise ValueError('OCR model hash mismatch '+e['name'])
        models[e['name']]=actual
    log=io.Log(Image.open(case['remote_image']),worker.read(case['remote_annotation'])['bounding_boxes'],case['log'])
    native=adapter._native();legend_seconds=0.;parsed=[]
    if not native.get(log.name):
        t=time.monotonic();parsed=legend.parse(log);legend_seconds=time.monotonic()-t
        augmented=dict(native);augmented[log.name]=[(e.label,e.swatch) for e in parsed if e.is_lithology and e.swatch is not None]
        adapter._native=lambda:augmented
    calibration=None;depth_error=None;cache_verified=False;t=time.monotonic()
    candidate=ROOT/'depth-inputs'/f'{case_id}.json'
    if candidate.exists():
        cached=worker.read(candidate)
        if cached.get('version')==depth.VERSION and cached.get('signature')==depth.cache_signature(log):
            from math import isfinite
            c=cached['calibration']
            if c['a']>0 and all(isfinite(c[k]) for k in ['a','b','rmse_px']):calibration=c;cache_verified=True
    if calibration is None:
        try:calibration=depth.calibrate(log).__dict__
        except Exception as exc:depth_error=f'{type(exc).__name__}: {exc}'
    depth_seconds=time.monotonic()-t
    out=ROOT/'exports'/case_id;out.mkdir(parents=True,exist_ok=True)
    worker.write(out/'preprocess.json',dict(case_id=case_id,image_sha256=case['image_sha256'],annotation_sha256=case['annotation_sha256'],legend_source='frozen-native-cache' if native.get(log.name) else 'shared-machine-parser',parsed_legend=[dict(id=e.id,label=e.label,box=e.swatch_box) for e in parsed],calibration=calibration,depth_error=depth_error,depth_cache_verified=cache_verified,legend_seconds=legend_seconds,depth_seconds=depth_seconds,seconds=time.monotonic()-start,ocr_model_sha256=models,metric_sha256=worker.sha(ROOT/'source/harness/score.py'),grading_sha256=worker.sha(ROOT/'source/strategies/grading_policy.py')))
    log.close();env['ocr_models']=models
    return io,adapter,api,env


worker.init=init_with_preprocessing
if __name__=='__main__':
    if sys.argv[1]=='export':worker.export(sys.argv[2])
    elif sys.argv[1]=='evaluate':worker.evaluate(sys.argv[2])
    else:raise ValueError('Only export and evaluate are supported')
