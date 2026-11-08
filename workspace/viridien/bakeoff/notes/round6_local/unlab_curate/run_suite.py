"""Run frozen TEST diagnostics; use plain Python for the harness DLLs.

All new result and parser caches stay in the assigned lane. Model files are
read from their normal locations. No model settings are selected here.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
LANE=Path(__file__).resolve().parent
BAKEOFF=LANE.parents[2]
RUN_CACHE=Path(os.environ.get('OUTSIDE_RUN_CACHE',str(LANE/'cache')))
sys.path.insert(0,str(BAKEOFF))
from harness import bench, unlabelled_metrics

_ORIGINAL_INIT=bench._init_worker
_S19_ENTRIES=None
_S19_CALIB=None
_S19_RUN=None
_S19_CHECKED_DEPTH=None
_DEPTH_CALIBRATE=None


@lru_cache(maxsize=1)
def budget_rapid_reader():
    from rapidocr_onnxruntime import RapidOCR
    # Same OCR model and image settings; bound compute per worker only.
    return RapidOCR(intra_op_num_threads=1,inter_op_num_threads=1,
                    det_limit_side_len=3200,det_limit_type='max')


def cached_depth_calibration(log,engine='rapidocr'):
    from harness import io
    # bench holds the image's preprocessing lock here. Its root is keyed by
    # inference files and exact image/annotation bytes. Cache deterministic
    # ValueErrors only; do not cache transient I/O or runtime failures.
    path=io.ROOT/f'unlab_depth_failure.{engine}.json'
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'))
        raise ValueError(data['message'])
    try:
        return _DEPTH_CALIBRATE(log,engine=engine)
    except ValueError as exc:
        path.write_text(json.dumps(dict(type='ValueError',message=str(exc))),encoding='utf-8')
        raise


@lru_cache(maxsize=1)
def owned_easy_reader():
    import easyocr
    return easyocr.Reader(['en'],gpu=False,detector=False,verbose=False,download_enabled=False,
                          user_network_directory=str(RUN_CACHE/'easyocr_user'))


def owned_s19_parser(log):
    global _S19_ENTRIES
    from harness import legend_v3
    from ml.combine import runtime
    # Same parser and frozen settings as runtime.parsed; only cache locations differ.
    with runtime._PARSER_LOCK:
        old_home,old_easy=legend_v3.HOME,legend_v3._easy
        try:
            legend_v3.HOME=RUN_CACHE/'s19_legend'
            legend_v3._easy=owned_easy_reader
            _S19_ENTRIES=legend_v3.parse(log,config=runtime.PARSER_CONFIG)
            return _S19_ENTRIES
        finally:
            legend_v3.HOME,legend_v3._easy=old_home,old_easy


def captured_depth(log,calib):
    global _S19_CALIB
    _S19_CALIB=_S19_CHECKED_DEPTH(log,calib)
    return _S19_CALIB


def captured_s19(log,window,ctx):
    global _S19_ENTRIES,_S19_CALIB
    _S19_ENTRIES=_S19_CALIB=None
    rows=_S19_RUN(log,window,ctx)
    return rows,dict(_unlabelled_legend=_S19_ENTRIES,
                     _unlabelled_calibration=_S19_CALIB,
                     _unlabelled_legend_parser='s19 legend_v3 with frozen PARSER_CONFIG')


def initialise_worker():
    global _S19_RUN,_S19_CHECKED_DEPTH,_DEPTH_CALIBRATE
    _ORIGINAL_INIT()
    from harness import depth,legend
    from ml.combine import runtime
    from strategies import s19_combine
    _S19_RUN=s19_combine.run
    _S19_CHECKED_DEPTH=runtime.checked_depth
    runtime.parsed=owned_s19_parser
    runtime.checked_depth=captured_depth
    s19_combine.run=captured_s19
    legend._ocr=budget_rapid_reader
    _DEPTH_CALIBRATE=depth.calibrate
    depth.calibrate=cached_depth_calibration


def main():
    global RUN_CACHE
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--suite',default='unlab_test')
    parser.add_argument('--strategies',default='s11,s17,s19,s22')
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--clean-dir',type=Path)
    parser.add_argument('--items',type=Path)
    args=parser.parse_args()
    if not 1 <= args.workers <= 2: parser.error('workers must be 1 or 2')
    RUN_CACHE=(args.cache_dir or args.out/'_cache').resolve()
    os.environ['OUTSIDE_RUN_CACHE']=str(RUN_CACHE)
    if not (args.dataset/'curation.json').exists() and not args.clean_dir:
        raise RuntimeError('No curation contract')
    if (args.dataset/'curation.json').exists():
        contract=json.loads((args.dataset/'curation.json').read_text(encoding='utf-8'))
        if contract['status']!='FROZEN TEST':raise RuntimeError('Visual review must finish before models run')
    os.environ['LITHOLOGY_DATASET']=str(args.dataset)
    os.environ['BENCH_UNLAB_TEST_DATASET']=str(args.dataset)
    os.environ['BENCH_UNLAB_TEST_ITEMS']=str(args.items or args.dataset/'eval_set.json')
    os.environ['BENCH_UNLAB_TEST_MANIFEST']=str(args.dataset/'manifest.csv')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'compute_contract.json').write_text(json.dumps(dict(workers=args.workers,strategies=args.strategies,
        torch_threads_per_worker=1,rapidocr_intra_threads=1,rapidocr_inter_threads=1,
        deterministic_depth_errors_cached=True,model_thresholds_changed=False),indent=2),encoding='utf-8')
    fingerprints={p.relative_to(bench.REPO).as_posix():bench.file_hash(p)
                  for p in bench.dependency_files() if not p.name.startswith('test_')
                  and p.name not in ('unlabelled_metrics.py','bench.py')}
    if args.clean_dir:
        previous=json.loads((args.clean_dir/'model_fingerprints.json').read_text(encoding='utf-8'))
        if previous!=fingerprints:
            changed=sorted(k for k in set(previous)|set(fingerprints) if previous.get(k)!=fingerprints.get(k))
            raise RuntimeError('Clean/scan inference files differ: '+', '.join(changed))
    (args.out/'model_fingerprints.json').write_text(json.dumps(fingerprints,indent=2),encoding='utf-8')
    bench._init_worker=initialise_worker
    frame=bench.evaluate(args.strategies,suite=args.suite,workers=args.workers,
                         out_dir=args.out,cache_dir=args.cache_dir or args.out/'_cache',registry=args.registry,render=False,progress=False)
    report=unlabelled_metrics.write_report(frame,args.out,clean_dir=args.clean_dir)
    after={p.relative_to(bench.REPO).as_posix():bench.file_hash(p)
           for p in bench.dependency_files() if not p.name.startswith('test_')
           and p.name not in ('unlabelled_metrics.py','bench.py')}
    (args.out/'model_files_unchanged.json').write_text(json.dumps(dict(unchanged=after==fingerprints),indent=2),encoding='utf-8')
    if after!=fingerprints:raise RuntimeError('Inference files changed during the TEST run')
    print('Results',len(frame),'execution failures',int(frame.failure.sum()),flush=True)
    print('Written',args.out,flush=True)


if __name__=='__main__':main()
