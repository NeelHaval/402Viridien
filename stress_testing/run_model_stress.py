"""Bounded local model degradation runs. No dataset downloads or notebook execution.

The default is a procedural construction test. An optional labelled manifest
uses real interval crops, retaining their split/group lineage. Every invocation
gets a new directory and keeps errors separate from measured low accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0,str(ROOT/'src'))

from viridian402.data import load_manifest, read_rgb, sha256
from viridian402.evaluate import evaluate
from viridian402.report import _inline
from viridian402.synthetic import STRESSES, _finish, _write_example, generate_demo, generate_synthetic

PROFILES={'quick':1,'extended':3,'soak':10}
MAX_CASES=100


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=tuple(PROFILES),default='extended')
    p.add_argument('--manifest',help='Optional labelled real interval source; no data is fetched')
    p.add_argument('--model',default='builtin:baseline',help='Trusted builtin or module:function interval predictor')
    p.add_argument('--config',help='Model JSON configuration; UTF-8 BOM accepted')
    p.add_argument('--mode',choices=('full_image','supplied_legend'),default='full_image',
                   help='Supplied legend is a separate assisted diagnostic')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--height',type=int,default=800)
    p.add_argument('--cases',type=int,help='Cases per degradation; all nine together must total at most 100')
    p.add_argument('--timeout',type=float,default=10,help='Per-worker hard timeout, 0.1 to 60 seconds')
    p.add_argument('--max-worker-rss-mb',type=float,default=512,
                   help='Sampled process-tree RSS threshold in MiB; not an allocator cap')
    p.add_argument('--min-score',type=float,help='Optional interval score threshold, independent of operational checks')
    p.add_argument('--output',type=Path,default=ROOT/'stress_testing/.runs',help='Parent directory; a unique child is always created')
    p.add_argument('--previews',action='store_true',help='Render annotated previews; off for faster stress iteration')
    return p


def _write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def _relative(path,base):
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def _procedural_manifest(source,output,stress,*,cases,height,seed):
    parent=load_manifest(source)['records'][0]
    entries=parent['legend']
    swatches=[read_rgb(parent['image'],entry['swatch_box']) for entry in entries]
    labels=[entry['label'] for entry in entries]
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for index in range(cases):
        lineage={'parent_id':parent['id'],'parent_group':parent['group'],'parent_split':parent['split'],
                 'parent_image_sha256':parent['image_sha256'],'parent_truth_sha256':parent['truth_sha256'],
                 'source_manifest_sha256':sha256(source),'origin':'procedural_fixture','stress':stress}
        records.append(_write_example(output,f'procedural-{index:03d}',swatches,labels,height=height,
            seed=seed+index,split=parent['split'],group=parent['group'],lineage=lineage,stress=stress))
    return _finish(output,records,'procedural-stress-'+stress)


def run(args):
    cases=PROFILES[args.profile] if args.cases is None else args.cases
    if type(cases) is not int or cases<1 or cases*len(STRESSES)>MAX_CASES:
        raise ValueError(f'cases per degradation must be positive with total ≤{MAX_CASES}; maximum {MAX_CASES//len(STRESSES)} per degradation')
    if not 64<=args.height<=1600:
        raise ValueError('height must be between 64 and 1600 pixels')
    if not 0<=args.seed<=2**32-1-len(STRESSES)*cases:
        raise ValueError('seed must be a nonnegative 32-bit value with room for per-case offsets')
    if not math.isfinite(args.timeout) or not .1<=args.timeout<=60:
        raise ValueError('timeout must be finite and between 0.1 and 60 seconds')
    if not math.isfinite(args.max_worker_rss_mb) or args.max_worker_rss_mb<=0:
        raise ValueError('max-worker-rss-mb must be finite and positive')
    if args.min_score is not None and (not math.isfinite(args.min_score) or not 0<=args.min_score<=1):
        raise ValueError('min-score must be between zero and one')
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {}
    if not isinstance(config,dict):raise ValueError('model config must be a JSON object')
    config.setdefault('seed',args.seed)
    source=Path(args.manifest).resolve() if args.manifest else None
    if source is not None:
        parents=load_manifest(source)
        if not any(row['truth'] and row['source']!='synthetic' for row in parents['records']):
            raise ValueError('--manifest needs labelled nonsynthetic real parents; omit it for procedural stress')
    output=args.output.resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    if source is None:
        source=generate_demo(output/'procedural-source',seed=args.seed,n=1)
    metadata={'schema_version':1,'task':'model_degradation_stress','profile':args.profile,'seed':args.seed,
              'model':args.model,'config':config,'mode':args.mode,'cases_per_degradation':cases,
              'height':args.height,'stresses':list(STRESSES),'source_manifest':str(source),
              'source_manifest_sha256':sha256(source),'source_kind':'real_interval_crops' if args.manifest else 'procedural_fixture',
              'timeout_seconds':args.timeout,'sampled_worker_rss_limit_mib':args.max_worker_rss_mb,
              'min_score':args.min_score,'runner_sha256':sha256(__file__)}
    _write(output/'request.json',metadata)
    results=[]
    for stress in STRESSES:
        stage_started=time.perf_counter()
        item={'degradation':stress,'cases_requested':cases,'generation_error':None,'evaluation_error':None}
        try:
            directory=output/'datasets'/stress
            manifest=generate_synthetic(source,directory,seed=args.seed,n=cases,height=args.height,stress=stress) if args.manifest else _procedural_manifest(source,directory,stress,cases=cases,height=args.height,seed=args.seed)
            item.update(manifest=_relative(manifest,output),generation_seconds=time.perf_counter()-stage_started)
            # Gate accuracy ourselves so low score is distinguishable from worker/
            # output failures. The evaluator still records its full diagnostics.
            run_path,summary=evaluate(manifest,model=args.model,config=config,output=output/'evaluations'/stress,
                mode=args.mode,timeout=args.timeout,max_worker_rss_mb=args.max_worker_rss_mb,
                min_score=None,min_completion=1,previews=args.previews)
            overall=summary['overall']
            operational=bool(summary['quality_gates']['passed'])
            accuracy=None if args.min_score is None else bool(overall.get('score') is not None and overall['score']>=args.min_score)
            item.update(run=_relative(run_path,output),per_case=_relative(run_path/'per_case.jsonl',output),
                        metrics=_relative(run_path/'metrics.json',output),report=_relative(run_path/'summary.md',output),
                        overall=overall,operational_passed=operational,accuracy_passed=accuracy,
                        passed=operational and accuracy is not False)
        except Exception as error:
            key='evaluation_error' if item.get('manifest') else 'generation_error'
            item[key]=f'{type(error).__name__}: {error}'
            item.update(operational_passed=False,accuracy_passed=None,passed=False)
        item['total_seconds']=time.perf_counter()-stage_started
        results.append(item)
        _write(output/'degradations.json',results)  # keep completed cases if interrupted
        print(f"{stress}: operational={'PASS' if item['operational_passed'] else 'FAIL'}, accuracy={item['accuracy_passed']}",flush=True)
    summary={**metadata,'run_id':output.name,'results':results,'total_seconds':time.perf_counter()-started,
             'operational_passed':all(item['operational_passed'] for item in results),
             'accuracy_gate_configured':args.min_score is not None,
             'accuracy_passed':None if args.min_score is None else all(item['accuracy_passed'] is True for item in results),
             'passed':all(item['passed'] for item in results),
             'limitations':['Procedural and constructed stress scores do not establish real scanned-log accuracy.',
                            'Real-pattern derivatives inherit parent labels and well/split lineage; they are not independent holdouts.',
                            'Sampled worker RSS can miss short peaks and excludes parent generation/scoring memory.',
                            'No notebook cells, training, model downloads or paid APIs are invoked by this runner; custom plugins are trusted code.']}
    _write(output/'summary.json',summary)
    lines=['# Model degradation stress','',f"Source: {metadata['source_kind']}; model {_inline(args.model)}; mode `{args.mode}`; seed {args.seed}.",'',
           f"{cases} cases × {len(STRESSES)} degradations. Operational checks: {'PASS' if summary['operational_passed'] else 'FAIL'}. Accuracy threshold: {args.min_score if args.min_score is not None else 'not configured'}.",'',
           '| Degradation | Operational | Accuracy gate | Score | Worker seconds | Peak worker MiB | Cases |',
           '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for item in results:
        metrics=item.get('overall',{})
        value=lambda key:'n/a' if metrics.get(key) is None else f'{metrics[key]:.4f}'
        case_link=f"[per-case JSONL]({item['per_case']})" if item.get('per_case') else _inline(item.get('generation_error') or item.get('evaluation_error'))
        lines.append(f"| {item['degradation']} | {'PASS' if item['operational_passed'] else 'FAIL'} | {item['accuracy_passed'] if item['accuracy_passed'] is not None else 'not configured'} | {value('score')} | {value('seconds')} | {value('peak_rss_mb')} | {case_link} |")
    lines+=['',*summary['limitations']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output,summary


def main(argv=None):
    options=parser();args=options.parse_args(argv)
    try:
        output,summary=run(args)
    except (ValueError,OSError,ImportError,KeyError,TypeError) as error:
        options.exit(2,f'error: {error}\n')
    print(json.dumps({'run':str(output),'operational_passed':summary['operational_passed'],
                      'accuracy_passed':summary['accuracy_passed'],'passed':summary['passed']},indent=2))
    return 0 if summary['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
