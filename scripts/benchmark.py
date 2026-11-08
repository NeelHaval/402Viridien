"""Reproducible local benchmark matrix; writes both readable and JSON statistics.

Example: python scripts/benchmark.py --model builtin:baseline --model builtin:legacy-s11
         --suite human --suite organizer --suite external-dev --suite external-diagnostic
Install optional legacy dependencies before requesting S11. Datasets stay local.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
import uuid

from viridian402.evaluate import evaluate
from viridian402.report import load_run

SUITES=('human','organizer','external-dev','external-diagnostic','challenge-unlabelled')


def display(value,percent=False):
    return 'n/a' if value is None else f'{value*100:.2f}%' if percent else f'{value:.3f}'


def render(results):
    lines=['# Benchmark matrix','',
           '| Model | Suite | Completed | Pixel accuracy | Boundary F1 | Composite score | Median worker seconds | p95 worker seconds | Peak worker MiB |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in results:
        stats=item['overall']
        lines.append(f"| {item['model']} | {item['suite']} | {stats['completed']}/{stats['cases']} | {display(stats.get('row_acc_weighted'),True)} | {display(stats.get('boundary_f1'))} | {display(stats.get('score'))} | {display(stats.get('latency_median_seconds'))} | {display(stats.get('latency_p95_seconds'))} | {display(stats.get('peak_rss_mb'))} |")
    lines+=['','Worker time includes startup/loading/inference. Parent audit/hash/score/preview time is separate in each run. RSS is sampled process-tree memory, not a hard allocation bound.','',
            'Human data cover two development pages; the organizer example is repeatedly exposed. External development is visible and the diagnostic well was previously tuned. Unlabelled data have no accuracy. These results are descriptive, not blinded generalisation estimates.','',
            'Composite score is the mean per-case row accuracy × boundary F1. Pixel accuracy above is pooled over labelled rows, so it can be dominated by thick single-rock intervals. Inspect macro F1, per-class and per-well statistics in the individual runs.','',
            '## Runs','']
    for item in results:lines.append(f"- {item['suite']} / {item['model']}: `{item['run']}`; configured checks {'PASS' if item['quality_gates']['passed'] else 'FAIL'}")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',action='append',help='Repeat model IDs; default builtin:baseline')
    parser.add_argument('--suite',action='append',choices=SUITES,help='Repeat suites; default human')
    parser.add_argument('--data-root',default='data/local')
    parser.add_argument('--config',help='Optional shared model config; defaults to the matching built-in config')
    parser.add_argument('--output',default='runs/benchmarks')
    parser.add_argument('--timeout',type=float,default=90)
    parser.add_argument('--max-worker-rss-mb',type=float)
    parser.add_argument('--previews',action='store_true')
    args=parser.parse_args(argv)
    root=Path(args.output).resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True,exist_ok=False)
    results=[]
    for model in args.model or ['builtin:baseline']:
        config_path=Path(args.config or ('configs/legacy-s11.json' if model in ('builtin:legacy-s11','legacy:s11') else 'configs/baseline.json'))
        config=json.loads(config_path.read_text(encoding='utf-8-sig'))
        for suite in args.suite or ['human']:
            started=time.perf_counter()
            run,summary=evaluate(Path(args.data_root)/f'{suite}.json',model=model,config=config,
                                 output=root/model.replace(':','_').replace('/','_')/suite,
                                 timeout=args.timeout,previews=args.previews,max_worker_rss_mb=args.max_worker_rss_mb)
            load_run(run)
            item={'suite':suite,'model':model,'run':str(run),'overall':summary['overall'],
                  'total_wall_seconds':time.perf_counter()-started,'quality_gates':summary['quality_gates'],
                  'dataset_sha256':summary['dataset_sha256'],'scorer_sha256':summary['scorer_sha256']}
            results.append(item)
            (root/'benchmark.json').write_text(json.dumps({'task':'benchmark_matrix','platform':platform.platform(),'runs':results},indent=2,allow_nan=False)+'\n',encoding='utf-8')
            (root/'report.md').write_text(render(results),encoding='utf-8')
            print(f"{model} / {suite}: {summary['overall']['completed']}/{summary['overall']['cases']} complete -> {run}",flush=True)
    print(root/'report.md')
    return 0 if all(item['quality_gates']['passed'] for item in results) else 1


if __name__=='__main__':raise SystemExit(main())
