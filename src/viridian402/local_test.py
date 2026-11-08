"""Local setup check, private data download, and explicit development tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['demo','quick-dev','dev'],default='demo')
    p.add_argument('--download',choices=['all','dev','human','synthetic','unlabelled','heldout-inputs','heldout-references'])
    p.add_argument('--data-dir',type=Path,default=Path('data/local/review-v1'))
    p.add_argument('--allow-heldout-references',action='store_true')
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--check',action='store_true',help='Validate setup and inputs without running a model')
    p.add_argument('--run',action='store_true',help='Explicitly run the selected model on the selected development inputs')
    p.add_argument('--model',default='builtin:baseline');p.add_argument('--config',type=Path,default=Path('configs/baseline.json'))
    p.add_argument('--output',type=Path,default=Path('runs/local'))
    p.add_argument('--timeout',type=float,default=60)
    a=p.parse_args(argv)
    try:
        from .data import load_manifest
        from .review_data import download,development_manifest
        root=Path.cwd().resolve()
        if a.download:
            profiles=['dev','human','synthetic','unlabelled','heldout-inputs'] if a.download=='all' else [a.download]
            for profile in profiles:
                download(profile,a.data_dir,root,allow_heldout_references=a.allow_heldout_references)
            return 0
        if a.suite=='demo':
            from .synthetic import generate_demo
            manifest=generate_demo(a.data_dir/'demo',seed=2026,n=1)
        else:
            download('dev',a.data_dir,root)
            manifest=development_manifest(a.data_dir,quick=a.suite=='quick-dev')
        loaded=load_manifest(manifest)
        config=json.loads(a.config.read_text(encoding='utf-8'))
        if not isinstance(config,dict):raise ValueError('Model config must be a JSON object')
        print(f'LOCAL_INPUT_CHECK_OK suite={a.suite} cases={len(loaded["records"])} manifest={manifest}')
        if a.check or a.download_only or not a.run:
            print('Model not run. Add --run to request this evaluation; no model API or auto-label call is made by setup.')
            return 0
        from .evaluate import evaluate
        run,summary=evaluate(manifest,model=a.model,config=config,output=a.output,timeout=a.timeout)
        print(f'REPORT={run / "report.html"}')
        print(f'SUMMARY={run / "summary.md"}')
        print('Interpretation: execution/development diagnostic; not independent accuracy or official scoring.')
        return 0 if summary['quality_gates']['passed'] else 1
    except (ValueError,OSError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
        print('Local test failed: '+str(error),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
