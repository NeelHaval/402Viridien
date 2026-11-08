"""Command-line experiment interface, identical locally and in CI."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


SUITES = ('demo','human','organizer','external-dev','external-diagnostic','challenge-unlabelled','real-crops')


def doctor(*, root=None, suite=None, manifest=None, data_root=None, model=None, config=None):
    """Read local readiness only. Never fetch data, inspect credentials or run plugins."""
    root=Path.cwd().resolve() if root is None else Path(root).resolve()
    checks=[]
    def add(name,ready,detail,action=None,required=True):
        check={'name':name,'ready':bool(ready),'required':required,'detail':detail}
        if action:check['action']=action
        checks.append(check)
    add('python',sys.version_info>=(3,10),'.'.join(map(str,sys.version_info[:3]))+'; CI tests Python 3.12',
        'Create a Python 3.12 virtual environment.' if sys.version_info[:2]!=(3,12) else None)
    modules=[('numpy','numpy'),('Pillow','PIL'),('psutil','psutil')]
    if model in ('builtin:legacy-s11','legacy:s11'):
        modules += [('scipy','scipy'),('opencv-python-headless','cv2'),('scikit-learn','sklearn'),
                    ('ruptures','ruptures'),('rapidocr-onnxruntime','rapidocr_onnxruntime'),
                    ('onnxruntime','onnxruntime'),('tifffile','tifffile')]
    for distribution,module in modules:
        try:
            importlib.import_module(module)  # known dependency only; never the user's plugin
            available=True;failure=''
        except Exception as error:
            available=False;failure=f'{type(error).__name__}: {error}'
        try:version=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:version='version unavailable'
        action='python -m pip install -c requirements.lock -e ".[test]"'
        if module not in ('numpy','PIL','psutil'):
            action='python -m pip install -c requirements.lock -c requirements-legacy.lock -e ".[legacy]"'
        add('dependency:'+module,available,version if available else failure,None if available else action)
    if model:
        builtins={'builtin:baseline','builtin:empty','builtin:legacy-s11','legacy:s11','builtin:prototype'}
        module='viridian402.crops' if model=='builtin:prototype' else 'viridian402.legacy' if model in ('builtin:legacy-s11','legacy:s11') else 'viridian402.models' if model.startswith('builtin:') else model.split(':',1)[0]
        source=next((candidate for directory in sys.path for candidate in
                    (Path(directory or '.').joinpath(*module.split('.')).with_suffix('.py'),Path(directory or '.').joinpath(*module.split('.'))/'__init__.py')
                    if candidate.is_file()),None)
        custom=model.split(':')
        custom_valid=len(custom)==2 and all(part.isidentifier() for part in custom[0].split('.')) and custom[1].isidentifier() and not model.startswith('builtin:')
        valid=(model in builtins or custom_valid) and source is not None
        add('model',valid,'source discoverable; plugin execution and callable validation happen inside evaluation workers' if valid else 'model source was not found',
            None if valid else 'Install your src/ model package with python -m pip install -e . and use module:function.')
    if config:
        try:
            settings=json.loads(Path(config).read_text(encoding='utf-8-sig'))
            if not isinstance(settings,dict):raise ValueError('config must be a JSON object')
            from .evaluate import discover_config_assets
            assets=discover_config_assets(settings)
            add('config',True,f'JSON object is readable; {len(assets)} declared model asset files found (not hashed)')
        except (OSError,ValueError,ImportError) as error:
            add('config',False,str(error),'Correct the JSON config and restore its declared model inputs; asset paths resolve from the working directory.')
    installed={'evaluation':root/'data/local/human.json','challenge':root/'data/local/challenge-unlabelled.json',
               'crops':root/'data/local/pr54/viridien/data/real/manifest.json'}
    for bundle,path in installed.items():
        command=f'python scripts/datasets.py fetch {bundle}'+(' --destination data/local/pr54' if bundle=='crops' else '')
        add('bundle:'+bundle,path.is_file(),'manifest present; use an explicit suite/data check to verify referenced assets' if path.is_file() else 'optional bundle is not installed',
            None if path.is_file() else command,required=False)
    if suite and (manifest or data_root):
        raise ValueError('doctor accepts --suite, --manifest or --data-root; choose one dataset input')
    if manifest and data_root:
        raise ValueError('doctor accepts either --manifest or --data-root')
    if suite:
        if suite=='real-crops':data_root=root/'data/local/pr54'
        elif suite=='demo':manifest=root/'examples/demo/manifest.json'
        else:manifest=root/'data/local'/f'{suite}.json'
    if manifest:
        try:
            from .data import load_manifest
            data=load_manifest(manifest)
            add('dataset',True,f"validated {len(data['records'])} interval records and all referenced assets")
        except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
            action='Check the manifest path and its asset paths; paths are relative to the manifest directory.'
            if suite=='demo':action='python -m viridian402 synthetic --output examples/demo; then rerun doctor.'
            elif suite:
                bundle='challenge' if suite=='challenge-unlabelled' else 'evaluation'
                action=f'python scripts/datasets.py fetch {bundle}; then rerun doctor.'
            add('dataset',False,str(error),action)
    if data_root:
        try:
            from .crops import audit_crops
            audit=audit_crops(data_root)
            add('dataset',True,f"validated {audit['stats']['train']} train, {audit['stats']['test']} test crops; {audit['stats']['needs_review']} holds excluded")
        except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
            add('dataset',False,str(error),'python scripts/datasets.py fetch crops --destination data/local/pr54; then rerun doctor.')
    return {'ready':all(check['ready'] for check in checks if check['required']),
            'checks':checks,'network_used':False,
            'note':'Optional bundle presence is informational. Selected datasets are fully audited; models are not executed.'}


def parser():
    p=argparse.ArgumentParser(prog="viridian402",description="Provenance-aware lithology model experiments")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser('doctor',help='Check local dependencies and selected data without network access or running a model')
    d.add_argument('--suite',choices=SUITES);d.add_argument('--manifest');d.add_argument('--data-root')
    d.add_argument('--model');d.add_argument('--config')
    v=sub.add_parser("validate",help="Validate dataset paths, intervals, provenance and split leakage")
    v.add_argument("--manifest",required=True)
    e=sub.add_parser("evaluate",help="Run isolated model workers and save an immutable run")
    e.add_argument("--manifest",required=True);e.add_argument("--model",default="builtin:baseline")
    e.add_argument("--config",help="Model JSON config file")
    e.add_argument("--output","--out",default="runs");e.add_argument("--split",choices=("train","dev","test","verify"))
    e.add_argument("--mode",choices=("full_image","supplied_legend"),default="full_image")
    e.add_argument("--timeout",type=float,default=60)
    e.add_argument("--min-score",type=float);e.add_argument("--min-completion",type=float,default=1)
    e.add_argument("--no-previews",action="store_true")
    e.add_argument('--case-id',action='append',help='Exact record ID; repeat for OR matching')
    e.add_argument('--tag',action='append',help='Record tag; repeat for OR matching (AND with case/split filters)')
    e.add_argument('--max-worker-rss-mb',type=float,help='Stop a worker when sampled process-tree RSS exceeds this MiB limit; not an allocator cap')
    crop=sub.add_parser("classify",help="Evaluate PR54 fixed real crop classification, separately from intervals")
    crop.add_argument("--data-root",required=True);crop.add_argument("--model",default="builtin:prototype")
    crop.add_argument("--config");crop.add_argument("--output",default="runs/crops");crop.add_argument("--timeout",type=float,default=60)
    crop.add_argument("--min-score",type=float,help="Minimum exact-set classification accuracy")
    i=sub.add_parser("import-source",help="Copy permitted source inputs into a portable manifest")
    i.add_argument("--source-root",required=True);i.add_argument("--dataset-root","--challenge-dataset",required=True);i.add_argument("--output","--out",required=True)
    i.add_argument("--no-unlabelled",action="store_true")
    s=sub.add_parser("synthetic",help="Generate deterministic labelled demo or interval-derived stress data")
    s.add_argument("--output","--out",required=True);s.add_argument("--manifest");s.add_argument("--seed",type=int,default=0)
    s.add_argument("--n",type=int,default=3);s.add_argument("--height",type=int,default=800);s.add_argument("--stress",default="clean")
    r=sub.add_parser("report",help="Print a saved run's Markdown report");r.add_argument("run")
    c=sub.add_parser("compare",help="Compare matched runs; reject incompatible inputs/modes")
    c.add_argument("left");c.add_argument("right");c.add_argument("--output")
    l=sub.add_parser("runs",help="List completed runs");l.add_argument("--output",default="runs")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='doctor':
            result=doctor(suite=args.suite,manifest=args.manifest,data_root=args.data_root,model=args.model,config=args.config)
            print(json.dumps(result,indent=2))
            return 0 if result['ready'] else 1
        elif args.command=="validate":
            from .data import load_manifest
            data=load_manifest(args.manifest)
            print(json.dumps({"valid":True,"records":len(data["records"]),"name":data.get("name")},indent=2))
        elif args.command=="evaluate":
            from .evaluate import evaluate
            config=json.loads(Path(args.config).read_text(encoding="utf-8-sig")) if args.config else {}
            if not isinstance(config,dict): raise ValueError("config must be a JSON object")
            if not 0<=args.min_completion<=1 or args.min_score is not None and not 0<=args.min_score<=1:
                raise ValueError("quality gates must be between 0 and 1")
            run,summary=evaluate(args.manifest,model=args.model,config=config,output=args.output,split=args.split,timeout=args.timeout,
                                 mode=args.mode,min_score=args.min_score,min_completion=args.min_completion,previews=not args.no_previews,
                                 case_ids=args.case_id,tags=args.tag,max_worker_rss_mb=args.max_worker_rss_mb)
            print(json.dumps({"run":str(run),"overall":summary["overall"],"quality_gates":summary["quality_gates"]},indent=2))
            return 0 if summary["quality_gates"]["passed"] else 1
        elif args.command=="import-source":
            from .importers import import_source
            print(import_source(args.source_root,args.dataset_root,args.output,include_unlabelled=not args.no_unlabelled))
        elif args.command=="classify":
            from .crops import classify
            config=json.loads(Path(args.config).read_text(encoding="utf-8-sig")) if args.config else {}
            if not isinstance(config,dict):raise ValueError("config must be a JSON object")
            run,summary=classify(args.data_root,model=args.model,config=config,output=args.output,timeout=args.timeout,min_score=args.min_score)
            print(json.dumps({"run":str(run),"overall":summary["overall"],"quality_gates":summary["quality_gates"]},indent=2))
            return 0 if summary["quality_gates"]["passed"] else 1
        elif args.command=="synthetic":
            from .synthetic import generate_demo,generate_synthetic
            if args.manifest:
                print(generate_synthetic(args.manifest,args.output,seed=args.seed,n=args.n,height=args.height,stress=args.stress))
            else: print(generate_demo(args.output,seed=args.seed,n=args.n))
        elif args.command=="report":
            print((Path(args.run)/"summary.md").read_text(encoding="utf-8"))
        elif args.command=="runs":
            from .report import list_runs
            results=list_runs(args.output)
            from .crops import load_crop_run
            for file in Path(args.output).rglob("summary.json"):
                try:summary,_=load_crop_run(file.parent)
                except (ValueError,OSError,KeyError):continue
                if summary.get("task")=="crop_classification":
                    results.append({"path":str(file.parent.resolve()),"task":"crop_classification","model":summary["model"],"exact_set_accuracy":summary["overall"]["exact_set_accuracy"],"passed":summary["quality_gates"]["passed"]})
            print(json.dumps(results,indent=2))
        elif args.command=="compare":
            from .report import compare_runs
            if (Path(args.left)/"summary.json").exists() or (Path(args.right)/"summary.json").exists():
                from .crops import compare_crops
                result=compare_crops(args.left,args.right)
            else:result=compare_runs(args.left,args.right)
            text=json.dumps(result,indent=2,allow_nan=False)+"\n"
            if args.output:
                path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
            print(text,end="")
        return 0
    except (ValueError,OSError,ImportError,KeyError,TypeError) as exc:
        print(f"error: {exc}",file=sys.stderr)
        return 2
