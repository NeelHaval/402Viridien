"""Command-line experiment interface, identical locally and in CI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def parser():
    p=argparse.ArgumentParser(prog="viridian402",description="Provenance-aware lithology model experiments")
    sub=p.add_subparsers(dest="command",required=True)
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
    i=sub.add_parser("import-source",help="Copy permitted source inputs into a portable manifest")
    i.add_argument("--source-root",required=True);i.add_argument("--dataset-root","--challenge-dataset",required=True);i.add_argument("--output","--out",required=True)
    i.add_argument("--no-unlabelled",action="store_true")
    s=sub.add_parser("synthetic",help="Generate deterministic labelled demo or legend-derived stress data")
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
        if args.command=="validate":
            from .data import load_manifest
            data=load_manifest(args.manifest)
            print(json.dumps({"valid":True,"records":len(data["records"]),"name":data.get("name")},indent=2))
        elif args.command=="evaluate":
            from .evaluate import evaluate
            config=json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(config,dict): raise ValueError("config must be a JSON object")
            if not 0<=args.min_completion<=1 or args.min_score is not None and not 0<=args.min_score<=1:
                raise ValueError("quality gates must be between 0 and 1")
            run,summary=evaluate(args.manifest,model=args.model,config=config,output=args.output,split=args.split,timeout=args.timeout,
                                 mode=args.mode,min_score=args.min_score,min_completion=args.min_completion,previews=not args.no_previews)
            print(json.dumps({"run":str(run),"overall":summary["overall"],"quality_gates":summary["quality_gates"]},indent=2))
            return 0 if summary["quality_gates"]["passed"] else 1
        elif args.command=="import-source":
            from .importers import import_source
            print(import_source(args.source_root,args.dataset_root,args.output,include_unlabelled=not args.no_unlabelled))
        elif args.command=="synthetic":
            from .synthetic import generate_demo,generate_synthetic
            if args.manifest:
                print(generate_synthetic(args.manifest,args.output,seed=args.seed,n=args.n,height=args.height,stress=args.stress))
            else: print(generate_demo(args.output,seed=args.seed,n=args.n))
        elif args.command=="report":
            print((Path(args.run)/"summary.md").read_text(encoding="utf-8"))
        elif args.command=="runs":
            from .report import list_runs
            print(json.dumps(list_runs(args.output),indent=2))
        elif args.command=="compare":
            from .report import compare_runs
            result=compare_runs(args.left,args.right)
            text=json.dumps(result,indent=2,allow_nan=False)+"\n"
            if args.output:
                path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
            print(text,end="")
        return 0
    except (ValueError,OSError,ImportError,KeyError,TypeError) as exc:
        print(f"error: {exc}",file=sys.stderr)
        return 2
