"""Workflow entry point. Inputs travel as environment data, never shell code."""
from __future__ import annotations

import json
import argparse
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from urllib.parse import unquote, urlsplit

from viridian402.evaluate import evaluate
from viridian402.synthetic import generate_demo, generate_synthetic

SUITES = ("demo", "human", "organizer", "external-dev", "external-diagnostic",
          "challenge-unlabelled", "synthetic", "stress", "real-crops", "all")


def checkout_file(root, value, label):
    file=(root/value).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        raise ValueError(f'{label} must be a readable file inside this checkout')
    return file


def selection_options(value):
    selection=json.loads(value or '{}')
    if not isinstance(selection,dict) or set(selection)-{'case_ids','tags','split','max_worker_rss_mb'}:
        raise ValueError('Selection must be a JSON object containing only case_ids, tags, split, max_worker_rss_mb')
    for key in ('case_ids','tags'):
        if key in selection and (not isinstance(selection[key],list) or not selection[key] or
                any(not isinstance(item,str) or not item for item in selection[key])):
            raise ValueError(f'Selection {key} must be a nonempty list of exact strings')
    if 'split' in selection and selection['split'] not in ('train','dev','test','verify'):
        raise ValueError('Selection split must be train, dev, test or verify')
    memory=selection.get('max_worker_rss_mb')
    if memory is not None and (isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory<=0):
        raise ValueError('Selection max_worker_rss_mb must be finite and positive')
    return selection


def summary_links(text, run, combined_root, *, github=False):
    """Rebase artifact-relative links, or make them labels in a web summary.

    Individual run reports are never edited. URLs and in-page anchors retain
    their Markdown; only local file destinations are changed.
    """
    pattern=re.compile(r'(?P<image>!?)\[(?P<label>[^\]\n]*)\]\((?P<target><[^>\n]+>|[^\s)]+)(?P<title>\s+"[^"\n]*")?\)')
    def replace(match):
        target=match['target'].strip('<>')
        windows_absolute=PureWindowsPath(target).is_absolute()
        parts=urlsplit(target) if not windows_absolute else None
        if target.startswith(('#','//')) or parts is not None and parts.scheme not in ('','file'):
            return match[0]
        if github:
            return match['label']
        if windows_absolute or PurePosixPath(target).is_absolute() or parts.scheme=='file':
            return match[0]
        relative=Path(os.path.relpath((Path(run)/unquote(parts.path)).resolve(),Path(combined_root).resolve())).as_posix()
        destination=relative+('?' + parts.query if parts.query else '')+('#' + parts.fragment if parts.fragment else '')
        if any(char.isspace() or char in '()' for char in destination):destination='<'+destination+'>'
        return f"{match['image']}[{match['label']}]({destination}{match['title'] or ''})"
    return pattern.sub(replace,text)


def fetch_data():
    """Fetch trusted lock entries only; credentials stay in this workflow step."""
    import runpy
    root=Path.cwd().resolve()
    suite=os.environ.get('EVAL_SUITE','demo')
    if suite not in SUITES:raise ValueError('Unknown evaluation suite')
    override=os.environ.get('EVAL_DATA_BUNDLE','').strip()
    if suite=='real-crops' and override not in ('','crops'):
        raise ValueError('Custom data_bundle overrides apply to interval suites; real-crops uses its pinned crops snapshot')
    if override:
        names=[override]
    elif suite=='demo':
        names=[]
    elif suite=='real-crops':
        names=['crops']
    elif suite=='challenge-unlabelled':
        names=['evaluation','challenge']
    else:
        names=['evaluation']
    lock_file=root/'data/datasets.lock.json'
    lock=json.loads(lock_file.read_text(encoding='utf-8-sig'))
    if any(name not in lock.get('bundles',{}) for name in names):
        raise ValueError('Data bundle must name a pinned entry in data/datasets.lock.json')
    fetch=runpy.run_path(str(root/'scripts/datasets.py'))['fetch']
    for name in names:
        destination=root/'data/local/pr54' if name=='crops' else root/'data/local'
        fetch(name,destination,lock_file)
    return 0


def main():
    suite = os.environ.get("EVAL_SUITE", "demo")
    if suite not in SUITES:
        raise ValueError("Unknown evaluation suite")
    model = os.environ.get("EVAL_MODEL", "builtin:baseline")
    config_path = os.environ.get("EVAL_CONFIG", "configs/baseline.json")
    root = Path.cwd().resolve()
    config_file = checkout_file(root, config_path, 'Config')
    config = json.loads(config_file.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Model config must be a JSON object")
    timeout = float(os.environ.get("EVAL_TIMEOUT", "60"))
    if not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError("Per-case timeout must be between 1 and 600 seconds")
    mode = os.environ.get("EVAL_MODE", "full_image")
    if mode not in ("full_image", "supplied_legend"):
        raise ValueError("Invalid input mode")
    minimum = os.environ.get("EVAL_MIN_SCORE", "").strip()
    minimum = float(minimum) if minimum else None
    if minimum is not None and (not math.isfinite(minimum) or not 0 <= minimum <= 1):
        raise ValueError("Minimum score must be between 0 and 1")
    override=os.environ.get('EVAL_MANIFEST','').strip()
    selection=selection_options(os.environ.get('EVAL_SELECTION','').strip())
    if suite=='real-crops' and (override or selection):
        raise ValueError('Manifest overrides and interval selection cannot be used with the fixed real-crops classification suite')
    if suite=='real-crops' and os.environ.get('EVAL_DATA_BUNDLE','').strip() not in ('','crops'):
        raise ValueError('Custom data_bundle overrides apply to interval suites; real-crops uses its pinned crops snapshot')
    if suite=='all' and override:
        raise ValueError('A manifest override requires one interval suite, not all')
    custom_manifest=checkout_file(root,override,'Manifest override') if override else None
    selected = ["human", "organizer", "external-dev", "external-diagnostic"] if suite == "all" else [suite]
    summaries = []
    github_summaries = []
    passed = True
    for name in selected:
        if name == "real-crops":
            from viridian402.crops import classify
            classifier = "builtin:prototype" if model == "builtin:baseline" else model
            run, summary = classify("data/local/pr54", model=classifier, config=config,
                                    output=Path("runs") / name, timeout=timeout, min_score=minimum)
            passed = passed and summary["quality_gates"]["passed"]
            report=f"## {name}\n\n" + (run / "summary.md").read_text(encoding="utf-8")
            summaries.append(summary_links(report,run,root/'runs'))
            github_summaries.append(summary_links(report,run,root/'runs',github=True))
            print(f"{name}: {run}")
            continue
        if custom_manifest is not None:
            manifest=custom_manifest
        elif name == "demo":
            manifest = generate_demo("data/local/demo", seed=2026, n=3)
        elif name in ("synthetic", "stress"):
            manifest = generate_synthetic("data/local/human.json", f"data/local/{name}",
                                          seed=2026, n=3, height=1600,
                                          stress="noise" if name == "stress" else "clean")
        else:
            manifest = Path("data/local") / f"{name}.json"
        run, summary = evaluate(manifest, model=model, config=config,
                                output=Path("runs") / name, mode=mode, timeout=timeout,
                                min_score=minimum, min_completion=1, previews=True,**selection)
        passed = passed and summary["quality_gates"]["passed"]
        report=f"## {name}\n\n" + (run / "summary.md").read_text(encoding="utf-8")
        summaries.append(summary_links(report,run,root/'runs'))
        github_summaries.append(summary_links(report,run,root/'runs',github=True))
        print(f"{name}: {run}")
    combined = "\n\n".join(summaries)
    Path("runs/workflow-summary.md").write_text(combined, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write('\n\n'.join(github_summaries)+'\n\nDownload the evaluation artifact for CSVs/previews.\n')
    return 0 if passed else 1


if __name__ == "__main__":
    options=argparse.ArgumentParser(description=__doc__)
    options.add_argument('--fetch-data',action='store_true',help='Fetch pinned release bundles selected by workflow inputs')
    arguments=options.parse_args()
    raise SystemExit(fetch_data() if arguments.fetch_data else main())
