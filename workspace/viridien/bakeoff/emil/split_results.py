"""Aggregate only the available dev/test windows; never evaluate removed wells."""
import json
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
MODES = ['detect', 'labels', 's11']


def read(path):
    return Path(path).read_text(encoding='utf-8')


def membership():
    splits = json.loads(read(WORK/'external_tests/splits.json'))['logs']
    assert not (set(splits['dev']) & set(splits['test']))
    assert not (set(splits['verify']) & set(splits['dev']+splits['test']))
    return {name: split for split, names in splits.items() for name in names}


def selected(records, split):
    if split not in {'dev', 'test', 'dev_test'}:
        raise ValueError('Only dev and test are available for evaluation')
    lookup = membership()
    allowed = {'dev', 'test'} if split == 'dev_test' else {split}
    return [r for r in records if lookup[r['log']] in allowed]


def evaluate_split(root, records, split, mode):
    from emil.evaluate import run_evaluator
    from harness.run import aggregate
    rr = [r for r in selected(records, split) if r['mode'] == mode]
    # Evaluators are unchanged; constrain their input files before execution.
    with tempfile.TemporaryDirectory(prefix='emil_eval_') as temp:
        temp = Path(temp)
        for folder in ['truth', 'predictions', 'annotations', 'output']:
            (temp/folder).mkdir()
        for r in rr:
            filename = r['key']+'.json'
            for src, dst in [(root/'truth', temp/'truth'),
                             (root/mode/'predictions', temp/'predictions'),
                             (root/'annotations', temp/'annotations')]:
                shutil.copyfile(src/filename, dst/filename)
        result = {kind: run_evaluator(kind, temp/'truth', temp/'predictions', temp/'annotations', temp/'output')
                  for kind in ['segmentation', 'classification', 'legend']}
        result = dict(harness=aggregate(rr), emil=result)
        out = root/split/mode
        out.mkdir(parents=True, exist_ok=True)
        for file in (temp/'output').iterdir():
            shutil.copyfile(file, out/file.name)
        (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result


def summarize_external(root, records, modes=MODES):
    for split in ['dev', 'test', 'dev_test']:
        for mode in modes:
            evaluate_split(root, records, split, mode)
    # Retire old mixed-split aggregates while preserving per-window predictions.
    for mode in modes:
        for name in ['metrics.json', 'classification.json', 'legend.json', 'segmentation.json',
                     'classification.txt', 'legend.txt', 'segmentation.txt']:
            path = root/mode/name
            if path.exists():
                path.unlink()
