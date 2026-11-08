"""Extract the LIVE 56-cell notebook; preserve algorithm and evaluator source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
NOTEBOOK = WORK / 'emil/drive/HTB2026.ipynb'
EVALUATORS = {'segmentation': 'evaluate_layers_colab_cell.py',
              'classification': 'evaluate_segment_classes_cell.py',
              'legend': 'evaluate_legend_cell.py'}


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert len(nb['cells']) == 56
    chunks = []
    cells = [3, 5, 6, 8, 10, 13, 16, 18, 20, 24, 27, 29, 31,
             32, 34, 37, 39, 42, 45, 47, 54, 55]
    for i in cells:
        src = '\n'.join(s for s in ''.join(nb['cells'][i]['source']).splitlines()
                        if not s.startswith('!')) + '\n'
        kept = []
        for node in ast.parse(src).body:
            if isinstance(node, (ast.For, ast.Expr)):
                continue
            if isinstance(node, ast.Assign):
                names = {t.id for t in node.targets if isinstance(t, ast.Name)}
                if i == 5 and names & {'ROOT', 'IMAGE_PATH', 'ANNOTATION_PATH', 'LABEL_PATH'}:
                    continue
                if i == 8:
                    continue
            start = node.lineno
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.decorator_list:
                start = node.decorator_list[0].lineno
            kept.append('\n'.join(src.splitlines()[start - 1:node.end_lineno]))
        chunks.append(f'# Live notebook cell {i}\n' + '\n\n'.join(kept))
    original = '\n\n'.join(chunks) + '\n'
    code = original.replace("ROOT / 'dataset'", 'DATASET').replace("ROOT / 'test'", 'LABEL_DIR')
    code = code.replace('img.crop(', '_crop_image(img, ')
    code = code.replace("for i, (b, n) in enumerate(zip(boxes, names))])",
                        "for i, (b, n) in enumerate(zip(boxes, names))], columns=['legend_id', 'lithology', 'box'])")
    code = code.replace('dist[order[1]] - dist[order[0]]',
                        "(dist[order[1]] - dist[order[0]] if len(order) > 1 else float('inf'))")
    code = code.replace("return np.array(crop_rgb(img, (x0, y0, x1, y1))), y0",
                        "result = np.array(crop_rgb(img, (x0, y0, x1, y1)))\n    _close_image(img)\n    return result, y0")
    code = code.replace("MODEL_PATH = ROOT / 'boundary_net.pt'", "MODEL_PATH = DEFAULT_MODEL")
    code = code.replace('def read_swatch_labels(img, boxes, legend_box):',
                        "def read_swatch_labels(img, boxes, legend_box):\n    from emil.paths import cached_swatch_names\n    cached = cached_swatch_names(img, boxes, legend_box)\n    if cached is not None: return cached")
    code = code.replace("easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)",
                        "__import__('emil.paths', fromlist=['ocr_reader']).ocr_reader()")
    # Preserve the actual TRAIN_PAGES selection as an explicit, parameterised call.
    train = ''.join(nb['cells'][21]['source'])
    clean_node = ast.parse(train).body[0]
    chunks_clean = '\n'.join(train.splitlines()[clean_node.lineno-1:clean_node.end_lineno])
    train_body = '\n'.join(train.splitlines()[clean_node.end_lineno:])
    train_body = train_body.replace("ROOT / 'dataset'", 'DATASET').replace("ROOT / 'test'", 'LABEL_DIR')
    train_body = train_body.replace("ROOT / 'splits.json'", 'SPLITS_PATH')
    train_body = train_body.replace("print(len(TRAIN_PAGES), 'train pages')", 'return TRAIN_PAGES')
    code += '\n' + chunks_clean + '\n\ndef build_train_pages():\n' + '\n'.join('    '+s for s in train_body.splitlines()) + '\n'
    preamble = '''"""Port of emil/drive/HTB2026.ipynb (56 cells). See CHANGES.md."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
import sys
sys.path.insert(0, str(ROOT.parent))
from emil.paths import dataset_path
DATASET = dataset_path()
LABEL_DIR = WORK / 'emil/labels'
SPLITS_PATH = WORK / 'emil/splits.json'
DEFAULT_MODEL = WORK / 'emil/drive/boundary_net.pt'

def configure(dataset=None, labels=None, model=None, run_dir=None, root=None):
    global DATASET, LABEL_DIR, MODEL_PATH, GT_DIR, RUN_DIR, TEST_STEMS, net, ROOT, EXPERIMENTS
    if root is not None:
        ROOT = Path(root).resolve()
        EXPERIMENTS = ROOT / 'experiments.csv'
        if run_dir is None: RUN_DIR = ROOT / 'predictions' / RUN_NAME
    if dataset is not None: DATASET = Path(dataset).resolve()
    if labels is not None:
        LABEL_DIR = Path(labels).resolve()
        GT_DIR = LABEL_DIR
        TEST_STEMS = sorted(p.stem for p in GT_DIR.glob('*.json'))
    if model is not None:
        MODEL_PATH = Path(model).resolve()
        net = None
    if run_dir is not None: RUN_DIR = Path(run_dir).resolve()
    PAGES.clear()

def _crop_image(img, box):
    from harness import io
    if not hasattr(img, '_emil_log'):
        img._emil_log = io.Log(img, {}, 'emil')
    return io._crop(img._emil_log, tuple(map(int, box)))

def _close_image(img):
    if hasattr(img, '_emil_log'): img._emil_log.close()
    else: img.close()

def score_run(run_dir):
    # The live notebook calls score_run but does not define it.
    from emil.evaluate import score_prediction_run
    return score_prediction_run(Path(run_dir), GT_DIR, DATASET / 'bbox_annotation')

'''
    (HERE / 'pipeline.py').write_text(preamble + code, encoding='utf-8')
    (HERE / 'notebook.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), code.splitlines(True), fromfile='live extracted notebook', tofile='local pipeline')), encoding='utf-8')
    hashes = {}
    for kind, filename in EVALUATORS.items():
        data = (WORK / 'emil/drive/eval_cells' / filename).read_bytes()
        (HERE / 'evaluators' / f'{kind}.py').write_bytes(data)
        hashes[kind] = hashlib.sha256(data).hexdigest()
    (HERE / 'port_manifest.json').write_text(json.dumps(dict(notebook=str(NOTEBOOK.relative_to(WORK)), notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(), cells=cells, evaluators=hashes), indent=2))
    print('Ported live notebook and byte-identical Drive eval scripts')


if __name__ == '__main__': main()
