"""Run with python -I bakeoff/emil/train.py; never imports external input code."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from emil import pipeline as p


def main():
    sp = json.loads((p.WORK / 'emil/splits.json').read_text())
    excluded = {f.stem for f in p.LABEL_DIR.glob('*.json')} | {p.clean(n['stem'] if isinstance(n, dict) else n) for n in sp['test']}
    pages = p.build_train_pages()
    assert all('hadrian' not in str(a) and '60811405120000' not in str(a) for a, _ in pages)
    p.torch.manual_seed(0)
    p.torch.set_num_threads(6)
    metadata = dict(pages=[a.stem for a, _ in pages], excluded=sorted(excluded), numpy_sampling_seed=0,
                    torch_seed=0, notebook_torch_seed=None, steps=2000, length=1024, batch=16,
                    pseudo_k=p.PSEUDO_K, device=p.DEVICE)
    (HERE / 'training.json').write_text(json.dumps(metadata, indent=2))
    print('Training on', len(pages), 'pages', flush=True)
    start = time.perf_counter()
    net = p.train_cnn(pages)
    p.torch.save(net.state_dict(), HERE / 'boundary_net.pt')
    metadata.update(seconds=time.perf_counter()-start, checkpoint_sha256=hashlib.sha256((HERE/'boundary_net.pt').read_bytes()).hexdigest())
    (HERE / 'training.json').write_text(json.dumps(metadata, indent=2))
    print('Saved checkpoint', metadata, flush=True)


if __name__ == '__main__':
    main()
