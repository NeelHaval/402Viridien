"""Quick colour filter for mined column stretches.

A stretch passes when the legend swatch's main colour is one of the stretch's two most common colours,
using colours quantised to steps of 32 and a tolerance of 48 per channel.
Black-and-white swatches (main colour near-white, no strong colour) always pass, because colour tells nothing there.
Writes data/colour_gate.json: {pid: {"hidden": [candidate numbers], "kept": n}}. The review server hides these.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.common import HOME  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'


def resolve(f):
    p = HOME / f
    return p if p.exists() else HOME.parent / f


def top_colours(path, k=2):
    a = np.asarray(Image.open(path).convert('RGB'))
    if a.shape[0] > 8 and a.shape[1] > 12:
        a = a[2:-2, 4:-4]  # drop the column border
    q = a.reshape(-1, 3) // 32
    values, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)
    return [values[i].astype(int) * 32 + 16 for i in order[:k]]


def is_monochrome(colours):
    return all(int(np.ptp(c)) < 40 for c in colours)


def passes(swatch_main, stretch_colours):
    return any(int(np.abs(swatch_main - c).max()) <= 48 for c in stretch_colours)


def main():
    index = json.loads((DATA / 'sheets' / 'index.json').read_text(encoding='utf-8'))
    catalog = json.loads((DATA / 'catalog.json').read_text(encoding='utf-8'))
    patterns = {p['pid']: p for p in catalog['patterns']}
    out, fam = {}, {}
    for pid, entry in index.items():
        sw = patterns[pid]['swatches'][0]
        main_colours = top_colours(resolve(sw.get('native_file', sw['file'])))
        mono = is_monochrome(main_colours)
        hidden = []
        for n, c in entry.get('candidates', {}).items():
            if not mono and not passes(main_colours[0], top_colours(resolve(c['file']))):
                hidden.append(int(n))
        total = len(entry.get('candidates', {}))
        out[pid] = dict(hidden=sorted(hidden), kept=total - len(hidden), monochrome=mono)
        f = fam.setdefault(patterns[pid]['family'], [0, 0, 0])
        f[0] += total; f[1] += total - len(hidden); f[2] += (total - len(hidden)) == 0
    (DATA / 'colour_gate.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for k, (t, kept, empty) in fam.items():
        print(f'{k}: kept {kept}/{t}, patterns left with none: {empty}')


if __name__ == '__main__':
    main()
