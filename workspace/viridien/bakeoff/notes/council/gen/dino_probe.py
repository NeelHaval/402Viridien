"""Zero-shot probe: frozen DINOv2 at native scale (tile, no resize) vs the hand classifier, on the v3 blind-LLM windows."""
import sys, csv, json, pickle, collections, os
import numpy as np, torch
sys.path.insert(0, '.')
from labeltool import legend_classify as C
from transformers import AutoModel
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'facebook/dinov2-small'
dev = 'cuda'
m = AutoModel.from_pretrained(MODEL, local_files_only=True).to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def canvas(a, S=224):
    a = np.ascontiguousarray(a[..., :3])
    a = a[:, :S] if a.shape[1] > S else a
    reps = (int(np.ceil(S / a.shape[0])), int(np.ceil(S / a.shape[1])), 1)
    return np.tile(a, reps)[:S, :S]
@torch.no_grad()
def emb(arrs):
    x = torch.tensor(np.stack([canvas(a) for a in arrs]), device=dev).permute(0, 3, 1, 2).float() / 255
    h = m(pixel_values=(x - mean) / std).last_hidden_state
    e = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    return torch.nn.functional.normalize(e, dim=1).cpu().numpy()
D = 'analysis/categories/llmrev/'
R = [r for r in json.load(open(D + 'results.json', encoding='utf-8'))['rows'] if not r.get('dup') and r['llm'] and r['llm'] not in ('mixed', 'none')]
W = pickle.load(open(D + 'windows.pkl', 'rb'))
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(x['log']): x['family'] for x in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8'))}
items = pickle.load(open('notes/council/gen/lofo_items.pkl', 'rb'))  # same order as R (built from the same filter)
assert len(items) == len(R)
legE = {}
for log in {r['log'] for r in R}:
    labs, arrs = [], []
    for label, a in C._native().get(log, []):
        if C.NOT_ROCK.search(label):
            continue
        t = 2 if min(a.shape[:2]) > 12 else 0
        a = C.strip_borders(a[t:a.shape[0] - t or None, t:a.shape[1] - t or None])
        if a.size and min(a.shape[:2]) >= 3:
            labs.append(label); arrs.append(a)
    legE[log] = (labs, emb(arrs) if arrs else None)
res = collections.defaultdict(collections.Counter)
for r, it in zip(R, items):
    f = fam.get(key(r['log']), '?'); res[f]['n'] += 1
    labs, E = legE[r['log']]
    if E is None or not it['ok']:
        continue
    a = C.strip_borders(W[(r['log'], r['y'])][..., :3])
    s = E @ emb([a])[0]
    tw = C.twins(r['log'])
    dino = {}
    for lb, v in zip(labs, s):
        n = tw.get(lb, lb); dino[n] = max(v, dino.get(n, -9))
    hand = {}
    d = it['comp'] @ np.array([1, 1, 2, 1])
    for n, v in zip(it['names'], d):
        hand[n] = min(v, hand.get(n, 1e9))
    ok = lambda lab: r['llm'] in lab.split(' / ')
    res[f]['hand'] += ok(min(hand, key=hand.get))
    res[f]['dino'] += ok(max(dino, key=dino.get))
    # fusion: z-scored hand distance minus z-scored dino similarity
    names = list(hand)
    hv = np.array([hand[n] for n in names]); dv = np.array([dino.get(n, 0) for n in names])
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    for wgt in (0.5, 1.0):
        res[f][f'fuse{wgt}'] += ok(names[int(np.argmin(z(hv) - wgt * z(dv)))])
tot = collections.Counter()
for f in sorted(res):
    tot.update(res[f]); c = res[f]
    print(f"{f:26s} n={c['n']:3d} hand {c['hand']:3d}  dino {c['dino']:3d}  fuse.5 {c['fuse0.5']:3d}  fuse1 {c['fuse1.0']:3d}")
print('ALL', dict(tot))
