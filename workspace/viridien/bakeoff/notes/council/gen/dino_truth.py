"""Hand classifier vs frozen DINOv2 (native scale) vs fusion, scored on HUMAN/GOLD truth windows (inside one truth bed)."""
import sys, csv, re, collections, os
import numpy as np, torch
sys.path.insert(0, '.')
os.environ['HF_HUB_OFFLINE'] = '1'
from harness import io
from labeltool import legend_classify as C
exec(open('notes/council/gen/dino_probe.py', encoding='utf-8').read().split("D = 'analysis")[0].split("from transformers")[1].join(["from transformers", ""]) if False else "")
from transformers import AutoModel
dev = 'cuda'; m = AutoModel.from_pretrained('facebook/dinov2-small', local_files_only=True).to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def canvas(a, S=224):
    a = np.ascontiguousarray(a[..., :3]); a = a[:, :S] if a.shape[1] > S else a
    return np.tile(a, (int(np.ceil(S / a.shape[0])), int(np.ceil(S / a.shape[1])), 1))[:S, :S]
@torch.no_grad()
def emb(arrs):
    x = torch.tensor(np.stack([canvas(a) for a in arrs]), device=dev).permute(0, 3, 1, 2).float() / 255
    h = m(pixel_values=(x - mean) / std).last_hidden_state
    return torch.nn.functional.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1), dim=1).cpu().numpy()
norm = lambda s: re.sub(r'[^a-z]', '', s.lower())
alias = {'sandandsandstone': 'sandstone'}
def match(truth, pred):
    ts = {alias.get(norm(x), norm(x)) for x in truth.split('/')} | {norm(truth)}
    ps = {alias.get(norm(x), norm(x)) for x in pred.split(' / ')} | {norm(pred)}
    return bool(ts & ps)
truths = {'D-582 gold log06': 'gt/gold/0000000009_d-582_0000020435-0001_log06_fe_5in_md.csv',
          'Gulf Neel 1inch': '../neel_dataset/human_labels/truth/60811405120000_1inch tvd mudlog.csv',
          'Gulf Neel Shenzi': '../neel_dataset/human_labels/truth/60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400.csv'}
for tag, p in truths.items():
    name = os.path.splitext(os.path.basename(p))[0]
    T = [(int(float(t['top_y'])), int(float(t['bottom_y'])), t['lithology']) for t in csv.DictReader(open(p, encoding='utf-8'))]
    log = io.open_log(name)
    labs, arrs = [], []
    for label, a in C._native().get(name, []):
        if C.NOT_ROCK.search(label): continue
        t = 2 if min(a.shape[:2]) > 12 else 0
        a = C.strip_borders(a[t:a.shape[0] - t or None, t:a.shape[1] - t or None])
        if a.size and min(a.shape[:2]) >= 3: labs.append(label); arrs.append(a)
    E = emb(arrs); tw = C.twins(name); feats = dict(C.legend(name))
    wins = []
    for y0, y1, lab in T:
        for y in range(y0 + 4, y1 - 52, 48):
            wins.append((y, lab))
    rng = np.random.default_rng(0)
    if len(wins) > 400: wins = [wins[i] for i in sorted(rng.choice(len(wins), 400, replace=False))]
    c = collections.Counter()
    for y, lab in wins:
        a = C.strip_borders(io.read_rgb(log, 'interpreted_lithology_column', y0=y, y1=y + 48)[..., :3])
        ai = a.astype(int)
        if (ai.max(2) < 90).mean() < 0.012 and ((ai.max(2) >= 140) & (ai.min(2) <= 235)).mean() < 0.05:
            continue
        c['n'] += 1
        fw = C.features(a)
        hand = {}
        for lb, fs in C.legend(name):
            d = C.distance(fw, fs) + C.P['gamma'] * (1 - C.ncc(fw['mask'], fs['mask']))
            n = tw.get(lb, lb); hand[n] = min(d, hand.get(n, 1e9))
        s = E @ emb([a])[0]; dino = {}
        for lb, v in zip(labs, s):
            n = tw.get(lb, lb); dino[n] = max(v, dino.get(n, -9))
        names = list(hand); hv = np.array([hand[n] for n in names]); dv = np.array([dino.get(n, 0) for n in names])
        z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
        c['hand'] += match(lab, min(hand, key=hand.get)); c['dino'] += match(lab, max(dino, key=dino.get))
        c['fuse'] += match(lab, names[int(np.argmin(z(hv) - z(dv)))])
    print(f"{tag:18s} windows {c['n']:3d}  hand {c['hand'] / c['n']:.2f}  dino {c['dino'] / c['n']:.2f}  fuse {c['fuse'] / c['n']:.2f}", flush=True)
