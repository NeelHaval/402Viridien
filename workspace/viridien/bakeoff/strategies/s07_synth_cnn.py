"""Per-log swatch-trained CNN with sequence decoding and ink boundary snapping."""
import time
import cv2
import numpy as np
import torch
from torch import nn
from harness import io, viz

NAME = 'S7 swatch-trained CNN'
NEEDS_GPU = True
PAID_API = None


class Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.features(x))


def train(entries, device, mono):
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    swatches = [e.swatch for e in entries] + [np.full((32, 32, 3), 255, np.uint8)]
    # Periodic extension models the repeated motifs used in printed columns.
    tiles = [np.tile(s, (max(2, 100 // s.shape[0] + 1), max(2, 100 // s.shape[1] + 1), 1)) for s in swatches]
    samples, targets = [], []
    for idx, tile in enumerate(tiles):
        for j in range(96):
            size = 32 if j % 4 else int(rng.integers(29, 36))
            y = int(rng.integers(0, tile.shape[0] - size + 1))
            x = int(rng.integers(0, tile.shape[1] - size + 1))
            im = cv2.resize(tile[y:y+size, x:x+size], (32, 32), interpolation=cv2.INTER_AREA)
            if j % 8 == 0:
                matrix = cv2.getRotationMatrix2D((16, 16), float(rng.uniform(-5, 5)), 1)
                im = cv2.warpAffine(im, matrix, (32, 32), borderMode=cv2.BORDER_REFLECT)
            if j % 7 == 0:
                im[int(rng.integers(0, 32))] = rng.integers(0, 90)
            if mono and j % 3 == 0:
                im = (im > int(rng.integers(150, 220))).astype(np.uint8) * 255
            if j % 5 == 0:
                im = np.clip(im.astype(np.int16) + rng.integers(-8, 9), 0, 255).astype(np.uint8)
            samples.append(im); targets.append(idx)
    x = torch.from_numpy(np.asarray(samples).transpose(0, 3, 1, 2).copy()).to(device, dtype=torch.float32).div_(255)
    y = torch.tensor(targets, device=device)
    net = Net(len(swatches)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    start = time.perf_counter()
    for step in range(160):
        if time.perf_counter() - start > 20: break
        idx = torch.randint(len(y), (96,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(net(x[idx]), y[idx])
        loss.backward(); opt.step()
    net.eval()
    return net


def run(log, window, ctx):
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    if not entries: return []
    torch.set_num_threads(2)
    device = ctx['device']
    net = train(entries, device, log.mode == '1')
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *window)
    h, w = rgb.shape[:2]
    # Remove edge rules and average independent patch votes across the track.
    left, right = int(w*.08), max(int(w*.92), int(w*.08)+1)
    core = rgb[:, left:right]
    padded = cv2.copyMakeBorder(core, 16, 16, 16, 16, cv2.BORDER_REPLICATE)
    centers = np.arange(0, h, 8)
    xs = np.linspace(0, max(0, core.shape[1]-32), min(5, max(1, core.shape[1]//24)), dtype=int)
    patches = np.asarray([padded[y:y+32, x+16:x+48] for y in centers for x in xs], dtype=np.uint8)
    if patches.shape[2] != 32:
        patches = np.asarray([cv2.resize(p, (32, 32)) for p in patches])
    logits = []
    with torch.no_grad():
        for start in range(0, len(patches), 512):
            batch = torch.from_numpy(patches[start:start+512].transpose(0, 3, 1, 2).copy()).to(device, dtype=torch.float32).div_(255)
            logits.append(net(batch).cpu().numpy())
    emissions = np.concatenate(logits).reshape(len(centers), len(xs), -1).mean(axis=1)
    emissions -= np.max(emissions, axis=1, keepdims=True)
    # Viterbi discourages short, noisy runs while retaining strong evidence.
    penalty = 12.0
    scores = emissions[0].copy(); back = np.zeros(emissions.shape, np.int32)
    for i in range(1, len(centers)):
        transition = scores[:, None] - penalty * (1 - np.eye(len(scores)))
        back[i] = transition.argmax(axis=0)
        scores = emissions[i] + transition.max(axis=0)
    states = np.empty(len(centers), np.int32); states[-1] = scores.argmax()
    for i in range(len(centers)-1, 0, -1): states[i-1] = back[i, states[i]]
    # Strong horizontal rules and blank/nonblank transitions localize edges.
    dark = (core.mean(axis=2) < 150).mean(axis=1)
    white = (core.min(axis=2) > 245).mean(axis=1)
    edges = [0]
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    for i in changes:
        nominal = int((centers[i-1] + centers[i]) // 2)
        a, b = max(edges[-1]+1, nominal-20), min(h, nominal+21)
        if b <= a: edges.append(nominal); continue
        local = dark[a:b]
        if local.max() > .55:
            edge = a + int(np.argmax(local))
        else:
            grad = np.abs(np.diff(white, prepend=white[0]))
            edge = a + int(np.argmax(grad[a:b])) if grad[a:b].max() > .25 else nominal
        edges.append(edge)
    edges.append(h)
    rowstates = states[np.r_[0, changes]]
    rows = []
    for a, b, state in zip(edges[:-1], edges[1:], rowstates):
        if state >= len(entries) or b <= a: continue
        e = entries[state]
        rows.append(dict(top_y=window[0]+int(a), bottom_y=window[0]+int(b), lithology=e.label, cluster_id=e.id))
    view = cv2.resize(rgb, (min(180, w), min(800, h)), interpolation=cv2.INTER_AREA)
    track = np.full((h, 36, 3), 255, np.uint8)
    raw = emissions.argmax(axis=1)
    for j, y in enumerate(centers):
        for x0, state in [(0, raw[j]), (18, states[j])]:
            if state < len(entries): track[y:min(y+8,h), x0:x0+18] = viz.colour(entries[state].label)
    track = cv2.resize(track, (36, min(800,h)), interpolation=cv2.INTER_NEAREST)
    sw = [cv2.resize(e.swatch, (48, 32)) for e in entries[:20]]
    gallery = np.concatenate(sw, axis=1) if sw else np.full((32,32,3),255,np.uint8)
    return rows, {'1 swatches': gallery, '2 input': view, '3 CNN / Viterbi': track, '4 intervals': track[:, 18:].copy()}
