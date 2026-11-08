"""Small CPU boundary models and portable, truth-free row feature extraction."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from torch import nn

HERE = Path(__file__).resolve().parent


def raw_features(rgb):
    """Emil's six features, after trimming persistent vertical borders."""
    dark = rgb.mean(2, dtype=np.float32) < 100
    idx = np.flatnonzero(dark.mean(0) > .6)
    mid = rgb.shape[1] // 2
    left, right = idx[idx < mid], idx[idx >= mid]
    a = int(left.max()) + 3 if left.size else 0
    b = int(right.min()) - 2 if right.size else rgb.shape[1]
    col = rgb[:, a:b] if b-a >= 10 else rgb
    dark = col.mean(2, dtype=np.float32) < 128
    return np.column_stack([
        col.mean(1, dtype=np.float32)/255, dark.mean(1),
        (dark[:, 1:] != dark[:, :-1]).mean(1) if col.shape[1] > 1 else np.zeros(len(col)),
        np.r_[0., (dark[1:] != dark[:-1]).mean(1)],
    ]).astype(np.float32)


def standardize(x):
    return ((x-x.mean(0))/np.maximum(x.std(0), .03)).astype(np.float32)


def features(rgb, kind='rich'):
    raw = raw_features(rgb)
    if kind == 'six':
        # Match Emil's window-wise normalization exactly.
        scale = raw.std(0)
        scale[scale < 1e-6] = 1
        return ((raw-raw.mean(0))/scale).astype(np.float32)
    w = rgb.shape[1]
    inset = max(1, w//16)
    interior = rgb[:, inset:-inset] if w > 2*inset else rgb
    gray = cv2.cvtColor(interior, cv2.COLOR_RGB2GRAY)
    panels = [raw_features(p) for p in np.array_split(interior, min(3, interior.shape[1]), axis=1)]
    # Preserve panel colour changes and rule extent; multiscale differences
    # distinguish a real change in texture from one dark row in a hatch.
    base = np.column_stack([raw, *panels, (gray < 65).mean(1), (gray < 220).mean(1)])
    z = standardize(base)
    changes = []
    for radius in (4, 12, 32):
        sm = uniform_filter1d(z, size=radius, axis=0, mode='nearest')
        above = sm[np.maximum(0, np.arange(len(sm))-radius//2)]
        below = sm[np.minimum(len(sm)-1, np.arange(len(sm))+radius//2)]
        changes.extend([np.abs(below-above).mean(1), np.abs(below-above).max(1)])
    return np.column_stack([z, *changes]).astype(np.float32)


class BoundaryNet(nn.Module):
    def __init__(self, channels=32, hidden=24):
        super().__init__()
        layers = []
        for dilation in (1, 2, 4, 8, 16, 32):
            layers.extend([nn.Conv1d(channels, hidden, 3, padding=dilation, dilation=dilation), nn.ReLU()])
            channels = hidden
        layers.append(nn.Conv1d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.inference_mode()
def probabilities(net, feat):
    """Overlap chunks so even full-height logs have bounded activation memory."""
    out = np.empty(len(feat), np.float32)
    for a in range(0, len(feat), 8192):
        b = min(len(feat), a+8192)
        lo, hi = max(0, a-128), min(len(feat), b+128)
        x = torch.from_numpy(feat[lo:hi].T.copy()[None])
        out[a:b] = net(x).sigmoid().numpy()[0, a-lo:b-lo]
    return out


def peaks(prob, threshold=.5, gap=12):
    return find_peaks(prob, height=threshold, distance=gap, prominence=.03)[0]


@lru_cache(maxsize=1)
def load_model():
    torch.set_num_threads(min(3, torch.get_num_threads()))
    payload = torch.load(HERE/'boundary_net.pt', map_location='cpu', weights_only=True)
    net = BoundaryNet(payload['channels'], payload['hidden'])
    net.load_state_dict(payload['state_dict'])
    return net.eval(), payload['config']
