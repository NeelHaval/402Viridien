"""Load the S21 models. No truth files are used."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from ml.boundary.model import BoundaryNet
from ml.match.features import LinearPairScorer

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load():
    torch.set_num_threads(min(3, torch.get_num_threads()))
    cfg = json.loads((HERE/'selected.json').read_text())
    p = torch.load(HERE/'boundary_net.pt', map_location='cpu', weights_only=True)
    net = BoundaryNet(p['channels'], p['hidden'])
    net.load_state_dict(p['state_dict'])
    with np.load(HERE/'matcher.npz', allow_pickle=False) as z:
        matcher = LinearPairScorer(**{k:z[k] for k in ('mean','scale','coef','intercept')})
    return net.eval(), matcher, cfg
