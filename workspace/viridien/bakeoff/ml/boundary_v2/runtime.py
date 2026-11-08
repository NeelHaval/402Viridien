"""Portable CPU inference. This module does not read truth or split files."""
from functools import lru_cache
from pathlib import Path
import torch
from ml.boundary.model import BoundaryNet, features, peaks, probabilities

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load():
    torch.set_num_threads(min(3, torch.get_num_threads()))
    payload = torch.load(HERE / 'boundary_net.pt', map_location='cpu', weights_only=True)
    net = BoundaryNet(payload['channels'], payload['hidden'])
    net.load_state_dict(payload['state_dict'])
    return net.eval(), payload['config']


def propose(rgb, net, config):
    prob = probabilities(net, features(rgb, 'rich'))
    return peaks(prob, config['threshold'], config['gap']), prob
