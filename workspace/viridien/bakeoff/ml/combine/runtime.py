"""S19 inference. This module does not open labels or split files."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import threading

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from harness import io, legend_v3, depth
from ml.boundary.model import BoundaryNet, features, probabilities
from ml.match.features import LinearPairScorer
from ml.integrate.core import entries_only, evidence, boundary_points, decode
from strategies.s14_learned_match import proposals

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
PARSER_CONFIG = dict(easy=True, cleanup=True, easy_weight=.9)
_PARSER_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _easy_reader():
    import easyocr
    # Recognition weights are preinstalled. Keep the user-network folder owned.
    return easyocr.Reader(['en'],gpu=False,detector=False,verbose=False,
                          download_enabled=False,
                          user_network_directory=str(HERE/'easyocr_user'))


def parsed(log):
    """Use v3 with fixed defaults and keep each new cache in this lane."""
    rgb = io.read_rgb(log, 'lithology_legend')
    digest = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    filename = f'v{legend_v3.VERSION}_{digest}.json'
    destination = HERE/'legend_cache'/'cache'/filename
    # Reuse raw image OCR only. Do not load the old selected parser settings.
    source = BAKEOFF/'ml'/'legend_names'/'cache'/filename
    if not destination.exists() and source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    with _PARSER_LOCK:
        old = legend_v3.HOME
        old_easy = legend_v3._easy
        try:
            legend_v3.HOME = HERE/'legend_cache'
            legend_v3._easy = _easy_reader
            return legend_v3.parse(log, config=PARSER_CONFIG)
        finally:
            legend_v3.HOME = old
            legend_v3._easy = old_easy


@lru_cache(maxsize=1)
def load():
    torch.set_num_threads(min(3, torch.get_num_threads()))
    config = json.loads((HERE/'selected.json').read_text(encoding='utf-8'))
    payload = torch.load(HERE/'boundary_net.pt', map_location='cpu', weights_only=True)
    net = BoundaryNet(payload['channels'], payload['hidden'])
    net.load_state_dict(payload['state_dict'])
    with np.load(HERE/'matcher.npz', allow_pickle=False) as data:
        matcher = LinearPairScorer(**dict(data))
    if config.get('model_mix',1.) < 1.:
        real_payload=torch.load(HERE/'boundary_real.pt',map_location='cpu',weights_only=True)
        real_net=BoundaryNet(real_payload['channels'],real_payload['hidden'])
        real_net.load_state_dict(real_payload['state_dict'])
        net=(net.eval(),real_net.eval())
        with np.load(HERE/'matcher_real.npz',allow_pickle=False) as data:
            real_matcher=LinearPairScorer(**dict(data))
        matcher=MixedScorer(matcher,real_matcher,config['model_mix'])
        return net,matcher,config
    return net.eval(), matcher, config


class MixedScorer:
    """Blend two clean visual models. No labels are read at inference."""
    def __init__(self,mixed,real,weight):
        self.mixed=mixed;self.real=real;self.weight=float(weight)

    def predict_proba(self,x):
        return self.weight*self.mixed.predict_proba(x)+(1-self.weight)*self.real.predict_proba(x)


def boundary_probabilities(net,feat,config):
    if isinstance(net,tuple):
        weight=config['model_mix']
        return weight*probabilities(net[0],feat)+(1-weight)*probabilities(net[1],feat)
    return probabilities(net,feat)


def fit_depth_points(points):
    """Rank equal-support fits by pixel error before anchor span.

    Span breaks ties only. A long span must not beat a much more exact scale.
    Count distinct pixel rows, so rotated OCR cannot add independent votes.
    """
    points = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        raise ValueError('Need two depth anchors')
    y, d = points.T
    pairs = [(i, j) for i in range(len(points)) for j in range(i)]
    if len(pairs) > 4000:
        rng = np.random.default_rng(19)
        pairs = [pairs[i] for i in rng.choice(len(pairs), 4000, replace=False)]
    best = None
    for i, j in pairs:
        if abs(y[i]-y[j]) < max(80, np.ptp(y)*.08):
            continue
        a = (d[j]-d[i])/(y[j]-y[i])
        if not .0001 < a < 20:
            continue
        b = d[i]-a*y[i]
        good = np.abs((d-b)/a-y) < 16
        # Refit before ranking. All hypotheses get the same residual measure.
        for _ in range(2):
            a, b = np.polyfit(y[good], d[good], 1)
            if not .0001 < a < 20:
                break
            revised = np.abs((d-b)/a-y) < 16
            if revised.sum() < 2:
                break
            good = revised
        if not .0001 < a < 20:
            continue
        residual = np.abs((d[good]-b)/a-y[good])
        count = len(np.unique(np.round(y[good]/5)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        rank = (count, -round(rmse, 2), float(np.ptp(y[good])))
        if best is None or rank > best[0]:
            best = (rank, float(a), float(b), count, rmse)
    minimum = 2 if len(points) == 2 else 3
    if best is None or best[3] < minimum or best[4] > 16:
        raise ValueError('No consistent depth scale')
    return best[1:]


def checked_depth(log, calib):
    """Audit weak calibration from bounded depth-track crops, using no labels."""
    if calib is None or 'depth_column' not in log.boxes:
        return calib
    x0, y0, x1, y1 = map(int, log.boxes['depth_column'])
    suspicious = (calib.n_points <= 3 or calib.rmse_px > 4 or
                  calib.to_depth(y0) < 0)
    if not suspicious:
        return calib
    points = []
    starts = np.unique(np.linspace(y0, max(y0, y1-1800), 3, dtype=int))
    for top in starts:
        rgb = io.read_rgb(log, [x0,y0,x1,y1], int(top), min(y1,int(top)+1800))
        points.extend(depth._read_points(rgb, int(top), 'rapidocr'))
    try:
        a, b, n, rmse = fit_depth_points(points)
    except ValueError:
        return calib
    # Do not infer units from a file name or a nearby well.
    if '+m-to-ft' in calib.method:
        a *= depth.METRES_TO_FEET
        b *= depth.METRES_TO_FEET
    improved = (rmse < calib.rmse_px*.5 or
                (n > calib.n_points and rmse <= max(1.,calib.rmse_px)))
    if n >= 3 and improved and a*y0+b >= 0:
        return depth.Calib(a,b,n,rmse,'s19+residual-consensus'+
                           ('+m-to-ft' if '+m-to-ft' in calib.method else '+ft'))
    return calib


def predict(rgb, entries, prob, s11, rules, config, matcher):
    """Run the same decoder in validation and deployment."""
    entries = entries_only(entries)
    if not entries:
        return []
    points = boundary_points(prob, s11, config['mode'], config['threshold'], config['gap'])
    ev = evidence(rgb, points, entries)
    return decode(ev, entries, prob, rules, config, model=matcher)


@threadpool_limits.wrap(limits=3)
def run(log, window, ctx):
    cv2.setNumThreads(1)
    torch.set_num_threads(min(3, torch.get_num_threads()))
    net, matcher, config = load()
    y0, y1 = map(int, window)
    entries = parsed(log)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    prob = boundary_probabilities(net, features(rgb, 'rich'),config)
    s11, rules = proposals(log, window, ctx, rgb)
    rows = predict(rgb, entries, prob, s11, rules, config, matcher)
    calib = checked_depth(log, ctx.get('calib'))
    for row in rows:
        row['top_y'] += y0
        row['bottom_y'] += y0
        if calib is not None:
            for edge in ('top', 'bottom'):
                row[edge+'_depth'] = round(calib.to_depth(row[edge+'_y']), 1)
    return rows
