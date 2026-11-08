"""Native RGB palettes and CIE76 (ordinary, unscaled Lab) fill checks."""
import sys
sys.dont_write_bytecode = True
import cv2
import numpy as np

DELTA_E = 15.
MIN_SHARE = .08
SHARE_RATIO = .5


def lab(rgb):
    return cv2.cvtColor(np.asarray(rgb, np.float32).reshape(1, -1, 3) / 255.,
                        cv2.COLOR_RGB2LAB).reshape(-1, 3)


def profile(rgb):
    pixels = np.asarray(rgb, np.uint8).reshape(-1, 3)
    # Keep exact pixel multiplicities for deltaE shares. Quantization is used
    # only to identify palette modes, never to approximate gate distances.
    packed = pixels.astype(np.uint32) @ np.array([65536, 256, 1], np.uint32)
    unique, counts = np.unique(packed, return_counts=True)
    colours = np.stack((unique >> 16, (unique >> 8) & 255, unique & 255), axis=1).astype(np.uint8)
    weights = counts / counts.sum()
    codes = (colours.astype(int) // 32) @ np.array([64, 8, 1])
    mass = np.bincount(codes, weights=weights, minlength=512)
    centers = np.stack([np.bincount(codes, weights=weights * colours[:, j], minlength=512)
                        for j in range(3)], axis=1) / np.maximum(mass[:, None], 1e-12)
    order = np.argsort(-mass)
    order = [i for i in order if mass[i] > 0 and centers[i].max() >= 60]
    modes = lab(centers[order]) if order else np.empty((0, 3))
    palette = modes[mass[order] >= MIN_SHARE]
    labs = lab(colours)
    main = modes[0] if len(modes) else None
    share = float(weights[np.linalg.norm(labs-main, axis=1) <= DELTA_E].sum()) if main is not None else 0.
    binary = bool(weights[np.all((colours <= 8) | (colours >= 247), axis=1)
                          & (np.ptp(colours.astype(int), axis=1) <= 8)].sum() >= .995)
    return dict(labs=labs, weights=weights, palette=palette, main=main,
                share=share, binary=binary, modes=modes, mode_shares=mass[order])


def colour_gate(reference, crop, shift=None):
    """Return (passes, main-fill share, palette similarity); ink is excluded."""
    shift = np.zeros(3) if shift is None else np.asarray(shift)
    if reference['main'] is None or crop['main'] is None:
        return reference['binary'], 0., 0.
    distances = np.linalg.norm(crop['labs'] - (reference['main'] + shift), axis=1)
    share = float(crop['weights'][distances <= DELTA_E].sum())
    palette = reference['palette'] + shift
    dominant_ok = bool(len(palette) and np.min(np.linalg.norm(palette-crop['main'], axis=1)) <= DELTA_E)
    passed = share >= SHARE_RATIO * reference['share'] and dominant_ok
    # Symmetric nearest-colour transport rewards matching secondary colours,
    # so two patterns with the same background can still be distinguished.
    a, b = reference['modes'] + shift, crop['modes']
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=2)
    wa, wb = reference['mode_shares'], crop['mode_shares']
    error = .5 * (np.average(distances.min(axis=1), weights=wa)
                  + np.average(distances.min(axis=0), weights=wb))
    similarity = float(np.exp(-error / DELTA_E))
    return bool(reference['binary'] or passed), share, similarity


def contrastive(scores, pid, twins=()):
    rivals = [value for other, value in scores.items() if other != pid and other not in twins]
    # A legend consisting entirely of twins has no discriminable alternative.
    return float(scores[pid] - max(rivals)) if rivals else float(scores[pid])


class PaletteMatcher:
    """Batch the identical gate over a whole legend/family during wide search."""
    def __init__(self, references, shifts):
        self.refs = references
        self.shifts = np.asarray(shifts)
        self.main = np.array([r['main'] if r['main'] is not None else [1e5]*3 for r in references]) + self.shifts
        self.required = np.array([r['share']*.5 for r in references])
        self.binary = np.array([r['binary'] for r in references])
        self.valid = [i for i, r in enumerate(references) if len(r['modes'])]
        self.starts = np.cumsum([0]+[len(references[i]['modes']) for i in self.valid])[:-1]
        self.palette_ids = np.array([i for i, r in enumerate(references) for _ in r['palette']], int)
        self.palette = np.array([v+self.shifts[i] for i, r in enumerate(references) for v in r['palette']]).reshape(-1,3)
        self.modes = np.array([v+self.shifts[i] for i in self.valid for v in references[i]['modes']]).reshape(-1,3)
        self.weights = np.array([v/references[i]['mode_shares'].sum() for i in self.valid for v in references[i]['mode_shares']])

    def __call__(self, crop):
        distances = np.linalg.norm(self.main[:, None]-crop['labs'][None, :], axis=2)
        shares = (distances <= DELTA_E) @ crop['weights']
        nearest = np.full(len(self.refs), np.inf)
        scores = np.zeros(len(self.refs))
        if crop['main'] is not None and len(self.palette):
            np.minimum.at(nearest, self.palette_ids, np.linalg.norm(self.palette-crop['main'], axis=1))
        if len(crop['modes']) and self.valid:
            distances = np.linalg.norm(self.modes[:, None]-crop['modes'][None, :], axis=2)
            forward = np.add.reduceat(distances.min(axis=1)*self.weights, self.starts)
            reverse = np.average(np.minimum.reduceat(distances, self.starts, axis=0), axis=1,
                                 weights=crop['mode_shares'])
            scores[self.valid] = np.exp(-.5*(forward+reverse)/DELTA_E)
        passed = self.binary | ((shares >= self.required) & (nearest <= DELTA_E))
        return passed, shares, scores
