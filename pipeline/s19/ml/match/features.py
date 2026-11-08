"""Portable, phase-tolerant patch descriptions and learned pair features.

No labels, filenames, or truth paths enter these functions. Tiny swatches are
repeated at native resolution, rather than stretched to the column width.
"""
import cv2
import numpy as np
from scipy.special import expit
from strategies.s11_fft_legend import descriptor as fft_descriptor, distance

RUN_BINS = [1, 2, 3, 4, 6, 9, 13, 21, 33, 1000000]
FEATURE_NAMES = ['fft_distance', 'fft_multiscale', 'fft_spectrum', 'ink82',
                 'colour_hist', 'median_lab', 'peak', 'light_lab', 'mean_lab',
                 'gray_hist', 'ink50', 'ink70', 'ink90', 'h_dark', 'v_dark',
                 'h_gap', 'v_gap', 'orientation', 'lbp', 'colourfulness',
                 'texture', 'ref_ink', 'query_ink', 'ref_texture', 'query_texture',
                 'ref_saturation', 'query_saturation']


class LinearPairScorer:
    """Numeric checkpoint format avoids scikit-learn pickle version coupling."""
    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    def predict_proba(self, x):
        # Match StandardScaler's two in-place float32 operations exactly.
        z = np.array(x, dtype=np.float32, copy=True)
        z -= self.mean
        z /= self.scale
        p = expit((z @ self.coef.reshape(-1, 1)).ravel() + self.intercept)
        return np.column_stack((1-p, p))


def run_hist(mask):
    m = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    lengths = np.where(d == -1)[1] - np.where(d == 1)[1]
    h = np.histogram(lengths, bins=RUN_BINS, weights=lengths)[0].astype(np.float32)
    return h / max(1, h.sum())


def describe(rgb, trim=False):
    rgb = np.asarray(rgb, np.uint8)
    if trim:
        t = min(4, len(rgb)//8)
        e = max(1, rgb.shape[1]//12)
        if t: rgb = rgb[t:-t]
        if rgb.shape[1] > e*2: rgb = rgb[:, e:-e]
    # Bound descriptor work while retaining native pixels (no tall RGB copies).
    if len(rgb) > 512:
        starts = np.linspace(0, len(rgb)-128, 4, dtype=int)
        rgb = np.concatenate([rgb[s:s+128] for s in starts])
    h, w = rgb.shape[:2]
    if min(h,w) < 2: rgb = np.tile(rgb, (2,2,1)); h,w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)/255
    dark = gray < .5
    light = lab[~dark]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ink = np.array([(gray<t).mean() for t in (.5,.7,.82,.9)], np.float32)
    ch = np.histogramdd(lab.reshape(-1,3), bins=(4,5,5),range=((0,1),(0,1),(0,1)))[0].ravel().astype(np.float32)
    ch /= max(1, ch.sum())
    gh = np.histogram(gray, bins=12,range=(0,1))[0].astype(np.float32); gh/=gh.sum()
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(gx,gy); angle=(np.arctan2(gy,gx)%np.pi)/np.pi
    orient=np.histogram(angle,bins=12,range=(0,1),weights=mag)[0];orient/=max(1e-8,orient.sum())
    center=gray[1:-1,1:-1]; code=np.zeros(center.shape,np.uint8)
    for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        code += (gray[1+dy:h-1+dy,1+dx:w-1+dx] > center+.025).astype(np.uint8)
    lbp=np.bincount(code.ravel(),minlength=9).astype(np.float32);lbp/=max(1,lbp.sum())
    patches=[]
    for y in np.linspace(0,max(0,h-64),min(4,max(1,h//64)),dtype=int):
        x=max(0,(w-64)//2);patches.append(rgb[y:y+64,x:x+64])
    fft=[fft_descriptor(p,False) for p in patches]
    middle=patches[len(patches)//2]
    scales=[fft_descriptor(cv2.resize(middle,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR),False) for s in (.7,1.4)]
    return dict(fft=fft, scales=scales, ink=ink, hist=ch, gray=gh,
                light=np.median(light,axis=0) if len(light) else np.zeros(3),
                mean=lab.mean(axis=(0,1)), median=np.median(lab.reshape(-1,3),axis=0),
                runs=[run_hist(dark),run_hist(dark.T),run_hist(~dark),run_hist(~dark.T)],
                orient=orient,lbp=lbp,saturation=float(np.mean(hsv[:,:,1])/255),texture=float(mag.mean()))


def pair_features(q, r, mono=False):
    tv=lambda a,b: float(.5*np.abs(a-b).sum())
    norm=lambda a,b: float(np.linalg.norm(a-b))
    ds=[distance(d,r['fft'][0],mono) for d in q['fft']]
    multi=min(float(np.median([distance(d,ref,mono) for d in q['fft']])) for ref in r['fft'][:1]+r['scales'])
    spec=np.median([min(norm(d[0],np.roll(r['fft'][0][0],s,axis=0)) for s in (-1,0,1)) for d in q['fft']])
    return np.asarray([np.median(ds),multi,spec,abs(q['ink'][2]-r['ink'][2]),
        tv(q['hist'],r['hist']),norm(q['median'],r['median']),abs(q['fft'][0][4]-r['fft'][0][4]),
        norm(q['light'],r['light']),norm(q['mean'],r['mean']),tv(q['gray'],r['gray']),
        abs(q['ink'][0]-r['ink'][0]),abs(q['ink'][1]-r['ink'][1]),abs(q['ink'][3]-r['ink'][3]),
        *[tv(a,b) for a,b in zip(q['runs'],r['runs'])],tv(q['orient'],r['orient']),tv(q['lbp'],r['lbp']),
        abs(q['saturation']-r['saturation']),abs(q['texture']-r['texture']),r['ink'][0],q['ink'][0],
        r['texture'],q['texture'],r['saturation'],q['saturation']],np.float32)


def matrix(query, refs, mono=False):
    return np.asarray([pair_features(query,ref,mono) for ref in refs])


def emil_cost(x):
    # Fully dark solid colours have no "light" pixels. Their median Lab
    # remains informative, whereas the empty light-pixel median is zero.
    solid = (x[:,23] < .05) & (x[:,24] < .05)
    colour = np.where(solid, x[:,5], x[:,7])
    return 2*colour + 4*x[:,10] + x[:,13:17].mean(axis=1)
