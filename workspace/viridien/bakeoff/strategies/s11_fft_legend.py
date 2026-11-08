"""Phase-tolerant legend matching using directional Fourier magnitudes."""
import cv2
import numpy as np
from PIL import Image, ImageDraw
from harness import io
from strategies.s02_features_pelt import segment

NAME = 'S11 FFT hatch + Lab legend'
NEEDS_GPU = False
PAID_API = None

N = 64
fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing='ij')
rad = np.hypot(fx, fy)
ang = (np.arctan2(fy, fx) % np.pi) / np.pi
window2 = np.outer(np.hanning(N), np.hanning(N))


def descriptor(patch, mono):
    if min(patch.shape[:2]) < 2:
        patch = np.tile(patch, (2, 2, 1))
    h, w = patch.shape[:2]
    # Repetition preserves the native motif scale for small legend interiors.
    if h < N or w < N:
        patch = np.tile(patch, (max(1, (N+h-1)//h), max(1, (N+w-1)//w), 1))
    patch = patch[:N, :N]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    ink = float(np.mean(gray < .82))
    f = np.abs(np.fft.fft2((gray-gray.mean()) * window2))
    # Divide by the dominant radial frequency: the angular/radial pattern is
    # insensitive to a small global scale change while preserving direction.
    radial = np.array([f[(rad >= a) & (rad < a+1/N)].sum() for a in np.arange(1/N, .4, 1/N)])
    peak = (np.argmax(radial) + 1) / N if radial.any() else .1
    rel = rad / max(peak, 2/N)
    bins = np.zeros((8, 12), np.float32)
    rb = np.clip((np.log2(np.maximum(rel, .25)) + 2) * 2, 0, 7.999).astype(int)
    ab = np.minimum((ang * 12).astype(int), 11)
    valid = (rad >= 2/N) & (rad < .48)
    np.add.at(bins, (rb[valid], ab[valid]), f[valid])
    bins = np.sqrt(bins)
    bins /= np.linalg.norm(bins) + 1e-8
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    hist = np.histogramdd(lab.reshape(-1, 3), bins=(4, 5, 5), range=((0,256),(0,256),(0,256)))[0].ravel().astype(np.float32)
    hist /= hist.sum() + 1e-8
    colour = np.median(lab.reshape(-1,3), axis=0).astype(np.float32) / 255
    return bins, ink, hist, colour, peak


def distance(d, ref, mono):
    # Jitter in the radial dimension models residual uncertainty in peak period.
    spec = min(float(np.linalg.norm(d[0] - np.roll(ref[0], shift, axis=0))) for shift in (-1, 0, 1)) / 1.414
    value = .55 * spec + .45 * min(1., abs(d[1]-ref[1]) * 2)
    if not mono:
        hd = .5 * np.abs(d[2]-ref[2]).sum()
        cd = min(1., np.linalg.norm(d[3]-ref[3]) * 2)
        value = .45 * value + .35 * hd + .20 * cd
    return float(value)


def panel(images, labels, size=(90,90)):
    width = min(6, max(1, len(images)))
    canvas = Image.new('RGB', (width*size[0], max(1, (len(images)+width-1)//width)*(size[1]+18)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i%width)*size[0], (i//width)*(size[1]+18)
        im = Image.fromarray(im).convert('RGB').resize(size, Image.Resampling.NEAREST)
        canvas.paste(im, (x,y)); draw.text((x,y+size[1]),str(label)[:13],fill='black')
    return np.asarray(canvas)


def spectrum_image(d):
    a = d[0]; a = (255*a/(a.max()+1e-8)).astype(np.uint8)
    return cv2.applyColorMap(a, cv2.COLORMAP_VIRIDIS)[:,:,::-1].copy()


def run(log, window, ctx):
    y0, y1 = map(int, window)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    if not entries:
        return [], {'1 column': rgb}
    # RGB synthetic images may still originate from a monochrome source.
    mono = log.mode == '1' or np.mean(np.ptp(rgb[::max(1,len(rgb)//1000)].astype(np.int16), axis=2) > 12) < .01
    refs = [descriptor(e.swatch, mono) for e in entries]
    intervals = segment(log, window, ctx)
    # Very dark, nearly full-width rules are reliable boundary proposals.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inset = min(max(1, rgb.shape[1]//12), max(1, rgb.shape[1]//3))
    dark = (gray[:, inset:-inset] < 65).mean(axis=1) if rgb.shape[1] > 2*inset else (gray < 65).mean(axis=1)
    candidate = np.flatnonzero((dark > .88) & (np.r_[0,dark[:-1]] <= .88))
    # Reject rules repeated at regular hatch spacing, and rules within solid fill.
    candidate = np.array([p for i,p in enumerate(candidate)
                          if (i == 0 or p-candidate[i-1] > 35)
                          and (i+1 == len(candidate) or candidate[i+1]-p > 35)
                          and np.mean(dark[max(0,p-8):min(len(dark),p+9)]) < .65], dtype=int)
    points = sorted(set([y0, y1] + [a for a,b in intervals] + [int(y0+i) for i in candidate if 0 < i < len(rgb)-1]))
    # Fuse proposals that refer to the same line.
    merged = [points[0]]
    for p in points[1:-1]:
        if p - merged[-1] >= 20 and y1-p >= 20: merged.append(p)
    merged.append(y1)
    rows, scores, descs, margins = [], [], [], []
    unknown_refs = []
    for a,b in zip(merged,merged[1:]):
        strip = rgb[a-y0:b-y0]
        if not strip.size: continue
        edge = max(1, strip.shape[1]//12)
        interior = strip[:,edge:-edge] if strip.shape[1] > edge*2 else strip
        if np.mean(np.min(interior,axis=2)<235) < .006: continue
        h = len(strip)
        starts = np.linspace(max(0,min(5,h//8)), max(0,h-min(64,h)-min(5,h//8)), min(8,max(1,h//64)), dtype=int)
        ds = [descriptor(strip[s:min(h,s+64), max(0,(strip.shape[1]-64)//2):max(0,(strip.shape[1]-64)//2)+64],mono) for s in starts]
        dist = np.median(np.array([[distance(d,r,mono) for r in refs] for d in ds]),axis=0)
        order = np.argsort(dist); best = int(order[0])
        margin = float(dist[order[1]]-dist[best]) if len(order)>1 else 1.
        # A small synthetic-calibrated margin rejects essentially tied matches.
        unknown = margin < .001 or dist[best] > .65
        if unknown:
            d = ds[len(ds)//2]
            nearest = [distance(d,r,mono) for r in unknown_refs]
            base = max(e.id for e in entries)+1
            if nearest and min(nearest)<.12: cid = base+int(np.argmin(nearest))
            else: cid = base+len(unknown_refs); unknown_refs.append(d)
            label = 'Unknown'
        else:
            cid = entries[best].id; label = entries[best].label
        rows.append(dict(top_y=a,bottom_y=b,lithology=label,cluster_id=cid))
        scores.append(dist); descs.append(ds[len(ds)//2]); margins.append(margin)
    # Same label runs are merged unless a clear line explicitly separates them.
    result = []
    lines = set((candidate+y0).tolist())
    for row in rows:
        if result and result[-1]['bottom_y']==row['top_y'] and result[-1]['lithology']==row['lithology'] and not any(abs(row['top_y']-p)<=2 for p in lines):
            result[-1]['bottom_y']=row['bottom_y']
        else: result.append(row)
    sw = panel([spectrum_image(d) for d in refs], [e.label for e in entries])
    inter = panel([spectrum_image(d) for d in descs[:24]], [f'{i}: {m:.3f}' for i,m in enumerate(margins[:24])])
    if scores:
        heat = np.asarray(scores,dtype=np.float32)
        heat = np.uint8(np.clip(heat*255,0,255))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)[:,:,::-1].copy()
        heat = cv2.resize(heat,(max(120,len(entries)*15),max(80,min(500,len(scores)*12))),interpolation=cv2.INTER_NEAREST)
    else: heat = np.full((40,120,3),255,np.uint8)
    viewheight = min(1000, max(200, len(rgb)))
    final = Image.fromarray(rgb).resize((max(120, rgb.shape[1]), viewheight))
    draw = ImageDraw.Draw(final)
    for r in result:
        yy=round((r['top_y']-y0)*viewheight/(y1-y0))
        draw.line((0,yy,final.width,yy), fill='red', width=2)
        draw.text((2,yy+2),r['lithology'][:18],fill='blue',stroke_width=1,stroke_fill='white')
    return result, {'1 swatch spectra': sw, '2 interval spectra (margin)': inter, '3 distances': heat, '4 final labels': np.asarray(final)}
