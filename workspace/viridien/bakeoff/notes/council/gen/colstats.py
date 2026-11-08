"""Per-log column statistics from 120 sampled 48-row windows (read only)."""
import sys, csv, collections
import numpy as np
sys.path.insert(0, '.')
from harness import io
from labeltool import legend_classify as C
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(x['log']): x['family'] for x in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8'))}
out = []
for name in io.list_logs():
    log = io.open_log(name)
    b = log.boxes['interpreted_lithology_column']
    y0, y1 = int(b[1]), int(b[3])
    st = collections.Counter(); col = []; ratio_a = []; ratio_g = []; d1 = []; marg = []
    for y in np.linspace(y0, y1 - 48, 120).astype(int):
        a = io.read_rgb(log, 'interpreted_lithology_column', y0=int(y), y1=int(y) + 48)[..., :3]
        ai = a.astype(int)
        paper = ai.min(2) > 235
        fill = (ai.max(2) >= C.P['fill_dark']) & ~paper
        col.append((fill.mean(1) > 0.2).mean())
        w0 = a.shape[1]; s = C.strip_borders(a)
        state = 'blank' if s.shape[1] < 0.6 * w0 else C.window_state(s)
        st[state] += 1
        if state == 'ok':
            r = C.classify(a, name)
            if r.get('label'):
                fw = C.features(s)
                leg = dict(C.legend(name))
                tw = C.twins(name)
                cands = [fs for lb, fs in C.legend(name) if tw.get(lb, lb) == r['label']]
                if cands and cands[0]['blob_area'] > 0 and fw['blob_area'] > 0:
                    ratio_a.append(fw['blob_area'] / cands[0]['blob_area'])
                    if cands[0]['blob_gap'] > 0: ratio_g.append(fw['blob_gap'] / cands[0]['blob_gap'])
                d1.append(r['d1']); marg.append(r['margin'])
    log.close() if hasattr(log, 'close') else None
    n = sum(st.values())
    out.append((fam.get(key(name), '?'), name[:34], int(np.median([b[2] - b[0]])), round(float(np.mean(col)), 2),
                round(st['ok'] / n, 2), round(st['mixed'] / n, 2), round(st['blank'] / n, 2),
                round(float(np.median(ratio_a)), 2) if ratio_a else None, round(float(np.median(ratio_g)), 2) if ratio_g else None,
                round(float(np.median(d1)), 2) if d1 else None, round(float(np.median(marg)), 2) if marg else None))
    print(*out[-1], sep='\t', flush=True)
