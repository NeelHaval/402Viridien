"""Blind LLM review of the v3 classifier across confidence bands, one image per LLM call.

  build    sample windows (stratified by confidence band and log), draw one sheet per batch:
           ONE log per sheet = one numbered legend + up to 8 numbered windows; the classifier's call is NOT
           shown (blind pick, no anchoring). ~10 % of windows appear twice on different sheets (self-agreement).
  run      one Terra job per sheet through sol_jobs.py, N at a time (default 6), resumable.
  collect  parse answers -> analysis/categories/llmrev/results.json and a printed summary.

Usage: python notes/llm_review.py build|run|collect [--per-band 200] [--par 6]"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / 'analysis' / 'categories'
OUT = D / 'llmrev'; SH = OUT / 'sheets'
JOBS = 'C:/Users/hssil/.claude/skills/sol-fleet/scripts/sol_jobs.py'
RUN = 'lrev1108'
BANDS = [('under 50%', 0, .5), ('50-80%', .5, .8), ('80-95%', .8, .95), ('95%+', .95, 1.01)]
PER_SHEET = 8
try:
    FONT = ImageFont.truetype('arial.ttf', 15); FONTB = ImageFont.truetype('arialbd.ttf', 22)
except OSError:
    FONT = FONTB = ImageFont.load_default()

PROMPT = """You are labelling small crops of a scanned geological mud-log lithology column. Look at ONE image: {sheet}
(in the working folder). Do not write code; just look at the image (open it with your image viewer).

The image has two parts:
1. TOP: this log's legend. Every legend entry is a numbered box (red number) with its rock name underneath.
2. BOTTOM: crops cut from the same log's lithology column, each with a big label W1, W2, ...

For EACH crop W, pick the single legend entry whose picture matches it best. Compare the fill colour AND the drawn marks:
their shape (dots, dashes, bricks, ticks, crosses, V or L marks), direction, spacing and density. The legend picture may be
drawn at a somewhat different size than the column, so match the kind of marks, not their exact size.
Answer for each crop:
- "pick": the legend number, or "mixed" if the crop clearly holds two different patterns/colours (a bed boundary)
  or text/a white gap over a large part of it, or "none" if it matches no entry (blank paper, grid lines only, text only).
- "second": the next best legend number, or null.
- "sure": "high", "medium" or "low".
Judge every crop on its own; the crops are in random order and do not continue one another.
Reply ONLY with JSON, one entry per crop shown:
{{"crops":{{"W1":{{"pick":3,"second":7,"sure":"high"}},"W2":{{"pick":"mixed","second":null,"sure":"medium"}}}}}}"""


def key(l):
    return l.replace('#', '_').replace(' ', '_')[:90]


def build(per_band, seed=7, src='classified'):
    from labeltool import legend_classify as LC
    from labeltool import jev2
    from harness import io
    rng = random.Random(seed)
    SH.mkdir(parents=True, exist_ok=True)
    names = {key(n): n for n in io.list_logs()}
    iso = fit_calibration() if src == 'classified' else None
    pools = defaultdict(lambda: defaultdict(list))  # band -> log -> windows
    for f in sorted((D / src).glob('*.json')):
        log = names[f.stem]
        for o in json.loads(f.read_text(encoding='utf-8')):
            if o['state'] == 'ok':
                c = o['conf'] if o.get('conf') is not None else float(iso.predict([o['margin'] or 0.0])[0])
                b = next(n for n, lo, hi in BANDS if lo <= c < hi)
                pools[b][log].append(dict(log=log, y=o['y'], h=o['h'], label=o['label'], conf=round(c, 3), band=b, state='ok'))
            elif o['state'] == 'mixed':
                pools['rejected: two beds/gap/text'][log].append(dict(log=log, y=o['y'], h=o['h'], label=None, conf=None,
                                                                    band='rejected: two beds/gap/text', state='mixed'))
    picked = []
    for b, by_log in pools.items():
        want = per_band if b != 'rejected: two beds/gap/text' else per_band // 2
        logs = list(by_log); rng.shuffle(logs)
        for L in logs:
            rng.shuffle(by_log[L])
        while want > 0 and any(by_log[L] for L in logs):   # round-robin over logs: every log represented
            for L in logs:
                if by_log[L] and want > 0:
                    picked.append(by_log[L].pop()); want -= 1
    for i, w in enumerate(picked):
        w['id'] = i
    dups = rng.sample(picked, len(picked) // 10)
    items = picked + [dict(w, dup=True) for w in dups]
    by_log = defaultdict(list)
    for w in items:
        by_log[w['log']].append(w)
    sheets = []
    for log, ws in by_log.items():
        rng.shuffle(ws)
        # keep a window and its duplicate on different sheets
        chunks = [[]]
        for w in ws:
            c = next((c for c in chunks if len(c) < PER_SHEET and all(x['id'] != w['id'] for x in c)), None)
            if c is None:
                chunks.append([]); c = chunks[-1]
            c.append(w)
        legend = LC._native().get(log, [])
        ctx = jev2.Context.get(log)
        for c in chunks:
            if not c:
                continue
            n = len(sheets); fn = f's{n:03d}.png'
            with ctx['lock']:
                wins = [ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=w['y'], y1=w['y'] + w['h'])[..., :3] for w in c]
            draw_sheet(SH / fn, legend, wins)
            for k, w in enumerate(c):
                w['w'] = f'W{k + 1}'
            sheets.append(dict(sheet=fn, log=log, legend=[lb for lb, _ in legend], windows=c))
    (OUT / 'manifest.json').write_text(json.dumps(sheets, indent=0), encoding='utf-8')
    for s in sheets:
        (OUT / (s['sheet'][:-4] + '.txt')).write_text(PROMPT.format(sheet=s['sheet']), encoding='utf-8')
    print(len(picked), 'windows +', len(dups), 'repeats on', len(sheets), 'sheets;', Counter(w['band'] for w in picked))


def fit_calibration():
    """Isotonic P(LLM says right | margin) on the v3 full-legend check, the same curve as notes/conf_data.py."""
    from sklearn.isotonic import IsotonicRegression
    by = {}
    for f in (D / 'classified').glob('*.json'):
        for o in json.loads(f.read_text(encoding='utf-8')):
            by[(f.stem, o['y'])] = o
    X, Y = [], []
    for r in json.loads((D / 'pairs_llm3.json').read_text(encoding='utf-8'))['rows']:
        o = by.get((key(r['log']), r['y'])); v = r['llm'].split(':')[0]
        if o and o['state'] == 'ok' and v in ('right', 'wrong', 'mixed', 'none'):
            X.append(o['margin'] or 0.0); Y.append(v == 'right')
    return IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds='clip').fit(X, Y)


def draw_sheet(path, legend, wins):
    W = 1400
    cells = []
    for n, (lb, a) in enumerate(legend):
        a = np.ascontiguousarray(a[..., :3]).astype(np.uint8)
        im = Image.fromarray(a); s = min(2.0, 112 / im.height, 260 / im.width)  # same 2x scale as the crops when it fits
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.NEAREST)
        cells.append((n + 1, lb, im))
    # legend grid
    x, y, rows, rh = 10, 64, [], 0
    for n, lb, im in cells:
        cw = max(im.width, 130)
        if x + cw > W - 10:
            x, y, rh = 10, y + rh + 52, 0
        rows.append((n, lb, im, x, y)); x += cw + 16; rh = max(rh, im.height)
    ly = y + rh + 52
    # windows: 2x, boxed, big labels
    wims = []
    for a in wins:
        im = Image.fromarray(np.ascontiguousarray(a).astype(np.uint8))
        im = im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST)
        if im.width > 320:
            im = im.crop((0, 0, 320, im.height))
        wims.append(im)
    x, y2, pos = 10, ly + 84, []
    for im in wims:
        if x + im.width + 20 > W:
            x, y2 = 10, y2 + 96 + 60
        pos.append((x, y2)); x += max(im.width, 90) + 40
    H = y2 + 96 + 30
    s = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(s)
    dr.text((10, 8), 'LEGEND of this log (numbered entries)', fill=(0, 0, 0), font=FONTB)
    for n, lb, im, x, y in rows:
        dr.text((x, y - 26), str(n), fill=(190, 20, 20), font=FONTB)
        s.paste(im, (x, y)); dr.rectangle((x - 1, y - 1, x + im.width, y + im.height), outline=(120, 120, 120))
        dr.text((x, y + im.height + 4), lb[:22], fill=(0, 0, 0), font=FONT)
    dr.line((0, ly, W, ly), fill=(0, 0, 0), width=3)
    dr.text((10, ly + 10), 'CROPS from the column (judge each one separately)', fill=(0, 0, 0), font=FONTB)
    for k, (im, (x, y)) in enumerate(zip(wims, pos)):
        dr.text((x, y - 28), f'W{k + 1}', fill=(20, 20, 160), font=FONTB)
        s.paste(im, (x, y)); dr.rectangle((x - 2, y - 2, x + im.width + 1, y + im.height + 1), outline=(20, 20, 160), width=2)
    s.save(path)


def jobstate(job):
    r = subprocess.run([sys.executable, JOBS, 'status', '--job', job], capture_output=True, text=True, encoding='utf-8',
                       env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
    m = re.search(r'SOLJOB (\{.*\})', r.stdout)
    return json.loads(m.group(1)) if m else None


def run(par):
    sheets = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    todo = [s['sheet'][:-4] for s in sheets]
    live, done, failed = set(), set(), []
    t0 = time.time()
    while todo or live:
        for j in list(live):
            st = jobstate(f'{RUN}-{j}')
            if st and st['state'] in ('done', 'failed', 'cancelled', 'error'):
                live.discard(j); (done.add(j) if st.get('ok') else failed.append((j, st.get('error'))))
        while todo and len(live) < par:
            j = todo.pop(0); st = jobstate(f'{RUN}-{j}')
            if st and st.get('ok'):
                done.add(j); continue
            subprocess.run([sys.executable, JOBS, 'start', '--job', f'{RUN}-{j}', '--run', RUN, '--model', 'gpt-6.1-terra',
                            '--mode', 'medium', '--account', 'kira181204', '--cwd', str(SH), '--sandbox', 'read-only',
                            '--prompt-file', str(OUT / f'{j}.txt'), '--wait', '1'], capture_output=True, text=True, encoding='utf-8',
                           env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
            live.add(j)
        print(f'{time.time() - t0:5.0f}s done {len(done)} live {len(live)} todo {len(todo)} failed {len(failed)}', flush=True)
        time.sleep(10)
    print('failed:', failed)


def collect():
    import os
    sheets = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    H = Path(os.path.expanduser('~/.codex/sol-jobs'))
    rows, missing = [], 0
    for s in sheets:
        f = H / f'{RUN}-{s["sheet"][:-4]}' / 'result.txt'
        try:
            ans = json.loads(re.search(r'\{.*\}', f.read_text(encoding='utf-8'), re.S).group(0))['crops']
        except Exception:
            missing += 1; continue
        for w in s['windows']:
            a = ans.get(w['w']) or {}
            p = a.get('pick'); sec = a.get('second')
            nm = lambda v: s['legend'][int(v) - 1] if isinstance(v, (int, str)) and str(v).isdigit() and 0 < int(v) <= len(s['legend']) else None  # noqa: E731
            pick = nm(p) or (p if p in ('mixed', 'none') else None)
            rows.append(dict(w, sheet=s['sheet'], llm=pick, llm_second=nm(sec), sure=a.get('sure'),
                             agree=(pick == w['label']) if w['state'] == 'ok' else None))
    res = dict(rows=rows, missing_sheets=missing)
    (OUT / 'results.json').write_text(json.dumps(res, indent=0), encoding='utf-8')
    # summary
    print('sheets missing', missing, 'rows', len(rows))
    first = [r for r in rows if not r.get('dup')]
    for b, *_ in BANDS:
        R = [r for r in first if r['band'] == b and r['llm']]
        lab = [r for r in R if r['llm'] not in ('mixed', 'none')]
        print(f'{b:10s} n={len(R):4d}  agree {sum(r["agree"] for r in R) / max(1, len(R)):.0%}  '
              f'(of labelled by LLM {sum(r["agree"] for r in lab) / max(1, len(lab)):.0%})  mixed {sum(r["llm"] == "mixed" for r in R)}  none {sum(r["llm"] == "none" for r in R)}  '
              f'mean conf {np.mean([r["conf"] for r in R]):.2f}')
    M = [r for r in first if r['state'] == 'mixed' and r['llm']]
    print('rejected windows: LLM says mixed', sum(r['llm'] == 'mixed' for r in M), 'none', sum(r['llm'] == 'none' for r in M), 'a single label', sum(r['llm'] not in ('mixed', 'none') for r in M), 'of', len(M))
    by = defaultdict(list)
    for r in rows:
        by[r['id']].append(r['llm'])
    pairs = [v for v in by.values() if len(v) == 2 and all(v)]
    print('LLM self-agreement on repeated windows', sum(a == b for a, b in pairs), '/', len(pairs))
    hi = [r for r in first if r['sure'] == 'high' and r['state'] == 'ok' and r['llm']]
    print('when the LLM says high sure:', f'agree {sum(r["agree"] for r in hi) / max(1, len(hi)):.0%} of {len(hi)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--per-band', type=int, default=200); ap.add_argument('--par', type=int, default=6)
    ap.add_argument('--src', default='classified'); ap.add_argument('--tag', default='lrev1108'); ap.add_argument('--out', default='llmrev')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    RUN = a.tag; OUT = D / a.out; SH = OUT / 'sheets'
    {'build': lambda: build(a.per_band, a.seed, a.src), 'run': lambda: run(a.par), 'collect': collect}[a.cmd]()
