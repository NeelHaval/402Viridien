"""How good is each LLM as a judge?  Score blind picks against HUMAN interval truth.

Truth: Neel 1inch + Shenzi (Gulf, QA'd), hadrian (Geoservices, not QA'd), gold log06 (D-582).
Items: crops inside one human bed ("pure") and crops straddling a human boundary (truth = mixed); 10% repeats.
Sheets (Codex models): the raw printed legend image + up to 8 full-width crops; numbered options from the parsed
legend plus "other: NAME" (the parse can miss entries). JEV: one call per crop through labeltool.jev2 (no examples).

    python -X utf8 notes/judge_eval.py build
    python -X utf8 notes/judge_eval.py run --model terra|sol|luna [--mode medium] [--par 8]
    python -X utf8 notes/judge_eval.py jev [--par 6]
    python -X utf8 notes/judge_eval.py score        -> analysis/categories/judge_eval/score.json
"""
import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
W = ROOT.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'analysis' / 'categories' / 'judge_eval'
SOLJOBS = 'C:/Users/hssil/.claude/skills/sol-fleet/scripts/sol_jobs.py'
MODELS = {'terra': 'gpt-6.1-terra', 'sol': 'gpt-6.1-sol', 'luna': 'gpt-6-luna', 'astra': 'gpt-6-astra'}
TRUTH = {
    '60811405120000_1inch tvd mudlog': ('Gulf 3-panel', W / 'neel_dataset/human_labels/truth/60811405120000_1inch tvd mudlog.csv'),
    '60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400': ('Gulf 3-panel', W / 'neel_dataset/human_labels/truth/60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400.csv'),
    '60808400140000_hadrian 2_masterlog_2in_10000-21962': ('Geoservices grid', W / 'neel_labels/2026-11-07/labelling/labels/60808400140000_hadrian 2_masterlog_2in_10000-21962.json'),
    '0000000009_d-582_0000020435-0001_log06_fe_5in_md': ('D-582 1-bit grid', ROOT / 'gt/gold/0000000009_d-582_0000020435-0001_log06_fe_5in_md.csv'),
}
PER_LOG, MIXED_PER_LOG, PER_SHEET = 44, 8, 8
norm = lambda s: re.sub(r'[^a-z]', '', (s or '').lower())  # noqa: E731


def truth(log):
    p = TRUTH[log][1]
    if p.suffix == '.json':
        d = json.loads(p.read_text(encoding='utf-8'))
        return [(int(l['top_px']), int(l['bottom_px']), l['composition'][0]['type']) for l in d['layers']]
    return [(int(float(r['top_y'])), int(float(r['bottom_y'])), r['lithology']) for r in csv.DictReader(open(p, encoding='utf-8'))]


def options(log):
    from labeltool import legend_classify as C
    seen, out = set(), []
    for label, _ in C._native().get(log, []):
        if C.NOT_ROCK.search(label) or label in seen:
            continue
        seen.add(label); out.append(label)
    return out


def ctx(log):
    from labeltool import jev2
    return jev2.Context.get(log)


def read(log, box, y0, y1):
    c = ctx(log)
    with c['lock']:
        return c['io'].read_rgb(c['log'], box, y0=y0, y1=y1)[..., :3]


def build(a):
    rnd = random.Random(a.seed)
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    items = []
    for log, (fam, _) in TRUTH.items():
        T = sorted(truth(log))
        pure = []
        for t, b, lab in T:  # windows inside one bed: trim 3 rows, 24..96 rows tall, several per tall bed
            t2, b2 = t + 3, b - 3
            if b2 - t2 < 16:
                continue
            h = min(96, b2 - t2)
            for y in range(t2, b2 - h + 1, max(h, 200)):
                pure.append(dict(y=y, h=h, truth=lab))
        by = defaultdict(list)
        for p in pure:
            by[p['truth']].append(p)
        for v in by.values():
            rnd.shuffle(v)
        pick = []
        while len(pick) < PER_LOG and any(by.values()):  # round-robin over labels so Halite/Shale don't dominate
            for k in list(by):
                if by[k] and len(pick) < PER_LOG:
                    pick.append(by[k].pop())
        mixed = []
        for (t1, b1, l1), (t2, b2, l2) in zip(T[:-1], T[1:]):
            if b1 == t2 and l1 != l2 and b1 - t1 >= 14 and b2 - t2 >= 14:
                h = min(30, b1 - t1 - 2, b2 - t2 - 2)
                mixed.append(dict(y=b1 - h, h=2 * h, truth='mixed', pair=[l1, l2]))
        rnd.shuffle(mixed)
        for it in pick + mixed[:MIXED_PER_LOG]:
            items.append(dict(it, log=log, family=fam))
    for i, it in enumerate(items):
        it['id'] = f'i{i:03d}'
    rnd.shuffle(items)
    dups = [dict(it, dup=True) for it in rnd.sample(items, len(items) // 10)]
    # sheets: per log, PER_SHEET crops; repeats go to a different sheet than their original
    sheets = []
    for log in TRUTH:
        L = [it for it in items if it['log'] == log]
        D = [it for it in dups if it['log'] == log]
        chunks = [L[i:i + PER_SHEET] for i in range(0, len(L), PER_SHEET)]
        for d in D:
            for ch in reversed(chunks):
                if all(x['id'] != d['id'] for x in ch) and len(ch) < PER_SHEET + 2:
                    ch.append(d); break
        for ch in chunks:
            sheets.append(dict(log=log, items=ch))
    meta = dict(sheets=[], items=items + dups)
    for s_i, s in enumerate(sheets):
        log = s['log']; name = f's{s_i:02d}'
        opts = options(log)
        lp = OUT / 'sheets' / f'legend_{s_i:02d}.png'
        c = ctx(log); box = c['log'].boxes['lithology_legend']
        Image.fromarray(read(log, 'lithology_legend', int(box[1]), int(box[3]))).save(lp)
        tiles = []
        for k, it in enumerate(s['items']):
            a_ = read(log, 'interpreted_lithology_column', it['y'], it['y'] + it['h'])
            sc = max(2, int(np.ceil(120 / max(1, a_.shape[1]))))
            sc = min(sc, max(1, 400 // a_.shape[1]))
            im = Image.fromarray(a_).resize((a_.shape[1] * sc, a_.shape[0] * sc), Image.NEAREST)
            tiles.append((f'W{k + 1}', im)); it.setdefault('slots', {})[name] = f'W{k + 1}'
        cw = max(t.width for _, t in tiles) + 20
        Hs = sum(t.height + 40 for _, t in tiles) + 10
        sheet = Image.new('RGB', (cw, Hs), 'white'); dr = ImageDraw.Draw(sheet); y = 5
        for lab, t in tiles:
            dr.text((10, y), lab, fill='black'); y += 18
            sheet.paste(t, (10, y)); dr.rectangle([9, y - 1, 10 + t.width, y + t.height], outline=(255, 0, 0)); y += t.height + 22
        sheet.save(OUT / 'sheets' / f'crops_{s_i:02d}.png')
        meta['sheets'].append(dict(name=name, idx=s_i, log=log, options=opts, slots=[(it['id'], it['slots'][name], bool(it.get('dup'))) for it in s['items']]))
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(len(items), 'items +', len(dups), 'repeats on', len(sheets), 'sheets;', Counter(it['truth'] == 'mixed' for it in items))


PROMPT = """You are checking a scanned mud log. Two images are in the current folder:
- {legend}: the log's lithology LEGEND exactly as printed (each pattern swatch with its printed name).
- {crops}: {n} crops (W1..W{n}) cut from the same log's interpreted-lithology column, enlarged; each is outlined in red.

Open and look at both images carefully (zoom in if your tools allow). For each crop decide which ONE legend entry
it shows. Compare colour, pattern (dashes, dots, bricks, hatching), their size, spacing and orientation. The crop
may be drawn at a different scale than the legend swatch.

Answer with one of these numbered options:
{opts}
If the crop shows TWO or more different patterns (a boundary inside the crop), answer "mixed".
If it is blank paper / grid only / text, answer "none".
If the right legend entry is missing from the list, answer "other: <printed name from the legend image>".

Reply with ONLY this JSON (no prose):
{{"crops": {{"W1": {{"pick": "<number | mixed | none | other: NAME>", "second": "<number or null>", "sure": "high|medium|low"}}, ...}}}}"""


def jobstate(job):
    r = subprocess.run([sys.executable, SOLJOBS, 'status', '--job', job], capture_output=True, text=True, encoding='utf-8', env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
    m = re.search(r'SOLJOB (\{.*\})', r.stdout)
    return json.loads(m.group(1)) if m else {}


def run(a):
    meta = json.loads((OUT / 'meta.json').read_text(encoding='utf-8'))
    model = MODELS[a.model]; tag = f'je{a.tag}-{a.model}{a.mode[0]}'
    (OUT / 'prompts').mkdir(exist_ok=True)
    todo = []
    for s in meta['sheets']:
        opts = '\n'.join(f'{i + 1}. {o}' for i, o in enumerate(s['options']))
        p = PROMPT.format(legend=f"legend_{s['idx']:02d}.png", crops=f"crops_{s['idx']:02d}.png", n=len(s['slots']), opts=opts)
        pf = OUT / 'prompts' / f"{s['name']}.txt"; pf.write_text(p, encoding='utf-8')
        todo.append((f"{tag}-{s['name']}", pf))
    live, t0 = [], time.time()
    env = {**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'}
    while todo or live:
        while todo and len(live) < a.par:
            job, pf = todo.pop(0)
            st = jobstate(job)
            if st.get('state') not in ('done',):
                subprocess.run([sys.executable, SOLJOBS, 'start', '--job', job, '--run', tag, '--model', model, '--mode', a.mode,
                                '--account', 'kira181204', '--cwd', str(OUT / 'sheets'), '--sandbox', 'read-only',
                                '--prompt-file', str(pf), '--wait', '1'], capture_output=True, env=env)
            live.append(job)
        time.sleep(10)
        for job in list(live):
            st = jobstate(job).get('state')
            if st in ('done', 'error', 'cancelled', 'failed', None):
                live.remove(job); print(f'{time.time() - t0:5.0f}s {job} {st}', flush=True)
    print('done', a.model, a.mode)


def jev(a):
    from labeltool import jev2
    meta = json.loads((OUT / 'meta.json').read_text(encoding='utf-8'))
    out = OUT / 'jev.json'
    res = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
    todo = [it for it in meta['items'] if not it.get('dup') and it['truth'] != 'mixed' and it['id'] not in res]

    def one(it):
        t0 = time.time()
        try:
            r = jev2.ask(it['log'], it['y'], it['y'] + it['h'], use_examples=False)
        except Exception as ex:  # noqa: BLE001
            r = dict(status='error', error=type(ex).__name__)
        return it['id'], dict(label=r.get('raw_label') or r.get('label'), p=r.get('probability'), status=r.get('status'),
                              ranking=r.get('ranking'), s=round(time.time() - t0, 1))
    with ThreadPoolExecutor(a.par) as ex:
        for k, (i, r) in enumerate(ex.map(one, todo)):
            res[i] = r
            if k % 20 == 0:
                out.write_text(json.dumps(res), encoding='utf-8'); print(k, len(todo), r['status'], flush=True)
    out.write_text(json.dumps(res), encoding='utf-8'); print('jev done', Counter(r['status'] for r in res.values()))


SYN = [{'clay', 'shale', 'clayshale'}, {'sand', 'sandstone', 'sandandsandstone'}, {'shalegrey', 'shale'}]


def same(truth_lab, pred):
    if not pred:
        return False
    t, p = norm(truth_lab), norm(pred)
    if t == p or t in {norm(x) for x in re.split(r'[/,]', pred)} or p in {norm(x) for x in re.split(r'[/,]', truth_lab)}:
        return True
    return any(t in g and p in g for g in SYN)


def parse_answer(txt):
    m = re.search(r'\{.*\}', txt or '', re.S)
    try:
        return json.loads(m.group(0)).get('crops', {}) if m else {}
    except ValueError:
        return {}


def score(a):
    meta = json.loads((OUT / 'meta.json').read_text(encoding='utf-8'))
    items = {(it['id'], bool(it.get('dup'))): it for it in meta['items']}
    home = Path.home() / '.codex' / 'sol-jobs'
    runs = sorted({p.name.rsplit('-s', 1)[0] for p in home.glob(f'je{a.tag}-*-s*')})
    S = {}
    for run_ in runs:
        ans, usage, secs = {}, 0, []
        for s in meta['sheets']:
            d = home / f"{run_}-{s['name']}"
            A = parse_answer((d / 'result.txt').read_text(encoding='utf-8')) if (d / 'result.txt').exists() else {}
            st = jobstate(f"{run_}-{s['name']}")
            usage += (st.get('billed') or 0); secs.append(st.get('elapsed_s') or 0)
            for iid, slot, dup in s['slots']:
                r = A.get(slot)
                if not isinstance(r, dict):
                    continue
                pk = str(r.get('pick', '')).strip()
                if pk.isdigit() and 1 <= int(pk) <= len(s['options']):
                    lab = s['options'][int(pk) - 1]
                elif pk.lower().startswith('other'):
                    lab = pk.split(':', 1)[-1].strip()
                else:
                    lab = pk.lower()
                ans[(iid, dup)] = dict(label=lab, sure=r.get('sure'))
        S[run_] = summarise(items, ans, usage, secs)
    jf = OUT / 'jev.json'
    if jf.exists():
        J = json.loads(jf.read_text(encoding='utf-8'))
        ans = {(i, False): dict(label=r['label'], sure=('high' if (r['p'] or 0) >= 0.8 else 'medium' if (r['p'] or 0) >= 0.5 else 'low'), p=r['p'])
               for i, r in J.items() if r['status'] == 'ok'}
        S['jev'] = summarise(items, ans, None, [r['s'] for r in J.values()])
    (OUT / 'score.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    for k, v in S.items():
        print(f"\n== {k}: answered {v['answered']}/{v['n_items']}  pure acc {v['pure_acc']:.1%} (n {v['pure_n']})  mixed caught {v['mixed_caught']}  "
              f"said-mixed on pure {v['mixed_on_pure']}  self-agree {v['self_agree']}  tokens {v['tokens']}  median s/call {v['median_s']}")
        print('   by family', v['family']); print('   by sure', v['sure'])
        print('   top errors', v['errors'][:6])


def summarise(items, ans, usage, secs):
    pure = [(k, it) for k, it in items.items() if not k[1] and it['truth'] != 'mixed']
    mixed = [(k, it) for k, it in items.items() if not k[1] and it['truth'] == 'mixed']
    fam, sure, errs = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), Counter()
    ok_n = n = mp = 0
    for k, it in pure:
        r = ans.get(k)
        if not r:
            continue
        if r['label'] in ('mixed', 'none'):
            mp += r['label'] == 'mixed'
        good = same(it['truth'], r['label'])
        n += 1; ok_n += good
        fam[it['family']][0] += good; fam[it['family']][1] += 1
        sure[r.get('sure') or '?'][0] += good; sure[r.get('sure') or '?'][1] += 1
        if not good:
            errs[(it['family'][:8], it['truth'], r['label'])] += 1
    mc = sum(1 for k, it in mixed if ans.get(k, {}).get('label') == 'mixed')
    pairs = [(ans.get((k[0], False)), ans.get((k[0], True))) for k in items if k[1]]
    pairs = [(x['label'], y['label']) for x, y in pairs if x and y]
    return dict(n_items=len(pure) + len(mixed), answered=sum(1 for k, _ in pure + mixed if k in ans), pure_n=n, pure_acc=ok_n / max(1, n),
                mixed_caught=f'{mc}/{len(mixed)}', mixed_on_pure=mp,
                self_agree=f'{sum(x == y for x, y in pairs)}/{len(pairs)}',
                family={f: f'{v[0]}/{v[1]} ({v[0] / max(1, v[1]):.0%})' for f, v in fam.items()},
                sure={s: f'{v[0]}/{v[1]} ({v[0] / max(1, v[1]):.0%})' for s, v in sure.items()},
                errors=[(list(e), c) for e, c in errs.most_common(12)], tokens=usage,
                median_s=float(np.median(secs)) if secs else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--model', default='terra')
    ap.add_argument('--mode', default='medium'); ap.add_argument('--par', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--tag', default='1108')
    a = ap.parse_args()
    dict(build=build, run=run, jev=jev, score=score)[a.cmd](a)
