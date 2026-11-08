"""Two-stage detector test: (1) JEV names the query's visual family, (2) only this log's legend entries
in that family compete (JEV v2 same-scale panel, then look-alike tie-break).

Run after labeltool/families.py:  python -X utf8 -B labeltool/twostage.py
Writes analysis/twostage_test.html and analysis/twostage_test.json.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool.common import ROOT, CACHE, GOLD, HUMAN, normalize  # noqa: E402
from labeltool import jev2  # noqa: E402
from labeltool.families import FAMILIES, KEYS, PROMPT, data_uri, inner  # noqa: E402

ANALYSIS = ROOT / 'analysis'
OUT = CACHE / 'twostage'
VERSION = 'two-1'


def family_index():
    data = json.loads((ANALYSIS / 'pattern_families.json').read_text(encoding='utf-8'))
    index = {}
    for r in data['reps']:
        for m in r['members']:
            index[(m['log'], m['id'])] = dict(pattern=r['pattern'], probs=r['probs'])
    return index


def query_family(query, client):
    from strategies._jev import image_asset
    big = Image.fromarray(query).resize((query.shape[1] * 3, query.shape[0] * 3), Image.Resampling.NEAREST)
    key = hashlib.sha256(VERSION.encode() + big.tobytes() + str(big.size).encode()).hexdigest()[:24]
    OUT.mkdir(parents=True, exist_ok=True)
    cached = OUT / (key + '.json')
    if cached.exists():
        return json.loads(cached.read_text(encoding='utf-8'))
    resp = client.ask({'task': PROMPT, 'image': 'symbol'}, {'symbol': image_asset(big)},
                      {'pattern': {'type': 'choice', 'instructions': PROMPT, 'criteria': {k: FAMILIES[k] for k in KEYS}}})
    probs = (resp or {}).get('answers', {}).get('pattern', {}).get('probabilities', {})
    p = {k: float(probs.get(k, 0) or 0) for k in KEYS}
    total = sum(p.values()) or 1
    p = {k: round(v / total, 3) for k, v in p.items()}
    out = dict(best=max(p, key=p.get), probs=p)
    cached.write_text(json.dumps(out), encoding='utf-8')
    return out


def detect(name, top, bottom, index, client):
    ctx = jev2.Context.get(name)
    with ctx['lock']:
        query = jev2.read_interval(ctx['io'], ctx['log'], top, bottom)
    fam = query_family(query, client)
    ranked = sorted(fam['probs'].items(), key=lambda kv: -kv[1])
    tile_fam = {t['raw']: index.get((name, t['id']), {}).get('pattern') for t in ctx['tiles']}
    # Stage 1: the families JEV gives real weight to (top family, plus any within 0.25 of it).
    keep = [k for k, v in ranked if v >= ranked[0][1] - 0.25 and v > 0.05]
    cands = {raw for raw, f in tile_fam.items() if f in keep}
    if not cands:
        cands = {raw for raw, f in tile_fam.items() if f in [k for k, _ in ranked[:3]]}
    stage2 = jev2.decide(name, top, bottom, jev2.ask(name, top, bottom, client=client, only=cands or None))
    return dict(query=query, family=fam, families_kept=keep, candidates=sorted(cands),
                tile_family=tile_fam, result=stage2)


def cases():
    out = []
    for r in csv.DictReader(open(ROOT / 'gt' / 'gold' / f'{GOLD}.csv', encoding='utf-8')):
        out.append(dict(log=GOLD, top=int(r['top_y']), bottom=int(r['bottom_y']), truth=r['lithology'], src='gold'))
    events = [json.loads(l) for l in (HUMAN / 'labels.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    undone = {e.get('target') for e in events if e.get('action') == 'undo'}
    for e in events:
        if e.get('action') == 'label' and e['event_id'] not in undone:
            out.append(dict(log=e['log'], top=e['top_y'], bottom=e['bottom_y'], truth=e['label'], src='human'))
    return out


def same(a, b):
    return normalize(jev2.fix_label(a or '')) == normalize(jev2.fix_label(b or ''))


def main():
    from strategies._jev import Client
    index = family_index()
    client = Client(timeout=45, retries=1)
    items = cases()

    def run(c):
        d = detect(c['log'], c['top'], c['bottom'], index, client)
        flat = jev2.decide(c['log'], c['top'], c['bottom'], jev2.ask(c['log'], c['top'], c['bottom'], client=client))
        return c, d, flat

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, items))
    rows, table = [], []
    score = dict(family=0, two=0, flat=0, two_top3=0, ambiguous=0)
    for c, d, flat in results:
        ctx = jev2.Context.get(c['log'])
        truth_tile = next((t for t in ctx['tiles'] if same(t['label'], c['truth'])), None)
        truth_fam = d['tile_family'].get(truth_tile['raw']) if truth_tile else None
        fam_ok = truth_fam in d['families_kept']
        two_ok = same(d['result']['label'], c['truth'])
        flat_ok = same(flat['label'], c['truth'])
        score['family'] += fam_ok
        score['two'] += two_ok
        score['flat'] += flat_ok
        score['two_top3'] += any(same(x['label'], c['truth']) for x in d['result'].get('ranking', [])[:3])
        score['ambiguous'] += bool(d['result'].get('ambiguous'))
        table.append(dict(log=c['log'], top=c['top'], bottom=c['bottom'], truth=c['truth'], src=c['src'],
                          truth_family=truth_fam, query_family=d['family']['best'], families_kept=d['families_kept'],
                          candidates=d['candidates'], two=d['result']['label'], two_p=d['result'].get('probability'),
                          ambiguous=d['result'].get('ambiguous'), why=d['result'].get('why'),
                          flat=flat['label'], family_ok=fam_ok, two_ok=two_ok, flat_ok=flat_ok))
        tiles = ''.join(
            f'<figure class="{"c" if t["raw"] in d["candidates"] else "x"}{" truth" if truth_tile and t["raw"] == truth_tile["raw"] else ""}">'
            f'<img src="{data_uri(inner(t["swatch"]), 1)}" alt=""><figcaption>{html.escape(t["label"])}<br>'
            f'<span>{html.escape(str(d["tile_family"].get(t["raw"])))}</span></figcaption></figure>' for t in ctx['tiles'])
        fam_txt = ', '.join(f'{k} {v:.0%}' for k, v in sorted(d['family']['probs'].items(), key=lambda kv: -kv[1])[:3])
        mark = lambda ok: '<b class="ok">right</b>' if ok else '<b class="no">wrong</b>'
        rows.append(f'<tr><td><img class="q" src="{data_uri(d["query"][:160], 1)}" alt=""><br><small>{html.escape(c["src"])} · '
                    f'{html.escape(c["log"][-22:])}<br>rows {c["top"]}-{c["bottom"]}</small></td>'
                    f'<td><b>{html.escape(c["truth"])}</b><br><small>family {html.escape(str(truth_fam))}</small></td>'
                    f'<td>{html.escape(fam_txt)}<br>{mark(fam_ok)}</td><td class="tiles">{tiles}</td>'
                    f'<td><b>{html.escape(d["result"]["label"] or "")}</b> {mark(two_ok)}<br><small>{html.escape(d["result"].get("why") or "")}</small></td>'
                    f'<td>{html.escape(flat["label"] or "")} {mark(flat_ok)}</td></tr>')
    n = len(results)
    css = ('body{font:13px system-ui,sans-serif;margin:22px;color:#1d2a24}table{border-collapse:collapse}td,th{border-bottom:1px solid #d5ddd9;'
           'padding:8px;vertical-align:top;text-align:left}th{background:#eef3f0}.q{image-rendering:pixelated;border:2px solid #b3261e;max-width:230px}'
           '.tiles{display:flex;flex-wrap:wrap;gap:4px;max-width:560px}figure{margin:0;width:100px;font-size:10px;opacity:.35}figure img{width:100px;height:68px;object-fit:cover;object-position:left top;border:1px solid #ccc;image-rendering:pixelated}'
           'figure.c{opacity:1}figure.truth img{border:3px solid #1f7a45}figure span{color:#5b6b63}.ok{color:#1f7a45}.no{color:#b3261e}small{color:#5b6b63}')
    head = (f'<h1>Two-stage detector test ({n} labelled intervals)</h1><p>Stage 1: JEV names the drawn family of the query '
            f'(the legend swatches were sorted the same way). Stage 2: only this log\'s legend entries in the kept families '
            f'compete (full colour = candidate, faded = ruled out, green border = truth).</p><p><b>Family right: {score["family"]}/{n}</b> · '
            f'<b>two-stage top-1: {score["two"]}/{n}</b> (top-3 {score["two_top3"]}/{n}, flagged ambiguous {score["ambiguous"]}) · '
            f'flat JEV v2 top-1: {score["flat"]}/{n}</p>')
    (ANALYSIS / 'twostage_test.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>Two-stage test</title><style>{css}</style>{head}<table><tr><th>Query (native)</th>'
        f'<th>Truth</th><th>Stage 1 family</th><th>This log\'s legend</th><th>Two-stage pick</th><th>Flat JEV v2</th></tr>{"".join(rows)}</table>',
        encoding='utf-8')
    (ANALYSIS / 'twostage_test.json').write_text(json.dumps(dict(score=score, n=n, calls=client.calls, rows=table), indent=1), encoding='utf-8')
    print(json.dumps(dict(score=score, n=n, calls=client.calls)))
    for t in table:
        print(f"{t['src']:5s} {t['top']}-{t['bottom']} truth={t['truth']:10s} fam={t['query_family']}/{t['truth_family']} "
              f"kept={t['families_kept']} cands={len(t['candidates'])} two={t['two']} flat={t['flat']}")


if __name__ == '__main__':
    main()
