"""Render the complete 13-log cohort from saved accepted-policy scores."""
from pathlib import Path
import hashlib
import html
import json

HERE = Path(__file__).resolve().parent
NAMES = {'s19_combine': 's19', 's20_v4': 'v4',
         's21_v4dino': 'Baseline DINO', 's23_v4dino_identity': 'S23: distinct labels'}
COLORS = ['#1778b7', '#138273', '#bb631c', '#865494']


def main():
    source = HERE / 'accepted_preview.json'
    source_bytes = source.read_bytes()
    data = json.loads(source_bytes.decode('utf-8'))
    by = {r['strategy']: r for r in data['accepted_summaries']
          if r['suite'] == 'viridien_eval' and r['family'] == 'all'}
    if set(by) != set(NAMES) or any(r['n_logs'] != 13 for r in by.values()):
        raise ValueError('This view needs the complete 13-log matched cohort')
    bars = []
    rows = []
    for (strategy, name), color in zip(NAMES.items(), COLORS):
        metrics = by[strategy]['metrics']
        value = metrics['score']
        bars.append(f'<div class="bar-row"><span>{html.escape(name)}</span>'
                    f'<div class="track"><div style="width:{100*value:.5f}%;background:{color}"></div></div>'
                    f'<strong>{value:.1%}</strong></div>')
        rows.append('<tr><th>'+html.escape(name)+'</th>'+''.join(
            f'<td>{metrics[key]:.1%}</td>' for key in ('row_acc', 'boundary_f1'))+
            f'<td>{metrics["cluster_ari"]:.3f}</td></tr>')
    pair = next(p for p in data['accepted_paired'] if p['suite'] == 'viridien_eval'
                and p['a'] == 's23_v4dino_identity' and p['b'] == 's21_v4dino')
    ci = pair['metrics']['score']['log_ci']
    css = '''body{margin:0;color:var(--foreground,#203847);font:15px/1.5 var(--font-sans,Arial,sans-serif)}
    h1{font-size:24px;line-height:1.15;margin:0 0 12px}h2{font-size:18px;margin:25px 0 10px}
    p{max-width:72ch}.note{background:var(--warning-surface,#fff0d6);padding:14px;border-radius:var(--radius,4px)}
    .bar-row{display:grid;grid-template-columns:135px minmax(70px,1fr) 48px;gap:9px;align-items:center;margin:13px 0;font-size:13px}
    .track{height:26px;background:var(--secondary,#e9eff3)}.track div{height:26px}.bar-row strong{text-align:right}
    table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:9px 6px;border-bottom:1px solid var(--border,#cbd8e0);text-align:right}th:first-child{text-align:left}
    a{color:var(--primary,#12699b)}.small{font-size:12px;overflow-wrap:anywhere;color:var(--muted-foreground,#4e626e)}
    @media(max-width:400px){.bar-row{grid-template-columns:121px minmax(55px,1fr) 45px;gap:7px}th,td{padding:9px 4px}}
    '''
    doc = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           '<style>'+css+'</style><body><h1>13 development logs · complete cohort</h1>'
           '<p class="note">The full cross-suite comparison is still incomplete. No champion is declared. '
           'These are local test scores. The official challenge materials give no numerical grading formula.</p>'
           '<h2>Mean local score</h2><p>Clay, Shale, Clay/Shale and Clay-Shale receive the approved full label credit. '
           'Bed contacts stay unchanged.</p>'+''.join(bars)+
           '<h2>Read the parts too</h2><table><tr><th>Pipeline</th><th>Row accuracy</th><th>Contact F1</th><th>ARI</th></tr>'+
           ''.join(rows)+'</table><p>Each window score is row accuracy × contact F1. Window scores are averaged within each log; '
           'then logs have equal weight. The displayed mean parts are not multiplied together.</p>'
           f'<p>S23 minus baseline DINO: <strong>{100*ci["mean"]:+.2f} percentage points</strong>. '
           f'Paired log bootstrap 95% interval: <strong>{100*ci["low"]:+.2f} to {100*ci["high"]:+.2f}</strong>. '
           'The interval includes zero. These logs do not establish a reliable gain over baseline DINO.</p>'
           '<p>Boundary, label and depth errors still matter. Row accuracy ignores unlabelled truth rows. '
           'ARI, depth error, time and memory are reported separately; they are not factors in this local product.</p>'
           '<p><a href="index.html">Full progress scoreboard</a> · <a href="CHALLENGE_ALIGNMENT.md">Official requirements and local rules</a></p>'
           '<p class="small"><a href="viridien_eval_preview_data.json">Saved score-data snapshot</a> SHA-256: '+hashlib.sha256(source_bytes).hexdigest()+'</p></body></html>')
    (HERE / 'viridien_eval_preview_data.json').write_bytes(source_bytes)
    (HERE / 'viridien_eval_preview.html').write_text(doc, encoding='utf-8')
    print('Rendered 13 matched logs × 4 candidates; paired interval copied from saved score data')


if __name__ == '__main__':
    main()
