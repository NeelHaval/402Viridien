"""Small, standalone, mobile-readable report with no external assets."""
from html import escape


def number(value):
    return 'NA' if value is None else f'{value:.3f}'


def interval(row, metric):
    value = number(row.get(metric))
    low, high = row.get(metric + '_ci_low'), row.get(metric + '_ci_high')
    return value if low is None else value + f' [{low:.3f}, {high:.3f}]'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def report(path, records, summary, drops):
    is_mock = any(r['mock'] for r in records)
    content = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Synthetic lithology benchmark</title><style>',
        'body{font:16px/1.5 system-ui,sans-serif;color:#17202a;background:#fff;margin:0 auto;padding:16px;max-width:1100px}',
        'h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:28px}a{color:#0645ad}',
        '.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}',
        'th,td{text-align:left;vertical-align:top;border-bottom:1px solid #b8bec5;padding:9px}',
        'th{background:#eef0f2}td:first-child{overflow-wrap:anywhere}p{max-width:80ch}',
        '.notice{border:2px solid #525b65;padding:12px}img{max-width:100%;height:auto}',
        '@media(max-width:600px){body{padding:10px}th,td{padding:6px;font-size:.8rem}}',
        '</style><main><h1>Synthetic lithology benchmark</h1>']
    if is_mock:
        content.append('<p class="notice"><strong>MOCK smoke test.</strong> These crops are gold/agreed-consensus test fixtures, '
                       'not human-approved pattern examples. These numbers do not establish real-log performance.</p>')
    content.append('<p>Only test-pool fake logs are scored, using harness.score. Strategies use their original run interface. '
        'Patch classifiers use a fixed row grid, without access to truth contacts. Hybrid uses the existing fixed FFT '
        'local-only fallback; it is not fitted on this test set. Donor legend parser results and the exact synthetic '
        'linear depth scale are supplied as warm caches; OCR and depth estimation are outside these scores.</p>')
    content.append('<p>Cells show mean [95% percentile bootstrap interval], resampling whole fake logs 2,000 times. '
        'These intervals describe variation conditional on this crop library and generator. Reused pixels mean '
        'they are not source-independent intervals for real wells. A group with one fake log has no interval. '
        'Unavailable methods have NA metrics and explicit coverage; algorithm errors score zero.</p>')
    for group, heading in [('overall', 'Overall'), ('family', 'By family'), ('nuisance', 'By applied nuisance')]:
        rows = [r for r in summary if r['group'] == group]
        content.append('<h2>' + heading + '</h2>')
        content.append(table(['Method', 'Group', 'Scored / logs', 'Source logs', 'Row accuracy [95% CI]',
            'Boundary F1 [95% CI]', 'Cluster ARI [95% CI]'], [[r['method'], r['value'],
                f'{r["n_scored"]} / {r["n_logs"]}', r['n_source_logs'], interval(r, 'row_acc'),
                interval(r, 'boundary_f1'), interval(r, 'cluster_ari')] for r in rows]))
    content.append('<h2>What breaks it</h2><p>Positive values mean accuracy fell. “Versus clean logs” compares within '
        'family and is unavailable when no fully clean logs were drawn. “Paired image drop” compares the same '
        'textures and bed plan with all image nuisances removed; it measures their combined effect, not the isolated '
        'effect of one nuisance. Interbedding, blank gaps and mandatory 1-bit conversion remain in paired controls, '
        'so their paired drop is NA. Use single-nuisance generation to estimate individual effects.</p>')
    content.append(table(['Method', 'Nuisance', 'Logs', 'Drop vs clean logs [95% CI]', 'Paired image drop [95% CI]'], [
        [r['method'], r['nuisance'], r['n_logs'], interval(dict(drop=r['drop_vs_clean'],
            drop_ci_low=r['drop_ci_low'], drop_ci_high=r['drop_ci_high']), 'drop'),
         interval(dict(drop=r['paired_image_drop'], drop_ci_low=r['paired_ci_low'],
                       drop_ci_high=r['paired_ci_high']), 'drop')] for r in drops]))
    status = {}
    for row in records:
        key = (row['method'], row['status'], row.get('error_type', ''))
        status[key] = status.get(key, 0) + 1
    content.append('<h2>Availability</h2>' + table(['Method', 'Status', 'Error type', 'Runs'],
        [list(k) + [n] for k, n in sorted(status.items())]))
    content.append('<p>External API calls: 0. Optional paid methods require --include-paid and capped, explicitly '
        'supplied offline response caches. Credentials are never read.</p>')
    content.append('<h2>Native aspect previews</h2><p>PNG files retain each column’s native width and height. '
                   'Open a link to inspect texture and contacts at full resolution.</p><ul>')
    for name in sorted({r['log'] for r in records}):
        filename = escape(name + '.png', quote=True)
        content.append(f'<li><a href="previews/{filename}">{escape(name)}</a></li>')
    content.append('</ul><p><a href="results.csv">Per-log results</a> · <a href="summary.csv">Summary with intervals</a> · '
                   '<a href="breaks.csv">Nuisance drops</a></p></main></html>')
    path.write_text(''.join(content), encoding='utf-8')
