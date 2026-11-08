"""Reproducible legend comparison and visual review artifacts (dataset read-only)."""
import json
from pathlib import Path
from time import perf_counter
import statistics
import sys

from PIL import Image, ImageDraw, ImageFont
from harness import io, legend

ROOT = io.ROOT / 'notes'


def discover():
    folder = ROOT / 'legfix_crops'
    folder.mkdir(exist_ok=True)
    records = []
    for name in io.list_logs():
        log = io.open_log(name)
        start = perf_counter()
        entries = legend.parse(log)
        labels = sorted({e.label for e in entries if e.is_lithology})
        row = dict(log=name, before=labels, baseline_seconds=perf_counter()-start)
        if not labels:
            row['crop'] = str(folder / f'{sum(not r["before"] for r in records)+1}.png')
            Image.fromarray(io.read_rgb(log, 'lithology_legend')).save(row['crop'])
        records.append(row)
        print(len(records), len(labels), name, flush=True)
        log.close()
    (ROOT / 'legfix_baseline.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def compare():
    from harness import legend_v2
    source = 'legfix_results.json' if any(flag in sys.argv for flag in ['--failed-only', '--layouts-only']) else 'legfix_baseline.json'
    records = json.loads((ROOT / source).read_text(encoding='utf-8'))
    sheets = ROOT / 'legfix_sheets'
    sheets.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    failed = 0
    for row in records:
        if '--failed-only' in sys.argv and row['before']:
            continue
        if '--layouts-only' in sys.argv and not row.get('layout'):
            continue
        log = io.open_log(row['log'])
        cache_path = legend_v2.CACHE / (log.name+'.json')
        if cache_path.exists():
            cache_path.unlink()
        start = perf_counter()
        entries = legend_v2.parse(log)
        row['uncached_seconds'] = perf_counter()-start
        samples = []
        for _ in range(3):
            start = perf_counter()
            cached = legend_v2.parse(log)
            samples.append(perf_counter()-start)
            assert [(e.label, e.is_lithology, e.swatch_box) for e in cached] == [
                (e.label, e.is_lithology, e.swatch_box) for e in entries]
        row['cached_seconds'] = statistics.median(samples)
        lith = [e for e in entries if e.is_lithology]
        row['after'] = sorted({e.label for e in lith})
        row['n_entries_after'] = len(lith)
        row['added'] = sorted(set(row['after'])-set(row['before']))
        row['removed'] = sorted(set(row['before'])-set(row['after']))
        row['layout'] = json.loads(cache_path.read_text(encoding='utf-8'))['layout']
        x0, y0, x1, y1 = log.boxes['lithology_legend']
        for entry in entries:
            assert isinstance(entry, legend.LegendEntry)
            a, b, c, d = entry.swatch_box
            assert x0 <= a < c <= x1 and y0 <= b < d <= y1
            assert entry.swatch.ndim == 3 and entry.swatch.shape[2] == 3
            assert entry.swatch.size and entry.swatch.dtype.name == 'uint8'
        if not row['before']:
            failed += 1
            sheet = Image.new('RGB', (1350, 70+((len(lith)+2)//3)*110), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 8), f'{failed}. {log.name}', fill='black', font=small)
            draw.text((12, 31), f'{len(lith)} lithology swatches (inner crop, enlarged for inspection)', fill='black', font=small)
            for i, entry in enumerate(lith):
                x, y = 12+(i % 3)*450, 72+(i//3)*110
                swatch = Image.fromarray(entry.swatch)
                swatch.thumbnail((100, 75))
                if swatch.width < 85:
                    swatch = swatch.resize((85, round(swatch.height*85/swatch.width)), Image.Resampling.NEAREST)
                sheet.paste(swatch, (x, y))
                draw.text((x+100, y+15), entry.label, font=font, fill='black')
                draw.text((x+100, y+45), f'id {entry.id}', font=small, fill='#555555')
            sheet.save(sheets/f'{failed}.png')
            row['sheet'] = failed
        print(len(row['before']), '->', len(lith), f'{row["uncached_seconds"]:.3f}s / {row["cached_seconds"]:.4f}s',
              row['log'], 'DIFF' if row['before'] and (row['added'] or row['removed']) else '', flush=True)
        log.close()
        (ROOT / 'legfix_results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def report():
    rows = json.loads((ROOT / 'legfix_results.json').read_text(encoding='utf-8'))
    assert len(rows) == 51 and all('after' in row for row in rows)
    fixed = [r for r in rows if not r['before']]
    existing = [r for r in rows if r['before']]
    differences = [r for r in existing if r['added'] or r['removed']]
    lines = [
        '# Legend parser repair', '',
        'Implementation: `harness/legend_v2.py`; original `LegendEntry` is imported from '
        '`harness.legend`. The original parser is unchanged. All v2 JSON, swatch PNGs, '
        'OCR records and outlined debug crops go to `out/_cache/legend_v2/`. '
        'The dataset is read-only throughout.', '',
        '## Discovery and cause', '',
        'The audit called `io.list_logs()` and `legend.parse(log)` on all 51 logs. '
        'Eleven had zero lithology entries. Every failing source crop was saved to '
        '`notes/legfix_crops/<sheet number>.png` and opened for inspection.', '',
        'The decisive failure is heading recognition: the original regular expression '
        'requires spaces and plural `symbols`, while OCR returns `LITHOLOGYSYMBOL` '
        'and `SHOW&DRILLINGSYMBOL`. Only `MODIFIERS` survives, and the nearest-heading '
        'rule assigns every swatch to it. Rectangle detection and right-hand label '
        'pairing already find the cells. Simply accepting the missing headings would '
        'still discard real rocks because the exporter has flattened the categories '
        'to five columns while retaining a three-title banner.', '',
        'The requested stem '
        '`60811407300100_bp_gc743_15607_dc141_st00bp01_mudlog_1md_10700-22872` '
        'is absent from the 51 names returned here. The actual failing DC141 log is '
        '`60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev`. '
        'The similarly named DC222 log is present and passes the regression unchanged.', '',
        '## Fix and visual review', '',
        'Headings are normalized independent of OCR whitespace, punctuation and '
        'singular/plural `symbol`. True panels use continuous vertical rules, or '
        'heading midpoint boundaries when unruled; only the Lithology section can '
        'produce lithology entries. A general flattened-grid test requires a complete '
        'rock row spanning more columns than the heading count, a technical row, '
        'and no continuous panel dividers. This layout uses positive rock vocabulary '
        'instead of banner proximity. Gas_Good, Dry, fresh/salted water, core, '
        'casing, tests and survey symbols remain non-lithology. There are no '
        'log-name conditions or per-log settings.', '',
        'For the recognized three-title layouts, batched recognition of tight OCR '
        'text crops recovers spaces without repeatedly running text detection. '
        'Other layouts retain the original detection, OCR and classification rules. '
        'Swatches remain RGB interior crops and boxes use full-image coordinates.', '',
        'All eleven final contact sheets were opened and checked against the source '
        'crops. A short technical row in the two Ardennes legends initially prevented '
        'flattened-grid recognition; the rule was corrected to accept that shorter row. '
        'Their final sheets include Tar and Sylvite as well as the other rocks. '
        'Printed truncations such as `Calcareous clay-shal` and `Argillaceous limesto` '
        'are preserved. Some sources themselves print `Silstone` (confirmed in an '
        'enlarged source crop, `notes/legfix_crops/text_detail.png`); the returned '
        'label preserves that spelling. No per-log spelling substitutions were introduced.', '',
        '## Per-log results', '',
        'Counts are entries, while regression comparisons use sets of exact labels. '
        'One existing log has 16 entries but 15 distinct labels. Sheet numbers identify '
        '`notes/legfix_sheets/<n>.png`. Labels below are the returned label sets.', '',
        '| Log | n_lith before | n_lith after | Labels after | Sheet |',
        '|---|---:|---:|---|---:|',
    ]
    for r in rows:
        original = json.loads((io.ROOT/'out/_cache/legend'/(r['log']+'.json')).read_text(encoding='utf-8'))
        n_before = sum(e['is_lithology'] for e in original['entries'])
        labels = '; '.join(r['after']).replace('|', '\\|')
        lines.append(f'| {r["log"]} | {n_before} | {r["n_entries_after"]} | {labels} | {r.get("sheet", "—")} |')
    lines += ['', '## Regression diff (all 40 previously working logs)', '',
              f'{len(existing)-len(differences)}/40 identical lithology label sets; '
              f'{len(differences)} differences.', '', '| Log | Added labels | Removed labels |',
              '|---|---|---|']
    for r in existing:
        lines.append(f'| {r["log"]} | {"; ".join(r["added"]) or "—"} | {"; ".join(r["removed"]) or "—"} |')
    lines += ['', '## Timing (seconds per legend)', '',
              'Measured with `perf_counter`. Uncached means the v2 JSON was absent: '
              'it includes reading the TIFF crop, OCR, detection and writing cache/debug PNGs. '
              'The model is initialized once per Python process. Cached is the median of '
              'three calls, including JSON and swatch PNG reads; no in-memory result cache. '
              'Original-cache OCR is never reused by v2.', '',
              '| Group | Uncached mean | Uncached median | Uncached range | Cached mean | Cached median | Cached range |',
              '|---|---:|---:|---|---:|---:|---|']
    for name, group in [('All 51', rows), ('11 repaired', fixed), ('40 regression', existing)]:
        a = [r['uncached_seconds'] for r in group]
        b = [r['cached_seconds'] for r in group]
        lines.append(f'| {name} | {statistics.mean(a):.3f} | {statistics.median(a):.3f} | '
                     f'{min(a):.3f}–{max(a):.3f} | {statistics.mean(b):.4f} | '
                     f'{statistics.median(b):.4f} | {min(b):.4f}–{max(b):.4f} |')
    lines += ['', '| Log | Uncached seconds | Cached seconds |', '|---|---:|---:|']
    for r in rows:
        lines.append(f'| {r["log"]} | {r["uncached_seconds"]:.3f} | {r["cached_seconds"]:.4f} |')
    lines += ['', '## Reproduction and checks', '',
              'Run from the bake-off root:', '', '```text',
              'python -X utf8 -m notes.legfix_audit',
              'python -X utf8 -m notes.legfix_audit --compare',
              'python -X utf8 -m notes.legfix_audit --report',
              'python -X utf8 -m notes.legfix_checks', '```', '',
              'Five focused checks cover compact/singular headings, full and shortened '
              'flattened rows, real ruled panel exclusion (even for rock labels), '
              'original LegendEntry type, full-image coordinates and exact cached swatch '
              'round trips. The full audit checks every entry stays inside the annotated '
              'legend box, RGB shape/dtype and cached label/flag/box equality. '
              'Machine-readable results: `notes/legfix_baseline.json` and '
              '`notes/legfix_results.json`.', '']
    content = '\n'.join(lines)
    with (ROOT/'legfix.md').open('w', encoding='utf-8') as handle:
        for start in range(0, len(content), 4096):
            handle.write(content[start:start+4096])


if __name__ == '__main__':
    if '--report' in sys.argv:
        report()
    elif '--compare' in sys.argv:
        compare()
    else:
        discover()
