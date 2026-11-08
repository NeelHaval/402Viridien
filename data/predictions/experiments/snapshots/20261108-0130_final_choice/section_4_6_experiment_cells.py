# %% [text] notebook cell 48
# ## 4.6 Classification Experiments
# 
# Hill climbing on the classification step (Section 4) alone. Every method names the **same segments with the same legend** (read from the page, `LEGEND_SOURCE`), writes prediction files in the 1.1 schema and is scored by the team's classification evaluator (`eval_cells/evaluate_segment_classes_cell.py`, run unchanged apart from its CONFIG paths).
# 
# - **Modes**: `oracle` = true layer boundaries from `test/` (naming only); `pipeline` = the Section 2 segments (`SAVE_METHOD`), i.e. what the notebook ships.
# - **Metrics** (from the evaluator, background excluded from macro means): macro-F1, pixel accuracy, segment accuracy, mIoU, pooled over the test pages and for `PRIMARY_PAGE` alone.
# - **Primary metric**: iteration 1 used oracle macro-F1; from iteration 2 on it is **pipeline macro-F1 on `PRIMARY_PAGE`**, the only test page whose label types exist in its legend (the mudlog is labelled only "unknown" / "13" over 342 rows, and oracle naming is already at 98% pixel accuracy). Guard: oracle pixel accuracy must not drop by more than 1 point.
# - **Snapshots**: every `run_batch` saves `predictions/experiments/snapshots/<time>_<name>/`: `summary.md` (settings, metrics, leaderboard), `results.csv`, `leaderboard.png`, `methods.py` (the code / parameters of every method), `reports/` (the evaluator's full report per method and mode), and appends a line to `predictions/experiments/history.jsonl`.
# 
# The first cell is the harness and method library; each following cell is one iteration.

# %% [code] notebook cell 49
# 4.6 Experiment harness: page context, method library, run_batch (write predictions, run the team's classification
# evaluator, rank) and snapshots in predictions/experiments/snapshots/<time>_<name>/
import contextlib
import inspect
import io
from collections import Counter
from functools import partial

EXP_DIR = ROOT / 'predictions' / 'experiments'
CLS_SCRIPT = (ROOT / 'eval_cells' / 'evaluate_segment_classes_cell.py').read_text()
PRIMARY_PAGE = '60808400140000_hadrian 2_masterlog_2in_10000-21962'   # the test page with scorable lithology labels

# Everything a method may use for one page, built once: trimmed column, legend, rock swatches, both segment sets
EXP_CTX = {}

def exp_context(stem):
    if stem not in EXP_CTX:
        page = load_page(stem)
        img = Image.open(page_paths(stem)[0])
        legend = load_legend(stem, LEGEND_SOURCE)
        rock = legend[legend.lithology.map(is_rock)]
        label = json.loads((GT_DIR / f'{stem}.json').read_text())
        EXP_CTX[stem] = {
            'col': page['col'], 'cols': trim_borders(page['col']), 'y_offset': page['y_offset'],
            'legend': legend, 'img': img,
            'swatches': [(int(r.legend_id), r.lithology, swatch_pixels(img, r.box)) for r in rock.itertuples()],
            'segments': {
                'oracle': pd.DataFrame([(l['top_px'], l['bottom_px']) for l in label['layers']], columns=['y_top', 'y_bottom']),
                'pipeline': predict_segments(stem, SAVE_METHOD)[['y_top', 'y_bottom']].reset_index(drop=True)},
        }
    return EXP_CTX[stem]

# ---------------------------------------------------------------- shared helpers
# Pixels of a segment in the trimmed column, with a few rows cut off each end when it is tall enough
def seg_pixels(ctx, top, bottom, trim=2):
    a, b = max(top - ctx['y_offset'], 0), max(bottom - ctx['y_offset'], 0)
    if b - a > 4 * trim:
        a, b = a + trim, b - trim
    return ctx['cols'][a:b]

def blank_like(rgb):
    return rgb.size == 0 or ((rgb.mean(2) < 128).mean() < 0.01 and np.median(rgb.reshape(-1, 3), 0).min() > 235)

def named_table(segs, picks):
    return pd.DataFrame({'y_top': segs.y_top.to_numpy(), 'y_bottom': segs.y_bottom.to_numpy(),
                         'legend_id': [int(p[0]) for p in picks], 'lithology': [p[1] for p in picks]})

# Nearest swatch for every segment, with a feature function and a distance on its output
def classify_by(segs, ctx, featurise, distance):
    protos = [(i, n, featurise(px)) for i, n, px in ctx['swatches']]
    picks = []
    for s in segs.itertuples():
        px = seg_pixels(ctx, s.y_top, s.y_bottom)
        if blank_like(px):
            picks.append((-1, 'blank'))
            continue
        f = featurise(px)
        picks.append(protos[int(np.argmin([distance(f, p) for _, _, p in protos]))][:2])
    return named_table(segs, picks)

# 4.1 descriptor distance with its own colour / ink / texture weights
def weighted(wc, wi, wt):
    def dist(seg, sw):
        tv = lambda k: 0.5 * np.abs(seg[k] - sw[k]).sum()
        v = min(1.0, seg['rows'] / TEXTURE_FULL_PX)
        tex = (tv('h_dark') + tv('h_gap') + v * (tv('v_dark') + tv('v_gap'))) / 4
        return wc * np.linalg.norm(seg['lab'] - sw['lab']) + wi * abs(seg['ink'] - sw['ink']) + wt * tex
    return dist

# ---------------------------------------------------------------- the ten methods
def m01_baseline(segs, ctx):
    protos = prototypes(ctx['legend'], ctx['img'])
    segs = split_mixed(segs, ctx['col'], ctx['y_offset'], protos)
    return name_segments(segs, ctx['col'], ctx['y_offset'], protos)

def m02_no_split(segs, ctx):
    return name_segments(segs, ctx['col'], ctx['y_offset'], prototypes(ctx['legend'], ctx['img']))

def m03_colour_only(segs, ctx):
    return classify_by(segs, ctx, describe, weighted(1.0, 0.0, 0.0))

def m04_texture_only(segs, ctx):
    return classify_by(segs, ctx, describe, weighted(0.0, W_INK, W_TEXTURE))

def m05_colour_heavy(segs, ctx):
    return classify_by(segs, ctx, describe, weighted(3 * W_COLOR, W_INK, W_TEXTURE))

def m06_window_vote(segs, ctx, win=16):
    protos = [(i, n, describe(px)) for i, n, px in ctx['swatches']]
    picks = []
    for s in segs.itertuples():
        px = seg_pixels(ctx, s.y_top, s.y_bottom)
        votes = Counter()
        for y in range(0, max(len(px) - win, 0) + 1, win):
            w = px[y:y + win]
            if not blank_like(w):
                votes[int(np.argmin([descriptor_distance(describe(w), p) for _, _, p in protos]))] += len(w)
        picks.append(protos[votes.most_common(1)[0][0]][:2] if votes else (-1, 'blank'))
    return named_table(segs, picks)

def m07_scale_matched(segs, ctx, heights=(4, 8, 16, 24)):
    cache = {}
    def strips(k, h):                       # descriptors of swatch k cut into strips h rows tall
        sw = ctx['swatches'][k][2]
        h = min(h, len(sw))
        if (k, h) not in cache:
            cache[(k, h)] = [describe(sw[y:y + h]) for y in range(0, len(sw) - h + 1, max(1, h // 2))]
        return cache[(k, h)]
    picks = []
    for s in segs.itertuples():
        px = seg_pixels(ctx, s.y_top, s.y_bottom)
        if blank_like(px):
            picks.append((-1, 'blank'))
            continue
        d = describe(px)
        h = max([x for x in heights if x <= len(px)] or [len(px)]) if len(px) < max(heights) else 10 ** 6
        best = [min(descriptor_distance(d, st) for st in strips(k, h)) for k in range(len(ctx['swatches']))]
        picks.append(ctx['swatches'][int(np.argmin(best))][:2])
    return named_table(segs, picks)

def m08_colour_hist(segs, ctx, bins=6):
    def feat(px):
        p = px.reshape(-1, 3)
        p = p[p.mean(1) >= 128]
        h = np.histogramdd(p, bins=(bins,) * 3, range=((0, 256),) * 3)[0].ravel() if len(p) else np.zeros(bins ** 3)
        return {'hist': h / h.sum() if h.sum() else h, 'ink': float((px.mean(2) < 128).mean())}
    def dist(a, b):
        return 0.5 * np.sum((a['hist'] - b['hist']) ** 2 / (a['hist'] + b['hist'] + 1e-9)) + W_INK * abs(a['ink'] - b['ink'])
    return classify_by(segs, ctx, feat, dist)

def m09_column_adapted(segs, ctx, min_rows=24):
    first = m02_no_split(segs, ctx)
    sure = first[(first.legend_id >= 0) & (first.y_bottom - first.y_top >= min_rows)]
    sure = sure[sure.margin >= sure.margin.median()] if len(sure) else sure
    protos = {}
    for lid, grp in sure.groupby('legend_id'):
        px = np.concatenate([seg_pixels(ctx, r.y_top, r.y_bottom) for r in grp.itertuples()])
        protos[int(lid)] = (int(lid), grp.lithology.iloc[0], describe(px))
    for i, n, sw in ctx['swatches']:
        protos.setdefault(i, (i, n, describe(sw)))
    plist = list(protos.values())
    picks = []
    for s in segs.itertuples():
        px = seg_pixels(ctx, s.y_top, s.y_bottom)
        if blank_like(px):
            picks.append((-1, 'blank'))
            continue
        d = describe(px)
        picks.append(plist[int(np.argmin([descriptor_distance(d, p) for _, _, p in plist]))][:2])
    return named_table(segs, picks)

def m10_fft_texture(segs, ctx, nr=6, na=6):
    def feat(px):
        dark = (px.mean(2) < 128).astype(np.float32)
        h, w = dark.shape
        F = np.abs(np.fft.fftshift(np.fft.fft2(dark - dark.mean())))
        yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
        fy, fx = yy / max(h, 1), xx / max(w, 1)
        r, a = np.hypot(fy, fx), np.mod(np.arctan2(fy, fx), np.pi)
        m = (r > 0) & (r <= 0.5)
        H = np.histogram2d(r[m], a[m], bins=(nr, na), range=((0, 0.5), (0, np.pi)), weights=F[m] ** 2)[0].ravel()
        light = px[px.mean(2) >= 128]
        return {'spec': H / H.sum() if H.sum() else H, 'ink': float(dark.mean()),
                'lab': srgb_to_lab(np.median(light, 0) if len(light) else np.zeros(3))}
    def dist(a, b):
        return W_COLOR * np.linalg.norm(a['lab'] - b['lab']) + W_INK * abs(a['ink'] - b['ink']) + 0.5 * np.abs(a['spec'] - b['spec']).sum()
    return classify_by(segs, ctx, feat, dist)

# Split mixed segments with any descriptor distance, then name every piece with it (4.2 split_mixed, parameterised)
def split_then_name(segs, ctx, win=SPLIT_WINDOW, min_run=SPLIT_MIN_RUN, dist=descriptor_distance):
    protos = prototypes(ctx['legend'], ctx['img'])
    cols, off = ctx['cols'], ctx['y_offset']
    rows = []
    for s in segs.itertuples():
        a, n = s.y_top - off, (s.y_bottom - s.y_top) // win
        if n < 2 * min_run:
            rows.append((s.y_top, s.y_bottom))
            continue
        lab = [int(np.argmin([dist(describe(cols[a + i * win:a + (i + 1) * win]), p) for _, _, p in protos]))
               for i in range(n)]
        runs = []                                    # [label, first window, n windows]
        for i, l in enumerate(lab):
            if runs and runs[-1][0] == l:
                runs[-1][2] += 1
            else:
                runs.append([l, i, 1])
        kept = [r for r in runs if r[2] >= min_run] or [max(runs, key=lambda r: r[2])]
        starts = [r[1] for k, r in enumerate(kept) if k == 0 or r[0] != kept[k - 1][0]][1:]
        edges = [s.y_top] + [s.y_top + i * win for i in starts] + [s.y_bottom]
        rows += list(zip(edges[:-1], edges[1:]))
    pieces = pd.DataFrame(rows, columns=['y_top', 'y_bottom'])
    picks = []
    for s in pieces.itertuples():
        px = seg_pixels(ctx, s.y_top, s.y_bottom)
        if blank_like(px):
            picks.append((-1, 'blank'))
            continue
        d = describe(px)
        picks.append(protos[int(np.argmin([dist(d, p) for _, _, p in protos]))][:2])
    return named_table(pieces, picks)

METHODS = {'baseline': m01_baseline, 'no_split': m02_no_split, 'colour_only': m03_colour_only,
           'texture_only': m04_texture_only, 'colour_heavy': m05_colour_heavy, 'window_vote': m06_window_vote,
           'scale_matched': m07_scale_matched, 'colour_hist': m08_colour_hist,
           'column_adapted': m09_column_adapted, 'fft_texture': m10_fft_texture}

# ---------------------------------------------------------------- run, score, rank, snapshot
# Name every test page with one method and write the prediction files (1.1 schema, validated)
def write_predictions(batch, name, fn, mode):
    out = EXP_DIR / 'runs' / batch / f'{name}__{mode}'
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.json'):
        old.unlink()
    for stem in TEST_STEMS:
        ctx = exp_context(stem)
        layers = merge_named(fn(ctx['segments'][mode], ctx).sort_values('y_top'))
        rec = {'schema_version': SCHEMA_VERSION, 'stem': stem, 'image_name': load_record(stem)['image_name'],
               'written_by': {'experiment': {'batch': batch, 'method': name, 'mode': mode, 'legend_source': LEGEND_SOURCE}},
               'legend': legend_from_table(ctx['legend']),
               'layers': layers_from_table(layers[layers.legend_id >= 0])}
        errs = record_errors(rec)
        if errs:
            raise ValueError(f'{name}/{mode}/{stem}: ' + '; '.join(errs[:5]))
        (out / f'{stem}.json').write_text(json.dumps(rec, indent=1))
    return out

# The team's classification evaluator on one prediction folder (CONFIG paths swapped in, report captured)
def run_cls_eval(pred_dir):
    src = re.sub(r'^GT_PATH = .*$', f'GT_PATH = {str(GT_DIR)!r}', CLS_SCRIPT, count=1, flags=re.M)
    src = re.sub(r'^PRED_PATH = .*$', f'PRED_PATH = {str(pred_dir)!r}', src, count=1, flags=re.M)
    ns = {'__name__': 'classification_eval'}
    with contextlib.redirect_stdout(io.StringIO()) as report, contextlib.redirect_stderr(io.StringIO()):
        exec(compile(src, 'evaluate_segment_classes_cell.py', 'exec'), ns)
    return ns['AGGREGATE'], ns['RESULTS'], report.getvalue()

# One row per method: primary metric first, guards next, change vs the reference method
def leaderboard(df, primary=('pipeline', 'page_f1'), reference='baseline'):
    ok = df[df.get('error', pd.Series(index=df.index, dtype=object)).isna()] if 'error' in df else df
    board = ok.pivot(index='method', columns='mode', values=['page_f1', 'macro_f1', 'pixel_acc', 'segment_acc'])
    board.columns = [f'{mode} {metric}' for metric, mode in board.columns]
    key = f'{primary[0]} {primary[1]}'
    board = board.sort_values(key, ascending=False)
    if reference in board.index:
        board[f'{key} vs {reference}'] = board[key] - board.loc[reference, key]
        if 'oracle pixel_acc' in board:
            board['oracle pixel_acc vs ' + reference] = board['oracle pixel_acc'] - board.loc[reference, 'oracle pixel_acc']
    return board

def plot_leaderboard(board, key, title):
    b = board.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(b) + 1.2))
    ax.barh(b.index, b[key], color=['tab:orange' if m == 'baseline' else 'tab:blue' for m in b.index])
    for y, v in enumerate(b[key]):
        ax.text(v, y, f' {v:.3f}', va='center', fontsize=8)
    ax.set_xlabel(key)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, max(1.0, b[key].max() * 1.1))
    plt.tight_layout()
    return fig

# Save everything needed to look this batch up later, and append it to the history
def snapshot(batch, note, df, board, key, methods, reports, fig):
    snap = EXP_DIR / 'snapshots' / f"{datetime.datetime.now():%Y%m%d-%H%M}_{batch}"
    (snap / 'reports').mkdir(parents=True, exist_ok=True)
    df.to_csv(snap / 'results.csv', index=False)
    board.to_csv(snap / 'leaderboard.csv')
    fig.savefig(snap / 'leaderboard.png', dpi=130, bbox_inches='tight')
    for (m, mode), rep in reports.items():
        (snap / 'reports' / f'{m}__{mode}.txt').write_text(rep)
    src = []
    for m, fn in methods.items():
        try:
            src.append(f'# ---- {m}\n' + inspect.getsource(fn))
        except (OSError, TypeError):
            src.append(f'# ---- {m}\n# {fn!r}')
    (snap / 'methods.py').write_text('\n\n'.join(src))
    try:
        table = board.round(4).to_markdown()
    except ImportError:
        table = '    ' + board.round(4).to_string().replace('\n', '\n    ')
    (snap / 'summary.md').write_text('\n'.join([
        f'# {batch}', '', note, '',
        f"- time: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f'- legend source: {LEGEND_SOURCE}; Section 2 segments: {SAVE_METHOD}; test pages: {len(TEST_STEMS)}',
        f'- primary metric: {key} (page_f1 = macro-F1 on {PRIMARY_PAGE}); guard: oracle pixel accuracy',
        '- metrics from eval_cells/evaluate_segment_classes_cell.py; background excluded from macro means',
        '', table, '']))
    with open(EXP_DIR / 'history.jsonl', 'a') as f:
        f.write(json.dumps({'time': datetime.datetime.now().isoformat(timespec='seconds'), 'batch': batch, 'note': note,
                            'snapshot': str(snap), 'primary': key, 'legend_source': LEGEND_SOURCE,
                            'segments': SAVE_METHOD, 'results': df.to_dict('records')}, default=float) + '\n')
    return snap

# Run every method in every mode, score with the evaluator, rank and snapshot; returns (results, leaderboard)
def run_batch(batch, methods, note, modes=('pipeline', 'oracle'), primary=('pipeline', 'page_f1')):
    rows, reports = [], {}
    for name, fn in methods.items():
        for mode in modes:
            row = {'method': name, 'mode': mode}
            try:
                agg, per_file, reports[(name, mode)] = run_cls_eval(write_predictions(batch, name, fn, mode))
                page = per_file[f'{PRIMARY_PAGE}.json']
                row.update({'page_f1': page['macro_f1'], 'page_pixel_acc': page['pixel_accuracy'],
                            'macro_f1': agg['macro_f1'], 'pixel_acc': agg['pixel_accuracy'], 'miou': agg['miou'],
                            'mean_class_acc': agg['mean_class_accuracy'], 'fw_iou': agg['freq_weighted_iou'],
                            'segment_acc': agg['segment_level']['segment_accuracy']})
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'
            rows.append(row)
            print(f"{name:18s} {mode:9s} " + (row.get('error') or
                  f"page F1 {row['page_f1']:.3f}  macro-F1 {row['macro_f1']:.3f}  pixel {row['pixel_acc']:.3f}"))
    df = pd.DataFrame(rows)
    board = leaderboard(df, primary)
    key = f'{primary[0]} {primary[1]}'
    fig = plot_leaderboard(board, key, f'{batch}: {key}')
    snap = snapshot(batch, note, df, board, key, methods, reports, fig)
    plt.show()
    print('snapshot:', snap)
    return df, board

# %% [code] notebook cell 50
# Iteration 1: ten naming methods (re-run for its snapshot; primary at the time was oracle macro-F1)
it1, board1 = run_batch('it1_method_sweep', METHODS, 'Ten naming methods on the same segments and legend.',
                        primary=('oracle', 'macro_f1'))
display(board1.round(3))

# %% [text] notebook cell 51
# ### Iteration 2: splitting
# 
# Iteration 1 showed naming is near its ceiling on true boundaries (98% pixel accuracy) and that splitting is the only classification step that recovers boundaries Section 2 misses. This sweep tunes the split with the baseline distance: window height `win` (8-32 px) times the minimum run of matching windows `min_run` (1-4). Primary: pipeline macro-F1 on the hadrian page; guard: oracle pixel accuracy.

# %% [code] notebook cell 52
# Iteration 2: split window height x minimum run, baseline descriptor distance
split_grid = {f'split_w{w}_r{r}': partial(split_then_name, win=w, min_run=r)
              for w in (8, 12, 16, 24, 32) for r in (1, 2, 3, 4)}
it2, board2 = run_batch('it2_split_sweep', {'baseline': m01_baseline, 'no_split': m02_no_split, **split_grid},
                        'Split window height x minimum run (baseline distance).')
display(board2.round(3).head(12))

# %% [text] notebook cell 53
# ### Results
# 
# The latest run of every iteration (read from `predictions/experiments/history.jsonl`; interrupted iterations are listed in `IGNORED_BATCHES`). Columns:
# 
# - **pipeline page_f1** (primary): macro-F1 on the hadrian page with the Section 2 segments, i.e. what the notebook ships.
# - **pipeline pixel_acc**: share of rows named right with the Section 2 segments (both test pages).
# - **oracle pixel_acc / segment_acc**: naming alone on the true layer boundaries: rows named right, and true layers whose main predicted rock is right.
# - **macro_f1** columns pool both pages; they are dragged down by the mudlog's "unknown" / "13" labels and by rocks predicted on a handful of rows, so read them with care.
# 
# The table and chart are also saved to `predictions/experiments/results_summary.csv` / `.png`.

# %% [code] notebook cell 54
# Results of the classification iterations: latest run of each, as a table and a chart
IGNORED_BATCHES = {'it2_split_sweep'}          # interrupted, not compared

runs = {}
for line in (EXP_DIR / 'history.jsonl').read_text().splitlines():
    h = json.loads(line)
    if h.get('batch') and h['batch'] not in IGNORED_BATCHES:
        runs[h['batch']] = h                    # a later run of the same iteration replaces the earlier one
res = pd.DataFrame([{'iteration': b, **r} for b, h in runs.items() for r in h['results']
                    if not isinstance(r.get('error'), str)])

results = res.pivot_table(index=['iteration', 'method'], columns='mode',
                          values=['page_f1', 'macro_f1', 'pixel_acc', 'segment_acc'])
results.columns = [f'{mode} {metric}' for metric, mode in results.columns]
order = ['pipeline page_f1', 'pipeline pixel_acc', 'oracle pixel_acc', 'oracle segment_acc',
         'oracle page_f1', 'pipeline macro_f1', 'oracle macro_f1', 'pipeline segment_acc']
results = results[[c for c in order if c in results]].sort_values('pipeline page_f1', ascending=False)
results.to_csv(EXP_DIR / 'results_summary.csv')
display(results.round(3))

# One bar group per method: the primary metric and the three guards
show = {'pipeline page_f1': 'shipped: macro-F1 on hadrian (primary)', 'pipeline pixel_acc': 'shipped: rows named right',
        'oracle pixel_acc': 'naming only: rows named right', 'oracle segment_acc': 'naming only: layers named right'}
names = [m if results.index.get_level_values(0).nunique() == 1 else f'{it}: {m}' for it, m in results.index][::-1]
y, h = np.arange(len(names)), 0.2
fig, ax = plt.subplots(figsize=(11, 0.6 * len(names) + 1.6))
for k, (col, label) in enumerate(show.items()):
    ax.barh(y + (1.5 - k) * h, results[col].to_numpy()[::-1], h, label=label)
ax.set_yticks(y, names)
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold' if tick.get_text().endswith('baseline') else 'normal')
ax.set_xlim(0, 1)
ax.set_xlabel('score (higher is better)')
ax.set_title('Classification methods, sorted by the primary metric (top = best)', fontsize=10)
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
fig.savefig(EXP_DIR / 'results_summary.png', dpi=130, bbox_inches='tight')
plt.show()