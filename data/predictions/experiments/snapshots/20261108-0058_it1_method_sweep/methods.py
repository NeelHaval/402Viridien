# ---- baseline
def m01_baseline(segs, ctx):
    protos = prototypes(ctx['legend'], ctx['img'])
    segs = split_mixed(segs, ctx['col'], ctx['y_offset'], protos)
    return name_segments(segs, ctx['col'], ctx['y_offset'], protos)


# ---- no_split
def m02_no_split(segs, ctx):
    return name_segments(segs, ctx['col'], ctx['y_offset'], prototypes(ctx['legend'], ctx['img']))


# ---- colour_only
def m03_colour_only(segs, ctx):
    return classify_by(segs, ctx, describe, weighted(1.0, 0.0, 0.0))


# ---- texture_only
def m04_texture_only(segs, ctx):
    return classify_by(segs, ctx, describe, weighted(0.0, W_INK, W_TEXTURE))


# ---- colour_heavy
def m05_colour_heavy(segs, ctx):
    return classify_by(segs, ctx, describe, weighted(3 * W_COLOR, W_INK, W_TEXTURE))


# ---- window_vote
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


# ---- scale_matched
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


# ---- colour_hist
def m08_colour_hist(segs, ctx, bins=6):
    def feat(px):
        p = px.reshape(-1, 3)
        p = p[p.mean(1) >= 128]
        h = np.histogramdd(p, bins=(bins,) * 3, range=((0, 256),) * 3)[0].ravel() if len(p) else np.zeros(bins ** 3)
        return {'hist': h / h.sum() if h.sum() else h, 'ink': float((px.mean(2) < 128).mean())}
    def dist(a, b):
        return 0.5 * np.sum((a['hist'] - b['hist']) ** 2 / (a['hist'] + b['hist'] + 1e-9)) + W_INK * abs(a['ink'] - b['ink'])
    return classify_by(segs, ctx, feat, dist)


# ---- column_adapted
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


# ---- fft_texture
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
