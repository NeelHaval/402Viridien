"""Render generated pattern strips and stacked synthetic columns.

Layers sit directly against each other with no drawn boundary line, as in
the real columns. The doc's frame rails are painted on top afterwards.
"""
from __future__ import annotations

import numpy as np
from PIL import Image



def layer(data, x0, x1, y0, height, rng):
    """Fill page rows y0..y0+height and columns x0..x1 with one pattern."""
    width = x1 - x0
    if data['kind'] in ('jitter', 'texture'):
        return symbol_fill(data, width, height, rng)
    if data['kind'] == 'tile':
        tile = data['tile']
        py, px = tile.shape[:2]
        fy = data.get('phase', (0,0))[0] if data.get('anchored_y',data.get('anchored',False)) else int(rng.integers(py))
        fx = data.get('phase', (0,0))[1] if data.get('anchored_x',data.get('anchored',False)) else int(rng.integers(px))
        rows = (np.arange(y0, y0 + height) - fy) % py
        if data.get('x_repeat', True):
            cols = (np.arange(x0, x1) - fx) % px
        else:
            # An unmeasured lane is a page-positioned patch, never a torus.
            cols = np.clip(np.arange(x0,x1)-data.get('x_origin',x0),0,px-1)
        return tile[rows][:, cols].copy()
    supported=min(data.get('max_layer_height',min(len(s) for s in data['strips'])),min(len(s) for s in data['strips']))
    if height>supported:
        raise ValueError(f'Quilt layer height {height} exceeds supported maximum {supported}')
    index=int(rng.integers(len(data['strips'])))
    strip = data['strips'][index]
    top = int(rng.integers(strip.shape[0] - height + 1))
    origin=data.get('strip_origins',[x0 if data.get('quilt_local') else data.get('x_origin',x0)]*len(data['strips']))[index]
    cols=np.arange(x0,x1)-origin
    valid=(cols>=0)&(cols<strip.shape[1])
    result=np.empty((height,width,3),np.uint8);result[:]=data['bg']
    result[:,valid]=strip[top:top+height][:,cols[valid]]
    return result


def symbol_fill(data, width, height, rng):
    """Native glyph bootstrap with observed density or a jittered lattice."""
    from .score import foreground
    rgb = np.empty((height,width,3),np.uint8)
    rgb[:] = data['bg']
    occupied = np.zeros((height,width),bool)
    glyphs = data['glyphs']
    if data['kind']=='texture':
        # Separate ink budgets prevent missing clasts being replaced by fines.
        # Place native large components first; collision tests use their masks,
        # rather than rejecting an entire mostly-empty bounding rectangle.
        groups=[[g for g in glyphs if max(g.shape[:2])>=8],
                [g for g in glyphs if max(g.shape[:2])<8]]
        large=data.get('large_density')
        if large is None:
            areas=[sum(int(foreground(g,data['bg']).sum()) for g in pool) for pool in groups]
            large=data.get('density',.1)*areas[0]/max(1,sum(areas))
        budgets=[large,data.get('small_density',data.get('density',.1)-large)]
        for pool,density in zip(groups,budgets):
            if not pool or density<=0:continue
            masks=[foreground(g,data['bg']) for g in pool]
            target=density*height*width;placed=0
            for _ in range(max(500,height*width*2)):
                i=int(rng.integers(len(pool)));g=pool[i];mask=masks[i];gh,gw=g.shape[:2]
                # Every intersecting origin is possible at either boundary.
                # The previous half-glyph left margin gave the right edge a
                # systematically larger supply of clipped components.
                y=int(rng.integers(1-gh,height));x=int(rng.integers(1-gw,width))
                top,left=max(0,y),max(0,x);bottom,right=min(height,y+gh),min(width,x+gw)
                if bottom<=top or right<=left:continue
                part=mask[top-y:bottom-y,left-x:right-x]
                area=int(part.sum())
                if area==0 or (occupied[top:bottom,left:right]&part).any():continue
                if placed+area>target+max(1,height*width*.002):continue
                # Avoid merging disconnected source glyphs into fake components.
                from scipy.ndimage import binary_dilation
                if (binary_dilation(occupied[max(0,top-1):min(height,bottom+1),max(0,left-1):min(width,right+1)],structure=np.ones((3,3)))[
                    top-max(0,top-1):bottom-max(0,top-1),left-max(0,left-1):right-max(0,left-1)]&part).any():continue
                rgb[top:bottom,left:right][part]=g[top-y:bottom-y,left-x:right-x][part]
                occupied[top:bottom,left:right]|=part;placed+=area
                if placed>=target:break
        return rgb
    if data['kind'] == 'jitter':
        py,px = data['spacing']
        jy,jx = data.get('jitter',(2,2))
        origins = [(y+int(rng.integers(-jy,jy+1)),x+int(rng.integers(-jx,jx+1)))
                   for y in range(-py,height+py,py) for x in range(-px,width+px,px)]
    else:
        origins = ((int(rng.integers(height)),int(rng.integers(width))) for _ in range(height*width))
    target = data.get('density',.1)*height*width
    for y,x in origins:
        g = glyphs[int(rng.integers(len(glyphs)))]; gh,gw = g.shape[:2]
        if y<0 or x<0 or y+gh>height or x+gw>width:
            continue
        if occupied[max(0,y-1):min(height,y+gh+1),max(0,x-1):min(width,x+gw+1)].any():
            continue
        mask = foreground(g, data['bg'])
        rgb[y:y+gh,x:x+gw][mask] = g[mask]
        occupied[y:y+gh,x:x+gw] |= mask
        if data['kind']=='texture' and occupied.sum() >= target:
            break
    return rgb


def glyph_model(rgb, kind='texture', spacing=None):
    """Keep complete connected components; refuse cut motifs and empty pools."""
    from .learn import background
    from .score import foreground
    from scipy.ndimage import label, find_objects
    bg = background(rgb)
    mask = foreground(rgb,bg)
    ids,n = label(mask,structure=np.ones((3,3)))
    glyphs = [];total=[0,0];retained=[0,0];excluded=[]
    for i,sl in enumerate(find_objects(ids),1):
        if sl is None:
            continue
        ys,xs = sl
        member=ids[sl]==i;area=int(member.sum());group=0 if max(ys.stop-ys.start,xs.stop-xs.start)>=8 else 1
        total[group]+=area
        if ys.start==0 or xs.start==0 or ys.stop==len(rgb) or xs.stop==rgb.shape[1]:
            excluded.append(dict(area=area,large=group==0,reason='clipped source component'))
            continue
        if area<2:
            excluded.append(dict(area=area,large=group==0,reason='single pixel'))
            continue
        g = np.empty_like(rgb[sl]);g[:] = bg
        member = ids[sl]==i;g[member] = rgb[sl][member]
        glyphs.append(g)
        retained[group]+=area
    if not glyphs:
        return None
    evidence=dict(source_ink=total,retained_ink=retained,excluded=excluded,
                  large_share=total[0]/max(1,sum(total)),
                  complete_large_motifs=sum(max(g.shape[:2])>=8 for g in glyphs),
                  morphology_supported=not total[0] or retained[0]/total[0]>=.8)
    return dict(kind=kind,glyphs=glyphs,bg=bg,density=float(mask.mean()),
                large_density=total[0]/mask.size,small_density=total[1]/mask.size,glyph_evidence=evidence,
                spacing=spacing or (12,12),jitter=(2,2),phase=(0,0),anchored=False)

def frame(rgb, rails):
    mask = np.asarray(rails['mask'], bool)
    if mask.shape[0] == rgb.shape[1]:
        rgb[:, mask] = rails['ink']
    return rgb


def sample(data, rails, height, rng, y0=None):
    """One labelled pattern strip at the doc's real column width."""
    y0 = int(rng.integers(0, 10_000)) if y0 is None else y0
    rgb = layer(data, rails['x0'], rails['x1'], y0, height, rng)
    return frame(rgb, rails)


def column(models, rails, thickness, transitions, height, rng, min_rows=4):
    """Stack layers drawn from real thickness and real label-to-label odds.

    models: {legend_id: data}; thickness: list of real layer heights (px);
    transitions: {legend_id: {next_id: count}} learned from the real column.
    Returns (rgb, truth) where truth rows use output pixel coordinates.
    """
    ids = sorted(models, key=str)
    width = rails['x1'] - rails['x0']
    rgb = np.empty((height, width, 3), np.uint8)
    truth, y, prev = [], 0, None
    base = int(rng.integers(0, 20_000))
    while y < height:
        options = transitions.get(prev, {}) if prev is not None else {}
        options = {k: v for k, v in options.items() if k in models and k != prev}
        if options:
            keys = list(options)
            p = np.array([options[k] for k in keys], float)
            lid = keys[int(rng.choice(len(keys), p=p / p.sum()))]
        else:
            pool = [i for i in ids if i != prev] or ids
            lid = pool[int(rng.integers(len(pool)))]
        distribution = thickness[lid] if isinstance(thickness, dict) else thickness
        h = int(distribution[int(rng.integers(len(distribution)))])
        h = min(max(min_rows, h), height - y)
        rgb[y:y + h] = layer(models[lid], rails['x0'], rails['x1'], base + y, h, rng)
        truth.append(dict(top=y, bottom=y + h, legend_id=lid))
        y += h
        prev = lid
    return frame(rgb, rails), truth
def repeat_preview(data, width, x_repeat=True, x0=None):
    """Native strip width, 2–3 repeat periods, and a separate 1px period marker."""
    from PIL import ImageDraw
    py, px = data['tile'].shape[:2]
    height = min(240, max(48, py * 3))
    x0=data.get('x_origin',0) if x0 is None else x0
    # A documented zero vertical phase makes the displayed period readable;
    # horizontal permissions and page placement still use the normal renderer.
    preview=dict(data,phase=(0,data.get('phase',(0,0))[1]),anchored_y=True)
    rgb=layer(preview,x0,x0+width,0,height,np.random.default_rng(0))
    image = Image.fromarray(rgb)
    marker_width = min(width, px) if x_repeat else width
    ImageDraw.Draw(image).rectangle((0, 0, marker_width-1, min(height, py)-1), outline=(195, 45, 45), width=1)
    return np.asarray(image)

