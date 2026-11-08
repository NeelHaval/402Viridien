"""Exact free-layer schema, grouping, calibration, validation and challenge export.

The browser mirrors depth and grouping; the server recomputes derived depths on save.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from labelling_setup import DATASET

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "labels"
LABEL_SPLITS = ("test", "test_bw")

TOP_KEYS = ("image_name", "split", "depth_calibration", "legend", "layers")
LEGEND_KEYS = ("id", "box", "label_printed", "same_symbol_as")
LAYER_KEYS = ("top_px", "bottom_px", "top_ft", "bottom_ft", "top_m", "bottom_m", "composition", "flags")
COMPOSITION_KEYS = ("legend_id", "type", "proportion", "x_range")
FLAGS = ("uncertain",)

# --------------------------------------------------------------------------- data access
@lru_cache(maxsize=1)
def load_vocab() -> tuple[str, ...]:
    return tuple(json.loads((ROOT / "vocab.json").read_text()))


def load_splits() -> dict:
    return json.loads((ROOT / "splits.json").read_text())


def label_stems() -> dict[str, str]:
    """{stem: split} for every page that gets labelled, test first."""
    sp = load_splits()
    out = {}
    for split in LABEL_SPLITS:
        for s in sp.get(split, []):
            out.setdefault(s, split)
    return out


@lru_cache(maxsize=None)
def page_info(stem: str) -> dict:
    """bbox annotation + image file name for a stem."""
    d = json.loads((DATASET / "bbox_annotation" / f"{stem}.json").read_text())
    imgs = [p for p in (DATASET / "images").glob(f"{glob_escape(stem)}.tif*") if p.stem == stem]
    return {"image_name": imgs[0].name if imgs else d.get("image_name"),
            "width": d["width"], "height": d["height"], "boxes": d["bounding_boxes"]}


def glob_escape(s: str) -> str:
    return "".join(f"[{c}]" if c in "[]*?" else c for c in s)


def bbox(stem: str) -> list[int]:
    return [int(v) for v in page_info(stem)["boxes"]["interpreted_lithology_column"]]


def label_path(stem: str) -> Path:
    return LABELS_DIR / f"{stem}.json"


def empty_label(stem: str, split: str) -> dict:
    return {"image_name": page_info(stem)["image_name"], "split": split,
            "depth_calibration": {"anchors": []}, "legend": [], "layers": []}


def dumps(label):
    return json.dumps(label, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- depth model
def _sorted_ticks(ticks):
    return sorted(((float(t["y"]), float(t.get("depth_ft", t.get("depth")))) for t in ticks), key=lambda p: p[0])


def depth_fn(ticks):
    """Piecewise-linear depth(y); end segments extrapolate with the nearest slope. Needs >= 2 ticks."""
    pts = _sorted_ticks(ticks)
    if len(pts) < 2:
        return None
    ys = [p[0] for p in pts]
    ds = [p[1] for p in pts]

    def f(y: float) -> float:
        if y <= ys[0]:
            i = 0
        elif y >= ys[-1]:
            i = len(ys) - 2
        else:
            lo, hi = 0, len(ys) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ys[mid] <= y:
                    lo = mid
                else:
                    hi = mid
            i = lo
        y0, y1, d0, d1 = ys[i], ys[i + 1], ds[i], ds[i + 1]
        if y1 == y0:
            return d0
        return d0 + (y - y0) * (d1 - d0) / (y1 - y0)
    return f


def linear_fit(ticks) -> dict | None:
    pts = _sorted_ticks(ticks)
    if len(pts) < 2:
        return None
    n = len(pts)
    my = sum(p[0] for p in pts) / n
    md = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - my) ** 2 for p in pts)
    if sxx == 0:
        return None
    slope = sum((p[0] - my) * (p[1] - md) for p in pts) / sxx
    icpt = md - slope * my
    # residuals in the label's tick order, so the UI can show them per tick
    res = [float(t.get("depth_ft", t.get("depth"))) - (slope * float(t["y"]) + icpt) for t in ticks]
    return {"slope": slope, "intercept": icpt, "residuals": res,
            "max_residual_ft": max(abs(r) for r in res)}


# --------------------------------------------------------------------------- legend classes
def class_of(legend) -> dict[int, int]:
    """{legend id: smallest id in its same_symbol_as class} (symmetric, transitive)."""
    ids = [e["id"] for e in legend]
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in legend:
        for o in e.get("same_symbol_as", []):
            if o in parent:
                a, b = find(e["id"]), find(o)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = {i: find(i) for i in ids}
    mins = {}
    for i, r in roots.items():
        mins[r] = min(mins.get(r, i), i)
    return {i: mins[r] for i, r in roots.items()}


def class_names(legend) -> dict[int, str]:
    cls = class_of(legend)
    canon = {e["id"]: e["label_printed"] for e in legend}
    members: dict[int, set] = {}
    for i, c in cls.items():
        members.setdefault(c, set()).add(canon[i])
    return {i: "|".join(sorted(members.get(c, {canon[i]}))) for i, c in cls.items()}


# --------------------------------------------------------------------------- free boxes
import copy
import math


def sync_legend_labels(label):
    """Legend names are authoritative; accept legacy canonical fields on upgrade."""
    out = copy.deepcopy(label)
    names = {}
    for entry in out.get('legend', []) if isinstance(out.get('legend'), list) else []:
        if not isinstance(entry, dict):
            continue
        entry.pop('canonical', None)
        if _is_int(entry.get('id')) and isinstance(entry.get('label_printed'), str):
            names[entry['id']] = entry['label_printed']
    for layer in out.get('layers', []) if isinstance(out.get('layers'), list) else []:
        if not isinstance(layer, dict):
            continue
        for comp in layer.get('composition', []) if isinstance(layer.get('composition'), list) else []:
            if isinstance(comp, dict) and _is_int(comp.get('legend_id')) and comp['legend_id'] in names:
                comp['type'] = names[comp['legend_id']]
        comps = layer.get('composition')
        if isinstance(comps, list) and comps and isinstance(comps[0], dict):
            first = comps[0]
            first['proportion'] = 1.0
            layer['composition'] = [first]
        if isinstance(layer.get('flags'), list):
            layer['flags'] = [flag for flag in layer['flags'] if flag != 'mixed']
    return out


def recompute_depths(label):
    """Recompute only derived fields. Invalid drafts remain saveable and validate later."""
    out = copy.deepcopy(label)
    try:
        anchors = out['depth_calibration']['anchors']
        valid = all(_is_int(a['y']) and _is_num(a['depth_ft']) for a in anchors)
        pts = sorted(anchors, key=lambda a: a['y'])
        valid = valid and all(a['y'] < b['y'] and a['depth_ft'] < b['depth_ft'] for a,b in zip(pts,pts[1:]))
        f = depth_fn(anchors) if valid else None
    except (KeyError, TypeError, ValueError):
        f = None
    for layer in out.get('layers', []) if isinstance(out.get('layers'), list) else []:
        if not isinstance(layer, dict):
            continue
        for edge in ('top', 'bottom'):
            y = layer.get(edge+'_px')
            ft = f(y) if f and _is_int(y) else None
            layer[edge+'_ft'] = round(ft, 1) if ft is not None else None
            layer[edge+'_m'] = round(ft * .3048, 1) if ft is not None else None
    return out


def group_boxes(boxes, legend, anchors):
    """One rock per layer: keep its first box at 100%, independent of box width."""
    groups = {}
    names = {e['id']: e['label_printed'] for e in legend}
    for box in boxes:
        groups.setdefault((box['top_px'],box['bottom_px']), []).append(box)
    layers = []
    for (top,bottom), slices in sorted(groups.items()):
        first = slices[0]
        layers.append(dict(top_px=top,bottom_px=bottom,top_ft=None,bottom_ft=None,top_m=None,bottom_m=None,
            composition=[dict(legend_id=first['legend_id'],type=names.get(first['legend_id'],'unknown'),
                              proportion=1.0,x_range=list(first['x_range']))],
            flags=['uncertain'] if any('uncertain' in b.get('flags',[]) for b in slices) else []))
    return recompute_depths({'depth_calibration':{'anchors':anchors},'layers':layers})['layers']


def inner_range(stem, y):
    path = ROOT/'labelling_cache'/stem/'meta.json'
    if path.exists():
        for chunk in json.loads(path.read_text()).get('inner_edges',[]):
            if chunk['y0'] <= y < chunk['y1']:
                return chunk['x_range']
    x1,_,x2,_ = bbox(stem)
    return [x1,x2]


def _is_int(v): return isinstance(v,int) and not isinstance(v,bool)
def _is_num(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def validate(label, stem):
    errors,warnings=[],[]
    res={'errors':errors,'warnings':warnings,'n_unreviewed':0,'fit':None}
    def err(msg,y=None): errors.append({'msg':msg,'y':y})
    def warn(msg,y=None): warnings.append({'msg':msg,'y':y})
    def keys(obj,expected,where):
        if not isinstance(obj,dict) or set(obj)!=set(expected):
            err(where+': keys differ from schema'); return False
        return True
    def seq(v,where):
        if not isinstance(v,list): err(where+': expected a list'); return []
        return v
    if not keys(label,TOP_KEYS,'label'): return res
    stems=label_stems()
    if stem not in stems: err('unknown labelled page'); return res
    info=page_info(stem); x1,y1,x2,y2=bbox(stem)
    if label['split']!=stems[stem]: err('split mismatch with splits.json')
    if label['image_name']!=info['image_name']: err('image_name mismatch')
    legend={}
    for e in seq(label['legend'],'legend'):
        if not keys(e,LEGEND_KEYS,'legend entry'): continue
        lid=e['id']
        if not _is_int(lid): err('legend id must be a unique int'); continue
        if lid in legend: err('legend id must be a unique int')
        legend[lid]=e
        b=e['box']
        if not (isinstance(b,list) and len(b)==4 and all(_is_int(v) for v in b) and 0<=b[0]<b[2]<=info['width'] and 0<=b[1]<b[3]<=info['height']): err('legend box invalid')
        if not isinstance(e['label_printed'],str) or not e['label_printed'].strip(): err('legend label must be a non-empty string')
        if not isinstance(e['same_symbol_as'],list) or not all(_is_int(i) for i in e['same_symbol_as']): err('same_symbol_as must be a list of ints')
    for e in legend.values():
        if isinstance(e['same_symbol_as'],list):
            for i in e['same_symbol_as']:
                if not _is_int(i) or i not in legend or i==e['id']: err('same_symbol_as refers to bad id')
    anchors=[]; anchors_ok=True
    if keys(label['depth_calibration'],('anchors',),'depth_calibration'):
        for a in seq(label['depth_calibration']['anchors'],'anchors'):
            if not keys(a,('y','depth_ft'),'anchor') or not _is_int(a['y']) or not _is_num(a['depth_ft']):
                err('anchor needs integer y and finite depth_ft'); anchors_ok=False
            elif not 0<=a['y']<=info['height']: err('anchor outside image'); anchors_ok=False
            else: anchors.append(a)
    if len(anchors)<2: err('need >= 2 anchors')
    pts=sorted(anchors,key=lambda a:a['y'])
    for a,b in zip(pts,pts[1:]):
        if a['y']>=b['y'] or a['depth_ft']>=b['depth_ft']: err('depth must strictly increase with y',b['y']); anchors_ok=False
    f=depth_fn(anchors) if anchors_ok else None
    res['fit']=linear_fit(anchors) if anchors_ok else None
    if res['fit'] and res['fit']['max_residual_ft']>3: warn('non-linear page: anchor residual > 3 ft')
    layers=seq(label['layers'],'layers'); spans=[]; previous=-1
    if not layers: warn('no layers yet')
    for layer in layers:
        if not keys(layer,LAYER_KEYS,'layer'): continue
        t,b=layer['top_px'],layer['bottom_px']
        if not (_is_int(t) and _is_int(b)): err('layer pixels must be ints'); continue
        if t<previous: err('layers must be sorted by top_px',t)
        previous=t
        if not y1<=t<b<=y2: err('top_px >= bottom_px or box outside lithology bbox',t)
        spans.append((t,b))
        flags=layer['flags']
        if not isinstance(flags,list) or flags not in ([],['uncertain']): err('flags must be a subset of uncertain',t)
        comps=seq(layer['composition'],'composition'); shares=[]; ranges=[]
        if len(comps)>1: err('one rock per layer is required',t)
        if not comps: err('layer has no rock',t); res['n_unreviewed']+=1
        for c in comps:
            if not keys(c,COMPOSITION_KEYS,'composition entry'): continue
            lid=c['legend_id']
            if not _is_int(lid) or lid not in legend: err('box without a rock / legend_id not in legend',t); res['n_unreviewed']+=1
            elif c['type']!=legend[lid]['label_printed']: err('type != legend label',t)
            p=c['proportion']
            if not _is_num(p) or not 0<p<=1: err('proportion outside (0, 1]',t)
            else: shares.append(p)
            xr=c['x_range']
            if not (isinstance(xr,list) and len(xr)==2 and all(_is_int(v) for v in xr) and x1<=xr[0]<xr[1]<=x2): err('box outside lithology bbox / invalid x_range',t)
            else: ranges.append(xr)
        if abs(sum(shares)-1)>.01000001: err('proportions must sum to 1.0',t)
        if shares!=sorted(shares,reverse=True): err('composition must be dominant first',t)
        left,right=inner_range(stem,(t+b)/2); covered=0; end=left
        for a,z in sorted(ranges):
            a,z=max(a,left),min(z,right)
            covered+=max(0,z-max(a,end)); end=max(end,z)
        if covered<.9*(right-left): warn('layer coverage < 90%',t)
        for edge,y in (('top',t),('bottom',b)):
            ft=f(y) if f else None
            for unit,expected in (('ft',round(ft,1) if ft is not None else None),('m',round(ft*.3048,1) if ft is not None else None)):
                val=layer[edge+'_'+unit]
                if expected is None:
                    if val is not None: err('stored ft/m != recomputed',t)
                elif not _is_num(val) or abs(val-expected)>1e-8: err('stored ft/m != recomputed',t)
    for i,(a,b) in enumerate(spans):
        for c,d in spans[i+1:]:
            if c>=b: break
            if a<d and c<b: warn('partial overlap between layers',c)
    return res


CSV_HEADER = ('top_y','bottom_y','cluster_id','lithology','top_depth','base_depth')
def to_csv_rows(label):
    cls,names=class_of(label['legend']),class_names(label['legend'])
    label=recompute_depths(sync_legend_labels(label))
    rows=[]
    for layer in label['layers']:
        c=layer['composition'][0]; lid=c['legend_id']
        rows.append(dict(top_y=layer['top_px'],bottom_y=layer['bottom_px'],cluster_id=cls[lid],lithology=names[lid],
                         top_depth=f"{layer['top_ft']:.1f}",base_depth=f"{layer['bottom_ft']:.1f}"))
    return rows
