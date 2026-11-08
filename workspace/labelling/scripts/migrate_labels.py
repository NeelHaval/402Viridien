#!/usr/bin/env python3
"""Migrate interval labels to free layers. Originals are retained under labels/.backup/<run>/."""
import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import labelslib as L


def migrate(d, stem):
    if 'depth_calibration' in d:
        return L.recompute_depths(L.sync_legend_labels(d)), ['legend labels preserved; canonical generalisation removed']
    anchors=[{'y':int(t['y']),'depth_ft':float(t['depth'])} for t in d.get('depth_ticks',[])]
    x1,_,x2,_=L.bbox(stem)
    boxes=[{'top_px':int(v['top_y']),'bottom_px':int(v['bottom_y']),
            'legend_id':v.get('legend_id'),'x_range':[x1,x2],
            'flags':['uncertain'] if 'uncertain' in v.get('flags',[]) else []}
           for v in d.get('intervals',[]) if v.get('canonical') not in ('blank','unknown')]
    out={'image_name':d['image_name'],'split':L.label_stems().get(stem,d['split']),
         'depth_calibration':{'anchors':anchors},'legend':copy.deepcopy(d.get('legend',[])),
         'layers':L.group_boxes(boxes,d.get('legend',[]),anchors)}
    out = L.sync_legend_labels(out)
    return out,['intervals -> layers; blank/unknown dropped; mixed/thin_bed dropped']


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args(argv)
    backup=L.LABELS_DIR/'.backup'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    n=0
    for p in sorted(L.LABELS_DIR.glob('*.json')):
        old=p.read_text(); d=json.loads(old)
        new,_=migrate(d,p.stem)
        text=L.dumps(new)
        if text==old: continue
        if not args.dry_run:
            backup.mkdir(parents=True,exist_ok=True); shutil.copy2(p,backup/p.name)
            tmp=p.with_suffix('.json.tmp'); tmp.write_text(text); tmp.replace(p)
        n+=1
        print(('would migrate ' if args.dry_run else 'migrated ')+p.name)
    print(f'{n} changed; backups: {backup}' if n else 'No changes')


if __name__=='__main__': main()
