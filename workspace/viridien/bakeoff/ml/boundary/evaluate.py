"""Run the shared harness, with all generated files confined to this lane.

Examples, from bakeoff:
  python -m ml.boundary.evaluate --split dev --strategies s11,s12
  python -m ml.boundary.evaluate --split final --strategies s11,s12,s13

Final evaluation is guarded by a checkpoint hash and immutable run marker.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import torch

from harness import io, run as harness_run
from ml.boundary.model import HERE

BAKEOFF=HERE.parents[1]
REPO=BAKEOFF.parent
EXTERNAL=REPO/'external_tests'
WORKSPACE=HERE/'harness_workspace'


def seed_caches(names):
    for category in ('legend','depth'):
        source=BAKEOFF/'out'/'_cache'/category
        destination=WORKSPACE/'out'/'_cache'/category
        destination.mkdir(parents=True,exist_ok=True)
        for name in names:
            for path in source.glob(name+'.*'):
                target=destination/path.name
                if path.is_file() and not target.exists():
                    shutil.copy2(path,target)


def setup():
    # S12 contains a six-thread call: cap it without editing that lane's strategy.
    set_threads=torch.set_num_threads
    torch.set_num_threads=lambda n:set_threads(min(3,n))
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    io.ROOT=WORKSPACE
    original_strategy=harness_run._strategy
    def strategy(name):
        matches=sorted((BAKEOFF/'strategies').glob(name+'*.py'))
        if len(matches)!=1:
            raise ValueError(f'Strategy {name!r}: {len(matches)} files')
        return importlib.import_module('strategies.'+matches[0].stem)
    harness_run._strategy=strategy
    original_truth=harness_run._truth
    def truth(name,source,window):
        owned=io.ROOT
        try:
            io.ROOT=BAKEOFF
            return original_truth(name,source,window)
        finally:
            io.ROOT=owned
    harness_run._truth=truth
    from emil import pipeline as emil
    emil.MODEL_PATH=BAKEOFF/'emil'/'boundary_net.pt'
    emil.DEVICE='cpu'
    # Reuse only OCR caches, identified by image digest. No truth/predictions.
    destination=WORKSPACE/'out'/'_cache'/'s12_emil'
    destination.mkdir(parents=True,exist_ok=True)
    for parent in (BAKEOFF/'emil'/'cache',BAKEOFF/'out'/'_cache'/'s12_emil'):
        for path in parent.glob('legend_*.json'):
            target=destination/path.name
            if not target.exists():
                shutil.copy2(path,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','final'],required=True)
    parser.add_argument('--strategies',default='s11,s12,s13')
    args=parser.parse_args()
    if args.split=='final':
        marker=HERE/'FINAL_STARTED.json'
        if marker.exists():
            raise RuntimeError('Final evaluation already started; inspect artifacts, do not rerun for selection.')
        weights=HERE/'boundary_net.pt'
        payload=dict(weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                     config=json.loads((HERE/'selected_config.json').read_text()),strategies=args.strategies.split(','))
        payload['source_sha256']={str(p.relative_to(BAKEOFF)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (HERE/'model.py',BAKEOFF/'strategies'/'s13_boundary_net.py',
                                            BAKEOFF/'harness'/'legend.py',BAKEOFF/'harness'/'score.py',
                                            BAKEOFF/'harness'/'depth.py')}
        marker.write_text(json.dumps(payload,indent=2))
    setup()
    original_dataset=io.DATASET
    for strategy in args.strategies.split(','):
        io.DATASET=EXTERNAL
        split='dev' if args.split=='dev' else 'test'
        seed_caches([c['log'] for c in json.loads((EXTERNAL/f'eval_set_{split}.json').read_text())])
        harness_run.main(['--strategy',strategy,'--items',str(EXTERNAL/f'eval_set_{split}.json'),
                          '--gt-dir',str(EXTERNAL/'truth'),'--out-tag',f'ext_{split}'])
        if args.split=='final':
            io.DATASET=original_dataset
            seed_caches([c['log'] for c in json.loads((BAKEOFF/'eval_set.json').read_text())])
            harness_run.main(['--strategy',strategy,'--items',str(BAKEOFF/'eval_set.json'),
                              '--gt','consensus','--out-tag','viridien_eval'])
    if args.split=='final':
        (HERE/'FINAL_COMPLETE.json').write_text(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
