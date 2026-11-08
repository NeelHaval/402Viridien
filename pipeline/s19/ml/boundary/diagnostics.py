"""Dev-only checks separating boundary proposals from legend naming errors."""
import json

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
import torch

from emil import pipeline as emil
from harness import io, score
from ml.boundary.experiment import BAKEOFF, DATA, EXTERNAL, RUNS, legend_entries, proposal_f1, setup
from ml.boundary.model import features, probabilities
from strategies.s02_features_pelt import segment
from strategies.s13_boundary_net import name_intervals


def main():
    setup()
    io.DATASET=EXTERNAL
    cases=json.loads((DATA/'index.json').read_text())['cases']
    net=emil.BoundaryNet()
    net.load_state_dict(torch.load(BAKEOFF/'emil'/'boundary_net.pt',map_location='cpu',weights_only=True))
    net.eval()
    results=[]
    emil_records=[]
    s11_records=[]
    for case in cases:
        rgb=np.asarray(Image.open(DATA/(case['key']+'.png')).convert('RGB'))
        entries=legend_entries(case)
        y0,y1=case['window']
        pp=np.array(sorted(score.boundaries(case['truth'],(y0,y1))))-y0
        rows=name_intervals(rgb,y0,np.r_[0,pp,len(rgb)],entries)
        oracle=score.score(rows,case['truth'],(y0,y1))
        oracle.update(key=case['key'],well=case['well'])
        results.append(oracle)
        prob=probabilities(net,features(rgb,'six'))
        pp=find_peaks(prob,height=.5,distance=6)[0]
        emil_records.append(dict(key=case['key'],well=case['well'],boundary_f1=proposal_f1(pp,case)))
        log=io.open_log(case['log'])
        try:
            intervals=segment(log,(y0,y1),{})
        finally:
            log.close()
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        inset=min(max(1,rgb.shape[1]//12),max(1,rgb.shape[1]//3))
        dark=(gray[:,inset:-inset]<65).mean(1) if rgb.shape[1]>2*inset else (gray<65).mean(1)
        candidates=np.flatnonzero((dark>.88)&(np.r_[0,dark[:-1]]<=.88))
        candidates=np.array([p for i,p in enumerate(candidates)
                             if (i==0 or p-candidates[i-1]>35)
                             and (i+1==len(candidates) or candidates[i+1]-p>35)
                             and np.mean(dark[max(0,p-8):min(len(dark),p+9)])<.65],dtype=int)
        points=sorted(set([y0,y1]+[a for a,b in intervals]+[int(y0+p) for p in candidates if 0<p<len(rgb)-1]))
        merged=[points[0]]
        for p in points[1:-1]:
            if p-merged[-1]>=20 and y1-p>=20:
                merged.append(p)
        proposed=np.array(merged[1:])-y0
        s11_records.append(dict(key=case['key'],well=case['well'],boundary_f1=proposal_f1(proposed,case)))
    (RUNS/'dev_oracle_naming.json').write_text(json.dumps(results,indent=2))
    for path in (RUNS/'dev_baselines.json',RUNS/'selection.json'):
        data=json.loads(path.read_text())
        baselines=data['baselines'] if path.name=='selection.json' else data
        baselines['emil_proposals']=emil_records
        baselines['s11_full_proposals']=s11_records
        path.write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(oracle_naming_row_acc=float(np.mean([r['row_acc'] for r in results])),
                          emil_boundary_f1=float(np.mean([r['boundary_f1'] for r in emil_records])),
                          s11_proposal_f1=float(np.mean([r['boundary_f1'] for r in s11_records]))),indent=2))


if __name__=='__main__':
    main()
