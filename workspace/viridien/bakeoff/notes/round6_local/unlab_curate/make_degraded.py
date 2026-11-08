"""Apply the existing fixed medium scan preset, with unchanged geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dataset',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.root/'bakeoff'))
    from external.scanify import degrade
    source=json.loads((a.root/'external_tests_scanned/presets.json').read_text(encoding='utf-8'))
    preset=source['levels']['medium']
    out=a.dataset/'degraded';out.mkdir(exist_ok=True)
    for folder in ('images','bbox_annotation'):(out/folder).mkdir(exist_ok=True)
    items=json.loads((a.dataset/'eval_set.json').read_text(encoding='utf-8'))
    for i,item in enumerate(items):
        name=item['log'];seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)^source['seed']
        with Image.open(a.dataset/'images'/f'{name}.tif') as im:
            output=Image.new('RGB',im.size,'white')
            for y in range(0,im.height,1024):
                end=min(im.height,y+1024);top=max(0,y-8);bottom=min(im.height,end+8)
                rgb=np.asarray(im.crop((0,top,im.width,bottom)).convert('RGB'))
                scan=degrade(rgb,preset,seed+y)
                output.paste(Image.fromarray(scan[y-top:end-top]),(0,y))
            output.save(out/'images'/f'{name}.tif',compression='tiff_deflate')
        shutil.copyfile(a.dataset/'bbox_annotation'/f'{name}.json',out/'bbox_annotation'/f'{name}.json')
        if (i+1)%10==0:print('Degraded',i+1,flush=True)
    for name in ('eval_set.json','manifest.csv'):shutil.copyfile(a.dataset/name,out/name)
    (out/'scan_contract.json').write_text(json.dumps(dict(seed=source['seed'],preset=preset,
        implementation='bakeoff/external/scanify.py:degrade',geometry='unchanged; no shear or resampling',
        note='Fixed medium photometric stress, not an estimated physical scan. No TEST scores used.'),indent=2),encoding='utf-8')
    (out/'README.md').write_text('Unlabelled TEST scan stress copy. Never train or tune on it. '
        'The medium scanify photometric preset is fixed. Pixel geometry and windows are unchanged. '
        'No geometric shear or resolution scaling is applied. See scan_contract.json.\n',encoding='utf-8')
    print('Built degraded set',len(items),flush=True)


if __name__=='__main__':main()
