"""Patch classifiers, reusing the existing strategies without modifying them."""
import gc
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from strategies import s01_colour_runs as colour
from strategies import s05_dinov2 as dino
from strategies import s07_synth_cnn as cnn
from strategies import s11_fft_legend as fft

from classify.data import OUT, dump

LOCAL = ['colour', 'fft_hatch', 'dinov2', 'siglip', 'cnn_synth']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)
cv2.setNumThreads(2)


def softmax(values, temperature=1):
    x = np.asarray(values, dtype=np.float64)/temperature
    x -= np.max(x, axis=-1, keepdims=True)
    p = np.exp(x)
    return p/p.sum(axis=-1, keepdims=True)


def aggregate(case, scores):
    # Distinct swatches can share an OCR label; the label score is the best swatch.
    return np.stack([np.max(scores[:, [i for i,e in enumerate(case['entries']) if e.label == v]], axis=1)
                     for v in case['vocab']], axis=1)


def tiles(rgb, size=32):
    """Sample the full column, so a wide patch is not reduced to one corner."""
    h, w = rgb.shape[:2]
    if min(h,w) < size:
        rgb = np.tile(rgb, (max(1,(size+h-1)//h),max(1,(size+w-1)//w),1))
    h,w = rgb.shape[:2]
    return [rgb[y:y+size,x:x+size] for y in np.linspace(0,h-size,min(4,max(1,h//size)),dtype=int)
            for x in np.linspace(0,w-size,min(5,max(1,w//size)),dtype=int)]


def classify(case, name):
    entries = case['entries']
    if not entries:
        raise ValueError('No parsed rock swatches; local classification unavailable')
    patches = [p['rgb'] for p in case['patches']]
    mono = case['info']['mode'] == '1'
    if name == 'colour':
        def lab(rgb):
            rgb = colour._swatch_rgb(SimpleNamespace(swatch=rgb,mean_rgb=rgb.mean(axis=(0,1))))
            return cv2.cvtColor(rgb.astype(np.uint8).reshape(1,1,3),cv2.COLOR_RGB2LAB)[0,0].astype(float)
        refs = np.stack([lab(e.swatch) for e in entries])
        scores = -np.stack([np.linalg.norm(refs-lab(p),axis=1) for p in patches])
        return softmax(aggregate(case,scores),20), {}
    if name == 'fft_hatch':
        refs = [fft.descriptor(e.swatch,mono) for e in entries]
        scores = []
        for p in patches:
            desc = [fft.descriptor(t,mono) for t in tiles(p,64)]
            scores.append(-np.median([[fft.distance(d,r,mono) for r in refs] for d in desc],axis=0))
        return softmax(aggregate(case,np.asarray(scores)),.10), {}
    if name == 'dinov2':
        # Tile only swatches; resize the full query to preserve its full-width evidence.
        images = [cv2.resize(p,(224,224),interpolation=cv2.INTER_AREA) for p in patches]
        em = dino._embed(images+[dino._tile(e.swatch) for e in entries], DEVICE)
        return softmax(aggregate(case,em[:len(patches)] @ em[len(patches):].T),.07), {}
    if name == 'siglip':
        from strategies import s06_siglip as sig
        model, processor, tokenizer = sig._load_model()
        model.to(DEVICE)
        images = [Image.fromarray(p).resize((224,224),Image.Resampling.BICUBIC) for p in patches]
        em = sig._image_tensor(images+[sig._tile(e.swatch) for e in entries],processor,DEVICE)
        inp = tokenizer([f'a geological lithology symbol for {e.label}' for e in entries],
                        padding='max_length',truncation=True,return_tensors='pt')
        with torch.inference_mode():
            tx = model.get_text_features(**{k:v.to(DEVICE) for k,v in inp.items()})
        tx = torch.nn.functional.normalize(tx.float(),dim=-1).cpu().numpy()
        scores = .5*(em[:len(patches)] @ em[len(patches):].T)+.5*(em[:len(patches)] @ tx.T)
        return softmax(aggregate(case,scores),.07), {}
    if name == 'cnn_synth':
        path = OUT/'models'/(case['info']['key']+'.pt')
        path.parent.mkdir(exist_ok=True,parents=True)
        started = time.perf_counter()
        if path.exists():
            net = cnn.Net(len(entries)+1).to(DEVICE)
            net.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True)); net.eval()
            trained = False
        else:
            # S7 caps training itself at 20 seconds (well below the requested five minutes).
            net = cnn.train(entries,DEVICE,mono)
            torch.save(net.state_dict(),path)
            trained = True
        training = time.perf_counter()-started
        predictions = []
        for p in patches:
            t = np.asarray(tiles(p,32)).transpose(0,3,1,2).copy()
            x = torch.from_numpy(t).to(DEVICE,dtype=torch.float32)/255
            with torch.inference_mode():
                # Average tile logits, drop the synthetic blank class, rank legend rocks.
                predictions.append(net(x).mean(0)[:len(entries)].cpu().numpy())
        result = softmax(aggregate(case,np.asarray(predictions)))
        del net
        return result, dict(training_seconds=training,trained=trained)
    raise ValueError(name)


def release():
    dino._MODEL = None
    try:
        from strategies import s06_siglip as sig
        sig._load_model.cache_clear()
    except ImportError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
