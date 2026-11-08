"""Replay available predictions on CPU; retain recorded shared preprocessing outcomes."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch
from emil import pipeline as p
from emil.evaluate import dump, legend_frame
from emil.split_results import selected
from harness import io, score
from harness.run import Resources, aggregate
from strategies import s12_emil as s12


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(6)
    p.configure(model=p.DEFAULT_MODEL)
    sha = hashlib.sha256(p.DEFAULT_MODEL.read_bytes()).hexdigest()
    resources = []
    for domain in ['viridien', 'external']:
        root = HERE/'results_live'/domain
        rr = read(root/'records.json')
        if domain == 'external': rr = selected(rr, 'dev_test')
        io.DATASET = p.WORK/'external_tests' if domain == 'external' else p.DATASET
        for r in rr:
            if r['mode'] == 's11': continue
            log = io.open_log(r['log'])
            try:
                with Resources() as res:
                    start = time.perf_counter()
                    found = (s12.detect_legend(log, HERE/'cache') if r['mode'] == 'detect' else
                             legend_frame(read(root/'truth'/(r['key']+'.json'))))
                    layers, segments, beds, named = s12.predict(log, tuple(r['window']), found)
                    seconds = time.perf_counter()-start
                rows = [dict(top_y=int(x.y_top), bottom_y=int(x.y_bottom), lithology=x.lithology,
                             cluster_id=int(x.legend_id)) for x in layers.itertuples() if x.legend_id >= 0]
                prior = io.load_rows(root/r['mode']/'csv'/(r['key']+'.csv'))
                keys = ['top_y', 'bottom_y', 'lithology', 'cluster_id']
                equal = [[x[k] for k in keys] for x in rows] == [[x[k] for k in keys] for x in prior]
                resources.append(dict(domain=domain, log=r['log'], key=r['key'], mode=r['mode'],
                    window=r['window'], seconds=seconds, peak_rss_gb=res.peak/1024**3,
                    peak_gpu_gb=0.0, device='cpu', traced=False, ocr_cache='warm',
                    checkpoint_sha256=sha, matches_saved_predictions=equal))
                dump(HERE/'vm_resources.json', resources)
            finally:
                if hasattr(log.image, '_emil_log'): log.image._emil_log.close()
                log.close()
    direct = selected(read(HERE/'results_live/external/records.json'), 'dev_test')
    baseline = read(io.ROOT/'out/s11_fft_legend/external/metrics.json')
    by_window = {(r['log'], tuple(r['window'])): r for r in baseline['logs']}
    resources_by_key = {(r['key'],r['mode']):r for r in resources if r['domain']=='external'}
    records = []
    out = io.ROOT/'out/oracle_emil/external'
    out.mkdir(parents=True, exist_ok=True)
    for r in direct:
        if r['mode'] != 'labels': continue
        old = by_window[(r['log'], tuple(r['window']))]
        x = dict(r, preprocessing_source='Original s11 shared-harness preprocessing outcome',
                 inference_source='CPU-replayed direct s12 labels; identical predict() in oracle_emil.run')
        measured = resources_by_key[(r['key'],r['mode'])]
        x.update({k:measured[k] for k in ['seconds','peak_rss_gb','peak_gpu_gb']})
        rows = io.load_rows(HERE/'results_live/external/labels/csv'/(r['key']+'.csv'))
        if 'error' in old:
            truth = io.load_rows(HERE/'results_live/external/truth_csv'/(r['key']+'.csv'))
            x.update(score.score([], truth, tuple(r['window'])), error=old['error'], score=0.0,
                     seconds=None,peak_rss_gb=None,peak_gpu_gb=None)
            rows=[]
        io.save_rows(out/(r['key']+'.csv'),rows)
        records.append(x)
    dump(out/'metrics.json', dict(strategy='oracle_emil', logs=records,
        splits={split: aggregate(selected(records, split)) for split in ['dev', 'test', 'dev_test']},
        provenance='CPU-replayed oracle predictions with recorded original shared preprocessing outcomes; depth OCR not rerun on VM'))
    for strategy in ['s11_fft_legend', 's12_emil']:
        path = io.ROOT/'out'/strategy/'external/metrics.json'
        data = read(path)
        data['logs']=selected(data['logs'],'dev_test')
        data.pop('mean', None)
        data['splits'] = {split: aggregate(selected(data['logs'], split)) for split in ['dev', 'test', 'dev_test']}
        dump(path, data)
    print(json.dumps(dict(vm_inference_calls=len(resources), exact_prediction_matches=sum(r['matches_saved_predictions'] for r in resources))))


if __name__ == '__main__': main()
