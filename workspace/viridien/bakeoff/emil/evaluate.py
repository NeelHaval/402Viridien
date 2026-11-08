"""Run notebook evaluators with only their CONFIG paths changed, plus shared scoring."""
import argparse
from collections import Counter
import contextlib
import io as sio
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import tracemalloc
import hashlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from emil.bootstrap import installed_arrow
installed_arrow()
import numpy as np
import pandas as pd
import torch
from harness import io, legend as harness_legend, score
from harness.run import Resources, aggregate
from emil import pipeline as p
from emil.split_results import membership, summarize_external
from emil.paths import mapped_path
from strategies import s12_emil as s12, s11_fft_legend as s11


def serial(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=serial), encoding='utf-8')


def run_evaluator(kind, gt, pred, annotations, out):
    src_path = HERE / 'evaluators' / f'{kind}.py'
    source = src_path.read_text(encoding='utf-8')
    configs = dict(GT_PATH=str(gt), PRED_PATH=str(pred))
    if kind == 'legend':
        configs['ANNOTATIONS_DIR'] = str(annotations)
        configs['IMAGE_COORDS'] = True
    else: configs['SAVE_JSON_TO'] = str(out / f'{kind}.json')
    for key, value in configs.items():
        source, count = re.subn(r'^'+key+r'\s*=.*$', key+' = '+repr(value).replace('\\','\\\\'), source, count=1, flags=re.M)
        assert count == 1
    # Assert that no code outside the specified path assignments was changed.
    def stripped(s):
        return '\n'.join(line for line in s.splitlines() if not any(re.match('^'+k+r'\s*=', line) for k in configs))
    assert stripped(source) == stripped(src_path.read_text(encoding='utf-8'))
    namespace = dict(__name__='__main__', __file__=str(src_path))
    capture = sio.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        exec(compile(source, str(src_path), 'exec'), namespace)
    (out / f'{kind}.txt').write_text(capture.getvalue(), encoding='utf-8')
    result = namespace['legend_results'] if kind == 'legend' else namespace['_to_save']
    dump(out / f'{kind}.json', result)
    return result


def score_prediction_run(run_dir, gt, annotations):
    """Adapter missing in live run_experiment; evaluator bodies stay untouched."""
    out = run_dir.parent / (run_dir.name + '_eval')
    out.mkdir(parents=True, exist_ok=True)
    results = {kind: run_evaluator(kind, gt, run_dir, annotations, out)
               for kind in ['segmentation', 'classification', 'legend']}
    classification = results['classification']
    per_file = classification.get('per_file', classification.get('files', {}))
    if isinstance(per_file, dict):
        return pd.DataFrame([dict(stem=Path(k).stem, pixel_accuracy=v.get('pixel_accuracy'))
                             for k, v in per_file.items()])
    return pd.DataFrame([dict(stem=p.stem) for p in run_dir.glob('*.json')])


def truth_rows(labels):
    result = []
    for layer in labels['layers']:
        best = max(layer['composition'], key=lambda c:c['proportion'])
        row = dict(top_y=int(layer['top_px']),bottom_y=int(layer['bottom_px']),lithology=best['type'],cluster_id=best['legend_id'])
        for edge in ['top','bottom']:
            if layer.get(edge+'_ft') is not None: row[edge+'_depth']=layer[edge+'_ft']
        result.append(row)
    return result


def legend_frame(labels):
    return pd.DataFrame([dict(legend_id=e['id'], lithology=e['label_printed'], box=e['box']) for e in labels['legend']], columns=['legend_id','lithology','box'])


def export_legend(path, found, box):
    dump(path, dict(swatches=[dict(rock_name=r.lithology,bbox=[int(r.box[0]-box[0]),int(r.box[1]-box[1]),int(r.box[2]-box[0]),int(r.box[3]-box[1])]) for r in found.itertuples()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',choices=['viridien','external','incomplete'],required=True)
    parser.add_argument('--modes', nargs='+', default=['detect', 'labels', 's11'])
    parser.add_argument('--tag', default='live')
    parser.add_argument('--model', type=Path, default=p.DEFAULT_MODEL)
    parser.add_argument('--one-window', action='store_true')
    parser.add_argument('--refresh', action='store_true', help='Recompute selected modes, preserving other modes')
    parser.add_argument('--keys', nargs='+', help='Refresh only these window keys')
    parser.add_argument('--recompute-s11', action='store_true', help='Run s11 instead of loading earlier CSVs')
    args = parser.parse_args()
    domain = args.domain
    torch.set_num_threads(6)
    p.configure(model=args.model)
    checkpoint_sha256 = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if domain == 'external':
        io.DATASET = p.WORK / 'external_tests'
        labels_dir = HERE / 'external_labels'
        items = json.loads((io.DATASET/'eval_set.json').read_text())
        if args.one_window:
            items = list({w['log']:w for w in reversed(items)}.values())[::-1]
    else:
        labels_dir = p.WORK/('emil/drive/test' if domain=='incomplete' else 'emil/labels')
        items = []
        for path in sorted(labels_dir.glob('*.json')):
            labels = json.loads(path.read_text())
            if not labels.get('layers'): continue
            if domain=='incomplete' and len(labels['layers'])!=2: continue
            with (io.DATASET/'bbox_annotation'/path.name).open() as f: ann=json.load(f)
            box = ann['bounding_boxes']['interpreted_lithology_column']
            items.append(dict(log=path.stem,y0=box[1],y1=box[3]))
    counts = Counter(w['log'] for w in items)
    root = HERE/('results_'+args.tag)/domain
    records_path = root/'records.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    if domain == 'external':
        allowed = {(w['log'], tuple((w['y0'], w['y1']))) for w in items}
        records = [r for r in records if (r['log'], tuple(r['window'])) in allowed]
    baseline_errors = set()
    if domain=='external':
        baseline_errors={r['log'] for r in json.loads((io.ROOT/'out/s11_fft_legend/external/metrics.json').read_text())['logs'] if 'error' in r}
    def reusable(r):
        if 'error' in r or r.get('checkpoint_sha256') != checkpoint_sha256:
            return False
        out = root/r['mode']
        paths = [out/'predictions'/(r['key']+'.json'), out/'csv'/(r['key']+'.csv'), out/'legend_full'/(r['key']+'.json'), out/'section2'/(r['key']+'.json')]
        return all(f.exists() for f in paths) and not p.record_errors(json.loads(paths[0].read_text()))
    completed = set() if args.refresh else {(r['key'],r['mode']) for r in records if reusable(r)}
    splits = membership() if domain == 'external' else {}
    for r in records:
        if 'prediction_source' in r and ':' in r['prediction_source']:
            r['prediction_source'] = str(mapped_path(r['prediction_source']))
        if (r['key'], r['mode']) in completed:
            r.setdefault('execution_environment', 'Windows laptop (reused)')
    for item in items:
        name = item['log']; win = (item['y0'],item['y1'])
        key = name + (f'__y{win[0]}_{win[1]}' if counts[name]>1 else '')
        if args.keys and key not in args.keys: continue
        labels = json.loads((labels_dir/(name+'.json')).read_text())
        clipped = dict(labels, layers=[dict(l,top_px=max(win[0],l['top_px']),bottom_px=min(win[1],l['bottom_px'])) for l in labels['layers'] if min(win[1],l['bottom_px'])>max(win[0],l['top_px'])])
        dump(root/'truth'/(key+'.json'), clipped)
        dump(root/'annotations'/(key+'.json'),json.loads((io.DATASET/'bbox_annotation'/(name+'.json')).read_text()))
        truth = truth_rows(clipped)
        io.save_rows(root/'truth_csv'/(key+'.csv'),truth)
        log = io.open_log(name)
        try:
            for mode in args.modes:
                if (key,mode) in completed: continue
                out = root/mode
                record = dict(log=name,key=key,window=win,mode=mode,scored=True,
                              checkpoint_sha256=checkpoint_sha256, incomplete=domain=='incomplete', execution_environment=sys.platform, device=p.DEVICE)
                try:
                    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                    tracemalloc.start()
                    with Resources() as res:
                        start = time.perf_counter()
                        if mode == 'detect':
                            found = s12.detect_legend(log,HERE/'cache')
                        elif mode == 'labels':
                            found = legend_frame(labels)
                        else:
                            prior_legend = HERE/'results'/domain/'s11/legend_full'/(key+'.json')
                            if prior_legend.exists() and not args.recompute_s11:
                                from types import SimpleNamespace
                                entries = [SimpleNamespace(id=e['legend_id'], label=e['lithology'], swatch_box=e['box']) for e in json.loads(prior_legend.read_text())]
                            else:
                                entries = harness_legend.parse(log)
                            found = pd.DataFrame([dict(legend_id=e.id,lithology=e.label,box=e.swatch_box) for e in entries],columns=['legend_id','lithology','box'])
                        if mode == 's11':
                            prior = HERE/'results'/('viridien' if domain=='viridien' else 'external')/'s11/csv'/(key+'.csv')
                            if prior.exists() and not args.recompute_s11:
                                rows=io.load_rows(prior) or []
                                record['prediction_source']=str(prior)
                            elif domain == 'external' and name not in baseline_errors and not args.recompute_s11:
                                old = io.ROOT/'out/s11_fft_legend/external'/(key+'.csv')
                                rows = io.load_rows(old) or []
                                record['prediction_source']=str(old)
                            else:
                                rows,_ = s11.run(log,win,dict(legend=entries,calib=None,device=p.DEVICE,cache_dir=HERE/'cache'))
                                record['prediction_source']='s11.run (depth preprocessing bypass)'
                            seg = [dict(top_px=r['top_y'],bottom_px=r['bottom_y']) for r in rows]
                        else:
                            layers,segments,beds,named = s12.predict(log,win,found)
                            rows=[dict(top_y=int(r.y_top),bottom_y=int(r.y_bottom),lithology=r.lithology,cluster_id=int(r.legend_id)) for r in layers.itertuples() if r.legend_id>=0]
                            seg=[dict(top_px=int(r.y_top),bottom_px=int(r.y_bottom)) for r in segments.itertuples()]
                            record.update(n_beds=len(beds),n_segments=len(segments),n_named=len(named),n_layers=len(rows))
                            if domain=='viridien' and mode=='labels':
                                record['notebook_row_accuracy']=p.lithology_accuracy(named,labels_dir/(name+'.json'),win[0],win[1]-win[0])
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        record['seconds']=time.perf_counter()-start
                    record.update(peak_rss_gb=res.peak/1024**3,peak_gpu_gb=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else 0.0,n_detected=len(found))
                    record['peak_tracemalloc_mb']=tracemalloc.get_traced_memory()[1]/2**20
                    tracemalloc.stop()
                    record.update(score.score(rows,truth,win))
                    io.save_rows(out/'csv'/(key+'.csv'),rows)
                    rec=dict(schema_version=p.SCHEMA_VERSION, stem=key, image_name=name+'.tif', written_by={},
                             legend=p.legend_from_table(found),
                             layers=[dict(top_px=r['top_y'],bottom_px=r['bottom_y'],composition=[dict(legend_id=int(r['cluster_id']),type=r['lithology'],proportion=1.0)]) for r in rows])
                    if mode=='s11':
                        # S11 cluster IDs describe patterns and include Unknown; they
                        # are not physical legend IDs. The evaluator explicitly
                        # accepts lithology on a layer, so retain that distinction.
                        rec['layers']=[dict(top_px=r['top_y'],bottom_px=r['bottom_y'],
                                            lithology=r['lithology'],cluster_id=int(r['cluster_id'])) for r in rows]
                    p.RUN_DIR=out/'predictions'
                    p.save_record(rec,'local_evaluation',legend_source=mode,segments=p.SAVE_METHOD,checkpoint_sha256=checkpoint_sha256,window=list(win))
                    dump(out/'section2'/(key+'.json'),dict(layers=seg))
                    dump(out/'legend_full'/(key+'.json'),found.to_dict('records'))
                except Exception as exc:
                    if tracemalloc.is_tracing(): tracemalloc.stop()
                    record.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                    record.update(score.score([],truth,win))
                    print(record['traceback'],flush=True)
                    dump(out/'predictions'/(key+'.json'),dict(schema_version=1,stem=key,image_name=name+'.tif',written_by={},legend=[],layers=[]))
                records=[r for r in records if (r['key'],r['mode'])!=(key,mode)]+[record]
                dump(records_path,records)
                print(json.dumps(record,default=serial),flush=True)
            if domain=='viridien':
                # Naming-only reproduction uses exactly the supplied layer boundaries.
                oracle=legend_frame(labels)
                col=io.read_rgb(log,'interpreted_lithology_column')
                gt=pd.DataFrame([(l['top_px'],l['bottom_px']) for l in labels['layers']],columns=['y_top','y_bottom'])
                named,_=p.lithology_layers(gt,col,win[0],oracle,log.image)
                dump(root/'reproduction'/(name+'.json'),dict(true_segments=p.lithology_accuracy(named,labels_dir/(name+'.json'),win[0],len(col))))
        finally:
            if hasattr(log.image,'_emil_log'): log.image._emil_log.close()
            log.close()
    if domain == 'external':
        summarize_external(root, records, args.modes)
        dump(records_path, records)
        print('External dev, test, dev+test metrics saved; removed wells excluded.', flush=True)
        return
    for mode in args.modes:
        out=root/mode
        result={kind:run_evaluator(kind,root/'truth',out/'predictions',root/'annotations',out) for kind in ['segmentation','classification','legend']}
        dump(out/'metrics.json',dict(harness=aggregate([r for r in records if r['mode']==mode]),emil=result))


if __name__=='__main__': main()
