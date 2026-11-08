"""Reproducible shared calibration audit: python -m harness._check."""
import json
import time
from . import io, depth

def main():
    lines=['| Log | a (ft/px) | b (ft) | n_points | rmse_px | seconds | engine |',
           '|---|---:|---:|---:|---:|---:|---|']
    for item in json.loads((io.ROOT/'eval_set.json').read_text()):
        log=io.open_log(item['log']);start=time.perf_counter()
        try:
            c=depth.calibrate(log)
            cached=json.loads((io.ROOT/'out'/'_cache'/'depth'/(log.name+'.rapidocr.json')).read_text())
            line=f'| {log.name} | {c.a:.9f} | {c.b:.4f} | {c.n_points} | {c.rmse_px:.3f} | {cached["seconds"]:.2f} | {c.method} |'
        except Exception as exc:
            line=f'| {log.name} | ERROR | {exc} | 0 | | {time.perf_counter()-start:.2f} | rapidocr |'
        finally: log.close()
        lines.append(line);print(line,flush=True)
        (io.ROOT/'notes'/'depth_check.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log=io.open_log(json.loads((io.ROOT/'eval_set.json').read_text())[0]['log'])
    c=depth.calibrate(log);log.close()
    easy=io.ROOT/'out'/'_cache'/'depth'/(log.name+'.easyocr.json')
    if easy.exists():
        data=json.loads(easy.read_text());e=data['calibration']
        lines.append(f'| {log.name} (GPU alternative) | {e["a"]:.9f} | {e["b"]:.4f} | {e["n_points"]} | {e["rmse_px"]:.3f} | {data["seconds"]:.2f} | {e["method"]} |')
    lines.extend(['',f'log06: row 4409 -> {c.to_depth(4409):.2f} ft; row 10215 -> {c.to_depth(10215):.2f} ft; row 10415 -> {c.to_depth(10415):.2f} ft.',
                  'Gold CSV anchors: 4409 = 25400 ft, 10215 = 25690 ft, 10415 = 25700 ft. The requested 10415 = 25690 anchor disagrees with the supplied CSV.'])
    (io.ROOT/'notes'/'depth_check.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
