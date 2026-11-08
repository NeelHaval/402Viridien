"""Profile IngestService._process on one scan (warm: second run). Usage: profile_ingest.py SCAN [--bbox JSON] [--out DIR]"""
from __future__ import annotations
import argparse, json, sys, time, tempfile, shutil, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(); p.add_argument("scan"); p.add_argument("--bbox"); p.add_argument("--out", default="perf")
a = p.parse_args()
tmp = Path(tempfile.mkdtemp()); os.environ["APP_DATA_DIR"] = str(tmp)
from app.config import Settings
from app.pipeline.model import load_model
from app.services.ingest import IngestService
from app.store.db import Database
from app.validation import reference
s = Settings(); s.ensure_dirs(); shutil.copytree(ROOT / "data/reference", s.reference_dir, dirs_exist_ok=True)
reference.set_reference_dir(s.reference_dir)
db = Database(s.db_path); db.migrate(); st = load_model()
svc = IngestService(s, db, lambda: st)
scan = Path(a.scan); bbox = json.loads(Path(a.bbox).read_text()) if a.bbox else None
def run(copy_name):
    shutil.copy(scan, tmp / copy_name)
    r = svc.submit(tmp / copy_name, scan.name, bbox); svc.shutdown(drain=True)
    svc.jobs.__init__(svc.process)
    return r.id
t = time.perf_counter(); wid = run("a" + scan.suffix); print("cold", time.perf_counter() - t)
from pyinstrument import Profiler
db.execute if False else None
from app.store import wells as repo
repo.requeue(db, wid, repo.get(db, wid)["scan_path"], scan.name)
pr = Profiler(interval=0.005); pr.start(); t = time.perf_counter(); svc.process(wid); el = time.perf_counter() - t; pr.stop()
print("warm", el)
Path(a.out).mkdir(exist_ok=True)
(Path(a.out) / f"profile_{scan.stem[:30].replace(' ','_')}.txt").write_text(pr.output_text(unicode=False, color=False, show_all=False))
(Path(a.out) / f"profile_{scan.stem[:30].replace(' ','_')}.html").write_text(pr.output_html())
print(pr.output_text(unicode=False, color=False)[:6000])
