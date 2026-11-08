"""Generated demo-service checks; no OCR engine, private scans or network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
import importlib
import json
from pathlib import Path
import random
import threading
import time

import pytest
from PIL import Image

BACKEND=Path(__file__).resolve().parents[2]/"demo"/"backend"


@pytest.fixture
def modules(monkeypatch):
    pytest.importorskip("pandas",reason="Demo backend dependencies are required; run with the dedicated backend environment")
    monkeypatch.syspath_prepend(str(BACKEND))
    ingest=importlib.import_module("app.services.ingest")
    repo=importlib.import_module("app.store.wells")
    database=importlib.import_module("app.store.db")
    settings=importlib.import_module("app.config")
    assert Path(ingest.__file__).resolve().is_relative_to(BACKEND.resolve())
    return ingest,repo,database,settings


@pytest.fixture
def service(tmp_path,monkeypatch,modules):
    ingest,repo,database,settings=modules
    class RecordingJobs:
        def __init__(self,handler,**kwargs):self.ids=[]
        def admit(self):return self
        def __enter__(self):return self
        def __exit__(self,*exc):pass
        def enqueue(self,well_id):self.ids.append(well_id)
        def shutdown(self,drain=False):pass
    monkeypatch.setattr(ingest,"JobQueue",RecordingJobs)
    monkeypatch.setattr(ingest,"configure_tools",lambda settings:None)
    config=settings.Settings(data_dir=tmp_path/"runtime");config.ensure_dirs()
    db=database.Database(config.db_path);db.migrate()
    state=ingest.ModelState(True,None,"not-executed")
    svc=ingest.IngestService(config,db,lambda:state)
    yield svc,repo,ingest
    svc.shutdown()


def _scan(path,tweak=0):
    Image.new("RGB",(16,32),(31+tweak,61,91)).save(path,format="PNG")
    return path


def _bbox(bottom=32):
    return {"bounding_boxes":{"interpreted_lithology_column":[0,0,16,bottom]}}


def test_concurrent_identical_submissions_claim_once(service,tmp_path,monkeypatch):
    svc,repo,ingest=service
    paths=[_scan(tmp_path/f"source-{i}.png") for i in range(4)]
    original=repo.get_by_hash
    def delayed_lookup(*args):
        found=original(*args)
        # Widen the previously vulnerable read-before-insert interval. With
        # serialization the other calls wait, rather than reading a stale miss.
        if found is None:time.sleep(.1)
        return found
    monkeypatch.setattr(repo,"get_by_hash",delayed_lookup)
    barrier=threading.Barrier(len(paths))
    def submit(path):
        barrier.wait(timeout=5)
        return svc.submit(path,path.name,_bbox(),keep_source=True)
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures=[pool.submit(submit,path) for path in paths]
        results=[future.result(timeout=10) for future in futures]
    assert sum(result.created for result in results)==1
    assert len({result.id for result in results})==1
    assert len(repo.list_wells(svc.db))==1 and len(svc.jobs.ids)==1
    assert all(path.exists() for path in paths)
    stored=Path(repo.list_wells(svc.db)[0]["scan_path"])
    assert ingest.sha256_of(stored)==ingest.sha256_of(paths[0])


def test_changed_bbox_is_conflict_without_touching_first_inputs(service,tmp_path):
    svc,repo,ingest=service
    first=svc.submit(_scan(tmp_path/"first.png"),"first.png",_bbox(),keep_source=True)
    row=repo.get(svc.db,first.id);scan=Path(row["scan_path"]);sidecar=scan.with_suffix(".bbox.json")
    before=(scan.read_bytes(),sidecar.read_bytes(),deepcopy(row))
    with pytest.raises(ingest.InputConflict,match="different bounding boxes"):
        svc.submit(_scan(tmp_path/"second.png"),"second.png",_bbox(31),keep_source=True)
    assert (scan.read_bytes(),sidecar.read_bytes(),repo.get(svc.db,first.id))==before
    assert svc.jobs.ids==[first.id]


def test_bbox_key_order_does_not_create_false_conflict(service,tmp_path):
    svc,repo,ingest=service
    a={"note":"generated","bounding_boxes":_bbox()["bounding_boxes"]}
    b={"bounding_boxes":_bbox()["bounding_boxes"],"note":"generated"}
    first=svc.submit(_scan(tmp_path/"first.png"),"first.png",a,keep_source=True)
    again=svc.submit(_scan(tmp_path/"second.png"),"second.png",b,keep_source=True)
    assert first.id==again.id and not again.created


def test_no_bbox_and_empty_bbox_are_same_identity(service,tmp_path):
    svc,repo,ingest=service
    first=svc.submit(_scan(tmp_path/"first.png"),"first.png",None,keep_source=True)
    again=svc.submit(_scan(tmp_path/"second.png"),"second.png",{},keep_source=True)
    assert first.id==again.id and not again.created


def test_retry_preserves_original_scan_filename_and_bbox(service,tmp_path):
    svc,repo,ingest=service
    first=svc.submit(_scan(tmp_path/"first.png"),"first.png",_bbox(),keep_source=True)
    original=repo.get(svc.db,first.id);stored=Path(original["scan_path"])
    before=(stored.read_bytes(),stored.with_suffix(".bbox.json").read_bytes())
    repo.mark_failed(svc.db,first.id,"generated failure")
    again=svc.submit(_scan(tmp_path/"retry.png"),"renamed.tif",_bbox(),keep_source=True)
    assert again.created and again.id==first.id
    row=repo.get(svc.db,again.id)
    assert row["scan_path"]==original["scan_path"] and row["source_filename"]=="first.png"
    assert (stored.read_bytes(),stored.with_suffix(".bbox.json").read_bytes())==before
    assert row["status"]=="queued"


@pytest.mark.parametrize("bbox",[_bbox(33),{"depth_column":[0,0,17,32]},{"lithology_legend":[0,0,16.5,32]},{"depth_column":[False,0,16,32]},{"bounding_boxes":[]}])
def test_bbox_geometry_rejected_before_any_stored_file(service,tmp_path,bbox):
    svc,repo,ingest=service
    source=_scan(tmp_path/"source.png")
    with pytest.raises(ingest.ToolError) as error:svc.submit(source,"source.png",bbox)
    assert error.value.code=="invalid_bbox"
    assert source.exists() and not list(svc.settings.scans_dir.iterdir())
    assert repo.list_wells(svc.db)==[]


def test_orphan_bbox_refuses_upload_before_moving_source(service,tmp_path):
    svc,repo,ingest=service
    source=_scan(tmp_path/"source.png");well_id=ingest.sha256_of(source)[:16]
    sidecar=svc.settings.scans_dir/f"{well_id}.bbox.json";sidecar.write_text("canary",encoding="utf-8")
    with pytest.raises(ingest.InputConflict):svc.submit(source,"source.png",_bbox())
    assert source.exists() and sidecar.read_text()=="canary"
    assert not (svc.settings.scans_dir/f"{well_id}.png").exists()
    assert repo.list_wells(svc.db)==[]


def test_untrusted_filename_suffix_cannot_become_windows_stream(service,tmp_path):
    svc,repo,ingest=service
    submitted=svc.submit(_scan(tmp_path/"source.png"),"scan.png:stream",keep_source=True)
    row=repo.get(svc.db,submitted.id);stored=Path(row["scan_path"])
    assert stored.name==submitted.id+".scan" and stored.parent==svc.settings.scans_dir
    assert row["source_filename"]=="scan.png:stream"


def _ready_kwargs():
    return dict(name="generated",lat=12.0,lon=-20.0,block=None,total_depth_ft=100.0,csv_path="generated.csv",validation_confidence=.8,
                validation={"components":{"structure":.8},"flags":[]},scores={"values":{"drilling":80,"cost":None,"price":None,"co2":0},"logged_ft":100,"flags":[]},warnings=[],timings={"total":.1})


@pytest.mark.parametrize("field,value",[("validation_confidence",float("nan")),("validation_confidence",1.01),("total_depth_ft",float("inf")),("lat",91),("lon",-181)])
def test_store_rejects_nonfinite_or_out_of_range_scalars(service,field,value,tmp_path):
    svc,repo,ingest=service;sub=svc.submit(_scan(tmp_path/"source.png"),"source.png",keep_source=True)
    kwargs=_ready_kwargs();kwargs[field]=value
    with pytest.raises(ValueError):repo.mark_ready(svc.db,sub.id,**kwargs)
    assert repo.get(svc.db,sub.id)["status"]=="queued"


@pytest.mark.parametrize("field",["validation","scores","warnings","timings"])
def test_store_never_serializes_nan_json(service,tmp_path,field):
    svc,repo,ingest=service;sub=svc.submit(_scan(tmp_path/"source.png"),"source.png",keep_source=True)
    kwargs=_ready_kwargs()
    if field=="scores":kwargs[field]["values"]["drilling"]=float("nan")
    elif field=="warnings":kwargs[field]=[float("inf")]
    else:kwargs[field]={"bad":float("nan")}
    with pytest.raises(ValueError):repo.mark_ready(svc.db,sub.id,**kwargs)
    assert repo.get(svc.db,sub.id)["status"]=="queued"


def test_failure_recording_survives_invalid_timing_and_preserves_null_scores(service,tmp_path):
    svc,repo,ingest=service;sub=svc.submit(_scan(tmp_path/"source.png"),"source.png",keep_source=True)
    repo.mark_failed(svc.db,sub.id,"generated failure",{"total":float("nan")})
    row=repo.get(svc.db,sub.id)
    assert row["status"]=="failed" and row["scores"] is None and row["timings"] is None


def test_seeded_score_aggregation_matches_exact_rational_weights(modules,stress_seed,stress_iterations):
    scoring=importlib.import_module("app.services.scoring");rng=random.Random(stress_seed ^ 0xD001)
    for iteration in range(stress_iterations):
        wells=[]
        for _ in range(rng.randint(1,12)):
            wells.append({"scores":{"values":{key:rng.choice([None,rng.randint(0,100)]) for key in scoring.METRICS},"logged_ft":rng.randint(1,10000)}})
        actual=scoring.aggregate_scores(wells)
        for key in scoring.METRICS:
            known=[(well["scores"]["values"][key],well["scores"]["logged_ft"]) for well in wells if well["scores"]["values"][key] is not None]
            expected=round(Fraction(sum(value*weight for value,weight in known),sum(weight for _,weight in known))) if known else None
            assert actual[key]==expected,(stress_seed,iteration,key)


def test_seeded_generated_well_scores_are_finite_and_missing_location_stays_unknown(modules,stress_seed,stress_iterations):
    scoring=importlib.import_module("app.services.scoring");pd=importlib.import_module("pandas");rng=random.Random(stress_seed ^ 0xD002)
    for iteration in range(stress_iterations):
        tops=[];bases=[];rocks=[];cursor=rng.randint(-100,4000)
        for _ in range(rng.randint(1,12)):
            tops.append(cursor);cursor+=rng.randint(1,1000);bases.append(cursor);rocks.append(rng.choice(["Shale grey","Halite","Sandstone","Unknown"]))
        frame=pd.DataFrame({"top_depth":tops,"base_depth":bases,"lithology":rocks})
        result=scoring.score_well(frame,None,None,[],None)
        json.dumps(result.to_json(),allow_nan=False)
        assert result.values["cost"] is None and result.values["price"] is None
        assert all(value is None or type(value) is int and 0<=value<=100 for value in result.values.values()),(stress_seed,iteration)


def test_large_finite_logged_weights_do_not_overflow_aggregate(modules):
    scoring=importlib.import_module("app.services.scoring")
    wells=[{"scores":{"values":{key:score for key in scoring.METRICS},"logged_ft":1e308}} for score in (80,40)]
    assert scoring.aggregate_scores(wells)=={key:60 for key in scoring.METRICS}
    ties=[{"scores":{"values":{"drilling":value},"logged_ft":feet}} for value,feet in [(41,2),(2,10)]]
    assert scoring.aggregate_scores(ties)["drilling"]==8  # 102 / 12 = 8.5, ties to even.


@pytest.mark.parametrize("retry",[False,True])
def test_full_queue_refuses_before_changing_inputs_or_database(service,tmp_path,retry):
    svc,repo,ingest=service;jobs=importlib.import_module("app.services.jobs")
    release=threading.Event();started=threading.Event()
    def blocked(_):started.set();release.wait(5)
    svc.jobs=jobs.JobQueue(blocked,max_pending=1)
    try:
        first=svc.submit(_scan(tmp_path/"first.png"),"first.png",_bbox(),keep_source=True)
        assert started.wait(2)
        if retry:repo.mark_failed(svc.db,first.id,"generated failed attempt")
        before=repo.list_wells(svc.db)
        files={p.name:p.read_bytes() for p in svc.settings.scans_dir.iterdir()}
        source=_scan(tmp_path/"blocked.png",tweak=0 if retry else 1)
        with pytest.raises(ingest.ToolError) as exc:svc.submit(source,"blocked.png",_bbox())
        assert exc.value.code=="queue_busy"
        assert source.exists() and repo.list_wells(svc.db)==before
        assert {p.name:p.read_bytes() for p in svc.settings.scans_dir.iterdir()}==files
    finally:
        release.set();svc.shutdown()


def test_preparation_failure_releases_slot_and_restores_source(service,tmp_path,monkeypatch):
    svc,repo,ingest=service;jobs=importlib.import_module("app.services.jobs")
    svc.jobs=jobs.JobQueue(lambda _:None,max_pending=1)
    source=_scan(tmp_path/"source.png");original=source.read_bytes()
    real_insert=repo.insert_queued
    def broken_insert(*args,**kwargs):raise RuntimeError("generated insert failure")
    monkeypatch.setattr(repo,"insert_queued",broken_insert)
    with pytest.raises(RuntimeError,match="insert failure"):svc.submit(source,"source.png",_bbox())
    assert source.read_bytes()==original and not list(svc.settings.scans_dir.iterdir())
    assert repo.list_wells(svc.db)==[]
    monkeypatch.setattr(repo,"insert_queued",real_insert)
    assert svc.submit(source,"source.png",_bbox(),keep_source=True).created


def test_enqueue_failure_marks_failed_retains_inputs_and_can_retry(service,tmp_path,monkeypatch):
    svc,repo,ingest=service;jobs=importlib.import_module("app.services.jobs")
    svc.jobs=jobs.JobQueue(lambda _:None,max_pending=1)
    source=_scan(tmp_path/"source.png");real_submit=svc.jobs._pool.submit
    def broken_submit(*args,**kwargs):raise RuntimeError("generated executor failure")
    monkeypatch.setattr(svc.jobs._pool,"submit",broken_submit)
    with pytest.raises(RuntimeError,match="executor failure"):svc.submit(source,"source.png",_bbox(),keep_source=True)
    rows=repo.list_wells(svc.db)
    assert len(rows)==1 and rows[0]["status"]=="failed"
    assert Path(rows[0]["scan_path"]).is_file() and Path(rows[0]["scan_path"]).with_suffix(".bbox.json").is_file()
    monkeypatch.setattr(svc.jobs._pool,"submit",real_submit)
    assert svc.submit(source,"source.png",_bbox(),keep_source=True).created


def test_admission_releases_on_context_error_completion_and_cancel(modules):
    jobs=importlib.import_module("app.services.jobs");errors=importlib.import_module("app.tools.errors")
    release=threading.Event();started=threading.Event()
    def handler(_):started.set();release.wait(5)
    queue=jobs.JobQueue(handler,max_pending=2)
    try:
        first=queue.enqueue("first");assert started.wait(2)
        second=queue.enqueue("second")
        with pytest.raises(errors.QueueBusy):queue.enqueue("excess")
        assert second.cancel()
        with pytest.raises(RuntimeError,match="preparation"):
            with queue.admit():raise RuntimeError("preparation failed")
        third=queue.enqueue("third")
        release.set();queue.shutdown(drain=True)
        assert first.done() and third.done() and second.cancelled()
        assert queue._slots.acquire(blocking=False) and queue._slots.acquire(blocking=False)
        assert not queue._slots.acquire(blocking=False)
    finally:
        release.set();queue.shutdown()


def test_shutdown_cancel_releases_waiting_future_slots(modules):
    jobs=importlib.import_module("app.services.jobs")
    release=threading.Event();started=threading.Event()
    def handler(_):started.set();release.wait(5)
    queue=jobs.JobQueue(handler,max_pending=2)
    try:
        queue.enqueue("running");assert started.wait(2)
        waiting=queue.enqueue("waiting")
        with ThreadPoolExecutor(max_workers=1) as pool:
            stopped=pool.submit(queue.shutdown)
            deadline=time.monotonic()+2
            while not waiting.cancelled() and time.monotonic()<deadline:time.sleep(.005)
            release.set();stopped.result(timeout=3)
        assert waiting.cancelled()
        assert queue._slots.acquire(blocking=False) and queue._slots.acquire(blocking=False)
        assert not queue._slots.acquire(blocking=False)
    finally:
        release.set();queue.shutdown()
