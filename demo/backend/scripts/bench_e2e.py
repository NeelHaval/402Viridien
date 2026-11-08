"""End-to-end benchmark: HTTP upload -> well `ready` with scores AND located in GET /wells.

    python scripts/bench_e2e.py --label baseline [--runs 3] [--save ../../perf/baseline] [--scans key,key]

Starts its own uvicorn on a throwaway data dir (so run 1 of the first scan is a true cold request), uploads each scan
`runs` times (between runs the well row/files are removed so the pipeline really runs again) and reports cold and warm
medians. Writes perf/bench_<label>.json. Outputs of run 1 of each scan are saved to --save for accuracy comparison.
"""
from __future__ import annotations

import argparse, json, os, shutil, sqlite3, statistics, subprocess, sys, tempfile, time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parents[1] / "data" / "dataset"
# key -> (file, why it is in the set)
SCANS = {
    "shell08852_9k": "Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft.tif",      # 9,356 px, RGB, small
    "mc761_15k_pal": "60817411400100_mc761_1_st1_mudlog_1md_17133_22886.tiff",           # 15,400 px, palette
    "hadrian_50k": "60808400140000_hadrian 2_masterlog_2in_10000-21962.tif",             # 50,000 px, RGB
    "shenzi_73k_pal": "60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400.tif",      # 72,600 px, palette, widest (2550)
    "mdn1_188k": "60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft.tif",    # 187,600 px, RGB, tallest
}
PORT = int(os.environ.get("BENCH_PORT", "8765"))
BASE = f"http://127.0.0.1:{PORT}"


def start_server(data_dir: Path) -> subprocess.Popen:
    env = {**os.environ, "APP_DATA_DIR": str(data_dir)}
    log = open(data_dir / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=log, stderr=log)
    for _ in range(600):
        try:
            if httpx.get(f"{BASE}/health", timeout=1).status_code < 500:
                return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("server did not start")


def one_run(client: httpx.Client, path: Path, bbox: Path) -> dict:
    t0 = time.perf_counter()
    with open(path, "rb") as f, open(bbox, "rb") as b:
        r = client.post(f"{BASE}/upload", files={"file": (path.name, f), "bbox_json": (bbox.name, b, "application/json")})
    r.raise_for_status()
    t_upload = time.perf_counter() - t0
    wid = r.json()["id"]
    t_ready = t_located = t_pin = None
    while t_ready is None or t_located is None:
        now = time.perf_counter() - t0
        if now > 900:
            raise SystemExit(f"timeout on {path.name}")
        d = client.get(f"{BASE}/wells/{wid}").json()
        if d["status"] == "failed":
            raise SystemExit(f"{path.name} failed: {d['error']}")
        if t_pin is None and d.get("lat") is not None:
            t_pin = time.perf_counter() - t0
        if t_ready is None and d["status"] == "ready" and d.get("scores"):
            t_ready = time.perf_counter() - t0
        if t_located is None:
            row = next((w for w in client.get(f"{BASE}/wells").json() if w["id"] == wid), None)
            if row and row["located"]:
                t_located = time.perf_counter() - t0
        time.sleep(0.05)
    return {"id": wid, "upload_response_s": t_upload, "time_to_pin_s": t_pin, "time_to_ready_s": t_ready,
            "time_to_located_s": t_located, "total_s": max(t_ready, t_located)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--scans", default=",".join(SCANS))
    ap.add_argument("--outdir", type=Path, default=ROOT.parents[1] / "perf")
    a = ap.parse_args()
    data_dir = Path(tempfile.mkdtemp(prefix="bench-"))
    shutil.copytree(ROOT / "data" / "reference", data_dir / "reference")
    proc = start_server(data_dir)
    results: dict = {"label": a.label, "scans": {}}
    try:
        with httpx.Client(timeout=900) as client:
            for key in a.scans.split(","):
                path = DATASET / "images" / SCANS[key]
                bbox = DATASET / "bbox_annotation" / (path.stem + ".json")
                runs = []
                for i in range(a.runs):
                    res = one_run(client, path, bbox)
                    cold = (not results["scans"]) and i == 0 and not runs
                    res["cold_server"] = cold
                    con = sqlite3.connect(data_dir / "app.db")
                    res["stage_timings"] = json.loads(con.execute("select timings_json from wells where id=?", (res["id"],)).fetchone()[0] or "{}")
                    if i == 0 and a.save:
                        out = a.save / key
                        out.mkdir(parents=True, exist_ok=True)
                        shutil.copy(data_dir / "csv" / f"{res['id']}.csv", out / "corrected.csv")
                        shutil.copy(data_dir / "csv" / f"{res['id']}.raw.csv", out / "raw.csv")
                        detail = client.get(f"{BASE}/wells/{res['id']}").json()
                        detail.pop("column", None)
                        (out / "well.json").write_text(json.dumps(detail, indent=1))
                    runs.append(res)
                    print(f"{key} run{i + 1}{' (cold server)' if cold else ''}: total {res['total_s']:.2f}s pin {res['time_to_pin_s']:.2f}s upload {res['upload_response_s']:.2f}s", flush=True)
                    # make the next run real work again: forget the well and its files
                    con.execute("delete from wells where id=?", (res["id"],)); con.commit(); con.close()
                    for f in list((data_dir / "scans").glob(f"{res['id']}*")) + list((data_dir / "csv").glob(f"{res['id']}*")):
                        f.unlink()
                warm = [r["total_s"] for r in runs if not r["cold_server"]]
                results["scans"][key] = {"file": SCANS[key], "runs": runs, "warm_median_s": statistics.median(warm) if warm else None,
                                         "first_run_s": runs[0]["total_s"]}
    finally:
        proc.terminate(); proc.wait(timeout=30)
    cold = next(iter(results["scans"].values()))["runs"][0]
    results["cold_start_first_request_s"] = cold["total_s"]
    a.outdir.mkdir(exist_ok=True, parents=True)
    (a.outdir / f"bench_{a.label}.json").write_text(json.dumps(results, indent=1))
    print("\nscan  first_run  warm_median")
    for k, v in results["scans"].items():
        print(f"{k:16s} {v['first_run_s']:8.2f} {v['warm_median_s'] if v['warm_median_s'] is None else round(v['warm_median_s'], 2)}")


if __name__ == "__main__":
    main()
