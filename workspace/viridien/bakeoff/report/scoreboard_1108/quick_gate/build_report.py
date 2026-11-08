"""Build a portable advisory from cached contract-check metadata only."""
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = read(HERE / "result.json")
    receipt = read(HERE / "command_receipt.json")
    assert receipt["exit_code"] == data["test_exit"] == 0
    assert data["successful_dual_schema_exports"] == 271
    assert sha(HERE / "run.py") == data["source_sha256"]
    assert sha(HERE.parent / "availability_v2/release_receipt.json") == data["frozen_release_sha256"]
    runtime = data["historical_runtime_by_strategy"]["s19_combine"]["end_to_end_seconds"]
    stages = [
        ("Page + boxes", "Same source image and supplied column, legend and depth boxes."),
        ("Page context", "Legend parsing and depth calibration, once per exact context."),
        ("Whole column", "Process strips; keep bed contacts, labels and visual cluster IDs."),
        ("Deliverables", "One interval CSV and one annotated image per document."),
        ("Separate evaluation", "Strict and approved equivalence scores, depth, time and resources."),
    ]
    flow = "".join(f'<div class="stage"><b>{i}. {html.escape(title)}</b><p>{html.escape(note)}</p></div>'
                   for i, (title, note) in enumerate(stages, 1))
    body = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>*{{box-sizing:border-box}}html,body{{margin:0;font-family:var(--font-sans,Arial);color:var(--foreground)}}main{{max-width:760px;margin:auto}}h1{{font-size:24px}}h2{{font-size:19px;margin:0 0 10px}}p,li{{font-size:14px;line-height:1.45}}section{{padding:16px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius);margin:14px 0}}.warn{{border-left:4px solid var(--warning)}}.ok{{border-left:4px solid var(--success)}}.stage{{padding:10px 0;border-bottom:1px solid var(--border)}}.stage p{{margin:5px 0}}.bar{{height:14px;background:var(--chart-1)}}.track{{height:14px;background:var(--secondary);margin:8px 0}}.muted{{color:var(--muted-foreground)}}a{{color:var(--primary)}}.metric{{font-size:26px;font-weight:700}}.chart{{height:106px}}</style></head><body><main>
<h1>One workflow: current contract check</h1>
<section class="warn"><h2>Full challenge readiness is not verified</h2><p>The common input/output interface exists. Complete document coverage, working legend inputs, depth correctness and final annotated images are not all validated. The official numeric scoring formula is unknown.</p></section>
<section class="ok"><h2>New bounded check</h2><div class="metric">{data['elapsed_seconds']:.2f} seconds</div><p>272 cached records checked. 271 successful predictions round-trip through both published CSV formats. One failed/empty record stays a failure. All raw outputs are preserved.</p><p><b>5 contract tests passed in 0.85 seconds.</b> The background command exited 0. Its hard limit was 180 seconds. No model, truth image or API was used.</p></section>
<section><h2>One target flow</h2>{flow}</section>
<section><h2>Coverage is still partial</h2><p>61 physical documents per strategy. A saved window spans the full supplied lithology box on 44. Seventeen have only partial or unknown coverage.</p><div class="chart"><div class="track"><div class="bar" style="width:{44/61*100}%"></div></div><p>44 / 61 with a full-box record. This checks geometry, not label or depth correctness.</p></div></section>
<section><h2>Fastest useful next step</h2><p>Retain S19 as the broader reference candidate. Restore a working legend input path before judging v4/DINO on outside pages. Reuse exact page context, warm model state and strip decoding. Checkpoint each case.</p><p>The separate cache pilot reduced prediction time from 6.77 to 3.74 seconds with 612 identical intervals. That is a narrow pilot; the source packet remains separate from the frozen scores.</p></section>
<section><h2>What runtime can we promise?</h2><p>This cached gate: <b>{data['elapsed_seconds']:.2f} seconds</b>. A fresh full evaluation within five minutes: <b>not verified</b>.</p><p>Historical S19 end-to-end window time: median {runtime['median_seconds']:.1f} seconds, 95th percentile {runtime['p95_seconds']:.1f} seconds, over {runtime['n']} timed successful windows. One failed attempt has unknown inference time. Hardware and cache conditions differ.</p></section>
<section class="warn"><h2>Current execution state</h2><p>Dataset inference, tuning, label generation and new JEV calls remain held. Depth repair is UNASSIGNED. This check did not change model code, frozen scores or source ownership.</p></section>
<p><a href="FAST_PATH.md">Workflow and next PR notes</a> · <a href="result.json">Every check and timing count</a> · <a href="tests.log">Tests</a> · <a href="../availability_v2/index.html">Frozen corrected scores</a></p>
</main></body></html>'''
    (HERE / "index.html").write_text(body, encoding="utf-8")
    pins = {name: sha(HERE / name) for name in (
        "run.py", "background.py", "build_report.py", "FAST_PATH.md", "result.json",
        "command_receipt.json", "tests.log", "run.log", "index.html")}
    (HERE / "advisory_receipt.json").write_text(json.dumps({
        "files": pins, "frozen_release_sha256": data["frozen_release_sha256"],
        "frozen_release_modified": False, "inference_started": False,
        "scores_recomputed": False, "owner_hold_lifted": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("QUICK_REPORT: cached contract and workflow advisory saved; frozen release unchanged")


if __name__ == "__main__":
    main()
