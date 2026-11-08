"""Build the reproducible coverage report and inline comparison artifact."""
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import time

from PIL import Image
from jev_optimize import OUT, OLD, BAKE, HOME, locked, overlap, read, save, score


def build():
    data=locked();truth=read(OUT/'private/truth.json');report=read(OUT/'report.json');baseline=read(OUT/'baselines.json')
    old=read(OLD/'blind/packets.json');oldtruth=read(OLD/'private/answer-key.json')
    oldtruth={p:dict(v,status='match') for p,v in oldtruth.items()}
    jev=read(OUT/'test-criteria.json');control=read(OUT/'grid-control.json');frozen=read(OUT/'frozen.json')
    audit=read(HOME/'data/completion-audit-20261108/manifest.json')['records'];annotations={r['annotation_id']:r for r in audit}
    fam={r['log']:r['family'] for r in read(BAKE/'analysis/categories/conf.json')['logs']}
    test=[i for i in data['items'] if i['split']=='test'];pure=[i for i in test if truth[i['item_id']]['status']=='match']
    oldcells=[c for b in read(OLD/'results/jev-32.json')['batches'] for row in b['matrix'] for c in row if c]
    old_scores={'jev-grid32':score(oldcells,old['items'],oldtruth),**{m:score(c,old['items'],oldtruth) for m,c in baseline['methods'].items()}}
    by_doc=Counter(i['document_id'] for i in test)
    detail=dict(test_documents=len(by_doc),test_families=dict(Counter(fam[data['sources'][i['document_id']]['name']] for i in test)),
                reviewed_label_names=len({r['targets'][0]['key'] for r in audit if r['annotation_id'].startswith('queue-') and len(r['targets'])==1}),
                old_label_names=dict(Counter(annotations[t['annotation_id']]['targets'][0]['key'] for t in oldtruth.values())),
                old_scores=old_scores,control_timing_usable=False,control_long_context_calls=sum(c['long_context'] for c in control['calls']),
                test_candidate_recall=sum(truth[i['item_id']]['legend_ids'][0] in read(OUT/'candidate-ranks.json')['items'][i['item_id']] for i in pure),
                fresh_api_calls=sum(not c['cached'] for c in jev['calls']),api_calls=len(jev['calls']),usage=dict(Counter()))
    usage=Counter()
    for c in jev['calls']:
        for k,v in (c.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]+=v
    detail['usage']=dict(usage)
    # Check native dimensions and spatial exclusion from BOTH old test packs.
    excluded=defaultdict(list)
    for i in old['items']:excluded[i['document_id']].append(i['source_box'])
    first=read(HOME/'data/terra-grid-20261108/private/answer-key.json')
    for t in first.values():
        if t['annotation_id'] in annotations:
            r=annotations[t['annotation_id']];excluded[r['document_id']].append(r['box_xyxy'])
    collisions=[];bad_sizes=[]
    for i in test:
        if any(overlap(i['source_box'],b) for b in excluded[i['document_id']]):collisions.append(i['item_id'])
        b=i['source_box']
        with Image.open(i['crop']) as im:
            if im.size!=(b[2]-b[0],b[3]-b[1]):bad_sizes.append(i['item_id'])
    if collisions or bad_sizes:raise ValueError(f'Input integrity failure: overlaps={collisions}, sizes={bad_sizes}')
    detail['integrity']=dict(test_crops=len(test),spatial_overlaps=0,wrong_native_sizes=0,unique_ids=len({i['item_id'] for i in test}),unique_pixel_hashes=len({i['sha256'] for i in test}),manifest_sha256=read(OUT/'lock.json')['manifest_sha256'])
    cells={c['item_id']:c for c in jev['cells']};boundary=[i for i in test if truth[i['item_id']]['status']=='mixed']
    detail['boundary_flags']=dict(Counter(cells[i['item_id']]['status'] for i in boundary))
    detail['all_boundary_routed_review']=all(cells[i['item_id']]['route']=='review' for i in boundary)
    save(OUT/'audit-summary.json',detail)
    names={'jev':'Jev, new pipeline','jev-grid32-control':'Jev, old grid','cached-v4':'Cached v4','hand-crop':'Hand features, same crop','dino':'DINOv2 prototype','fusion':'DINO + hand prototype'}
    lines=['# Jev image pipeline: coverage and matched tests','',
           'Measured on 8 November 2026. All model outputs remain unapproved. No live reviews were changed.','',
           '## Coverage','',
           '- The dataset catalog has 51 documents. The saved audit has 350 new accepted boxes from 44 documents and 70 visual pattern sets.',
           '- The 678 supplied boxes come from Hadrian 2 and contain four labels. They were excluded from the large-grid benchmark.',
           '- The old 128-crop test covered 14 documents, 51 of the 70 reviewed sets, and only three of the seven observed layout families. Six documents supplied 120 crops (93.75%).',
           f'- The new test has 80 crops from {len(by_doc)} documents and all seven families: 48 truth-bed interiors, 12 known boundaries, and 20 accepted crops from 20 other documents.',
           '- The three truth logs are Neel 1-inch, Shenzi, and D-582 log06. The first two have reviewed truth; the last has gold truth. Hadrian was excluded from this truth test because its supplied labels have known quality issues.',
           '- Truth-bed sampling gives labels similar weight. It does not weight long layers by pixel area. The 20 saved reviews are a separate agreement test, not independent gold.',
           '- The 80 test boxes have no spatial overlap with the earlier 128-crop or 60-crop packs. Inputs and answer keys were hashed before API design tests.',
           '- Only 32 old crops were used to choose the input format. The three truth logs were already known to local model work. This is a new-window test, not a claim of generalisation to unseen wells. Shenzi truth may also retain model boundary anchoring.','',
           '## Matched crop results','',
           'Exact results require one correct current source-legend ID. A two-label answer does not receive exact credit. Unknown or unmapped outputs also receive no exact credit.','',
           '| Method | All 68 single-pattern crops | Truth interiors (48) | New reviewed crops (20) | Two-label answers on all 68 |',
           '|---|---:|---:|---:|---:|']
    for m in ['jev','jev-grid32-control','cached-v4','dino','fusion']:
        a=report['scores']['all-test'][m];g=report['scores']['truth-interior'][m];n=report['scores']['new-reviewed'][m]
        lines.append(f"| {names[m]} | {a['correct']}/68 ({a['agreement']:.1%}) | {g['correct']}/48 | {n['correct']}/20 | {a['ambiguous']} |")
    lines += ['',
        '- v4 has five two-label answers; all five contain the truth. Fusion has nine; all nine contain the truth. Count these as unresolved, rather than wrong visual matches. With this looser diagnostic, v4 contains the truth in 60/68 and fusion in 61/68. This is not exact classification accuracy.',
        '- The new Jev format improves exact agreement by 10/68 over its old grid on these same pixels. This is a change to several parts at once. It does not isolate which part caused the gain.',
        '- The old grid hit the normal context limit for its last batch and used the documented long-context retry. The first two calls were reused during recovery. Its combined wall time is not a fresh latency measurement.',
        f"- The new Jev run returned all 80 rows in {jev['seconds']:.2f} s, with {len(jev['calls'])} fresh requests, four concurrent requests, no failed calls, and model jev-1.14.0. This includes card creation, requests and parsing. It excludes TIFF crop extraction and local candidate ranking. It is one run, not a latency guarantee.",
        '- DINO baseline timing: 20.12 s to load the model, then 10.43 s for hand features and DINO inference on 240 crops (new, development, and old). It used CUDA. These scopes differ from Jev, so they are not a direct speed ratio.',
        '- Crop classification does not measure layer detection, row accuracy, or boundary F1. This result does not establish a higher end-to-end challenge score.','',
        '## Errors and review checks','',
        '- Jev has three exact-ID errors. Two are D-582 Shale versus Clay. The source legend draws these symbols alike. The third is Neel Sandstone versus Siltstone. All three went to review.',
        '- Jev marks eight boundary crops as mixed and four as bad crops. All 12 go to review. It flags 12 of the 48 truth interiors too: 11 are uncertain D-582 symbols and one is another quality flag.',
        '- At the thresholds fixed before the test, only 13/68 single-pattern crops qualify as strong suggestions. All 13 match the saved truth. This small sample does not justify automatic acceptance.',
        '- Candidate recall is 68/68 on the test and 30/32 on development. A shortlist can omit the true label. Keep the full source legend available and use the fit check to send weak cases to a person.',
        '- D-582 remains the hard layout. New Jev gives 14/16 exact labels, cached v4 8/16, and fusion 4/16 plus five two-label answers. Source hatch size differs from the column hatch size. The test is class-balanced and uses 72-pixel windows; earlier DINO reports used a different, often shale-heavy window sample. These results are not interchangeable.','',
        '## Implemented Jev path','',
        '1. Keep native source pixels, source coordinates, source document, legend IDs, and hashes.',
        '2. Group by document. Rank active legend symbols by local hatch features and background colour. Keep the top four hatch options plus the top two colour options.',
        '3. Send at most four queries per document request. Use source swatch images directly in Choice criteria. Upload identical images once with multipart.',
        '4. In the same request, ask whether any candidate fits and whether the crop is mixed, unclear, or unusable. Use the original crop for the quality question.',
        '5. Send a result to review unless fit is at least 0.9, top probability at least 0.8, confidence at least 0.9, and quality is single-pattern. These are uncalibrated review rules, not probabilities of correctness.',
        '6. Keep weak candidates, twins, boundary changes and missing legend options in the human queue. Keep all generated results separate from human labels and synthetic-data approval.',
        '',
        'The previous clustering experiment remains separate. This new test does not measure propagation of a cluster representative to all members. Do not use its 65/68 result to approve an entire cluster.','',
        '## Development tests','',
        '| Format | Exact agreement on 32 old crops | Measured run time |',
        '|---|---:|---:|']
    for v in ['raw','tile','criteria','grid']:
        s=frozen['dev_results'][v];r=read(OUT/f'dev-{v}.json');lines.append(f"| {v} | {s['correct']}/32 | {r['seconds']:.2f} s |")
    lines += ['',
        'The criteria format tied raw full-legend cards and used less time. Native-pixel tiling did not help Jev. Settings were frozen before the new test. These are single-run observations.','',
        '## Official guidance used','',
        '- [Images](https://docs.typesafe.ai/images): images can define Choice options; repeated references expand image context; state reuse and multipart reduce duplicate transfer. The API has size limits and an image feature header.',
        '- [Companion Noul](https://docs.typesafe.ai/patterns/companion-noul): a Choice always picks an option. Pair it with a fit check.',
        '- [Parallel questions](https://docs.typesafe.ai/patterns/fan-out): ask all needed questions in one request, then choose the route in code.',
        '- [Models and context](https://docs.typesafe.ai/models): pin the model version; avoid long context when normal context fits. The docs warn that long context is slower and less accurate.',
        '- [Confidence cookbook](https://docs.typesafe.ai/cookbooks/classification_using_confidence): check confidence on data from the target task. The cookbook does not calibrate our geology inputs.','',
        '## Reproduce','',
        'Run from the project root with Python UTF-8 enabled. The prepared pack is immutable. The configured project Jev credential is used without logging it.','',
        '```text',
        'python -X utf8 viridien_work/pattern_lab/jev_optimize.py rank',
        'python -X utf8 viridien_work/pattern_lab/jev_optimize.py dev --variant criteria',
        'python -X utf8 viridien_work/pattern_lab/jev_optimize.py test',
        'python -X utf8 viridien_work/pattern_lab/jev_optimize.py report',
        'python -X utf8 viridien_work/pattern_lab/jev_report.py',
        '```','',
        'Repeated API jobs use content-and-prompt caches. A cached rerun cannot establish fresh API latency. Do not refit the frozen pipeline using this test and then call the same cases a new holdout.','']
    (HOME/'JEV-OPTIMIZED.md').write_text('\n'.join(lines),encoding='utf-8')
    chart={k:{m:{x:v for x,v in s.items() if x not in ['errors']} for m,s in report['scores'][k].items()} for k in ['all-test','truth-interior','new-reviewed']}
    page='''<!doctype html><html><head><meta charset="utf-8"><style>
    body{font-family:var(--font-sans);color:var(--foreground);margin:0;font-size:14px}h3{font-size:17px;margin:16px 0 10px}p{line-height:1.5;color:var(--muted-foreground)}select{background:var(--card);color:var(--foreground);border:1px solid var(--border);padding:8px;border-radius:var(--radius);font:inherit;width:100%}.row{margin:15px 0}.lab{display:flex;justify-content:space-between;gap:12px;margin-bottom:6px}.bar{display:flex;height:19px;background:var(--muted);border-radius:4px;overflow:hidden}.exact{background:var(--chart-2)}.ambig{background:var(--chart-4)}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted-foreground)}.key{display:inline-block;width:10px;height:10px;margin-right:5px}#bars{height:260px}.note{padding:14px;background:var(--card);border-radius:var(--radius)}.cov{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cov div{background:var(--card);padding:16px;border-radius:var(--radius)}strong{color:var(--foreground)}.big{font-size:24px;font-weight:650;display:block;margin:6px 0}.small{font-size:12px}.result{margin-top:12px}@media(max-width:440px){.lab{font-size:12px}.cov{gap:8px}.cov div{padding:12px}}
    </style></head><body>
    <div class="cov"><div><span class="small">Earlier grid test</span><span class="big">14 / 51 docs</span>3 of 7 layout families<br><span class="small">94% of crops from 6 documents</span></div><div><span class="small">New fixed test</span><span class="big">23 / 51 docs</span>7 of 7 layout families<br><span class="small">80 crops; no old crop overlap</span></div></div>
    <h3>Exact source-legend matches</h3><label for="cohort" class="small">Select the test group</label><select id="cohort"><option value="all-test">68 single-pattern crops: all sources</option><option value="truth-interior">48 interiors from reviewed / gold truth</option><option value="new-reviewed">20 saved reviews from other documents</option></select><div id="bars"></div><div class="legend"><span><i class="key exact"></i>One correct label</span><span><i class="key ambig"></i>Two-label answer</span><span><i class="key" style="background:var(--muted)"></i>Other result</span></div>
    <p class="note"><strong>Jev: 5.4 seconds for all 80 crops.</strong><br>All 12 boundary crops went to review. Only 13 single-pattern crops passed the fixed strong-suggestion rules; all 13 matched. A person must still check the other cases.</p><p class="small">One run. Jev uses 4 concurrent requests. Time excludes source extraction and local ranking. Labels are compared on the same pixels. This does not measure full layer detection or the end-to-end challenge score.</p>
    <script>const data=DATA;const names={'jev':'Jev · new pipeline','jev-grid32-control':'Jev · old grid','cached-v4':'Cached v4','fusion':'DINO + hand prototype'};const keys=Object.keys(names);function draw(){const d=data[document.getElementById('cohort').value];document.getElementById('bars').innerHTML=keys.map(k=>{let x=d[k],a=x.correct/x.pure*100,b=x.ambiguous/x.pure*100;return `<div class="row"><div class="lab"><span>${names[k]}</span><strong>${x.correct}/${x.pure} · ${a.toFixed(1)}%</strong></div><div class="bar" aria-label="${names[k]}: ${x.correct} exact, ${x.ambiguous} ambiguous, ${x.pure-x.correct-x.ambiguous} other"><span class="exact" style="width:${a}%"></span><span class="ambig" style="width:${b}%"></span></div></div>`}).join('')}document.getElementById('cohort').addEventListener('change',draw);draw();</script></body></html>'''
    page=page.replace('DATA',json.dumps(chart));(OUT/'comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(integrity=detail['integrity'],test_families=detail['test_families'],control_long_context_calls=detail['control_long_context_calls'],files=['JEV-OPTIMIZED.md','comparison.html','audit-summary.json'])))


if __name__=='__main__':build()
