"""Write the development report. This module cannot run a final benchmark."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from ml.legend_v4.experiment import HERE,BAKEOFF,ROOT,allowed
from ml.legend_names.experiment import summarize_names


def mean(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def metric_table(rows,group_key):
    lines=['| Group | Strategy | Windows | Row accuracy | Boundary F1 | ARI | Score |',
           '|---|---|---:|---:|---:|---:|---:|']
    records=[]
    for group in sorted({r[group_key] for r in rows}):
        for strategy in ('s18','s18_v3','s20'):
            rr=[r for r in rows if r[group_key]==group and r['strategy']==strategy]
            m={k:mean(rr,k) for k in ('row_acc','boundary_f1','cluster_ari','score')}
            records.append(dict(group=group,strategy=strategy,n_windows=len(rr),**m))
            lines.append(f"| {group} | {strategy} | {len(rr)} | {m['row_acc']:.2%} | {m['boundary_f1']:.3f} | {m['cluster_ari']:.3f} | {m['score']:.3f} |")
    return lines,records


def main():
    items,meta=allowed()
    external=json.loads((HERE/'external_names.json').read_text());domain=json.loads((HERE/'domain_names.json').read_text())
    cv=json.loads((HERE/'strategy_cv.json').read_text());synthetic=json.loads((HERE/'synthetic_metrics.json').read_text())
    selected=json.loads((HERE/'selected.json').read_text())
    lines=['# Legend v4 and S20','',
           'Legend v4 keeps printed rock names. S20 calls S18 with the new legend entries. It keeps the S18 boundary, match, decode, and confidence code. The shared parser does not change.','',
           '## Result','',
           '| Data | Parser | Pages | Printed entries | Detection | Exact name | Exact name and usable |',
           '|---|---|---:|---:|---:|---:|---:|']
    for data,rr in [('External dev, grouped CV',external),('In-domain non-eval, development check',domain)]:
        for parser in ('v1','v3','v4'):
            m=summarize_names([r[parser] for r in rr])
            lines.append(f"| {data} | {parser} | {m['n_logs']} | {m['n_truth']} | {m['detected']:.2%} | {m['exact']:.2%} | {m['matchable']:.2%} |")
    lines+=['',
        'Exact agreement ignores case and spaces. It does not expand synonyms. Each truth sample can match one predicted sample at IoU 0.5 or more. Missed samples count as errors. The in-domain labels include two No Sample cells. Those cells can have a correct name but are not usable rocks. Page totals repeat some identical legends. They are not independent samples.','',
        '## Data limits','',
        'The current external dev file has 51 windows from five wells and 46 pages. Only these external pages were opened. ForsythB2-17 was excluded. The full old dev file was not read. No protected benchmark was run. No protected image or truth file was opened. The sealed wells were not searched for or restored.','',
        'The fixed in-domain eval list was read only to exclude its 13 log names. Six new legend label files come from manual reads of non-eval images. One existing label file supplies the Hadrian legend. The manual labels and source legend crops are in this folder. No real non-eval interval truth is present in the harness truth folder. The in-domain strategy results below therefore use synthetic columns.','',
        '## Changes','',
        '- Keep valid geological adjectives and plural rock names. Do not reduce Dolomites to Dolomite or Volcanics to Volcanic.',
        '- Keep readable page text on the right when tight OCR fails. This stops a label on the left from taking its place.',
        '- Keep printed separators when the letters agree across OCR engines.',
        '- Apply section headings for mineral, modifier, fossil, and technical cells. Reject technical names when OCR removes spaces or punctuation.',
        '- Split fused rows in a shallow Rock Types grid. This rule requires wide samples and unequal sample heights. Intact grids retain their geometry. Read each label inside its own row and column.',
        '- Cache raw OCR by exact pixels. Read an inherited cache only for the current image digest. Store new evidence inside this lane. No filename or well rule changes a runtime label.','',
        '## Legend names by well and family','',
        '| Group | Parser | Entries | Detection | Exact name | Exact name and usable |',
        '|---|---|---:|---:|---:|---:|']
    name_groups=[]
    for rr,key in ((external,'well'),(domain,'family')):
        for group in sorted({r[key] for r in rr}):
            for parser in ('v1','v3','v4'):
                m=summarize_names([r[parser] for r in rr if r[key]==group]);name_groups.append(dict(group=group,parser=parser,**m))
                lines.append(f"| {group} | {parser} | {m['n_truth']} | {m['detected']:.2%} | {m['exact']:.2%} | {m['matchable']:.2%} |")
    lines+=['',
        'The v4 OCR weight is selected without the held well. The objective gives each training well equal weight. The three candidate EasyOCR weights are 0.9, 1.0, and 1.15. Cleanup is on for all three. The deployed weight is '+str(selected['config']['easy_weight'])+'. The v3 reference keeps its inherited weight of 1.15. Thus the v4 grouped result includes the cost of setting selection. These are development estimates. Parser code changes used these development labels.','',
        '## Strategy metrics on external dev','',
        'Each held well is excluded from fresh boundary and matcher fits. Each CNN starts from new random weights. It uses rich features, 450 steps, and seed 100 plus the fold number. The matcher is a fresh logistic fit. Training pairs use only the other current dev wells and synthetic samples made from those wells. No old training-pair file or saved boundary probabilities are used.','',
        'S18 below uses parser v1. S18_v3 uses parser v3. S20 uses parser v4. The three variants share each fold model, proposals, and fixed S18 decoder. Confidence does not change labels and is not part of these CV scores.','']
    table,well_rows=metric_table(cv,'well');lines+=table
    lines+=['','| Mean | Strategy | Row accuracy | Boundary F1 | ARI | Score |','|---|---|---:|---:|---:|---:|']
    totals=[]
    for strategy in ('s18','s18_v3','s20'):
        rr=[r for r in cv if r['strategy']==strategy]
        for kind in ('window','well'):
            source=rr if kind=='window' else [r for r in well_rows if r['strategy']==strategy]
            m={k:mean(source,k) for k in ('row_acc','boundary_f1','cluster_ari','score')};totals.append(dict(mean=kind,strategy=strategy,**m))
            lines.append(f"| {kind} | {strategy} | {m['row_acc']:.2%} | {m['boundary_f1']:.3f} | {m['cluster_ari']:.3f} | {m['score']:.3f} |")
    lines+=['','## Strategy metrics by non-eval family','',
        'These are synthetic diagnostics, not real log scores or independent CV. Three fixed seeds make 3200-row columns for each labelled non-eval log. Each bed has 200 rows. Truth names come from the manual legend read. Patterns come from the inner sample pixels. The CNN and matcher are fitted on the 51 external dev windows only. The same models serve all families. The synthetic results did not select a model or threshold.','']
    table,family_rows=metric_table(synthetic,'family');lines+=table
    lines+=['',
        'Depth MAE is not available. These checks do not fit a depth calibration. The saved per-window records retain the null depth field.','',
        '## Inherited model history','',
        'The deployed S20 wrapper inherits the S18 weights. The old boundary training record lists 59 windows and includes ForsythB2-17. The old matcher preparation also used the former dev split. The S18 decoder settings and confidence fit were selected on that former split. These inherited assets have historical exposure to the new shared_verify well. S20 is not a clean retrain of the full S18 system.','',
        'The new CV fits exclude that well. However, the decoder settings, model form, logistic strength, and training schedule are fixed inherited choices. Their earlier selection included the former dev split. Treat the new CV scores as controlled parser comparisons with this history. Do not call them an independent clean estimate of the entire system. No evidence in the earlier training code shows test or in-domain eval truth in the boundary or matcher fit. This is a code review statement, not proof of all past runs.','',
        'Fresh full-dev weights are saved as clean_boundary.pt and clean_matcher.npz for review. They are not loaded by S20. No protected final run was used to select between old and new weights. The orchestrator owns the final benchmark.','',
        '## Checks and limits','',
        'Nine input and contract tests pass. They cover printed words, right-cell ownership, separators, technical cells, section headings, translated cache coordinates, the S20 call, intact wide grids, and fused wide rows.','']
    runtime=json.loads((HERE/'runtime_check.json').read_text())
    lines.append(f"The deployed S20 wrapper also passed a structure check on one allowed dev window: {runtime['intervals']} intervals, {runtime['seconds']:.2f} seconds, and {runtime['peak_rss_gb']:.2f} GiB peak RSS. This check read no truth.")
    lines.append('')
    resources=[]
    for action in ('names','cv','synthetic'):
        p=HERE/(action+'_resources.json')
        if p.exists():
            m=json.loads(p.read_text());resources.append(dict(action=action,**m))
            lines.append(f"- {action}: {m['seconds']:.1f} seconds; peak RSS {m['peak_rss_gb']:.2f} GiB; at most three compute threads.")
    lines+=['',
        'Limits: the manual sample has one log per family. Some non-eval logs can share a well or template with an excluded log. The manual name check is development data. Synthetic columns do not represent all real mixtures, blank zones, scale changes, or scan faults. The small grouped external name loss is retained as measured.','',
        '## Run again','',
        'From bakeoff, set LITHOLOGY_DATASET to the local dataset path. Use the local Python environment. Set OMP_NUM_THREADS, MKL_NUM_THREADS, and OPENBLAS_NUM_THREADS to 3. The code derives all output paths from this folder.','',
        '```text',
        'python -m ml.legend_v4.experiment names',
        'python -m ml.legend_v4.experiment prepare',
        'python -m ml.legend_v4.experiment cv',
        'python -m ml.legend_v4.experiment synthetic',
        'python -m pytest ml/legend_v4/test_legend_v4.py -q -o cache_dir=ml/legend_v4/pytest_cache',
        'python -m ml.legend_v4.report',
        '```','',
        'The experiment has no final action. All writes stay in the owned lane paths. No commits, pushes, child agents, external downloads, or team service calls were made.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'summary.json').write_text(json.dumps(dict(names=name_groups,cv_totals=totals,cv_wells=well_rows,synthetic_families=family_rows,resources=resources),indent=2))
    owned=[BAKEOFF/'harness/legend_v4.py',BAKEOFF/'strategies/s20_legend_v4.py',HERE/'selected.json',HERE/'experiment.py',HERE/'test_legend_v4.py']
    owned += [HERE/'clean_boundary.pt',HERE/'clean_matcher.npz',HERE/'report.py',HERE/'check_runtime.py']
    owned += sorted((HERE/'manual_truth').glob('*.json'))
    inherited=[BAKEOFF/'ml/boundary/boundary_net.pt',BAKEOFF/'ml/match/matcher.npz',BAKEOFF/'ml/integrate/selected.json',BAKEOFF/'ml/integrate/confidence.npz']
    lock=dict(dev_windows=len(items),well_names=sorted({r['well'] for r in meta.values()}),
              dev_allowlist_sha256=hashlib.sha256((ROOT/'external_tests/eval_set_dev.json').read_bytes()).hexdigest(),
              sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in owned+inherited})
    (HERE/'PROVENANCE.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
