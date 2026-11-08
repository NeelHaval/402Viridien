"""Record the current report and local grading source hashes, outside inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
PROJECT = BAKEOFF.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / 'report_source_freeze.json'
    previous = HERE / 'report_source_freeze_astra_checked.json'
    if not previous.exists():
        previous.write_bytes(target.read_bytes())
    names = ('build_report.py', 'report_inputs.py', 'report_integrity.py',
             'report_policy_gate.py', 'report_diagnostics.py', 'rescore_policy.py', 'audit_saved_predictions.py',
             'audit_run_integrity.py', 'reviewed_dev_wells.json', 'show_log06.py',
             'CHALLENGE_ALIGNMENT.md', 'scan_readiness.py', 'final_overview.py',
             'run_final_audits.py', 'freeze_report_sources.py', 'CLAY_SHALE_FIX.md',
             'CLAY_SHALE_FIX_MEASURED.md', '../../strategies/grading_policy.py',
             '../../strategies/legend_identity.py', '../../strategies/LEGEND_IDENTITY.md',
             '../../strategies/AGENTS.md', '../../strategies/challenge_export.py')
    pins = {name: sha(HERE / name) for name in names}
    target.write_text(json.dumps(pins, indent=2) + '\n', encoding='utf-8')
    challenge = PROJECT / 'viridien_challenge' / 'Viridien_HTB_challenge'
    sources = (challenge / 'Viridien_HackTheBurgh_Presentation.pptx',
               challenge / 'hack27' / 'HTB_Challenge_Slides.pptx',
               challenge / 'hack27' / 'project' / 'example_notebook.ipynb')
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        owner_policy='clay_shale_equivalent_v1',
        equivalent_labels=['Clay', 'Shale', 'Clay/Shale', 'Clay-Shale'],
        rule='Symmetric full label credit; preserve raw captions, every contact, depths and cluster IDs.',
        strict_reference='Unchanged harness.score.score; original metric scalars retained.',
        local_proxy='Per-window row accuracy times boundary F1; equal-log mean of products.',
        official_formula_verified=False, official_equivalence_verified=False,
        official_criteria='Correctness and efficiency; generalisation, automation, limitations and resources.',
        astra_scope='The two completed reviews predate this owner-configured grading policy. No new review was run.',
        challenge_source_sha256={str(path.relative_to(PROJECT)): sha(path) for path in sources},
        grading_source_sha256=pins['../../strategies/grading_policy.py'],
        shared_scorer_sha256=sha(BAKEOFF / 'harness' / 'score.py'),
        report_freeze_sha256=sha(target),
        raw_outputs_modified=False, inference_rerun_for_policy=False)
    (HERE / 'grading_policy_protocol.json').write_text(
        json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    print(f'Pinned {len(pins)} report/grading files and {len(sources)} challenge sources; inference manifest unchanged')


if __name__ == '__main__':
    main()
