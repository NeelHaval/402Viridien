"""Require the owner's configured local-proxy grade and its independent replay."""
from __future__ import annotations
import math
from pathlib import Path
from report_inputs import digest

QUALITY=('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft')


def require_policy(raw,audit,bakeoff):
    from strategies import grading_policy
    source=Path(grading_policy.__file__).resolve();scorer=Path(bakeoff)/'harness'/'score.py'
    expected=dict(id=grading_policy.POLICY_ID,source_path=str(source),source_sha256=digest(source),equivalent_labels=list(grading_policy.EQUIVALENT_LABELS),canonical_label='Clay/Shale',shared_scorer_path=str(scorer.resolve()),shared_scorer_sha256=digest(scorer),owner_configured=True,official_challenge_rule_verified=False)
    meta=raw.get('grading_policy',{})
    for key,value in expected.items():
        if meta.get(key)!=value:raise ValueError('Grading policy metadata mismatch: '+key)
    for rec in raw['records']:
        for scope,target in [('primary',rec),('pixel',rec.get('pixel_only'))]:
            if target is None:continue
            metrics=target.get('accepted_metrics',{})
            if set(metrics)!=set(QUALITY):raise ValueError('Missing accepted '+scope+' metric series')
            for key in QUALITY:
                value=metrics[key]
                if value is None and key=='depth_mae_ft':continue
                if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Nonfinite accepted metric')
            if abs(metrics['score']-metrics['row_acc']*metrics['boundary_f1'])>1e-10:raise ValueError('Accepted score product mismatch')
            for key in ('boundary_f1','depth_mae_ft'):
                a,b=target.get(key),metrics[key]
                if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-10):raise ValueError('Accepted grade changed contacts or depths')
            if scope=='primary' and rec.get('error'):
                if any(metrics[k]!=0 for k in QUALITY if k!='depth_mae_ft') or metrics['depth_mae_ft'] is not None:raise ValueError('Accepted failed record is nonzero')
    if audit.get('grading_policy_id')!=expected['id'] or audit.get('grading_policy_source_sha256')!=expected['source_sha256']:raise ValueError('Accepted replay policy binding is stale')
    if audit.get('accepted_rescored') is None or audit.get('accepted_rescored')!=audit.get('rescored'):raise ValueError('Accepted primary replay coverage differs from strict')
    if audit.get('accepted_pixel_diagnostics') is None or audit.get('accepted_pixel_diagnostics')!=audit.get('pixel_diagnostics_rescored'):raise ValueError('Accepted pixel replay coverage differs from strict')
    return dict(status='valid',grading_policy_id=expected['id'],grading_policy_source_sha256=expected['source_sha256'],accepted_rescored=audit['accepted_rescored'],accepted_pixel_diagnostics=audit['accepted_pixel_diagnostics'])
