"""Reproducible, read-only transfer analysis of fixed predictions.
No fitting, threshold selection, external-test access, or API use.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='3'
from pathlib import Path
import sys
import json
import csv
import hashlib
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau

HERE = Path(__file__).resolve().parent
BAKE = HERE.parents[1]
REPO = BAKE.parent
os.environ['MPLCONFIGDIR'] = str(HERE/'mplconfig')
sys.path.insert(0, str(BAKE))
from harness import io, score
METRICS = ['row_acc', 'boundary_f1', 'cluster_ari', 'score']
PREFIXES = ['b01', 's01', 's02', 's03', 's04', 's05', 's07', 's11', 's12']
NBOOT = 5000
SEED = 20261108

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

def sid(prefix):
    return next((BAKE/'strategies').glob(prefix+'*.py')).stem

def load_data():
    dev = json.loads((REPO/'external_tests/eval_set_dev.json').read_text())
    allowed = {i['log'] for i in dev}
    # Filter manifest immediately; no test truth or test result file is opened.
    with (REPO/'external_tests/manifest.csv').open(newline='') as f:
        manifest = {r['id']: r for r in csv.DictReader(f) if r['id'] in allowed}
    items = json.loads((BAKE/'eval_set.json').read_text())
    records, audit = [], []
    for prefix in PREFIXES:
        strategy = sid(prefix)
        original = BAKE/'out'/strategy
        archived = json.loads((original/'metrics.json').read_text())
        # Recompute every strategy from saved predictions under the same current scorer.
        for item in items:
            name = item['log']
            predpath = original/(name+'.csv')
            pred = io.load_rows(predpath)
            truth = io.load_rows(BAKE/'gt/silver/B'/(name+'.csv'))
            if truth is None or pred is None:
                raise RuntimeError(f'Missing archived prediction/truth: {strategy}, {name}')
            m = score.score(pred, truth, (item['y0'], item['y1']))
            records.append(dict(domain='viridien', strategy=prefix, log=name, key=name, source='viridien', well=name, truth_filled_rows=int(np.sum(score.raster(truth,(item['y0'],item['y1']))!='')), **m))
        audit.append(dict(strategy=prefix, archived_gt=archived['gt'], archived_n_scored=archived['mean']['n_scored'], rescored_n=13))
        synpath = HERE/'runtime/out'/strategy/'synthetic/metrics.json'
        if not synpath.exists():synpath=original/'synthetic/metrics.json'
        if synpath.exists():
            syn = json.loads(synpath.read_text())
            audit[-1].update(synthetic_n_windows=len(syn['logs']),synthetic_results=str(synpath.relative_to(REPO)))
            for r in syn['logs']:
                if all(k in r for k in METRICS):
                    records.append(dict(domain='synthetic', strategy=prefix, log=r['log'], key=r['key'], source='synthetic', well=r['log'], **{k:r[k] for k in METRICS}))
        extpath = HERE/'runtime/out'/strategy/'ext_dev/metrics.json'
        if not extpath.exists():
            continue
        ext = json.loads(extpath.read_text())
        if len(ext['logs']) != len(dev):
            raise RuntimeError(f'Incomplete run {prefix}: {len(ext["logs"])}/{len(dev)}')
        errors = Counter(r.get('error', '').split(':')[0] for r in ext['logs'] if 'error' in r)
        audit[-1].update(external_errors=dict(errors), n_external=len(ext['logs']), peak_rss_gb=max(r.get('peak_rss_gb',0) for r in ext['logs']))
        # Environment failure is not algorithmic evidence. Keep raw failures, exclude from ranking.
        dependency_error = any(any(token in r.get('error','') for token in ['ModuleNotFoundError','LocalEntryNotFoundError','cached files','OSError:','FileNotFoundError','ConnectionError','download']) for r in ext['logs'])
        audit[-1]['dependency_failure'] = dependency_error
        if dependency_error:
            continue
        for r, item in zip(ext['logs'], dev):
            assert r['log'] == item['log']
            meta = manifest[r['log']]
            m = {k:r.get(k,0.) for k in METRICS}
            records.append(dict(domain='ext_dev', strategy=prefix, log=r['log'], key=r.get('key',r['log']), source=meta['source'], well=meta['source']+':'+meta['well'], **m))
    frame = pd.DataFrame(records)
    frame.to_csv(HERE/'per_window_metrics.csv', index=False)
    dump(HERE/'audit.json', audit)
    return frame, audit, manifest

def corr_rows(x,y):
    """Vectorised Spearman and Kendall tau-b, retaining average-rank ties."""
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    rx, ry = rankdata(x,axis=1), rankdata(y,axis=1)
    rx -= rx.mean(axis=1,keepdims=True); ry -= ry.mean(axis=1,keepdims=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        rho = (rx*ry).sum(axis=1)/np.sqrt((rx*rx).sum(axis=1)*(ry*ry).sum(axis=1))
        i,j=np.triu_indices(x.shape[1],1)
        sx=np.sign(x[:,i]-x[:,j]);sy=np.sign(y[:,i]-y[:,j])
        tau=(sx*sy).sum(axis=1)/np.sqrt((sx*sx).sum(axis=1)*(sy*sy).sum(axis=1))
    return rho,tau

def boot_means(df, strategies, metric, rng, cluster='log', draws=None):
    groups = sorted(df[cluster].unique())
    sums=[]; counts=[]
    for group in groups:
        sub=df[df[cluster]==group].groupby('strategy')[metric]
        sums.append(sub.sum().reindex(strategies).to_numpy())
        counts.append(sub.count().reindex(strategies).to_numpy())
    sums, counts = np.array(sums), np.array(counts)
    assert np.isfinite(sums).all() and (counts>0).all()
    if draws is None:draws=rng.integers(len(groups),size=(NBOOT,len(groups)))
    return sums[draws].sum(axis=1)/counts[draws].sum(axis=1),draws

def correlation(df, domain, source=None, cluster='log', common=None):
    a=df[df.domain==domain]
    if source:a=a[a.source==source]
    b=df[df.domain=='viridien']
    strategies=sorted(set(a.strategy)&set(b.strategy))
    if common is not None:strategies=sorted(set(strategies)&set(common))
    a=a[a.strategy.isin(strategies)];b=b[b.strategy.isin(strategies)]
    rng=np.random.default_rng(SEED)
    rows=[]
    for metric in METRICS:
        am=a.groupby('strategy')[metric].mean().reindex(strategies).to_numpy()
        bm=b.groupby('strategy')[metric].mean().reindex(strategies).to_numpy()
        rho,tau=corr_rows(am,bm)
        ab,draws=boot_means(a,strategies,metric,rng,cluster)
        # Synthetic parents are the SAME 13 logs: paired parent-log resampling.
        same_parents=domain=='synthetic' and sorted(a.log.unique())==sorted(b.log.unique())
        bb,_=boot_means(b,strategies,metric,rng,'log',draws if same_parents else None)
        rb,tb=corr_rows(ab,bb)
        rec=dict(comparison=domain,source=source or 'all',bootstrap_unit=cluster,metric=metric,n_strategies=len(strategies),strategies=strategies,n_logs=len(a.log.unique()),n_wells=len(a.well.unique()),spearman=float(rho[0]),kendall=float(tau[0]))
        for label,values in [('spearman',rb),('kendall',tb)]:
            good=values[np.isfinite(values)]
            rec[label+'_lo'],rec[label+'_hi']=(map(float,np.percentile(good,[2.5,97.5])) if len(good) else (None,None))
            rec[label+'_valid_bootstraps']=len(good)
        rows.append(rec)
    return rows

def analyse():
    df,audit,manifest=load_data()
    external_strategies=sorted(df[df.domain=='ext_dev'].strategy.unique())
    synthetic_strategies=sorted(df[df.domain=='synthetic'].strategy.unique())
    common=sorted(set(external_strategies)&set(synthetic_strategies))
    correlations=[]
    for domain in ['ext_dev','synthetic']:
        correlations += correlation(df,domain)
        correlations += [dict(r, comparison=domain+'_common') for r in correlation(df,domain,common=common)]
    correlations += correlation(df,'ext_dev',cluster='well')
    balanced=df[df.domain=='ext_dev'].groupby(['well','strategy'])[METRICS].mean().reset_index()
    balanced['domain']='ext_dev';balanced['log']=balanced['well'];balanced['source']='balanced'
    balanced_df=pd.concat([df[df.domain!='ext_dev'],balanced],ignore_index=True)
    correlations += [dict(r,comparison='ext_dev_equal_well') for r in correlation(balanced_df,'ext_dev',cluster='well')]
    for source in ['kgs','idaho','nl','utah']:
        correlations += correlation(df,'ext_dev',source=source)
    image_stats=pd.read_csv(HERE/'image_statistics.csv')
    target=image_stats[(image_stats.domain=='viridien_all') & image_stats.in_eval]
    # Descriptive image-layout strata, never used to select a model or threshold.
    family_names={}
    for row in target.itertuples():
        family=('monochrome hatch' if row.one_bit else 'blank column' if row.ink_share<.05 else 'high-resolution colour' if row.px_per_ft>=5 else 'narrow low-resolution colour')
        family_names[row.log]=family
    family_rows=[]
    for family in sorted(set(family_names.values())):
        selected={name for name,assigned in family_names.items() if assigned==family}
        sub=df[(df.domain!='viridien') | df.log.isin(selected)]
        for source in [None,'kgs','idaho','nl','utah']:
            for r in correlation(sub,'ext_dev',source=source):
                family_rows.append(dict(target_family=family,n_target_logs=len(selected),**r))
    pd.DataFrame(family_rows).to_csv(HERE/'family_correlations.csv',index=False)
    dump(HERE/'target_families.json',family_names)
    c=pd.DataFrame(correlations)
    c.to_csv(HERE/'correlations.csv',index=False)
    dump(HERE/'correlations.json',correlations)
    means=df.groupby(['domain','strategy'])[METRICS].mean().reset_index()
    means.to_csv(HERE/'strategy_means.csv',index=False)
    df[df.domain=='ext_dev'].groupby(['source','strategy'])[METRICS].mean().to_csv(HERE/'source_means.csv')
    sensitivity=[]
    ext=df[df.domain=='ext_dev']
    for well in sorted(ext.well.unique()):
        sub=df[(df.domain!='ext_dev') | (df.well!=well)]
        for r in correlation(sub,'ext_dev'):
            sensitivity.append(dict(omitted_well=well,**r))
    pd.DataFrame(sensitivity).to_csv(HERE/'leave_one_well_out.csv',index=False)
    nonblank=df[(df.domain!='viridien') | (df.truth_filled_rows>0)]
    pd.DataFrame(correlation(nonblank,'ext_dev')).to_csv(HERE/'nonblank_target_correlations.csv',index=False)
    input_paths=list((BAKE/'strategies').glob('*.py'))+[BAKE/'eval_set.json',REPO/'external_tests/eval_set_dev.json']
    input_paths += list((REPO/'external_tests/truth').joinpath(name+'.csv') for name in manifest)
    input_paths += [BAKE/'gt/silver/B'/(name+'.csv') for name in df[df.domain=='viridien'].log.unique()]
    for prefix in PREFIXES:
        folder=BAKE/'out'/sid(prefix)
        input_paths += [folder/'metrics.json'] + [folder/(name+'.csv') for name in df[df.domain=='viridien'].log.unique()]
        if (folder/'synthetic/metrics.json').exists():input_paths += [folder/'synthetic/metrics.json']
    input_paths += list((HERE/'runtime/out').glob('*/ext_dev/metrics.json'))
    input_paths += list((HERE/'runtime/out').glob('*/synthetic/metrics.json'))
    input_paths += list((HERE/'frozen/harness').glob('*.py'))
    input_paths += list(HERE.glob('*.py'))
    dump(HERE/'provenance.json',dict(seed=SEED,bootstrap_replicates=NBOOT,inputs={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths if p.exists()}))
    charts(df,c,means)
    return df,c,means,audit

def charts(df,c,means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(4,1,figsize=(6,14),layout='constrained')
    colors={'ext_dev':'#0072B2','synthetic':'#D55E00'}
    common=sorted(set(df[df.domain=='ext_dev'].strategy)&set(df[df.domain=='synthetic'].strategy))
    for ax,metric in zip(axes,METRICS):
        for domain in ['ext_dev','synthetic']:
            r=c[(c.comparison==domain+'_common') & (c.source=='all') & (c.bootstrap_unit=='log') & (c.metric==metric)].iloc[0]
            y=0 if domain=='ext_dev' else 1
            ax.plot([r.spearman_lo,r.spearman_hi],[y,y],color=colors[domain],lw=3)
            ax.scatter([r.spearman],[y],color=colors[domain],s=70,zorder=3)
        ax.axvline(0,color='#999999',lw=1)
        ax.set(xlim=(-1.03,1.03),ylim=(-.7,1.7),yticks=[0,1],yticklabels=['Outside dev','Synthetic'],title=metric,xlabel='Spearman rank correlation with Viridien')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle(f'Rank transfer, same {len(common)} strategies\n95% log-bootstrap intervals',fontsize=15)
    fig.savefig(HERE/'rank_correlations.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(6,17),layout='constrained')
    pivot=means.set_index(['domain','strategy'])
    for ax,metric in zip(axes,METRICS):
        for domain in ['ext_dev','synthetic']:
            for s in common:
                x=pivot.loc[('viridien',s),metric];y=pivot.loc[(domain,s),metric]
                ax.scatter(x,y,c=colors[domain],s=35)
                ax.annotate(s,(x,y),xytext=(4,3),textcoords='offset points',fontsize=9,color=colors[domain])
        ax.set(xlabel='Viridien mean',ylabel='Proxy mean',title=metric)
        ax.grid(alpha=.2)
    axes[0].scatter([],[],c=colors['ext_dev'],label='Outside dev');axes[0].scatter([],[],c=colors['synthetic'],label='Synthetic');axes[0].legend(loc='best')
    fig.savefig(HERE/'strategy_scatter.png',dpi=160);plt.close(fig)
    src=df[df.domain=='ext_dev'].groupby(['source','strategy'])[METRICS].mean()
    fig,axes=plt.subplots(4,1,figsize=(6,17),layout='constrained')
    palette=['#0072B2','#D55E00','#009E73','#CC79A7']
    strategies=sorted(df[df.domain=='ext_dev'].strategy.unique())
    for ax,metric in zip(axes,METRICS):
        for j,source in enumerate(['kgs','idaho','nl','utah']):
            vals=src.loc[source,metric].reindex(strategies)
            ax.scatter(vals,np.arange(len(strategies))+(j-1.5)*.15,color=palette[j],s=35,label=source)
        ax.set(yticks=range(len(strategies)),yticklabels=strategies,xlabel='Mean '+metric,title=metric)
        ax.grid(axis='x',alpha=.2)
    axes[0].legend(ncols=2)
    fig.savefig(HERE/'source_scores.png',dpi=160);plt.close(fig)
    family=pd.read_csv(HERE/'family_correlations.csv')
    family=family[(family.source=='all') & (family.target_family!='blank column')]
    fig,axes=plt.subplots(4,1,figsize=(6,15),layout='constrained')
    for ax,metric in zip(axes,METRICS):
        part=family[family.metric==metric].sort_values('target_family')
        for y,row in enumerate(part.itertuples()):
            ax.plot([row.spearman_lo,row.spearman_hi],[y,y],color='#0072B2',lw=3)
            ax.scatter(row.spearman,y,color='#0072B2',s=45,zorder=3)
        labels=[r.target_family.replace('high-resolution','High-res').replace('narrow low-resolution','Narrow low-res').replace('monochrome hatch','Monochrome')+' (n='+str(r.n_target_logs)+')' for r in part.itertuples()]
        ax.set(yticks=range(len(part)),yticklabels=labels,xlim=(-1.03,1.03),xlabel='Spearman correlation',title=metric)
        ax.axvline(0,color='#999999',lw=1);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Outside dev versus target layout families\nExploratory; 95% log-bootstrap intervals',fontsize=14)
    fig.savefig(HERE/'family_correlations.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    analyse()
