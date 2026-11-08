"""Record an external validator-only change without changing predictions."""
from jev_e2e import OUT,BAKE,read,save,sha
from harness import io


def migrate():
    m=read(OUT/'manifest.json');old_hash=sha(OUT/'manifest.json')
    if old_hash!=read(OUT/'lock.json')['manifest_sha256']:raise ValueError('Manifest already changed')
    validator=str((BAKE/'strategies/legend_identity.py').resolve())
    changed=[p for p,h in m['source_hashes'].items() if sha(p)!=h]
    if changed!=[validator]:raise ValueError('This migration permits only the validator change: '+repr(changed))
    records=read(OUT/'results.json')['records']
    for r in records:
        if r.get('error'):continue
        captions={e['label'] for e in read(r['diagnostics'])['legend']}|{'Unknown'}
        if any(row['lithology'] not in captions for row in io.load_rows(r['prediction_csv'])):raise ValueError('Old output needs a class-policy change')
    save(OUT/'archive/manifest-v1.json',m);save(OUT/'archive/lock-v1.json',read(OUT/'lock.json'))
    previous=m['source_hashes'][validator];m['source_hashes'][validator]=sha(validator)
    grading=str((BAKE/'strategies/grading_policy.py').resolve());m['source_hashes'][grading]=sha(grading)
    m['version']=2;m['accepted_prior_manifest_hashes']=[old_hash]
    m['policy']['grading_supplement']='Score both strict_harness and clay_shale_equivalent_v1 from the same saved rows. Approved reference-thread grading rule; contacts, clusters, depths and inference labels unchanged.'
    m['migration']=dict(reason='Reference thread changed output validator to permit approved Clay/Shale grading family. All saved output labels already equal exact source captions or Unknown; no inference result changed.',prior_manifest_sha256=old_hash,validator_before=previous,validator_after=sha(validator),retained_records=len(records))
    save(OUT/'manifest.json',m);save(OUT/'lock.json',dict(manifest_sha256=sha(OUT/'manifest.json')))
    print('MIGRATED',len(records),'unchanged predictions; strict and equivalent grading will be reported')


if __name__=='__main__':migrate()
