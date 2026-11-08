import json
for i in range(6):
    p=f'data/genlab/jobs/review/lane{i}.json'; L=json.load(open(p))
    for it in L:
        if it.get('donor'): it['donor']={k:it['donor'][k] for k in ('doc_id','legend_id','label','match')}
        else: it.pop('donor',None)
    json.dump(L,open(p,'w'),indent=1,ensure_ascii=False)
    docs=sorted({it['doc_id'] for it in L})
    t=open('data/genlab/jobs/review/template.md',encoding='utf-8').read()
    t=t.replace('{I}',str(i)).replace('{N}',str(len(L))).replace('{D}',str(len(docs))).replace('{DOCS}',', '.join(docs))
    open(f'data/genlab/jobs/review/lane{i}.md','w',encoding='utf-8').write(t)
