"""Write a portable visual report of the 25-example review snapshot."""
import base64,html,json,sys
from pathlib import Path
from model import Store,Images,DATA
from review_queue import ReviewQueue
from review_media import ReviewMedia

def main():
    root=DATA/'quality-audit';rows=json.loads((root/'accepted-audit.json').read_text(encoding='utf-8'))
    media=ReviewMedia(ReviewQueue(Store()),Images());media.feedback.refresh()
    flags={r['id']:r for r in media.feedback.flagged};cards=[]
    for n,row in enumerate(rows,1):
        doc=media.queue.document(row['document_id']);entries=[e for e in doc['legend'] if e['id'] in row['labels']]
        source='data:image/png;base64,'+base64.b64encode((root/(row['id']+'.png')).read_bytes()).decode('ascii')
        legends=''.join('<figure><img src="'+media.image(doc,e['box'])+'"><figcaption>'+html.escape(e['label_printed'])+'</figcaption></figure>' for e in entries)
        link='http://127.0.0.1:8776/editor?doc='+doc['id']+'&segment='+row['id']
        flag=flags.get(row['id']);status='Needs another check' if flag else 'Broad visual agreement'
        cards.append(f'<article class="{"flagged" if flag else ""}"><h2>{n}. {html.escape(" / ".join(row["label_names"]))}</h2><p>{html.escape(doc["name"])}</p><strong>{status}</strong><div class="pair"><figure><img src="{source}"><figcaption>Your accepted box</figcaption></figure>{legends}</div><p>{html.escape(flag["reason"]) if flag else "Checked visually against the selected legend symbol. This is not independent ground truth."}</p><a href="{link}">Open this saved box in the editor</a></article>')
    body='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Saved label check</title><style>body{font:15px/1.6 system-ui;color:#23344c;background:#edf1f6;margin:0;padding:28px}main{max-width:1200px;margin:auto}h1{font-size:28px}h2{font-size:18px;margin:0}p{overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}article{padding:20px;background:white;border:1px solid #ccd7e5;border-radius:6px}article.flagged{border:2px solid #a26621}.pair{display:flex;gap:24px;align-items:center;flex-wrap:wrap;padding:15px 0}figure{margin:0}img{max-width:200px;max-height:240px;image-rendering:pixelated}figcaption{font-size:12px;color:#52657b}a{color:#1c5baa}</style><main><h1>Check of your five completed sets</h1><p>This snapshot contains 25 accepted boxes. Twenty-four look broadly consistent with their chosen legend symbols. One yellow dotted example is labelled Shale while its selected legend shows green lines. It is still saved as accepted, but is withheld from learning.</p><p>All 25 boxes were changed from the original 48-row seeds. Seven final label selections differ: six were initially blank, and one changed from Silty Sandstone to Shale. The old cluster grouping is based on visual similarity, so one completed set does not prove that five examples have the same rock type.</p><p>The new system uses accepted examples and each document's legend. It keeps label selection manual, and grows new seed boxes within strong colour changes and ruling lines. In a replay of the 25 original seeds, with all training examples from the test document excluded, mean box overlap with the saved boxes rose from 0.35 to 0.59. Some saved boxes moved to another layer; this is a small development check, not a general accuracy score.</p><p>Callaway's five merged machine choices have been replaced by 32 separate source-checked choices. The five old IDs remain for saved work. All source images below are included in this report. Editor links need the local service.</p><div class="grid">'''+''.join(cards)+'</div></main>'
    target=root/'report.html';target.write_text(body,encoding='utf-8');print('Saved report:',target);print('Checked:',len(rows),'flagged:',sum(r['id'] in flags for r in rows))

if __name__=='__main__':main()
