"""Clone current saved work for checks. Test edits never reach real review files."""
import shutil,sqlite3,sys,argparse,threading
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import DATA,Store,Images
from server import create_handler
root=DATA/'qa-feedback';root.mkdir(exist_ok=True)
for name in ('documents.json','pattern-queue.json','feedback-features.json'):
    if (DATA/name).exists():shutil.copyfile(DATA/name,root/name)
for folder in ('seeds','reviews','review-crops'):
    target=root/folder;target.mkdir(exist_ok=True)
    for file in (DATA/folder).glob('*'):
        if file.is_file():shutil.copyfile(file,target/file.name)
with sqlite3.connect(DATA/'shared-reviews.sqlite3') as src,sqlite3.connect(root/'shared-reviews.sqlite3') as dst:src.backup(dst)
print('Feedback test server: http://127.0.0.1:8777',flush=True)
parser=argparse.ArgumentParser();parser.add_argument('--lan');args=parser.parse_args()
handler=create_handler(Store(root),Images(),'http://'+(args.lan or '127.0.0.1')+':8777')
if args.lan:
    lan=ThreadingHTTPServer((args.lan,8777),handler)
    threading.Thread(target=lan.serve_forever,daemon=True).start()
ThreadingHTTPServer(('127.0.0.1',8777),handler).serve_forever()
