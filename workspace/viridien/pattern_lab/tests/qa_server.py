"""Start an isolated copy for browser checks. User reviews are not changed."""
import shutil
import sys
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import DATA,Store,Images,read,write
from server import create_handler
root=DATA/'qa-final'
root.mkdir(exist_ok=True)
catalog=read(DATA/'documents.json')[:2]
write(root/'documents.json',catalog)
for doc in catalog:
    target=root/'seeds'/(doc['id']+'.json')
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(DATA/'seeds'/(doc['id']+'.json'),target)
print('Isolated browser test server: http://127.0.0.1:8777',flush=True)
ThreadingHTTPServer(('127.0.0.1',8777),create_handler(Store(root),Images())).serve_forever()
