"""Disposable label store for verification UI checks; the real server stays running."""
import shutil
import sys
import threading
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import DATA, Images, Store, read, write
from background import Background
from review_media import ReviewMedia
import review_api
from server import create_handler

root=DATA/('qa-verification-'+uuid.uuid4().hex[:8]);root.mkdir()
assert root.resolve().is_relative_to(DATA.resolve())
catalog=read(DATA/'documents.json');write(root/'documents.json',catalog)
for name in ['seeds','reviews']:
    (root/name).mkdir()
    for p in (DATA/name).glob('*.json'):shutil.copyfile(p,root/name/p.name)
shutil.copyfile(DATA/'pattern-queue.json',root/'pattern-queue.json')
review_api.Background=lambda q,f:Background(q,f,autostart=False)
original=ReviewMedia.__init__
def init(self,*args,**kwargs):
    original(self,*args,**kwargs);self.cache=DATA/'review-crops'
ReviewMedia.__init__=init
handler=create_handler(Store(root),Images(),'http://127.0.0.1:8777')
class QAHandler(handler):
    def get(self):
        if self.path=='/qa/stop':
            self.respond(dict(ok=True));threading.Thread(target=self.server.shutdown,daemon=True).start();return
        return super().get()
server=ThreadingHTTPServer(('127.0.0.1',8777),QAHandler)
timer=threading.Timer(240,server.shutdown);timer.daemon=True;timer.start()
print('Verification QA on 8777; isolated store:',root.name,flush=True)
try:server.serve_forever()
finally:timer.cancel();server.server_close()
