import json, re, sys, time, base64, io, urllib.request
from PIL import Image
key = next(m.group(1).strip().strip('"\'') for l in open(sys.argv[1], encoding="utf8") if (m := re.match(r"\s*JEV_API_KEY\s*=\s*(.*)", l)))
im = Image.open("peek/col_legend.png").convert("RGB").crop((10, 40, 75, 95)); b = io.BytesIO(); im.save(b, "PNG")
b64 = base64.b64encode(b.getvalue()).decode()
Q = {"lith": {"type": "choice", "instructions": "What rock does the attached swatch image show?",
     "criteria": {"Coarse sandstone": "yellow with large black dots", "Bituminous limestone": "cyan with brick lines and black blobs", "Clay Shale": "green with black dashes"}}}
def call(body, tag, url="https://api.typesafe.ai/v1/systemone", method="POST"):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method=method)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r: out = r.read().decode()
    except urllib.error.HTTPError as e: out = f"HTTP {e.code}: {e.read().decode()[:500]}"
    except Exception as e: out = f"ERR {e}"
    print(tag, f"{time.time()-t:.1f}s", out[:500].replace(key, "<k>"))
src = {"type": "base64", "media_type": "image/png", "data": b64}
call({"model": "jev-latest", "state": {}, "images": [src], "questions": Q}, "top-images")
call({"model": "jev-latest", "state": {}, "images": [{"type": "image", "source": src}], "questions": Q}, "top-images-anth")
call({"model": "jev-latest", "state": {"swatch": {"type": "image", "source": src}}, "questions": Q}, "state-anth")
call({"model": "jev-latest", "state": {}, "attachments": [{"name": "swatch", "mime_type": "image/png", "data": b64}], "questions": Q}, "attachments")
call({"model": "jev-latest", "state": {}, "questions": {"box": {"type": "bbox", "instructions": "Box the swatch", "image": src}}}, "bbox-q")
for u in ["https://api.typesafe.ai/v1/models", "https://api.typesafe.ai/openapi.json", "https://api.typesafe.ai/docs", "https://api.typesafe.ai/v1"]:
    call(None, "GET " + u, u, "GET")
