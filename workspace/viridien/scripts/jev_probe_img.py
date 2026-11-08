import json, re, sys, time, base64, io, urllib.request
from PIL import Image
key = next(m.group(1).strip().strip('"\'') for l in open(sys.argv[1], encoding="utf8") if (m := re.match(r"\s*JEV_API_KEY\s*=\s*(.*)", l)))
im = Image.open("peek/col_legend.png").convert("RGB").crop((10, 40, 75, 95)); b = io.BytesIO(); im.save(b, "PNG")
b64 = base64.b64encode(b.getvalue()).decode()
def call(state, tag):
    body = {"model": "jev-latest", "state": state, "questions": {"lith": {"type": "choice", "instructions": "What rock does the swatch image show? Look at the image.",
            "criteria": {"Coarse sandstone": "yellow with large black dots", "Bituminous limestone": "cyan with brick lines and black blobs", "Clay Shale": "green with black dashes"}}}}
    req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r: out = r.read().decode()
    except urllib.error.HTTPError as e: out = f"HTTP {e.code}: {e.read().decode()[:600]}"
    print(tag, f"{time.time()-t:.1f}s", out[:600].replace(key, "<k>"))
call({"swatch": {"type": "image", "media_type": "image/png", "data": b64}}, "A")
call({"swatch_png_base64": b64}, "B")
call({"swatch": f"data:image/png;base64,{b64}"}, "C")
