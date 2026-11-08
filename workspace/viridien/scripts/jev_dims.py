import json, re, sys, base64, io, urllib.request
from PIL import Image
key = next(m.group(1).strip().strip('"\'') for l in open(sys.argv[1], encoding="utf8") if (m := re.match(r"\s*JEV_API_KEY\s*=\s*(.*)", l)))
def try_(w, h, q):
    b = io.BytesIO(); Image.new("RGB", (w, h), (255, 0, 0)).save(b, "PNG")
    body = {"model": "jev-latest", "state": "img", "questions": q, "assets": {"img": {"type": "image", "format": "png", "data_base64": base64.b64encode(b.getvalue()).decode()}}}
    req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return r.read().decode()[:160]
    except urllib.error.HTTPError as e: return f"{e.code} {e.read().decode()[:120]}"
C = {"c": {"type": "choice", "instructions": "What colour is the image?", "criteria": {"red": "red", "blue": "blue"}}}
B = {"b": {"type": "box", "instructions": "the red area"}}
for w, h in [(64, 64), (224, 224), (512, 512), (1024, 1024), (512, 1024), (256, 1024), (512, 2048), (2048, 2048)]:
    print(w, h, "choice:", try_(w, h, C)[:110], "| box:", try_(w, h, B)[:110])
