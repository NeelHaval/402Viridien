import json, re, sys, time, urllib.request
key = next(m.group(1).strip().strip('"\'') for l in open(sys.argv[1], encoding="utf8") if (m := re.match(r"\s*JEV_API_KEY\s*=\s*(.*)", l)))
def call(body):
    req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r: out = r.read().decode()
    except urllib.error.HTTPError as e: out = f"HTTP {e.code}: {e.read().decode()[:800]}"
    print(f"{time.time()-t:.1f}s", out[:1500].replace(key, "<k>"))
state = {"interval": {"mean_rgb": [250, 240, 20], "dark_fraction": 0.18, "pattern": "isolated round dots on a regular grid"}}
call({"model": "jev-latest", "state": state, "questions": {"lith": {"type": "choice", "instructions": "Which legend swatch does this interval fill match?",
      "criteria": {"Sandstone": "yellow fill with black dots", "Clay Shale": "green fill with black dashes", "Limestone": "cyan fill with brick lines"}}}})
