"""artifact/index.html -> artifact/lithology_explainer.html: images embedded as WebP, theme colours defined."""
import base64, io, re, pathlib
from PIL import Image
A = pathlib.Path(__file__).resolve().parent.parent / "artifact"
t = (A / "index.html").read_text(encoding="utf8")
THEME = """:root{--background:#fff;--foreground:#1f2328;--muted-foreground:#59636e;--card:#f6f8fa;--border:#d1d9e0;
--code-background:#eef1f4;--radius:8px;--success:#1a7f37;--destructive:#cf222e;--font-sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--font-mono:ui-monospace,Consolas,monospace;--chart-1:#2e7d32;--chart-2:#1565c0;--chart-3:#f9a825;--chart-4:#8e24aa;--chart-5:#00838f}
body{padding:16px;background:var(--background)}"""
t = t.replace("<style>", "<style>\n" + THEME, 1)
t = t.replace("<body><div class=\"wrap\">", "<body><div class=\"wrap\"><h1 style=\"font-size:26px;margin:8px 0 0\">Decoding the subsurface: methods, boxes and trade-offs</h1>", 1)
cache = {}
def emb(m):
    p = m.group(1)
    if p not in cache:
        im = Image.open(p).convert("RGB")
        if im.width > 1100: im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "WEBP", quality=82)
        cache[p] = "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()
    return f'src="{cache[p]}"'
t = re.sub(r'src="(C:/[^"]+)"', emb, t)
out = A / "lithology_explainer.html"; out.write_text(t, encoding="utf8")
print(out, round(out.stat().st_size / 1e6, 2), "MB,", len(cache), "images,", "C:/" in t and "PATH LEFT" or "no local paths left")
