"""Fill artifact/explainer.html placeholders -> artifact/index.html (images by absolute path)."""
import json, html, pathlib
W = pathlib.Path(__file__).resolve().parent.parent
A = W / "artifact"; IMG = (A / "img").as_posix(); HARD = (W / "figs" / "hard").as_posix()
man = json.loads((A / "manifest.json").read_text(encoding="utf8")) if (A / "manifest.json").exists() else {}
t = (A / "explainer.html").read_text(encoding="utf8")

def has(f): return (A / "img" / f).exists()
def cap(f, d=""): return html.escape(str(man.get(f, {}).get("caption", d)))

boxes = "".join(f'<div class="box"><img src="{IMG}/{f.name}"><div class="cap">{html.escape(man.get(f.name, {}).get("page", f.stem)[:60])}</div></div>'
                for f in sorted((A / "img").glob("f03_boxes_*.png")))
legs = "".join(f'<img src="{IMG}/{f.name}" style="margin-top:8px">' for f in sorted((A / "img").glob("f05_legend_parse_*.png"))[:2])

S = [
 ("S1", "Colour runs", "cpu", "CPU", "d_s1_colour_runs.png",
  "Most colour logs paint each rock a flat colour. Read the dominant colour of every pixel row and cut where it changes for long enough.",
  """col = rgb[:, int(.15*w):int(.85*w)]          # skip the borders
ink = col.max(2) < 60; paper = col.min(2) > 230
dom = mode_colour(col, mask=~ink & ~paper)       # one colour per row
cuts = changes(quantise_lab(dom), min_run=15)
name = nearest_swatch_colour(cluster_colour)""",
  "The baseline every other method must beat. Milliseconds a log, no model.",
  "Fastest; exact boundaries on flat fills", "Blind on the 9 black-and-white logs; same-colour rocks merge"),
 ("S2", "Hand-made features + change points", "cpu", "CPU", "d_s2_features_pelt.png",
  "Describe each row by ink fraction, horizontal and vertical stroke density, colour and pattern period, then let PELT find where that signal shifts.",
  """f = np.stack([ink, sobel_y_density, sobel_x_density,
              lab_mean, autocorr_period], 1)
f = uniform_filter1d(f, 40, axis=0)
cuts = ruptures.Pelt(model="rbf").fit(zscore(f)).predict(pen=8)
clusters = AgglomerativeClustering(distance_threshold=1.5).fit(seg_means)""",
  "Works on colour and hatch-only logs alike, and its segment() is reused by S8, S9 and S10 for boundaries.",
  "Good boundaries everywhere; explainable", "Naming two similar hatches is weak"),
 ("S3", "Template matching", "cpu", "CPU", "d_s3_template_ncc.png",
  "Cut each legend swatch, slide it down the column, score every tile with normalised cross-correlation and keep the best match.",
  """for e in rock_swatches:
    tpl = tile(strip_border(e.swatch), h=64)
    for s in (0.8, 1.0, 1.25):
        score[e.id] = max(score[e.id],
            cv2.matchTemplate(column, rescale(tpl, s), cv2.TM_CCOEFF_NORMED))
label = argmax(score, axis=0); merge_runs(label)""",
  "The most literal reading of the task: the legend is the answer key, so match against it.",
  "Names come straight from the legend; good on 1-bit hatches", "Breaks when the swatch is drawn at another scale or phase"),
 ("S4", "k-means + Viterbi smoothing", "cpu", "CPU", "d_s4_kmeans_viterbi.png",
  "Cut the column into 20-row tiles, cluster their texture features, then smooth along depth so one odd tile cannot create a fake interval.",
  """X = [lab_hist(t) + lbp_hist(t) + [ink(t)] for t in tiles]
k = best_silhouette(X, range(2, 11))
raw = KMeans(k).fit_predict(X)
clean = viterbi(raw, p_stay=0.98)     # long runs win
cuts = refine_to_row(clean)""",
  "Clustering is half the score (cluster_id), and smoothing removes the speckle that ruins boundary F1.",
  "Robust to noise; no legend needed to group", "Must guess k; naming is a separate step"),
 ("S5", "DINOv2 embeddings", "gpu", "GPU", "d_s5_dinov2.png",
  "A self-supervised vision model turns each 64-row tile into a vector. Big jumps between neighbouring vectors are boundaries.",
  """m = AutoModel.from_pretrained("facebook/dinov2-small").cuda()
E = m(pixel_values=tiles_224).pooler_output     # one vector per tile
d = 1 - cosine(E[1:], E[:-1])                   # change along depth
cuts = find_peaks(d, prominence=.15)
name = argmax(cos(seg_mean, swatch_embeds) + a*lab_sim)""",
  "Downloaded, pretrained, no training: tests whether general vision features already separate rock symbols.",
  "Sees texture, not just colour; strong grouping", "Swatches are tiny; 88 MB model; needs a GPU to be quick"),
 ("S6", "SigLIP image + text", "gpu", "GPU", None,
  "Same tiles, but SigLIP also scores text such as 'a lithology symbol for sandstone', so the label words help with naming.",
  """img = siglip.get_image_features(tiles)
txt = siglip.get_text_features(
    [f"a geological lithology symbol for {e.label}" for e in rocks])
score = w*cos(img, swatch_img) + (1-w)*cos(img, txt)""",
  "Checks whether a downloadable image-text model knows what rock symbols look like.",
  "Uses the label text, not only the swatch", "Hatch-only symbols mean little to it"),
 ("S7", "CNN trained on synthetic columns", "gpu", "GPU", None,
  "Stack this log's own legend swatches into fake columns with noise, train a small CNN for 25 s, then classify the real column.",
  """X, y = synth.make(log, n=400, jitter=True, jpeg=True, lines=True)
net = timm.create_model("resnet18", num_classes=len(rocks)).cuda()
train(net, X, y, seconds=25)
labels = viterbi(net(column_tiles).softmax(1))""",
  "Training data for free: the legend is a labelled sample of every class this log can contain.",
  "Only predicts names in this legend", "Slowest local option; synthetic-to-real gap"),
 ("S8", "JEV typed choice (Typesafe)", "api", "API", "d_s8_jev.png",
  "Find boundaries with S2, then send one crop per cluster to JEV with a choice question over this log's legend names.",
  """ans = jev(state={"crop": "img1"}, assets={"img1": png_512(crop)},
          questions={"lith": {"type": "choice",
              "instructions": "Which legend pattern fills this column?",
              "criteria": {e.label: describe(e) for e in rocks}}})
name = ans["lith"]["choice"]            # + probabilities""",
  "One call per cluster, not per row: about 5 calls a log, with typed probabilities we can threshold.",
  "Fast API (about 0.4 s); calibrated choices", "Needs network and a key; paid"),
 ("S9", "Vision LLM (Terra)", "api", "API", None,
  "Show a general vision LLM the cluster crop next to the numbered legend and ask for the swatch number, as a person would.",
  """png = side_by_side(upscale(crop), numbered_legend(rocks))
out = run(["codex", "exec", "-m", "gpt-6.1-terra", "-i", png,
           'Which numbered swatch matches the left strip? JSON {"n": int}'])
name = rocks[json.loads(out)["n"]].label""",
  "The human way of doing it. Shows how far a general multimodal model gets with no task code.",
  "Reads odd layouts and tiny text", "Tens of seconds a call; answers vary"),
 ("S10", "Hybrid: local first, API on ties", "gpu", "GPU + API", None,
  "S2 boundaries, DINOv2 + colour naming, and JEV only when the top two candidates are too close to call.",
  """s = .6*cos(dino(seg), dino(swatch)) + .4*lab_sim(seg, swatch)
top2 = s.topk(2)
if top2[0] - top2[1] < margin:          # tuned on synthetic
    name = jev_choice(seg_crop, candidates=top2.labels)
else:
    name = top2.label[0]""",
  "Spends API calls only where local models are unsure: the expected accuracy-per-second winner.",
  "Best expected accuracy for its cost", "Most moving parts to tune"),
]
MEAS = {
 "d_s1_colour_runs.png": "Measured: 0.27 s for 6,000 rows of the bp_gc743 colour log.",
 "d_s2_features_pelt.png": "Measured on log06: boundary F1 0.86 (all 6 true boundaries found, 2 extra), 3.6 s.",
 "d_s3_template_ncc.png": "Measured on log06: best correlation only 0.04 to 0.08, so raw NCC barely separates the hatches. It needs scale and phase search before it is useful.",
 "d_s4_kmeans_viterbi.png": "Measured on log06: Viterbi cut 48 label changes to 1 and raised row accuracy from 0.65 to 0.77, but it also merged short Sandstone beds.",
 "d_s5_dinov2.png": "Measured on log06: 3.1 s, 242 MB peak GPU. Distance peaks are dense, so they need PELT or a prominence threshold on top.",
 "d_s8_jev.png": "Measured: 2 of 3 right in 0.33 to 0.39 s a call. The Sandstone crop came back as Siltstone at 0.83 confidence, so confidence alone cannot be trusted.",
}
cards = []
for sid, title, cls, hw, img, what, code, why, pro, con in S:
    demo = (f'<img src="{IMG}/{img}"><div class="cap">{cap(img, "")}'
            + " " + MEAS.get(img, "")
            + '</div>') if img and has(img) else '<div class="cap" style="padding:30px 0;text-align:center">Demo comes from the bake-off run.</div>'
    cards.append(f'''<div class="box" style="margin-bottom:14px"><div class="strat"><div>
<span class="tag">{sid}</span><h3>{html.escape(title)}</h3>
<div class="chips"><span class="chip {cls}">{hw}</span></div>
<p style="margin:4px 0;font-size:14px">{html.escape(what)}</p>
<p style="margin:4px 0;font-size:14px"><b>Why it is here:</b> {html.escape(why)}</p>
<div class="pc"><div class="pro"><b>Strength</b>{html.escape(pro)}</div><div class="con"><b>Weakness</b>{html.escape(con)}</div></div>
<pre>{html.escape(code)}</pre></div><div>{demo}</div></div></div>''')

f04 = man.get("f04_labeller_vs_gold.png", {})
f04cap = (cap("f04_labeller_vs_gold.png") + " Measured on log06: labeller A got 74% of rows right with boundary F1 0.69; "
          "labeller B got boundary F1 0.94 but left 67% of rows as Unknown, so only 32% of rows are right. "
          "Agents are good at finding boundaries and unreliable at naming black-and-white hatches.")
t = (t.replace("__IMG__", IMG).replace("__BOXES__", boxes).replace("__STRATS__", "\n".join(cards))
      .replace("__LEGENDS__", legs).replace("__F04CAP__", f04cap))
hard = "".join(f'<div class="box"><img src="{HARD}/{n}"><div class="cap">{c}</div></div>' for n, c in [
 ("h1_pattern_only.jpg", "Pattern-only rocks: 1-bit logs have no colour at all."),
 ("h2_mixed_legend.jpg", "Legends mix rocks with modifier and drilling symbols."),
 ("h3_same_colour.jpg", "Same colour, different rock."),
 ("h4_tiny_text.jpg", "Legend text a few pixels tall."),
 ("h5_blank_runs.jpg", "Blank stretches and very long single intervals."),
 ("h6_overprints.jpg", "Overprinted lines and symbols break runs.")] if (W / "figs" / "hard" / n).exists())
if hard:
    t = t.replace('<h2>Ten strategies', f'<h2>Why it is hard</h2><div class="grid g3">{hard}</div>\n<h2>Ten strategies', 1)
(A / "index.html").write_text(t, encoding="utf8")
print("ok", len(t), "bytes; boxes", boxes.count("<img"), "legends", legs.count("<img"), "hard", hard.count("<img"))
