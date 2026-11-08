// Full-image coordinates are the only persistent geometry. UI boxes are grouped on save.
export function depthFn(anchors) {
  const a = [...anchors].sort((a, b) => a.y - b.y);
  if (a.length < 2 || a.some((v, i) => i && (v.y <= a[i - 1].y || v.depth_ft <= a[i - 1].depth_ft))) return null;
  return y => { let i = 0; while (i < a.length - 2 && y > a[i + 1].y) i++; return a[i].depth_ft + (y - a[i].y) * (a[i + 1].depth_ft - a[i].depth_ft) / (a[i + 1].y - a[i].y); };
}
export function linearFit(anchors) {
  if (anchors.length < 2) return null;
  const my = anchors.reduce((s, a) => s + a.y, 0) / anchors.length, md = anchors.reduce((s, a) => s + a.depth_ft, 0) / anchors.length;
  const den = anchors.reduce((s, a) => s + (a.y - my) ** 2, 0); if (!den) return null;
  const slope = anchors.reduce((s, a) => s + (a.y - my) * (a.depth_ft - md), 0) / den, intercept = md - slope * my;
  const residuals = anchors.map(a => a.depth_ft - slope * a.y - intercept);
  return { slope, intercept, residuals, max_residual_ft: Math.max(...residuals.map(Math.abs)) };
}
export function groupBoxes(boxes, legend, anchors) {
  const groups = new Map(), canon = new Map(legend.map(e => [e.id, e.label_printed]));
  for (const b of boxes) { const key = `${b.top_px},${b.bottom_px}`; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(b); }
  const f = depthFn(anchors), round = v => Math.round((v + Number.EPSILON) * 10) / 10;
  return [...groups.values()].map(bs => {
    const first = bs[0];
    const composition = [{legend_id:first.legend_id, type:canon.get(first.legend_id)||'unknown',
      proportion:1, x_range:[...first.x_range]}];
    const top = bs[0].top_px, bottom = bs[0].bottom_px;
    return {
top_px: top, bottom_px: bottom, top_ft: f ? round(f(top)) : null, bottom_ft: f ? round(f(bottom)) : null,
      top_m: f ? round(f(top) * .3048) : null, bottom_m: f ? round(f(bottom) * .3048) : null, composition,
      flags: bs.some(b => b.flags.includes('uncertain')) ? ['uncertain'] : []
};
  }).sort((a, b) => a.top_px - b.top_px || a.bottom_px - b.bottom_px);
}

// Screen pixels per second: scroll faster as a held pointer approaches an edge.
export function edgeScrollVelocity(y, top, bottom) {
  const band = Math.min(56, (bottom - top) / 4);
  if (band <= 0) return 0;
  if (y > bottom - band) return 850 * Math.min(1, (y - bottom + band) / band) ** 2;
  if (y < top + band) return -850 * Math.min(1, (top + band - y) / band) ** 2;
  return 0;
}

export function pageForAssignee(pages, owner, preferredStem) {
  const assigned = pages.filter(p => owner === 'all' || (owner === 'in_progress' ? p.in_progress : p.assignee === owner));
  return assigned.find(p => p.stem === preferredStem) ||
    (owner === 'all' ? assigned.find(p => p.in_progress) : null) || assigned[0];
}

export function nextPageForAssignee(pages, owner, currentStem) {
  const assigned = pages.filter(p => owner === 'all' || (owner === 'in_progress' ? p.in_progress : p.assignee === owner));
  if (!assigned.length) return null;
  const index = assigned.findIndex(p => p.stem === currentStem);
  return assigned[(index + 1) % assigned.length];
}

if (typeof document !== 'undefined') init();
function init() {
  const $ = s => document.querySelector(s), make = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
  const view = $('#view'), canvas = $('#canvas'), ctx = canvas.getContext('2d'), ruler = $('#ruler'), rc = ruler.getContext('2d'), mini = $('#minimap'), mc = mini.getContext('2d');
  const S = { pages: [], stem: null, meta: null, label: null, boxes: [], selected: new Set(), z: .2, undo: [], redo: [], version: '0', dirty: false, saving: null, blocked: false, revision: 0, hot: 0, lithOnly: false, images: new Map(), drag: null, cursor: null, space: false, pickerIndex: 0, validation: null };
  const color = name => {
    const text = String(name || 'unknown').toLowerCase();
    for (const [word, value] of [['brown', '#a57847'], ['green', '#68ae4d'], ['grey', '#929292'], ['gray', '#929292']]) {
      if (new RegExp(`\\b${word}\\b`).test(text)) return value;
    }
    let hash = 0;
    for (const char of text) hash = (hash * 31 + char.charCodeAt(0)) | 0;
    return `hsl(${Math.abs(hash) % 360} 45% 55%)`;
  };
  const api = async (path, options) => { const r = await fetch(path, options); const b = await r.json(); if (!r.ok) throw Object.assign(Error(b.error || r.statusText), { status: r.status }); return b; };
  const path = () => encodeURIComponent(S.stem), anchors = () => S.label.depth_calibration.anchors, lith = () => S.meta.boxes.interpreted_lithology_column;
  const say = t => $('#status').textContent = t;
  function updateToolHint() {
    const mode = $('#mode').value;
    $('#toolHint').textContent = {
      legend: 'Legend swatch: drag a rectangle around a lithology symbol on the large page, then enter its label.',
      draw: 'Draw beds: drag inside the lithology column. Click an existing box to select or resize it.',
      calibrate: 'Calibrate depth: click a printed depth number on the large page, then enter its value.',
      pan: 'Pan: drag the page to move it. Use the mode menu to return to drawing.'
    }[mode];
    $('#boxLegend').setAttribute('aria-pressed', String(mode === 'legend'));
  }
  function setMode(mode) {
    if (S.lithOnly && ['legend', 'calibrate'].includes(mode)) setLithOnly(false);
    $('#mode').value = mode; updateToolHint();
  }
  $('#mode').onchange = () => setMode($('#mode').value);
  function setLithOnly(enabled) {
    S.lithOnly = enabled;
    $('#lithOnly').setAttribute('aria-pressed', String(enabled));
    $('#lithOnly').textContent = enabled ? 'Show full page' : 'Lithology only';
    if (enabled && S.meta) {
      const [x1,y1,x2,y2] = lith();
      view.scrollLeft = (x1+x2)/2*S.z - view.clientWidth/2;
      if ((view.scrollTop+view.clientHeight)/S.z < y1 || view.scrollTop/S.z > y2) jumpY(y1);
    }
    draw();
  }
  $('#lithOnly').onclick = () => { if (S.meta) setLithOnly(!S.lithOnly); };
  function snapshot() { return JSON.stringify({ label: S.label, boxes: S.boxes }); }
  function restore(text) { const v = JSON.parse(text); S.label = v.label; S.boxes = v.boxes; S.selected.clear(); changed(); }
  function record(before) { if (before === snapshot()) return; S.undo.push(before); S.redo = []; changed(); }
  function change(fn) { const before = snapshot(); fn(); record(before); }
  function changed() {
    const kept = [], groups = new Map(), selected = new Set();
    S.boxes.forEach((box,index) => {
      const key = `${box.top_px},${box.bottom_px}`;
      if (!groups.has(key)) { groups.set(key,kept.length); kept.push(box); }
      const target = groups.get(key);
      if (box.flags.includes('uncertain')) kept[target].flags=['uncertain'];
      if (S.selected.has(index)) selected.add(target);
    });
    S.boxes=kept; S.selected=selected;
    S.label.layers = groupBoxes(S.boxes, S.label.legend, anchors()); S.revision++; S.dirty = true; say('Unsaved'); clearTimeout(save.timer); save.timer = setTimeout(() => save().catch(e => say(e.message)), 300); renderPanel(); draw(); }
  async function save() {
    clearTimeout(save.timer);
    if (S.saving) { await S.saving; if (S.dirty && !S.blocked) return save(); return; }
    if (!S.dirty) return;
    if (S.blocked) throw Error('Save conflict: reload this page before editing further');
    const revision = S.revision, stem = S.stem, label = structuredClone(S.label); S.dirty = false; say('Saving…');
    S.saving = (async () => {
      try {
const b = await api('/api/label/' + encodeURIComponent(stem), { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label, version: S.version }) });
        S.version = b.version; S.validation = b.validation;
        if (S.revision === revision) S.label = b.label;
        const page=S.pages.find(p=>p.stem===stem); if(page)page.done=!b.validation.errors.length; renderPagePicker();
        renderValidation(); say(S.dirty ? 'Unsaved' : 'Saved');
      } catch (e) { S.dirty = true; if (e.status === 409) { S.blocked = true; $('#banner').replaceChildren('Save conflict. ', make('button', { textContent: 'Reload page', onclick: () => location.reload() })); $('#banner').hidden = false; } say('Save failed: ' + e.message); throw e; }
      finally { S.saving = null; }
    })();
    await S.saving;
    if (S.dirty && !S.blocked) return save();
  }
  addEventListener('beforeunload', e => { if (S.dirty || S.saving) { e.preventDefault(); e.returnValue = ''; } });
  function loadBoxes() { S.boxes = S.label.layers.flatMap(l => l.composition.slice(0,1).map(c => ({ top_px: l.top_px, bottom_px: l.bottom_px, x_range: [...c.x_range], legend_id: c.legend_id, flags: [...l.flags] }))); }
  async function load(stem) {
    try {
if (S.stem) await save(); const b = await api('/api/page/' + encodeURIComponent(stem));
      if (!b.meta.layers.pyramid) throw Error('Run prep to build this page’s full-page pyramid');
      S.stem = stem; S.meta = b.meta; S.label = b.label; S.version = b.version; S.validation = b.validation; S.blocked = false; S.dirty = false; S.revision = 0; S.undo = []; S.redo = []; S.images.clear(); S.selected.clear(); stopEdgeScroll(); S.drag = null; loadBoxes(); setLithOnly(false);
      setMode(anchors().length < 2 ? 'calibrate' : 'draw'); $('#pages').value = stem; localStorage.setItem('labelling.currentPage',stem); renderPagePicker(); renderPanel(); renderValidation(); fit('width'); view.scrollTop = 0; say('Saved'); previewLegend();
    } catch (e) { say(e.message); if (S.stem) $('#pages').value = S.stem; }
  }
  function layout() { if (!S.meta) return; const w = view.clientWidth, h = view.clientHeight; canvas.width = w; canvas.height = h; canvas.style.marginBottom = `-${h}px`; $('#space').style.width = `${Math.max(w, S.meta.width * S.z)}px`; $('#space').style.height = `${Math.max(h, S.meta.height * S.z)}px`; ruler.width = ruler.clientWidth; ruler.height = h; mini.width = mini.clientWidth; mini.height = h; $('#zoom').textContent = (S.z * 100).toFixed(0) + '%'; draw(); }
  function setZoom(z, px = view.clientWidth / 2, py = view.clientHeight / 2) { const x = (view.scrollLeft + px) / S.z, y = (view.scrollTop + py) / S.z; S.z = Math.max(.002, Math.min(4, z)); layout(); view.scrollLeft = x * S.z - px; view.scrollTop = y * S.z - py; draw(); }
  function fit(which) { setZoom(which === 'width' ? view.clientWidth / S.meta.width : view.clientHeight / S.meta.height); }
  function jump(box) {
    if (S.lithOnly) {
      const b = lith();
      if (box[0]<b[0] || box[1]<b[1] || box[2]>b[2] || box[3]>b[3]) setLithOnly(false);
    }
    const [x1, y1, x2, y2] = box; setZoom(Math.min(2, Math.max(.05, Math.min(view.clientWidth / (x2 - x1 + 100), view.clientHeight / Math.min(y2 - y1 + 100, 1600))))); view.scrollLeft = ((x1 + x2) / 2) * S.z - view.clientWidth / 2; view.scrollTop = y1 * S.z - 30; draw(); }
  function jumpY(y) { view.scrollTop = y * S.z - view.clientHeight / 3; draw(); }
  function pos(e) { const r = canvas.getBoundingClientRect(); return { x: Math.round((e.clientX - r.left + view.scrollLeft) / S.z), y: Math.round((e.clientY - r.top + view.scrollTop) / S.z) }; }
  function imageFor(tile) { const key = tile.file; let im = S.images.get(key); if (!im) { im = new Image(); im.src = '/cache/' + path() + '/' + tile.file; im.onload = () => { im.ok = true; draw(); }; im.onerror = () => { im.failed = true; say('Tile missing: run prep again'); }; S.images.set(key, im); } return im; }
  function paintLayer(c, layer, z, ox, oy, w, h, clipBox = null) {
    const levels = layer.levels; let level = levels[0]; for (const v of levels) if (v.scale * z <= 1.25) level = v;
    for (const tile of level.tiles) {
const x = tile.x0 ?? layer.box[0], x1 = tile.x1 ?? layer.box[2], y = tile.y0, y1 = tile.y1;
      if (clipBox && (x1 <= clipBox[0] || x >= clipBox[2] || y1 <= clipBox[1] || y >= clipBox[3])) continue;
      if (x1 * z < ox || x * z > ox + w || y1 * z < oy || y * z > oy + h) continue;
      const im = imageFor(tile); if (im.ok) c.drawImage(im, x * z - ox, y * z - oy, (x1 - x) * z, (y1 - y) * z);
    }
  }
  let raf = 0;
  function draw() { if (!raf) raf = requestAnimationFrame(paint); }
  function paint() {
raf = 0; if (!S.meta) return; const z = S.z, ox = view.scrollLeft, oy = view.scrollTop, w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    if (S.lithOnly) {
      const [x1,y1,x2,y2] = lith();
      ctx.beginPath(); ctx.rect(x1*z-ox,y1*z-oy,(x2-x1)*z,(y2-y1)*z); ctx.clip();
    }
    ctx.fillStyle = 'white'; ctx.fillRect(-ox, -oy, S.meta.width * z, S.meta.height * z); paintLayer(ctx, S.meta.layers.pyramid, z, ox, oy, w, h, S.lithOnly ? lith() : null);
    ctx.lineWidth = 1; ctx.setLineDash([5, 5]); ctx.strokeStyle = '#168bb580'; for (const b of Object.values(S.meta.boxes)) ctx.strokeRect(b[0] * z - ox, b[1] * z - oy, (b[2] - b[0]) * z, (b[3] - b[1]) * z); ctx.setLineDash([]);
    function box(b, i, draft = false) {
const [a, c] = b.x_range, x = a * z - ox, y = b.top_px * z - oy, bw = (c - a) * z, bh = (b.bottom_px - b.top_px) * z; if (y + bh < 0 || y > h || x + bw < 0 || x > w) return;
      const e = S.label.legend.find(e => e.id === b.legend_id); ctx.fillStyle = color(e?.label_printed || 'unknown'); ctx.globalAlpha = +$('#opacity').value; ctx.fillRect(x, y, bw, bh); ctx.globalAlpha = 1;
      ctx.strokeStyle = S.selected.has(i) ? '#e34429' : b.flags.includes('uncertain') ? '#a639a6' : '#24516a'; ctx.lineWidth = S.selected.has(i) ? 3 : 1; ctx.strokeRect(x, y, bw, bh);
      if (!e) { ctx.save(); ctx.beginPath(); ctx.rect(x, y, bw, bh); ctx.clip(); ctx.strokeStyle = '#6669'; ctx.lineWidth = 1; ctx.beginPath(); for (let yy = Math.max(y, -bw);yy < Math.min(y + bh, h) + bw;yy += 9) { ctx.moveTo(x, yy); ctx.lineTo(x + bw, yy + bw); } ctx.stroke(); ctx.restore(); }
      if (bh > 17 && bw > 30) { ctx.fillStyle = '#173349'; ctx.font = '12px system-ui'; ctx.fillText(e?.label_printed || 'Unassigned', x + 3, Math.max(y + 14, 14), Math.max(10, bw - 6)); }
    }
    S.boxes.forEach((b, i) => box(b, i)); if (S.drag?.kind === 'draw' || S.drag?.kind === 'legend') box(S.drag.box, -1, true);
    ctx.lineWidth = 1; for (const a of anchors()) { const yy = a.y * z - oy; if (yy < 0 || yy > h) continue; ctx.strokeStyle = '#d42d2d'; ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(Math.min(w, 180), yy); ctx.stroke(); ctx.fillStyle = '#b52222'; ctx.fillText(`${a.depth_ft.toFixed(1)} ft`, 5, yy - 4); }
    ctx.restore();
    paintRuler(); paintMini();
    const picked = [...S.selected].map(i => S.boxes[i]).filter(Boolean);
    const layers = groupBoxes(S.boxes.filter(b => picked.some(p => p.top_px === b.top_px && p.bottom_px === b.bottom_px)), S.label.legend, anchors());
    $('#selectionInfo').textContent = layers.length ? layers.map(l => `${l.top_px}–${l.bottom_px} px` + (l.top_ft !== null ? ` · ${l.top_ft.toFixed(1)}–${l.bottom_ft.toFixed(1)} ft` : '') + ' · ' + (l.composition[0]?.type || 'Unassigned') + (l.flags.length ? ' · uncertain' : '')).join(' / ') : 'Click a bed to see its rock label.';
    // Keep only tiles near the current viewport, plus small legend/minimap tiles.
    if (S.images.size > 140) { for (const [key, im] of S.images) if (key.startsWith('pyramid/') && !ctxNeeded(key)) { im.onload = null; S.images.delete(key); } }
  }
  function ctxNeeded(key) { const lv = S.meta.layers.pyramid.levels.find(l => l.tiles.some(t => t.file === key)); const t = lv?.tiles.find(t => t.file === key); return t && t.y1 * S.z >= view.scrollTop - canvas.height && t.y0 * S.z <= view.scrollTop + canvas.height * 2; }
  function paintRuler() { const h = ruler.height, w = ruler.width; rc.clearRect(0, 0, w, h); rc.fillStyle = '#394c59'; rc.font = '11px system-ui'; const raw = 75 / S.z, pow = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].find(v => v * pow >= raw) * pow; const f = depthFn(anchors()); for (let y = Math.ceil(view.scrollTop / S.z / step) * step;y < (view.scrollTop + h) / S.z;y += step) { if (y > S.meta.height) break; const sy = y * S.z - view.scrollTop; rc.fillText(`${y} px`, 5, sy + 4); if (f) rc.fillText(`${f(y).toFixed(1)} ft`, 5, sy + 18); if (f) rc.fillText(`${(f(y) * .3048).toFixed(1)} m`, 5, sy + 31); rc.beginPath(); rc.moveTo(w - 8, sy); rc.lineTo(w, sy); rc.stroke(); } }
  function paintMini() { const [x1, y1, x2, y2] = lith(), h = mini.height, w = mini.width, scale = h / (y2 - y1); mc.clearRect(0, 0, w, h); mc.save(); mc.translate(0, -y1 * scale); mc.scale(w / (x2 - x1), scale); mc.translate(-x1, 0); paintLayer(mc, S.meta.layers.lith, 1, x1, 0, x2 - x1, y2); mc.restore(); for (const l of S.label.layers) { mc.globalAlpha = .6; mc.fillStyle = color(l.composition[0]?.type); mc.fillRect(0, (l.top_px - y1) * scale, w, (l.bottom_px - l.top_px) * scale); } mc.globalAlpha = 1; mc.strokeStyle = '#e53c25'; mc.lineWidth = 2; mc.strokeRect(1, (view.scrollTop / S.z - y1) * scale, w - 2, view.clientHeight / S.z * scale); }
  mini.onpointerdown = e => { mini.setPointerCapture(e.pointerId); mini.onpointermove = go; go(e); }; mini.onpointerup = () => mini.onpointermove = null;
  function go(e) { const y = lith()[1] + (e.clientY - mini.getBoundingClientRect().top) / mini.height * (lith()[3] - lith()[1]); jumpY(y); }
  function inner(y) { return S.meta.inner_edges?.find(e => e.y0 <= y && y < e.y1)?.x_range || [lith()[0], lith()[2]]; }
  function snap(p, exclude = -1) { const [x1, y1, x2, y2] = lith(), xs = [...inner(p.y)], ys = [y1, y2]; S.boxes.forEach((b, i) => { if (i === exclude) return; if (p.y >= b.top_px - 6 / S.z && p.y <= b.bottom_px + 6 / S.z) xs.push(...b.x_range); ys.push(b.top_px, b.bottom_px); }); const nearest = (v, arr) => { const q = arr.reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a, arr[0]); return Math.abs(q - v) <= 6 / S.z ? q : v; }; return { x: Math.round(Math.max(x1, Math.min(x2, nearest(p.x, xs)))), y: Math.round(Math.max(y1, Math.min(y2, nearest(p.y, ys)))) }; }
  function contains(b, p) { return p.x >= b.x_range[0] && p.x <= b.x_range[1] && p.y >= b.top_px && p.y <= b.bottom_px; }
  function hit(p) { for (let i = S.boxes.length - 1;i >= 0;i--)if (contains(S.boxes[i], p)) return i; return -1; }
  function edge(b, p) { const d = 6 / S.z; if (p.y >= b.top_px - d && p.y <= b.bottom_px + d) { if (Math.abs(p.x - b.x_range[0]) < d) return 'left'; if (Math.abs(p.x - b.x_range[1]) < d) return 'right'; } if (p.x >= b.x_range[0] - d && p.x <= b.x_range[1] + d) { if (Math.abs(p.y - b.top_px) < d) return 'top'; if (Math.abs(p.y - b.bottom_px) < d) return 'bottom'; } return null; }
  let edgeFrame = 0, edgePointer = null, edgeTime = null;
  const boxDragActive = () => S.drag && ['draw', 'legend', 'resize'].includes(S.drag.kind);
  function stopEdgeScroll() {
    cancelAnimationFrame(edgeFrame); edgeFrame = 0; edgeTime = null;
  }
  function updateBoxDrag(p) {
    const d = S.drag;
    if (!d) return;
    if (d.kind === 'draw' || d.kind === 'legend') {
      const q = d.kind === 'draw' ? snap(p) : {
        x: Math.max(0, Math.min(S.meta.width, p.x)), y: Math.max(0, Math.min(S.meta.height, p.y))
      };
      d.box = {...d.box, top_px: Math.min(d.start.y, q.y), bottom_px: Math.max(d.start.y, q.y),
        x_range: [Math.min(d.start.x, q.x), Math.max(d.start.x, q.x)]};
    }
    if (d.kind === 'resize') {
      const q = snap(p, d.i), b = S.boxes[d.i];
      if (d.edge === 'top') b.top_px = Math.min(b.bottom_px - 1, q.y);
      if (d.edge === 'bottom') b.bottom_px = Math.max(b.top_px + 1, q.y);
      if (d.edge === 'left') b.x_range[0] = Math.min(b.x_range[1] - 1, q.x);
      if (d.edge === 'right') b.x_range[1] = Math.max(b.x_range[0] + 1, q.x);
    }
  }
  function edgeScrollStep(time) {
    edgeFrame = 0;
    if (!boxDragActive() || !edgePointer) { edgeTime = null; return; }
    const rect = canvas.getBoundingClientRect();
    const speed = edgePointer.clientX >= rect.left && edgePointer.clientX <= rect.right
      ? edgeScrollVelocity(edgePointer.clientY, rect.top, rect.bottom) : 0;
    if (!speed) { edgeTime = null; return; }
    const dt = edgeTime === null ? 1 / 60 : Math.min(.05, (time - edgeTime) / 1000);
    edgeTime = time;
    const before = view.scrollTop;
    view.scrollTop += speed * dt;
    if (view.scrollTop === before) { edgeTime = null; return; }
    // The pointer can stay still while the page moves beneath it.
    const p = pos(edgePointer); S.cursor = p; updateBoxDrag(p); draw();
    edgeFrame = requestAnimationFrame(edgeScrollStep);
  }
  function trackEdgeScroll(e) {
    edgePointer = {clientX:e.clientX, clientY:e.clientY};
    if (boxDragActive() && !edgeFrame) edgeFrame = requestAnimationFrame(edgeScrollStep);
  }
  canvas.onpointerdown = e => {
    stopEdgeScroll();
    if (!S.label || e.button !== 0) return; e.preventDefault(); document.activeElement?.blur?.(); $('#rockPicker').hidden = true; const p = pos(e), mode = $('#mode').value, before = snapshot(); canvas.setPointerCapture(e.pointerId);
    const ai = anchors().findIndex(a => Math.abs(a.y - p.y) * S.z < 6 && p.x * S.z - view.scrollLeft < 180);
    if (S.space || mode === 'pan') S.drag = { kind: 'pan', x: e.clientX, y: e.clientY, left: view.scrollLeft, top: view.scrollTop };
    else if (ai >= 0) S.drag = { kind: 'anchor', i: ai, before };
    else if (mode === 'calibrate') S.drag = { kind: 'anchorClick', p, x: e.clientX, y: e.clientY };
    else if (mode === 'legend') S.drag = { kind: 'legend', start: p, box: { top_px: p.y, bottom_px: p.y, x_range: [p.x, p.x], flags: [] }, before };
    else {
      const selectedEdge = [...S.selected].map(i => ({ i, edge: edge(S.boxes[i], p) })).find(v => v.edge);
      if (selectedEdge && !e.altKey) { S.drag = { kind: 'resize', ...selectedEdge, before }; }
      else {
const i = e.altKey ? -1 : hit(p);
        if (i >= 0) { if (!e.shiftKey) S.selected.clear(); if (e.shiftKey && S.selected.has(i)) S.selected.delete(i); else S.selected.add(i); S.drag = { kind: 'select' }; draw(); }
        else if (contains({ top_px: lith()[1], bottom_px: lith()[3], x_range: [lith()[0], lith()[2]] }, p)) { const q = snap(p); S.selected.clear(); S.drag = { kind: 'draw', start: q, before, box: { top_px: q.y, bottom_px: q.y, x_range: [q.x, q.x], legend_id: null, flags: [] } }; }
        else S.drag = { kind: 'pan', x: e.clientX, y: e.clientY, left: view.scrollLeft, top: view.scrollTop };
      }
    }
    trackEdgeScroll(e);
  };
  canvas.onpointermove = e => {
    if (!S.label) return; const p = pos(e); S.cursor = p; const f = depthFn(anchors()); $('#info').textContent = `x ${p.x} · y ${p.y} px` + (f ? ` · ${f(p.y).toFixed(1)} ft · ${(f(p.y) * .3048).toFixed(1)} m` : '') + ` · ${S.boxes.length} boxes · ${S.label.layers.length} layers`;
    const d = S.drag;
    if (!d) { const i = hit(p); canvas.style.cursor = i >= 0 && S.selected.has(i) && edge(S.boxes[i], p) ? 'crosshair' : $('#mode').value === 'pan' ? 'grab' : 'crosshair'; return; }
    if (d.kind === 'pan') { view.scrollLeft = d.left - (e.clientX - d.x); view.scrollTop = d.top - (e.clientY - d.y); }
    updateBoxDrag(p); trackEdgeScroll(e);
    if (d.kind === 'anchor') anchors()[d.i].y = Math.max(0, Math.min(S.meta.height, p.y)); draw();
  };
  canvas.onpointerup = e => {
    stopEdgeScroll(); updateBoxDrag(pos(e));
    const d = S.drag; S.drag = null; if (!d) return;
    if (d.kind === 'anchorClick' && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 5) { openDepth(d.p.y); }
    if (d.kind === 'draw' && d.box.bottom_px > d.box.top_px && d.box.x_range[1] > d.box.x_range[0]) { S.boxes.push(d.box); S.selected = new Set([S.boxes.length - 1]); record(d.before); openPicker(e.clientX, e.clientY); }
    if (d.kind === 'legend' && d.box.bottom_px > d.box.top_px && d.box.x_range[1] > d.box.x_range[0]) { pendingLegend = [d.box.x_range[0], d.box.top_px, d.box.x_range[1], d.box.bottom_px]; $('#printed').value = ''; $('#legendDialog').showModal(); $('#printed').focus(); }
    if (d.kind === 'resize' || d.kind === 'anchor') { anchors().sort((a, b) => a.y - b.y); record(d.before); } draw();
  };
  canvas.onpointercancel = () => { stopEdgeScroll(); if (S.drag?.before) { const b = JSON.parse(S.drag.before); S.label = b.label; S.boxes = b.boxes; } S.drag = null; draw(); };
  canvas.addEventListener('lostpointercapture', stopEdgeScroll);
  addEventListener('blur', stopEdgeScroll);
  view.addEventListener('wheel', e => { if (e.ctrlKey || e.metaKey) { e.preventDefault(); const r = canvas.getBoundingClientRect(); setZoom(S.z * Math.exp(-e.deltaY * .002), e.clientX - r.left, e.clientY - r.top); } else if (e.shiftKey && e.deltaY) { e.preventDefault(); view.scrollLeft += e.deltaY; } }, { passive: false });
  view.onscroll = draw; new ResizeObserver(layout).observe(view);
  function assign(e) { if (!e) return; change(() => { for (const i of S.selected) S.boxes[i].legend_id = e.id; }); $('#rockPicker').hidden = true; $('#filter').blur(); }
  let pendingDepth = null, pendingLegend = null;
  function openDepth(y) { pendingDepth = y; $('#depthValue').value = ''; $('#depthDialog').showModal(); $('#depthValue').focus(); }
  $('#depthForm').onsubmit = e => { e.preventDefault(); const n = Number($('#depthValue').value); if (!Number.isFinite(n)) return; change(() => { S.label.depth_calibration.anchors = anchors().filter(a => a.y !== pendingDepth); anchors().push({ y: pendingDepth, depth_ft: n * ($('#depthUnit').value === 'm' ? 1 / .3048 : 1) }); anchors().sort((a, b) => a.y - b.y); }); $('#depthDialog').close(); if (anchors().length >= 2) setMode('draw'); };
  $('#cancelDepth').onclick = () => $('#depthDialog').close(); $('#cancelLegend').onclick = () => $('#legendDialog').close();
  $('#legendForm').onsubmit = e => {
    e.preventDefault();
    const name = $('#printed').value.trim();
    if (!name) return;
    change(() => S.label.legend.push({ id: Math.max(-1, ...S.label.legend.map(e => e.id)) + 1,
      box: pendingLegend, label_printed: name, same_symbol_as: [] }));
    $('#legendDialog').close(); setMode('draw');
  };
  function openPicker(x = canvas.getBoundingClientRect().left + 30, y = canvas.getBoundingClientRect().top + 30, text = '') {
    const p = $('#rockPicker'); p.hidden = false; p.style.left = Math.max(0, Math.min(innerWidth - 300, x)) + 'px'; p.style.top = Math.max(0, Math.min(innerHeight - 360, y)) + 'px'; $('#filter').value = text; S.pickerIndex = 0; renderChoices(); $('#filter').focus();
  }
  function matches() { const q = $('#filter').value.toLowerCase(); return S.label.legend.filter(e => e.label_printed.toLowerCase().includes(q)); }
  function renderChoices() { const es = matches(); S.pickerIndex = Math.max(0, Math.min(es.length - 1, S.pickerIndex)); $('#choices').replaceChildren(...es.map((e, i) => make('button', { textContent: e.label_printed, className: i === S.pickerIndex ? 'active' : '', onclick: () => assign(e) }))); $('#choices .active')?.scrollIntoView({ block: 'nearest' }); }
  $('#filter').oninput = () => { S.pickerIndex = 0; renderChoices(); };
  $('#filter').onkeydown = e => {
    if (['ArrowDown', 'ArrowUp', 'Enter', 'Escape', 'Tab'].includes(e.key)) { e.preventDefault(); e.stopPropagation(); }
    if (e.key === 'ArrowDown') { S.pickerIndex++; renderChoices(); } if (e.key === 'ArrowUp') { S.pickerIndex--; renderChoices(); }
    if (e.key === 'Enter') assign(matches()[S.pickerIndex]); if (e.key === 'Escape') { $('#rockPicker').hidden = true; $('#filter').blur(); }
    if (e.key === 'Tab') { S.hot = (S.hot + 1) % Math.max(1, Math.ceil(S.label.legend.length / 10)); renderPanel(); }
    // Number keys assign hotkeys when no filter text is present.
    if (/^\d$/.test(e.key) && !e.target.value) { e.preventDefault(); assign(S.label.legend[S.hot * 10 + (+e.key + 9) % 10]); }
  };
  function swatch(e) {
const b = make('button', { className: 'swatch', title: 'Assign ' + e.label_printed, onclick: () => assign(e) }), c = make('canvas', { width: 48, height: 40 }), cc = c.getContext('2d'); b.append(c); const [x1, y1, x2, y2] = e.box;
    for (const t of S.meta.layers.legend.levels[0].tiles) { const a = Math.max(x1, t.x0), z = Math.min(x2, t.x1), u = Math.max(y1, t.y0), v = Math.min(y2, t.y1); if (a >= z || u >= v) continue; const im = imageFor(t); const paint = () => { if (im.ok) cc.drawImage(im, a - t.x0, u - t.y0, z - a, v - u, (a - x1) / (x2 - x1) * 48, (u - y1) / (y2 - y1) * 40, (z - a) / (x2 - x1) * 48, (v - u) / (y2 - y1) * 40); }; paint(); if (!im.ok) im.addEventListener('load', paint, { once: true }); } return b;
}
  function previewLegend() { const p = $('#legendPreview'), layer = S.meta.layers.legend, [x1, y1, x2, y2] = layer.box, z = (p.clientWidth - 15) / (x2 - x1); const content = make('div', { style: `position:relative;height:${(y2 - y1) * z}px` }); for (const t of layer.levels[0].tiles) { content.append(make('img', { src: '/cache/' + path() + '/' + t.file, alt: 'Legend', draggable: false, style: `left:${(t.x0 - x1) * z}px;top:${(t.y0 - y1) * z}px;width:${(t.x1 - t.x0) * z}px;height:${(t.y1 - t.y0) * z}px` })); } p.replaceChildren(content); }
  function renderPanel() {
if (!S.label) return;
    const legend = S.label.legend, fit = linearFit(anchors()); $('#hotkeys').textContent = `Keys 1–9,0: entries ${S.hot * 10 + 1}–${Math.min(legend.length, S.hot * 10 + 10)} · Tab cycles`;
    $('#banner').hidden = !S.blocked && anchors().length >= 2; if (!$('#banner').hidden && !S.blocked) $('#banner').textContent = 'Calibrate depth: click 2 printed depth numbers';
    $('#undo').disabled = !S.undo.length; $('#redo').disabled = !S.redo.length;
    $('#legend').replaceChildren(...legend.map((e, i) => {
      const printed = make('input', { value: e.label_printed, title: 'Printed label', onchange: ev => change(() => e.label_printed = ev.target.value.trim()) });
      const same = make('details', {}, make('summary', { textContent: 'Same symbol as: ' + (e.same_symbol_as.join(', ') || '—') }));
      for (const other of legend) if (other.id !== e.id) { const input = make('input', { type: 'checkbox', checked: e.same_symbol_as.includes(other.id), onchange: ev => change(() => { const set = (a, id) => a.same_symbol_as = ev.target.checked ? [...new Set([...a.same_symbol_as, id])].sort((a, b) => a - b) : a.same_symbol_as.filter(v => v !== id); set(e, other.id); set(other, e.id); }) }); same.append(make('label', {}, input, other.label_printed)); }
      return make('div', { className: 'entry' }, swatch(e), printed, make('div', { className: 'tools' }, make('b', { textContent: Math.floor(i / 10) === S.hot ? 'Key ' + ((i % 10 + 1) % 10) : 'id ' + e.id }), make('button', { textContent: 'Show', onclick: () => jump(e.box) }), make('button', { textContent: 'Delete', onclick: () => { if (S.boxes.some(b => b.legend_id === e.id)) return say('Reassign boxes before deleting this legend entry'); change(() => { S.label.legend = legend.filter(v => v.id !== e.id); for (const v of S.label.legend) v.same_symbol_as = v.same_symbol_as.filter(id => id !== e.id); }); } })), same);
    }));
    $('#anchors').replaceChildren(...anchors().map((a, i) => make('div', { className: 'anchor' }, make('input', { type: 'number', value: a.y, title: 'y in full-image pixels', onchange: e => change(() => { a.y = Math.round(Number(e.target.value)); anchors().sort((a, b) => a.y - b.y); }) }), make('input', { type: 'number', step: 'any', value: a.depth_ft, title: 'Depth in feet', onchange: e => change(() => a.depth_ft = Number(e.target.value)) }), make('span', { textContent: 'ft' }), make('span', { title: 'Fit residual in feet', textContent: fit?.residuals[i] !== undefined ? fit.residuals[i].toFixed(1) : '' }), make('button', { textContent: '↗', onclick: () => jumpY(a.y) }), make('button', { textContent: '×', onclick: () => change(() => anchors().splice(i, 1)) }))));
    $('#fit').textContent = fit ? `${fit.slope.toFixed(5)} ft/px · max residual ${fit.max_residual_ft.toFixed(2)} ft${fit.max_residual_ft > 3 ? ' · non-linear page' : ''}` : 'Need 2 anchors for ft/m depths';
  }
  function renderValidation() { const v = S.validation; if (!v) return; $('#validation').textContent = `${v.errors.length ? '' : '✓ '}${v.errors.length} errors · ${v.warnings.length} warnings`; $('#messages').replaceChildren(...[...v.errors.map(e => ({ ...e, prefix: 'Error: ' })), ...v.warnings.map(e => ({ ...e, prefix: 'Warning: ' }))].map(e => make('button', { textContent: e.prefix + e.msg, onclick: () => { if (e.y !== null) jumpY(e.y); } }))); }
  $('#validation').onclick = () => $('#messages').hidden = !$('#messages').hidden;
  $('#pages').onchange = e => load(e.target.value);
  $('#nextPage').onclick = async () => {
    const next = nextPageForAssignee(S.pages, $('#assignee').value, S.stem);
    if (next && next.stem !== S.stem) await load(next.stem);
  };
  $('#in').onclick = () => setZoom(S.z * 1.3); $('#out').onclick = () => setZoom(S.z / 1.3); $('#width').onclick = () => fit('width'); $('#height').onclick = () => fit('height'); $('#opacity').oninput = draw;
  for (const b of document.querySelectorAll('[data-jump]')) b.onclick = () => jump(S.meta.boxes[b.dataset.jump]);
  let detectedSwatches = [];
  function alreadyBoxed(box) {
    return S.label.legend.some(entry => {
      const a = entry.box, overlap = Math.max(0, Math.min(a[2], box[2]) - Math.max(a[0], box[0])) *
        Math.max(0, Math.min(a[3], box[3]) - Math.max(a[1], box[1]));
      const union = (a[2]-a[0])*(a[3]-a[1]) + (box[2]-box[0])*(box[3]-box[1]) - overlap;
      return overlap / union > .5;
    });
  }
  $('#detectLegend').onclick = async () => {
    const stem = S.stem;
    if (!stem) return;
    $('#detectLegend').disabled = true;
    $('#detectionStatus').textContent = 'Detecting boxes and reading labels…';
    try {
      const result = await api('/api/legend-suggestions/' + encodeURIComponent(stem));
      if (S.stem !== stem) return;
      detectedSwatches = result.suggestions.map(s => ({...s, duplicate:alreadyBoxed(s.box)}));
      $('#detectWarnings').textContent = result.warnings.join(' ');
      $('#detectedEntries').replaceChildren(...detectedSwatches.map(s => {
        if (s.duplicate) s.include = false;
        const checkbox = make('input', {type:'checkbox',checked:s.include,disabled:s.duplicate,
          'ariaLabel':'Include swatch',onchange:event => s.include=event.target.checked});
        const preview = swatch(s); preview.disabled = true;
        const name = make('input', {type:'text',value:s.label_printed,placeholder:'Enter the legend label',
          disabled:s.duplicate,oninput:event => s.label_printed=event.target.value});
        return make('div', {className:'detected-entry'}, checkbox, preview, name,
          make('small', {textContent:s.duplicate?'Already in your legend — kept unchanged':s.note}));
      }));
      $('#detectionStatus').textContent = `${detectedSwatches.length} candidate swatches found. Review before adding.`;
      $('#detectDialog').showModal();
    } catch (error) {
      if (S.stem === stem) $('#detectionStatus').textContent = error.message + ' You can still box swatches manually.';
    } finally { $('#detectLegend').disabled = false; }
  };
  $('#cancelDetection').onclick = () => $('#detectDialog').close();
  $('#detectForm').onsubmit = event => {
    event.preventDefault();
    const selected = detectedSwatches.filter(s => s.include && !s.duplicate);
    if (selected.some(s => !s.label_printed.trim())) {
      $('#detectWarnings').textContent = 'Enter a name for every selected swatch, or uncheck it.'; return;
    }
    change(() => {
      let id = Math.max(-1, ...S.label.legend.map(e => e.id)) + 1;
      for (const swatch of selected) if (!alreadyBoxed(swatch.box)) {
        S.label.legend.push({id:id++,box:[...swatch.box],label_printed:swatch.label_printed.trim(),same_symbol_as:[]});
      }
    });
    $('#detectDialog').close();
    $('#detectionStatus').textContent = `Added ${selected.length} swatches. Undo reverses this import.`;
  };
  $('#boxLegend').onclick = () => { jump(S.meta.boxes.lithology_legend); setMode('legend'); };
  $('#undo').onclick = () => { if (!S.undo.length) return; S.redo.push(snapshot()); restore(S.undo.pop()); }; $('#redo').onclick = () => { if (!S.redo.length) return; S.undo.push(snapshot()); restore(S.redo.pop()); };
  $('#proposals').onclick = async () => {
if (S.boxes.length && !confirm('Add proposals only in unlabelled gaps? Existing boxes will be kept.')) return;
    try {
const stem = S.stem, proposals = await api('/api/proposals/' + path()); if (stem !== S.stem) return; change(() => {
        const occupied = S.boxes.map(b => [b.top_px, b.bottom_px]).sort((a, b) => a[0] - b[0]), added = [];
        for (const box of proposals) { let pieces = [[box.top_px, box.bottom_px]]; for (const [a, b] of occupied) pieces = pieces.flatMap(([t, z]) => b <= t || a >= z ? [[t, z]] : [[t, Math.min(a, z)], [Math.max(b, t), z]].filter(([u, v]) => v > u)); for (const [t, b] of pieces) added.push({ ...box, top_px: t, bottom_px: b, x_range: [...inner((t + b) / 2)] }); } S.boxes.push(...added);
      });
} catch (e) { say(e.message); }
};
  addEventListener('keydown', e => {
    if (!S.label || e.target.matches('input,select,textarea') || document.querySelector('dialog[open]')) return;
    const key = e.key.toLowerCase(); if ((e.ctrlKey || e.metaKey) && key === 'z') { e.preventDefault(); $('#' + (e.shiftKey ? 'redo' : 'undo')).click(); return; }
    if (e.code === 'Space') { e.preventDefault(); S.space = true; return; }
    if (e.key === 'Tab') { e.preventDefault(); S.hot = (S.hot + (e.shiftKey ? -1 : 1) + Math.max(1, Math.ceil(S.label.legend.length / 10))) % Math.max(1, Math.ceil(S.label.legend.length / 10)); renderPanel(); return; }
    if (/^\d$/.test(key)) { assign(S.label.legend[S.hot * 10 + (+key + 9) % 10]); return; }
    if (key === 'escape') { S.selected.clear(); $('#rockPicker').hidden = true; draw(); return; }
    if (key === 'delete' || key === 'backspace') { e.preventDefault(); change(() => { S.boxes = S.boxes.filter((_, i) => !S.selected.has(i)); S.selected.clear(); }); return; }
    if (key === 'u') { change(() => { const groups = new Set([...S.selected].map(i => S.boxes[i].top_px + ',' + S.boxes[i].bottom_px)); for (const g of groups) { const bs = S.boxes.filter(b => b.top_px + ',' + b.bottom_px === g), on = !bs.some(b => b.flags.includes('uncertain')); for (const b of bs) b.flags = on ? ['uncertain'] : []; } }); return; }
    if (key === 'f' && S.cursor) { const last = S.boxes.at(-1), top = last ? last.bottom_px : lith()[1], bot = snap(S.cursor).y; if (bot <= top) return; change(() => { S.boxes.push({ top_px: top, bottom_px: bot, x_range: [...inner((top + bot) / 2)], legend_id: null, flags: [] }); S.selected = new Set([S.boxes.length - 1]); }); openPicker(); return; }
    if (key === 'j' || key === 'k') { const layers = S.label.layers, sel = [...S.selected][0], b = S.boxes[sel]; let i = b ? layers.findIndex(l => l.top_px === b.top_px && l.bottom_px === b.bottom_px) : (key === 'j' ? -1 : layers.length); i = Math.max(0, Math.min(layers.length - 1, i + (key === 'j' ? 1 : -1))); if (layers[i]) { const l = layers[i]; S.selected = new Set(S.boxes.map((b, j) => b.top_px === l.top_px && b.bottom_px === l.bottom_px ? j : -1).filter(j => j >= 0)); jumpY(l.top_px); } return; }
    if (key === '+' || key === '=') { setZoom(S.z * 1.3); return; } if (key === '-') { setZoom(S.z / 1.3); return; }
    if (key === '/' || (/^[a-z]$/.test(key) && !e.ctrlKey && !e.metaKey)) { e.preventDefault(); openPicker(undefined, undefined, key === '/' ? '' : key); }
  }); addEventListener('keyup', e => { if (e.code === 'Space') S.space = false; });
  function renderPagePicker() {
    const owner=$('#assignee').value;
    const pages=S.pages.filter(p=>owner==='all'||(owner==='in_progress'?p.in_progress:p.assignee===owner));
    const picker=$('#pages');picker.replaceChildren();
    const caption=p=>(p.done?'✓ ':'')+(p.in_progress?'[In progress] ':p.assignee?`[${p.assignee}] `:'')+p.stem;
    const current=S.pages.find(p=>p.stem===S.stem);
    if(current&&!pages.includes(current)) {
      picker.append(make('optgroup',{label:'Current page — kept open'},make('option',{value:current.stem,textContent:caption(current)})));
    }
    for(const split of ['test','test_bw']) {
      const group=make('optgroup',{label:split});
      for(const p of pages.filter(p=>p.split===split))group.append(make('option',{value:p.stem,textContent:caption(p)}));
      if(group.children.length)picker.append(group);
    }
    if(S.stem)picker.value=S.stem;
  }
  const requestedAssignee=new URLSearchParams(location.search).get('assignee')?.toUpperCase();
  const storedAssignee=localStorage.getItem('labelling.assignee');
  $('#assignee').value=['E','B','N'].includes(requestedAssignee)?requestedAssignee:
    (['E','B','N','all','in_progress'].includes(storedAssignee)?storedAssignee:'all');
  $('#assignee').onchange=async()=>{
    const owner=$('#assignee').value;
    localStorage.setItem('labelling.assignee',owner);
    const current=pageForAssignee(S.pages,owner,S.stem);
    if(current && current.stem!==S.stem) await load(current.stem);
    renderPagePicker();
  };
  (async()=>{try{
    S.pages=await api('/api/pages');renderPagePicker();
    const last=localStorage.getItem('labelling.currentPage');
    const current=pageForAssignee(S.pages,$('#assignee').value,last);
    if(current)await load(current.stem);
  }catch(error){say(error.message);}})();
}
