/* Contract tests with a small DOM model. No browser, server, or packages. */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.listeners = [];
    this.attributes = {};
    this.value = "";
    this.style = {};
    this._text = "";
    this.disabled = false;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  set innerHTML(_) { throw new Error("Unsafe markup was used."); }
  append(...children) { for (const child of children) { this.children.push(child); child.parent = this; } }
  replaceChildren(...children) { this.children = []; this._text = ""; this.append(...children); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(type, handler, options) { this.listeners.push({ type, handler, signal: options.signal }); }
  emit(type) {
    for (const listener of this.listeners) {
      if (listener.type === type && !listener.signal.aborted) listener.handler({ preventDefault() {} });
    }
  }
  all() { return [this, ...this.children.flatMap(child => child.all())]; }
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}
async function settle() { for (let i = 0; i < 8; i++) await new Promise(done => setImmediate(done)); }
function response(body = {}) {
  return { ok: true, status: 200, async blob() { return {}; }, async json() { return body; } };
}

function harness(fetchOverride) {
  const requests = [];
  const revoked = [];
  let objectCounter = 0;
  class LocalURL extends URL {}
  LocalURL.createObjectURL = () => "blob:pattern-" + (++objectCounter);
  LocalURL.revokeObjectURL = value => revoked.push(value);
  const window = {
    document: { createElement: tag => new Element(tag) },
    location: { href: "http://localhost:8123/lab", origin: "http://localhost:8123" },
    Image: class { async decode() {} },
    async fetch(url, options = {}) {
      requests.push({ url, options });
      return fetchOverride ? fetchOverride(url, options) : response();
    }
  };
  const context = vm.createContext({ window, URL: LocalURL, URLSearchParams, AbortController });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "lab.js"), "utf8"), context);
  const container = new Element("main");
  const payload = { document: { id: "doc & 1", legend: [{ id: 8, label_printed: '<img src=x onerror="bad()">' }] },
    segment: { id: "layer/2", box: [525, 1800, 582, 1900], legend_ids: [8] } };
  function find(tag, text) {
    const element = container.all().find(item => item.tagName === tag && (!text || item.textContent === text));
    assert.ok(element, "Element exists: " + tag + " " + (text || ""));
    return element;
  }
  function input(name) { return container.all().find(item => item.name === name); }
  return { window, requests, revoked, container, payload, find, input };
}

test("mount uses encoded GET contracts and literal label text", async () => {
  const h = harness();
  const api = h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  assert.equal(h.requests.length, 2);
  const crop = new URL(h.requests[0].url, "http://localhost");
  const generated = new URL(h.requests[1].url, "http://localhost");
  assert.equal(crop.pathname, "/api/crop");
  assert.equal(crop.searchParams.get("doc"), "doc & 1");
  assert.equal(crop.searchParams.get("box"), "525,1800,582,1900");
  assert.equal(crop.searchParams.get("width"), "57");
  assert.equal(generated.pathname, "/api/generate");
  assert.equal(generated.searchParams.get("segment"), "layer/2");
  assert.equal(generated.searchParams.get("height"), "480");
  assert.equal(h.container.all().filter(element => element.tagName === "img").length, 2);
  assert.equal(h.find("p", h.payload.document.legend[0].label_printed).textContent, '<img src=x onerror="bad()">');
  assert.equal(h.find("img").style.width, "57px");
  assert.equal(h.find("button", "Approve sample").disabled, false);
  api.unmount();
  assert.equal(h.container.children.length, 0);
  assert.equal(h.revoked.length, 2);
});

test("reviews record the displayed parameters; changed controls block approval", async () => {
  const h = harness();
  const approvals = [];
  h.payload.onApprove = (payload, saved) => approvals.push({ payload, saved });
  h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  h.input("seed").value = "32";
  h.input("seed").emit("input");
  h.input("height").value = "600";
  h.input("variation").value = ".4";
  assert.equal(h.find("button", "Approve sample").disabled, true);
  h.find("button", "Approve sample").emit("click");
  assert.equal(h.requests.length, 2);
  h.find("form").emit("submit");
  await settle();
  h.find("textarea").value = "Symbols match the source.";
  h.find("button", "Approve sample").emit("click");
  await settle();
  const request = h.requests.find(item => item.url === "/api/generation-review");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(JSON.parse(request.options.body), {
    document_id: "doc & 1", segment_id: "layer/2", width: 180, height: 600, seed: 32,
    variation: .4, approved: true, comment: "Symbols match the source."
  });
  assert.equal(approvals.length, 1);
  assert.equal(h.find("button", "Approve sample").disabled, true);
});

test("reject saves a negative review without calling onApprove", async () => {
  const h = harness();
  let called = false;
  h.payload.onApprove = () => { called = true; };
  h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  h.find("button", "Reject sample").emit("click");
  await settle();
  assert.equal(JSON.parse(h.requests.at(-1).options.body).approved, false);
  assert.equal(called, false);
});

test("dataset export posts the source document, count, and seed", async () => {
  const h = harness(url => response(url === "/api/dataset" ? { manifest_url: "/exports/result/manifest.json" } : {}));
  h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  h.find("button", "Export dataset").parent.emit("submit");
  await settle();
  const request = h.requests.find(item => item.url === "/api/dataset");
  assert.deepEqual(JSON.parse(request.options.body), { document_id: "doc & 1", count_per_example: 3, seed: 0 });
  assert.equal(h.find("a").href, "http://localhost:8123/exports/result/manifest.json");
});

test("a failed source crop prevents review and retains its error", async () => {
  const h = harness(url => url.startsWith("/api/crop")
    ? { ok: false, status: 404, async json() { return { error: "Source missing." }; } } : response());
  h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  assert.equal(h.find("button", "Approve sample").disabled, true);
  assert.match(h.container.textContent, /source crop could not load/);
});

test("an older response cannot replace a newer sample", async () => {
  const pending = [];
  const h = harness(url => {
    if (url.startsWith("/api/generate")) { const task = deferred(); pending.push(task); return task.promise; }
    return response();
  });
  h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  h.input("seed").value = "9";
  h.input("seed").emit("input");
  h.find("form").emit("submit");
  await settle();
  pending[1].resolve(response());
  await settle();
  const shown = h.container.all().filter(element => element.tagName === "img")[1].src;
  pending[0].resolve(response());
  await settle();
  assert.equal(h.container.all().filter(element => element.tagName === "img")[1].src, shown);
  h.find("button", "Approve sample").emit("click");
  await settle();
  assert.equal(JSON.parse(h.requests.at(-1).options.body).seed, 9);
});

test("empty contexts and invalid boxes cause no requests", async () => {
  const h = harness();
  h.window.PatternLab.mount(h.container, {});
  assert.match(h.container.textContent, /Select a source segment/);
  h.payload.segment.box = [-1, 2, 3, 4];
  h.window.PatternLab.mount(h.container, h.payload);
  await settle();
  assert.match(h.container.textContent, /source box is not valid/);
  assert.equal(h.requests.length, 0);
});
