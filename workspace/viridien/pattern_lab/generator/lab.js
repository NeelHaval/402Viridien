/* Local pattern review. Load lab.css, then this file. No framework is needed. */
(function (global) {
  "use strict";
  const mounted = new WeakMap();
  let nextId = 0;

  function mount(container, context) {
    if (!container || typeof container.replaceChildren !== "function") {
      throw new TypeError("PatternLab.mount needs a DOM container.");
    }
    const previous = mounted.get(container);
    if (previous) previous.unmount();
    const dom = global.document;
    const root = dom.createElement("section");
    root.className = "plab";
    const prefix = "plab-" + (++nextId) + "-";
    const lifetime = new AbortController();
    let generationRequest = null;
    let generationVersion = 0;
    let generationBusy = false;
    let reviewBusy = false;
    let reviewed = false;
    let alive = true;
    let sourceLoaded = false;
    let sourceFailure = "";
    let currentSample = null;
    const urls = new Set();
    const source = context && context.document;
    const segment = context && context.segment;

    function node(tag, text, className) {
      const element = dom.createElement(tag);
      if (text !== undefined) element.textContent = String(text);
      if (className) element.className = className;
      return element;
    }
    function listen(element, event, handler) {
      element.addEventListener(event, handler, { signal: lifetime.signal });
    }
    const heading = node("h2", "Pattern lab");
    heading.id = prefix + "title";
    root.setAttribute("aria-labelledby", heading.id);
    root.append(heading);
    container.replaceChildren(root);
    const api = {
      unmount() {
        if (!alive) return;
        alive = false;
        lifetime.abort();
        if (generationRequest) generationRequest.abort();
        urls.forEach(url => URL.revokeObjectURL(url));
        urls.clear();
        root.remove();
        mounted.delete(container);
      },
      refresh: () => Promise.resolve()
    };
    mounted.set(container, api);
    if (!source || source.id === undefined || !segment || segment.id === undefined) {
      root.append(node("p", "Select a source segment to open the pattern lab."));
      return api;
    }
    const box = segment.box;
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isInteger)
        || box.some(value => value < 0) || box[0] >= box[2] || box[1] >= box[3]) {
      root.append(node("p", "The source box is not valid. Select another segment."));
      return api;
    }
    const legend = new Map((source.legend || []).map(item => [String(item.id), item.label_printed]));
    const labels = (segment.legend_ids || []).map(id => legend.get(String(id)) || "Legend " + id);
    root.append(node("p", labels.length ? labels.join(" / ") : "No legend label is set.", "plab-labels"));
    root.append(node("p", "Source " + segment.id + " · box " + box.join(", "), "plab-detail"));
    const status = node("p", "Loading the source crop and sample.", "plab-status");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");

    const form = node("form", undefined, "plab-controls");
    function field(labelText, name, initial, min, max, step) {
      const wrapper = node("div", undefined, "plab-field");
      const label = node("label", labelText);
      const input = node("input");
      input.id = prefix + name;
      input.name = name;
      input.type = "number";
      input.value = String(initial);
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      input.required = true;
      label.htmlFor = input.id;
      wrapper.append(label, input);
      form.append(wrapper);
      return input;
    }
    const seed = field("Seed", "seed", 0, 0, Number.MAX_SAFE_INTEGER, 1);
    const height = field("Height (px)", "height", 480, 16, 4096, 1);
    const variation = field("Variation (0 to 1)", "variation", 0, 0, 1, .05);
    const generateButton = node("button", "Generate sample");
    generateButton.type = "submit";
    form.append(generateButton);
    root.append(form);
    root.append(node("p", "Variation changes the source patches and their seams. Symbols keep their source size.", "plab-detail"));

    const pair = node("div", undefined, "plab-pair");
    function figure(title, alt) {
      const frame = node("figure");
      const caption = node("figcaption", title);
      const viewport = node("div", undefined, "plab-image-viewport");
      const image = node("img");
      image.alt = alt;
      image.width = 180;
      viewport.append(image);
      frame.append(caption, viewport);
      pair.append(frame);
      return image;
    }
    const realImage = figure("Real source crop", "Real source crop for " + labels.join(" / "));
    // /api/crop returns native source pixels. Keep the native display width so
    // its symbols can be compared with the unscaled generated pixels.
    realImage.style.width = Math.min(180, box[2] - box[0]) + "px";
    const generatedImage = figure("Generated sample", "Generated pattern for " + labels.join(" / "));
    root.append(pair, status);
    root.append(node("p", "Compare the symbol size, colour, and seams before you approve the sample.", "plab-detail"));
    const commentLabel = node("label", "Review comment");
    const comment = node("textarea");
    comment.id = prefix + "comment";
    comment.rows = 2;
    comment.maxLength = 2000;
    commentLabel.htmlFor = comment.id;
    root.append(commentLabel, comment);
    const reviewActions = node("div", undefined, "plab-actions");
    const approve = node("button", "Approve sample");
    const reject = node("button", "Reject sample", "plab-secondary");
    approve.type = reject.type = "button";
    approve.disabled = reject.disabled = true;
    reviewActions.append(approve, reject);
    root.append(reviewActions);

    const exportForm = node("form", undefined, "plab-export");
    const countLabel = node("label", "Samples per approved example");
    const count = node("input");
    count.id = prefix + "count";
    count.type = "number";
    count.min = "1";
    count.max = "100";
    count.step = "1";
    count.value = "3";
    count.required = true;
    countLabel.htmlFor = count.id;
    const exportButton = node("button", "Export dataset", "plab-secondary");
    exportButton.type = "submit";
    const exportResult = node("p", "Only approved source examples can enter a dataset.", "plab-detail");
    exportResult.setAttribute("role", "status");
    exportForm.append(countLabel, count, exportButton);
    root.append(exportForm, exportResult);

    function parameters() {
      const params = { width: 180, height: Number(height.value), seed: Number(seed.value), variation: Number(variation.value) };
      if (seed.value.trim() === "" || height.value.trim() === "" || variation.value.trim() === ""
          || !Number.isSafeInteger(params.seed) || params.seed < 0
          || !Number.isInteger(params.height) || params.height < 16 || params.height > 4096
          || !Number.isFinite(params.variation) || params.variation < 0 || params.variation > 1) return null;
      return params;
    }
    function matches() {
      const params = parameters();
      return params && currentSample && Object.keys(params).every(key => params[key] === currentSample[key]);
    }
    function syncReview() {
      approve.disabled = reject.disabled = !sourceLoaded || !matches() || generationBusy || reviewBusy || reviewed;
    }
    function query(path, params) {
      return path + "?" + new URLSearchParams(params).toString();
    }
    async function request(url, options = {}) {
      const response = await global.fetch(url, options);
      if (!response.ok) {
        let message = "Request failed (" + response.status + ").";
        try {
          const body = await response.json();
          if (typeof body.error === "string") message = body.error.slice(0, 400);
        } catch (_) { /* Keep the status if the server did not return JSON. */ }
        throw new Error(message);
      }
      return response;
    }
    async function imageURL(response) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      urls.add(url);
      const probe = new global.Image();
      probe.src = url;
      try {
        await probe.decode();
      } catch (error) {
        URL.revokeObjectURL(url);
        urls.delete(url);
        throw new Error("The image could not load. Generate the sample again.");
      }
      return url;
    }
    function release(url) {
      if (url && urls.has(url)) {
        URL.revokeObjectURL(url);
        urls.delete(url);
      }
    }
    async function loadSource() {
      try {
        const response = await request(query("/api/crop", { doc: source.id, box: box.join(","), width: box[2] - box[0] }),
                                       { signal: lifetime.signal });
        const url = await imageURL(response);
        if (!alive) { release(url); return; }
        realImage.src = url;
        sourceLoaded = true;
        sourceFailure = "";
        if (currentSample && matches() && !generationBusy) status.textContent = "Sample loaded. Compare it with the source crop.";
        syncReview();
      } catch (error) {
        if (alive && error.name !== "AbortError") {
          sourceFailure = "The source crop could not load. Reopen the lab to try again. " + error.message;
          status.textContent = sourceFailure;
          syncReview();
        }
      }
    }
    async function generateSample() {
      if (!alive) return;
      const params = parameters();
      if (!params) {
        status.textContent = "Set a whole seed, a height from 16 to 4096, and variation from 0 to 1.";
        syncReview();
        return;
      }
      const version = ++generationVersion;
      if (generationRequest) generationRequest.abort();
      generationRequest = new AbortController();
      generationBusy = true;
      currentSample = null;
      reviewed = false;
      generateButton.textContent = "Generating sample…";
      root.setAttribute("aria-busy", "true");
      status.textContent = "Generating the sample from real source pixels.";
      syncReview();
      try {
        const response = await request(query("/api/generate", { doc: source.id, segment: segment.id, ...params }),
                                       { signal: generationRequest.signal });
        const url = await imageURL(response);
        if (!alive || version !== generationVersion) { release(url); return; }
        release(generatedImage.src);
        generatedImage.src = url;
        currentSample = Object.freeze({ ...params });
        status.textContent = !matches() ? "Settings changed. Generate a sample before you review it."
          : sourceFailure || (sourceLoaded ? "Sample loaded. Compare it with the source crop." : "Sample loaded. Waiting for the source crop.");
      } catch (error) {
        if (alive && version === generationVersion && error.name !== "AbortError") {
          status.textContent = "The sample could not load. Select Generate sample to try again. " + error.message;
        }
      } finally {
        if (alive && version === generationVersion) {
          generationBusy = false;
          generateButton.textContent = "Generate sample";
          root.setAttribute("aria-busy", "false");
          syncReview();
        }
      }
    }
    async function review(approved) {
      if (approve.disabled || reject.disabled || !currentSample) return;
      const snapshot = currentSample;
      const payload = { document_id: source.id, segment_id: segment.id, ...snapshot, approved, comment: comment.value };
      reviewBusy = true;
      syncReview();
      status.textContent = "Saving the sample review.";
      try {
        const response = await request("/api/generation-review", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload), signal: lifetime.signal
        });
        const saved = response.status === 204 ? {} : await response.json();
        if (!alive) return;
        if (snapshot === currentSample) {
          reviewed = true;
          status.textContent = approved ? "Sample approval saved." : "Sample rejection saved.";
        }
        if (approved && typeof context.onApprove === "function") {
          try { context.onApprove(payload, saved); }
          catch (_) { status.textContent = "Review saved. Reopen the lab to update the editor."; }
        }
      } catch (error) {
        if (alive && error.name !== "AbortError") status.textContent = "The review was not saved. Try again. " + error.message;
      } finally {
        reviewBusy = false;
        if (alive) syncReview();
      }
    }
    listen(form, "submit", event => { event.preventDefault(); void generateSample(); });
    [seed, height, variation].forEach(input => listen(input, "input", () => {
      reviewed = false;
      status.textContent = "Settings changed. Generate a sample before you review it.";
      syncReview();
    }));
    listen(approve, "click", () => { void review(true); });
    listen(reject, "click", () => { void review(false); });
    listen(exportForm, "submit", async event => {
      event.preventDefault();
      const amount = Number(count.value);
      const params = parameters();
      if (!params || !Number.isInteger(amount) || amount < 1 || amount > 100 || exportButton.disabled) {
        exportResult.textContent = "Set a whole count from 1 to 100 and a valid seed.";
        return;
      }
      exportButton.disabled = true;
      exportResult.replaceChildren(node("span", "Building a dataset from approved source examples."));
      try {
        const response = await request("/api/dataset", {
          method: "POST", headers: { "Content-Type": "application/json" }, signal: lifetime.signal,
          body: JSON.stringify({ document_id: source.id, count_per_example: amount, seed: params.seed })
        });
        const result = await response.json();
        if (!alive) return;
        exportResult.replaceChildren(node("span", "Dataset export complete."));
        const href = result.download_url || result.manifest_url;
        if (typeof href === "string") {
          const url = new URL(href, global.location.href);
          if (url.origin === global.location.origin && /^(https?:)$/.test(url.protocol)) {
            const link = node("a", result.download_url ? "Download dataset" : "Open dataset manifest");
            link.href = url.href;
            exportResult.append(node("span", " "), link);
          }
        }
      } catch (error) {
        if (alive && error.name !== "AbortError") exportResult.textContent = "The dataset was not exported. Try again. " + error.message;
      } finally {
        if (alive) exportButton.disabled = false;
      }
    });
    api.refresh = generateSample;
    void loadSource();
    void generateSample();
    return api;
  }

  global.PatternLab = { mount };
})(window);
