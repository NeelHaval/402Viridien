# Native browser interaction check (macOS)

This isolated WebKit check clicks the legend tool, hit-tests the drawing surface,
and routes native mouse events through the real browser. It verifies that a
trusted pointer event reaches the canvas and opens the legend form, then cancels
without saving labels. It catches invisible layout elements intercepting gestures.

With the labelling server running:

```sh
swiftc -module-cache-path /tmp/htb-swift-cache tests/browser/WebKitHarness.swift -o /tmp/htb-webkit
/tmp/htb-webkit http://127.0.0.1:8765/ "$PWD/tests/browser/legend-interaction.js" /tmp/htb-legend.png
```

The harness uses a nonpersistent browser session and closes its window on completion.
Look for `{"ok":true,...}` and `"trusted":true`. A failure prints `FAIL ...`.

Automatic legend import is also exercised against an isolated Hadrian fixture:
24 candidates, 17 new rock entries after preserving one existing swatch and excluding
six accessory symbols; raw grey/brown/green shale names survive; one Undo restores
the original entry. The detector unit tests cover geometry, text orientation, blank
slots, accessory exclusion, coordinate translation and unavailable OCR.

`edge-scroll.js` checks drawing and resizing with native mouse input held at the
bottom edge. It verifies continued scrolling with a stationary pointer, geometry
following the page, immediate stop on release, and one undo step per gesture.
It refuses to run against a normal multi-page server: use an isolated fixture server
with a single prepped page, one legend entry and no beds, stored outside `labels/`.
The harness accepts optional `holdMs` on a `drag` message for this check.

`lithology-only.js` is read-only and can run against the regular server. It samples
the canvas to verify that surrounding image pixels are hidden while the column
remains visible, checks that depth/zoom stay unchanged, verifies reveal and automatic
reveal for depth/legend tools, and asserts that no label PUT is sent.
