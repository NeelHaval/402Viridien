// Run in an isolated WebKit harness that routes "drag" messages to native mouse
// events. This test never writes labels: it cancels the resulting swatch form.
(async () => {
  const report = value => window.webkit.messageHandlers.report.postMessage(value);
  const $ = selector => document.querySelector(selector);
  const wait = async predicate => {
    for (let i = 0; i < 150; i++) {
      if (predicate()) return;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    throw Error('Timed out waiting for browser state');
  };
  try {
    await wait(() => $('#status').textContent === 'Saved');
    if ($('#canonical') || $('#legend .entry select')) throw Error('Canonical generalisation controls must not appear');
    $('#boxLegend').click();
    if ($('#mode').value !== 'legend' || $('#boxLegend').getAttribute('aria-pressed') !== 'true') {
      throw Error('Legend button did not activate drawing mode');
    }
    const canvas = $('#canvas'), rect = canvas.getBoundingClientRect();
    const x1 = rect.left + 70, y1 = rect.top + 70;
    const x2 = rect.left + 120, y2 = rect.top + 110;
    for (const [x,y] of [[x1,y1],[x2,y2]]) {
      const target = document.elementFromPoint(x,y);
      if (target !== canvas) throw Error(`Drawing surface intercepted by #${target?.id}`);
    }
    const targets = [];
    canvas.addEventListener('pointerdown', event => targets.push({type:event.type, trusted:event.isTrusted, target:event.target.id}));
    report({action:'drag',x1,y1,x2,y2});
    await wait(() => $('#legendDialog').open);
    if (!targets.some(event => event.trusted && event.target === 'canvas')) {
      throw Error('No native pointer event reached the canvas');
    }
    $('#cancelLegend').click();
    if ($('#legendDialog').open) throw Error('Cancel did not close the form');
    report(JSON.stringify({ok:true, mode:$('#mode').value, targets, hint:$('#toolHint').textContent}));
  } catch (error) {
    report('FAIL ' + error.message);
  }
})();
