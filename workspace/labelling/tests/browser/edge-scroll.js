// Requires an isolated server exposing exactly one fixture page. Writes and undoes
// a temporary bed in that fixture; refuses to run against the normal page list.
(async () => {
  const report = value => window.webkit.messageHandlers.report.postMessage(value);
  const $ = selector => document.querySelector(selector);
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  const wait = async predicate => {
    for (let i=0; i<150; i++) { if (await predicate()) return; await pause(50); }
    throw Error('Timed out waiting for state');
  };
  try {
    const pages = await fetch('/api/pages').then(r=>r.json());
    if (pages.length!==1) throw Error('Use an isolated fixture server, not your annotations');
    await wait(()=>$('#status').textContent==='Saved');
    const read = ()=>fetch('/api/page/'+encodeURIComponent($('#pages').value)).then(r=>r.json());
    const page = await read();
    if (page.label.layers.length) throw Error('Fixture must have no initial beds');
    $('[data-jump="interpreted_lithology_column"]').click();
    for(let i=0;i<5;i++) $('#in').click();
    $('#mode').value='draw'; $('#mode').dispatchEvent(new Event('change'));
    const view=$('#view'),canvas=$('#canvas'),lith=page.meta.boxes.interpreted_lithology_column;
    const zoom=parseFloat($('#space').style.width)/page.meta.width;
    view.scrollTop=lith[1]*zoom;
    view.scrollLeft=(lith[0]+lith[2])/2*zoom-canvas.width/2;
    await pause(100);
    const rect=canvas.getBoundingClientRect();
    const x1=rect.left+(lith[0]+5)*zoom-view.scrollLeft;
    const x2=rect.left+(lith[2]-5)*zoom-view.scrollLeft;
    const y1=rect.top+120,y2=rect.bottom-3;
    if(document.elementFromPoint(x1,y1)!==canvas) throw Error('Canvas is not receiving input');
    const start=view.scrollTop;
    report({action:'drag',x1,y1,x2,y2,holdMs:1500});
    await pause(700); const held=view.scrollTop;
    await pause(700); const heldLater=view.scrollTop;
    if(held<=start+100||heldLater<=held+100) throw Error('Stationary held pointer did not keep scrolling');
    await wait(()=>!$('#rockPicker').hidden);
    await wait(()=>$('#status').textContent==='Saved');
    const stopped=view.scrollTop;
    await pause(350);
    if(view.scrollTop!==stopped) throw Error('Scrolling continued after release');
    const drawn=(await read()).label.layers[0];
    const expected=(y2-rect.top+stopped)/zoom;
    if(Math.abs(drawn.bottom_px-expected)>5) throw Error('Box bottom did not follow the scrolling page');
    $('#filter').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
    // Resize the selected box at the viewport edge, again holding still.
    const ry=rect.top+drawn.bottom_px*zoom-view.scrollTop;
    const rx=(x1+x2)/2;
    report({action:'drag',x1:rx,y1:ry,x2:rx,y2,holdMs:1000});
    await pause(1700);await wait(()=>$('#status').textContent==='Saved');
    const resized=(await read()).label.layers[0];
    if(resized.bottom_px<=drawn.bottom_px+100/zoom) throw Error('Edge scrolling failed while resizing');
    $('#undo').click();await wait(()=>$('#status').textContent==='Saved');
    if((await read()).label.layers[0].bottom_px!==drawn.bottom_px) throw Error('Resize was not one undo step');
    $('#undo').click();await wait(()=>$('#status').textContent==='Saved');
    if((await read()).label.layers.length) throw Error('Draw was not one undo step');
    report(JSON.stringify({ok:true,scrollWhileHeld:heldLater-start,stopsOnRelease:true,boxFollowsScroll:true,resize:true,undo:true}));
  } catch(error) { report('FAIL '+error.message); }
})();
