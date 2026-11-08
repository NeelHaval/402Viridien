// Read-only browser check: toggle the image mask, sample canvas pixels, and reveal.
(async () => {
  const report = value => window.webkit.messageHandlers.report.postMessage(value);
  const $ = selector => document.querySelector(selector);
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  const wait = async predicate => {
    for (let i=0;i<150;i++) { if (predicate()) return; await pause(50); }
    throw Error('Timed out waiting for page');
  };
  try {
    await wait(()=>$('#status').textContent==='Saved');
    let saves=0;
    const originalFetch=window.fetch;
    window.fetch=(url,options)=>{if(options?.method==='PUT')saves++;return originalFetch(url,options);};
    $('[data-jump="interpreted_lithology_column"]').click();
    for(let i=0;i<3;i++) $('#in').click();
    await pause(200);
    const view=$('#view'),canvas=$('#canvas'),context=canvas.getContext('2d');
    const alpha=(x,y)=>context.getImageData(x,y,1,1).data[3];
    const top=view.scrollTop,zoom=$('#zoom').textContent;
    if(alpha(10,Math.floor(canvas.height/2))!==255)throw Error('Full page was not initially painted');
    $('#lithOnly').click();await pause(250);
    if($('#lithOnly').textContent!=='Show full page')throw Error('Reveal control missing');
    const row=context.getImageData(0,Math.floor(canvas.height/2),canvas.width,1).data;
    let hidden=0,visible=0;
    for(let x=0;x<canvas.width;x++){if(row[x*4+3]===0)hidden++;else visible++;}
    if(hidden<100||visible<5)throw Error('Mask must hide surrounding page and keep the column visible');
    if(view.scrollTop!==top||$('#zoom').textContent!==zoom)throw Error('Focus changed depth or zoom');
    $('#lithOnly').click();await pause(150);
    if(alpha(10,Math.floor(canvas.height/2))!==255)throw Error('Full page not revealed');
    $('#lithOnly').click();$('[data-jump="depth_column"]').click();await pause(150);
    if($('#lithOnly').getAttribute('aria-pressed')!=='false')throw Error('Depth jump failed to reveal page');
    $('#lithOnly').click();$('#mode').value='legend';$('#mode').dispatchEvent(new Event('change'));
    if($('#lithOnly').getAttribute('aria-pressed')!=='false')throw Error('Legend mode failed to reveal page');
    if(saves)throw Error('View toggles unexpectedly saved labels');
    report(JSON.stringify({ok:true,hiddenPixels:hidden,visibleColumnPixels:visible,depthAndZoomPreserved:true,reveal:true,saves}));
  }catch(error){report('FAIL '+error.message);}
})();
