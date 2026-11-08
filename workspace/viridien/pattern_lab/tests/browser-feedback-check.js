async page => {
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const check=(v,m)=>{if(!v)throw Error(m);};
  await page.setViewportSize({width:1440,height:1000});
  await page.waitForFunction(()=>window.SharedReview?.current()&&document.getElementById('legend-view-tools'),null,{timeout:120000});
  await page.getByRole('button',{name:'Expand source legend',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.legend-dialog')?.open&&document.querySelector('.legend-dialog img')?.naturalWidth>0);
  const overview=await page.locator('.legend-dialog img').getAttribute('src');check(overview.startsWith('data:image/png;base64,'),'Full source legend must be inline for offline use.');
  await page.getByRole('slider',{name:'Legend zoom'}).fill('2');
  const zoomed=await page.locator('.legend-dialog img').evaluate(e=>e.getBoundingClientRect().width===e.naturalWidth*2);check(zoomed,'Source legend must zoom at native resolution.');
  await page.getByRole('button',{name:'Close',exact:true}).click();
  const before=await page.locator('#labels input:checked').count();await page.locator('#labels .legend-row button').first().click();
  check(await page.locator('#labels input:checked').count()===before,'Enlarge must not change selected labels.');await page.keyboard.press('Escape');
  await page.locator('#comment').fill('Feedback browser check.');await page.evaluate(()=>SharedReview.flush());
  const saved=await page.evaluate(()=>{const r=SharedReview.state(),id=SharedReview.current().id;return{id,box:[...r.answers[id].segment.box],labels:[...r.answers[id].segment.legend_ids]};});
  await page.evaluate(()=>SharedReview.improve());
  const unchanged=await page.evaluate(b=>{const a=SharedReview.state().answers[b.id];return JSON.stringify(a.segment.box)===JSON.stringify(b.box)&&JSON.stringify(a.segment.legend_ids)===JSON.stringify(b.labels);},saved);
  check(unchanged,'Improved guesses must preserve saved manual changes.');
  const repaired=await page.evaluate(async()=>{const docs=await(await fetch('/api/documents')).json();const meta=docs.find(d=>d.name==='Callaway 3-33 mudlog to TD.PDF');const d=await(await fetch('/api/document?doc='+meta.id)).json();return{active:d.legend.filter(e=>!e.deprecated).length,retired:d.legend.filter(e=>e.deprecated).length,revision:d.revision};});
  check(repaired.active===32&&repaired.retired===5,'Callaway must have 32 separate choices and retain the five old IDs.');
  await page.setViewportSize({width:390,height:844});check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Phone width must fit.');
  await page.locator('#labels .legend-row button').first().click();check(await page.locator('.legend-dialog').evaluate(e=>e.getBoundingClientRect().width<=innerWidth),'Expanded legend must fit on a phone.');await page.keyboard.press('Escape');
  const learning=await page.evaluate(()=>SharedReview.current().prediction.training_examples);check(learning>=1,'Suggestions must use current accepted examples.');
  check(errors.length===0,'Page errors: '+errors.join('; '));
  return {result:'PASS',fullLegend:true,nativeZoom:true,previewKeepsLabels:true,savedBoxesKept:true,callawayChoices:repaired.active,legacyIdsKept:repaired.retired,learningExamples:learning,phoneWidth:390,errors};
}
