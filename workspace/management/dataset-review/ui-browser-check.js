/* Run only in a scratch browser document. This restores the loaded catalogue. */
(() => {
  const api=window.DatasetReview,checks=[];
  const check=(name,ok)=>{checks.push({name,pass:!!ok});if(!ok)throw new Error(name);};
  const original=api.state.catalog,reviews=JSON.parse(JSON.stringify(api.state.reviews)),cropReviews=JSON.parse(JSON.stringify(api.state.suggestionReviews));
  const testImage='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="500"><rect width="200" height="500" fill="white"/><path d="M0 100H200M0 250H200M0 400H200" stroke="black"/></svg>');
  const fixture={generatedAt:'2026-11-08T07:40:00Z',records:Array.from({length:14},(_,i)=>({id:'ui-check-'+i,sourceKind:['current','synthetic','outside'][i%3],family:'Check family',split:i===13?'sealed':i===12?'shared_verify':'train',sealed:i===13,showTruthInReview:i!==12,truthStatus:i%3===2?'unlabelled':'human confirmed',ready:i!==1,image:testImage,thumbnail:testImage,imageWidth:200,imageHeight:500,boxes:[{label:'Depth',x0:0,y0:0,x1:30,y1:500},{label:'Lithology',x0:40,y0:0,x1:160,y1:500},{label:'Legend',x0:165,y0:0,x1:200,y1:100}],intervals:[{top_y:100,bottom_y:250,lithology:'Sandstone'}],baseWell:'check-well',metricReadiness:{depth:{ready:false,reason:'Fixture blocker'}},issues:i===1?[{code:'bad_depth',severity:'high',message:'Fixture issue'}]:[]})),suggestions:[{id:'ui-check-crop',recordId:'ui-check-0',bbox:[40,100,160,250],suggestedLithology:'Sandstone',modelConfidence:{value:'medium'},reviewPriority:100,sourceImageSha256:'fixture-source',evidenceCropPath:testImage,contextCropPath:testImage,contextBox:[0,50,200,300],legendContextPaths:[testImage],boundaryIssue:'Mixed lithology at contact',rawLegendIds:[1,2]}]};
  try {
    api.loadCatalog(fixture);
    check('All fixture records loaded',api.state.records.length===14);
    check('Contact sheet is paged',document.querySelectorAll('.record').length===12);
    check('Shared verification is visible',!api.sealed(fixture.records[12]));
    check('Sealed record remains sealed',api.sealed(fixture.records[13]));
    check('Technical blocked state retained',api.category(fixture.records[1])==='blocked');
    document.getElementById('source').value='synthetic';document.getElementById('source').dispatchEvent(new Event('change'));
    check('Source filter works',api.state.filtered.length===5);
    document.getElementById('source').value='';document.getElementById('source').dispatchEvent(new Event('change'));
    api.pinRecord('ui-check-0');api.pinRecord('ui-check-1');
    check('Two source images pin',document.querySelectorAll('[data-viewer]').length===2);
    api.openRecord('ui-check-1');api.setDecision('accept');
    check('Acceptance does not override blocker',api.category(api.state.records[1])==='blocked');
    check('Page triage saved',api.state.reviews['ui-check-1'].decision==='accept');
    document.getElementById('review-note').value='Review test';document.getElementById('review-note').dispatchEvent(new Event('input'));
    check('Review note saved',api.state.reviews['ui-check-1'].note==='Review test');
    ['x0','y0','x1','y1'].forEach((k,i)=>document.getElementById('box-'+k).value=[40,110,100,200][i]);document.getElementById('add-box').click();
    check('Original-pixel issue box saved',api.state.reviews['ui-check-1'].boxes[0].y0===110);
    check('Crop queue visible',!document.getElementById('crop-review').hidden);
    check('Source legend shown beside crop',document.querySelectorAll('.legend-contexts img').length===1);
    check('Neighbouring context shown',document.querySelectorAll('.crop-context img').length===2);
    check('Mixed boundary issue visible',!!document.querySelector('.crop-boundary-warning'));
    document.getElementById('crop-label').value='Shale';document.getElementById('crop-confirm').click();
    check('Explicit crop confirmation saved',api.state.suggestionReviews['ui-check-crop'].label==='Shale');
    check('Raw suggestion unchanged',fixture.suggestions[0].suggestedLithology==='Sandstone');
    api.openRecord('ui-check-12');
    check('Held-out truth overlay hidden',!document.querySelector('[data-viewer="ui-check-12"] .truth-mark'));
    api.openRecord('ui-check-13');
    check('Sealed inspector hides review controls',!document.getElementById('review-note')&&!document.querySelector('#inspector img'));
    check('No horizontal document overflow',document.documentElement.scrollWidth<=window.innerWidth+1);
    window.UI_CHECK_RESULT={passed:checks.length,failed:0,checks};console.log('UI_CHECK_PASS '+checks.length+'/'+checks.length);
  } catch(error) {window.UI_CHECK_RESULT={passed:checks.filter(c=>c.pass).length,failed:1,error:error.message,checks};console.error('UI_CHECK_FAIL '+error.message);}
  finally {api.state.reviews=reviews;api.state.suggestionReviews=cropReviews;api.loadCatalog(original);try{localStorage.setItem('viridien-dataset-review-v1',JSON.stringify(reviews));localStorage.setItem('viridien-dataset-review-v1-suggestions',JSON.stringify(cropReviews));}catch(_){};}
})();
