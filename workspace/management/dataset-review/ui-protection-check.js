/* Focused containment checks. Run in a scratch document; restore its catalogue. */
(() => {
  const api=window.DatasetReview,old=api.state.catalog,saved=JSON.parse(JSON.stringify(api.state.reviews)),savedCrop=JSON.parse(JSON.stringify(api.state.suggestionReviews));
  let checks=0;
  const check=(name,ok)=>{if(!ok)throw new Error(name);checks++;};
  const source='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="white"/><path d="M5 20H95" stroke="black"/></svg>');
  const protectedPage={id:'guard-protected',sourceKind:'current',family:'Guard test',split:'train',protectedWell:true,protectedForTraining:true,reviewPolicy:{sourceOnly:true,allowTruth:false,allowConfirm:false,allowTrain:false},image:source,imageKind:'source-only',thumbnail:source,thumbnailKind:'source-only',imageWidth:100,imageHeight:100,sourceBoxes:[{label:'Lithology',x0:10,y0:0,x1:90,y1:100}],truthStatus:'human confirmed',truth:'secret_truth.csv',notes:'__HIDDEN_PROTECTED_TRUTH__',intervals:[{top_y:20,bottom_y:40,lithology:'__HIDDEN_LABEL__'}],issues:[{message:'__HIDDEN_PROTECTED_TRUTH__'}],reviewSuggestions:[{id:'guard-suggestion',recordId:'guard-protected',suggestedLithology:'__HIDDEN_PROTECTED_SUGGESTION__',evidenceCropPath:'bench_vm/20261108T071306_dbc210ed/report.html',contextCropPath:'external_tests_round6/gallery/leak.png',legendContextPaths:['truth-overlay.png']}]};
  const heldout={id:'guard-heldout',sourceKind:'outside',family:'Guard test',split:'test_heldout',showTruthInReview:false,image:'external_tests_round6/gallery/log.png',imageKind:'source-only',thumbnail:'external_tests_round6/gallery/log.png',thumbnailKind:'source-only',existingReport:'bench_vm/20261108T071306_dbc210ed/report.html',imageWidth:100,imageHeight:100};
  try{
    check('Known DJS gallery denied despite source metadata',!api.reviewAsset(heldout,heldout.image,'source-only'));
    check('Scorer report denied in every image role',!api.reviewAsset(protectedPage,'bench_vm/20261108T071306_dbc210ed/report.html','source-only'));
    check('Encoded scorer path denied',!api.reviewAsset(protectedPage,'bench_vm%2f20261108T071306_dbc210ed%2freport.html','source-only'));
    check('Truth overlay context denied',!api.reviewAsset(protectedPage,'assets/truth-overlay.png','source-only'));
    check('Unclassified heldout preview denied',!api.reviewAsset(heldout,'assets/opaque.jpg',null));
    check('Explicit approved source preserved',!!api.reviewAsset(heldout,source,'source only (no truth)'));
    check('Hashed copy of known overlay denied by provenance',!api.asset({...heldout,image:'',thumbnail:'assets/opaque.jpg',thumbnailKind:'source-only',thumbnailSourcePath:'external_tests_round6/gallery/log.png'},true));
    check('No fallback to existing report',!api.asset(heldout));
    check('Protected current train cannot confirm',!api.canConfirm(protectedPage));
    check('Heldout cannot confirm',!api.canConfirm(heldout));
    check('Protected truth always hidden',!api.truthVisible({...protectedPage,showTruthInReview:true}));
    api.loadCatalog({records:[protectedPage,heldout],sources:[]});
    api.openRecord(protectedPage.id);
    check('Protected label not rendered',!document.getElementById('app').textContent.includes('__HIDDEN_LABEL__'));
    check('Protected notes and truth issues not rendered',!document.getElementById('app').textContent.includes('__HIDDEN_PROTECTED_TRUTH__'));
    check('Protected suggestions not rendered',!document.getElementById('app').textContent.includes('__HIDDEN_PROTECTED_SUGGESTION__'));
    check('Protected crop confirm button absent',!document.getElementById('crop-confirm'));
    check('Protected truth overlay absent',!document.querySelector('[data-viewer="guard-protected"] .truth-mark'));
    check('Protected truth link absent',![...document.querySelectorAll('#inspector a')].some(a=>/truth|report|bench_vm|gallery/.test(a.getAttribute('href'))));
    api.setDecision('accept');check('Protected accept blocked',api.state.reviews[protectedPage.id]?.decision!=='accept');
    api.setDecision('flag');check('Diagnostic flag remains available',api.state.reviews[protectedPage.id].decision==='flag');
    api.state.suggestionReviews['guard-stale']={recordId:protectedPage.id,status:'confirmed',humanConfirmed:true,humanTrainEligible:true,label:'Old label'};
    const exported=api.exportSuggestionReviews()['guard-stale'];check('Old confirmations cannot train',exported.status==='blocked-protected'&&!exported.humanConfirmed&&!exported.humanTrainEligible);
    api.openRecord(heldout.id);check('Blocked preview has visible source-only state',document.getElementById('app').textContent.includes('Source preview unavailable'));
    check('No known unsafe image or link rendered',![...document.querySelectorAll('img,image,a')].some(el=>/bench_vm|external_tests_round6\/gallery|truth-overlay/.test(el.getAttribute('src')||el.getAttribute('href')||'')));
    window.UI_PROTECTION_RESULT={passed:checks,failed:0};console.log('UI_PROTECTION_PASS '+checks+'/'+checks);
  }catch(error){window.UI_PROTECTION_RESULT={passed:checks,failed:1,error:error.message};console.error('UI_PROTECTION_FAIL '+error.message);}
  finally{api.state.reviews=saved;api.state.suggestionReviews=savedCrop;api.loadCatalog(old);try{localStorage.setItem('viridien-dataset-review-v1',JSON.stringify(saved));localStorage.setItem('viridien-dataset-review-v1-suggestions',JSON.stringify(savedCrop));}catch(_){} }
})();
