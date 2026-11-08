/* Offline dataset review. No labels or source files are changed by this page. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ESC = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const SOURCE_LABELS = {current:'Current data',synthetic:'Synthetic data',outside:'Outside data'};
  const CATEGORY_LABELS = {accuracy:'Accuracy-evaluable',diagnostic:'Synthetic diagnostic',robustness:'Unlabelled robustness',blocked:'Blocked',sealed:'Sealed'};
  const STORAGE = 'viridien-dataset-review-v1';
  const PAGE_SIZE = 12;
  const state = {catalog:{records:[],sources:[]},records:[],filtered:[],reviews:{},pins:{},page:0,selected:null,views:{},drag:null,suggestionReviews:{},cropIndex:0};
  try { state.reviews = JSON.parse(localStorage.getItem(STORAGE) || '{}'); } catch (_) { state.reviews = {}; }
  try { state.suggestionReviews = JSON.parse(localStorage.getItem(STORAGE+'-suggestions') || '{}'); } catch (_) { state.suggestionReviews = {}; }
  const sealed = r => r.sealed === true || /^(sealed|locked|custodian.only)$/i.test(String(r.access || '')) || /^(sealed|locked|verify|final.verify|sealed.verify)$/i.test(String(r.split || ''));
  const protectedRecord = r => r.protectedForTraining === true || r.protectedWell === true || r.protected === true || r.registryProtected === true || r.protectedWellRegistry?.protected === true || r.reviewPolicy?.allowConfirm === false || r.reviewPolicy?.allowTrain === false;
  const sourceOnly = r => protectedRecord(r) || r.sourceOnly === true || r.reviewPolicy?.sourceOnly === true || r.showTruthInReview === false || /(^|[ _-])(test|heldout|held.out|verify|verification)([ _-]|$)/i.test(String(r.split||''));
  const truthVisible = r => !sealed(r) && !sourceOnly(r) && r.reviewPolicy?.allowTruth !== false;
  const canConfirm = r => !sealed(r) && !protectedRecord(r) && !sourceOnly(r) && r.reviewPolicy?.allowConfirm !== false;
  const issuesOf = r => Array.isArray(r.issues) ? r.issues : [];
  const num = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const recordKey = r => String(r.id);
  function decision(r) { const value=state.reviews[recordKey(r)]?.decision || 'unreviewed';return value==='accept'&&sourceOnly(r)?'unreviewed':value; }
  function category(r) {
    if (sealed(r)) return 'sealed';
    if (r.ready === false || r.blocked === true || r.evidenceCategory === 'blocked' || r.readiness?.status === 'blocked') return 'blocked';
    if (r.sourceKind === 'synthetic' || r.diagnosticOnly === true) return 'diagnostic';
    const truth = String(r.truthStatus || '').toLowerCase();
    if (/unlabelled|unlabeled|none|unknown|suggest|ai.only|silver/.test(truth) || !truth) return 'robustness';
    if (/human|confirm|ground.truth|labelled|labeled|validated|verified/.test(truth)) return 'accuracy';
    return 'robustness';
  }
  function safePath(value) {
    if (!value || typeof value !== 'string') return '';
    if (/^(?:javascript|vbscript|data:text):/i.test(value.trim())) return '';
    if (/^[A-Za-z]:[\\/]/.test(value)) return 'file:///'+value.replace(/\\/g,'/').split('/').map((s,i)=>i===0?s:encodeURIComponent(s)).join('/');
    return value;
  }
  function asset(r, thumb=false) {
    if (sealed(r)) return '';
    const imageOrigin=r.imageSourcePath||r.imageProvenance?.sourcePath||r.originalImage,thumbOrigin=r.thumbnailSourcePath||r.thumbnailProvenance?.sourcePath||r.existingThumbnail;
    const candidates=thumb?[[r.sourceThumbnail,r.sourceThumbnailKind||'source-only'],[r.thumbnail,r.thumbnailKind,thumbOrigin],[r.sourceImage,r.sourceImageKind||'source-only'],[r.image,r.imageKind,imageOrigin]]:[[r.sourceImage,r.sourceImageKind||'source-only'],[r.image,r.imageKind,imageOrigin],[r.sourceThumbnail,r.sourceThumbnailKind||'source-only'],[r.thumbnail,r.thumbnailKind,thumbOrigin]];
    for(const [path,kind,origin] of candidates){const accepted=reviewAsset(r,path,kind,origin);if(accepted)return accepted;}
    return '';
  }
  function unsafeReviewPath(value) {
    if(typeof value!=='string')return true;
    let path=value;try{path=decodeURIComponent(path);}catch(_){}
    path=path.replace(/\\/g,'/').toLowerCase();
    if(path.startsWith('data:image/'))return false;
    return /(^|\/)(bench_vm|scorer[-_]only)(\/|$)|(^|\/)external_tests_round6\/gallery\/|(^|\/)(?:comparison|report|results?)[^/]*\.html?(?:[?#]|$)|(?:truth[-_ ]?overlay|ground[-_ ]?truth|annotated|(?:^|[\/_-])truth(?:[\/_.-]|$)|overlay)/i.test(path);
  }
  function scorerPath(value){let p=String(value||'');try{p=decodeURIComponent(p);}catch(_){}return /(^|\/)(bench_vm|scorer[-_]only)(\/|$)|(^|\/)(?:comparison|report|results?)[^/]*\.html?(?:[?#]|$)/i.test(p.replace(/\\/g,'/'));}
  function reviewAsset(r,value,kind,origin) {
    if(!value||sealed(r)||unsafeReviewPath(value)||(origin&&unsafeReviewPath(origin)))return '';
    if(sourceOnly(r)&&!['source','source-only','source_only','source only (no truth)','raw-source','raw_source','public-input','input-source'].includes(String(kind||'').toLowerCase()))return '';
    return safePath(value);
  }
  function assetBounds(r) {
    const raw = r.image ? r.imageBounds || r.imageTransform : r.thumbnailBounds || r.thumbnailTransform;
    if (Array.isArray(raw) && raw.length >= 4) return raw.map(num);
    if (raw && typeof raw === 'object') {
      if (raw.x1 != null) return [num(raw.x0),num(raw.y0),num(raw.x1),num(raw.y1)];
      if (raw.width != null) return [num(raw.x || raw.x0),num(raw.y || raw.y0),num(raw.x || raw.x0)+num(raw.width),num(raw.y || raw.y0)+num(raw.height)];
    }
    return [0,0,Math.max(1,num(r.imageWidth)||1000),Math.max(1,num(r.imageHeight)||1000)];
  }
  function normalizeBox(box) {
    if (Array.isArray(box) && box.length >= 4) return {x0:num(box[0]),y0:num(box[1]),x1:num(box[2]),y1:num(box[3])};
    if (!box || typeof box !== 'object') return null;
    const b = box.box || box.bbox || box;
    if (Array.isArray(b)) return normalizeBox(b);
    const x0 = num(b.x0 ?? b.x ?? b.left), y0 = num(b.y0 ?? b.y ?? b.top);
    const x1 = num(b.x1 ?? b.right ?? (x0+num(b.width))), y1 = num(b.y1 ?? b.bottom ?? (y0+num(b.height)));
    if (x1 <= x0 || y1 <= y0) return null;
    return {x0,y0,x1,y1,label:box.label || b.label || box.name || ''};
  }
  function boxesOf(r) {
    const boxes = sourceOnly(r) ? r.sourceBoxes || r.inputBoxes || r.publicInputBoxes || (r.boxesKind==='source-only'?r.boxes:[]) || [] : r.sourceBoxes || r.inputBoxes || r.boxes || r.bboxes || [];
    return (Array.isArray(boxes) ? boxes : Object.entries(boxes).map(([label,b])=>({...normalizeBox(b),label}))).map(normalizeBox).filter(Boolean);
  }
  function priority(r) {
    const p = num(r.reviewPriority ?? r.review_priority);
    const structural = issuesOf(r).reduce((s,i)=>s+(/high|error|critical|block/i.test(i.severity || '') ? 30 : 5),0);
    const suggestion = r.suggestion || r.modelReview || {};
    let confidence = r.modelConfidence ?? r.confidence ?? suggestion.confidence;
    if(confidence && typeof confidence==='object')confidence=confidence.value;
    if (confidence != null && Number.isFinite(Number(confidence))) { confidence=Number(confidence); if(confidence>1)confidence/=100; }
    const uncertainty = confidence != null ? Math.max(0,1-confidence)*20 : 0;
    const disagreement = r.disagreement || suggestion.disagreement ? 25 : 0;
    return p+structural+uncertainty+disagreement+(decision(r)==='unreviewed'?10:0);
  }
  function notice(text,error=false) { $('notice').textContent=text; $('notice').classList.toggle('error',error); }
  function storeReviews() {
    try { localStorage.setItem(STORAGE,JSON.stringify(state.reviews));localStorage.setItem(STORAGE+'-suggestions',JSON.stringify(state.suggestionReviews)); notice('Review saved in this browser. Save the review file to share it.'); }
    catch (_) { notice('Browser storage is unavailable. Save the review file now to keep your decisions.',true); }
  }
  function download(name,data) {
    const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a'); a.href=url;a.download=name;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function loadCatalog(raw) {
    if (!raw || !Array.isArray(raw.records)) throw new Error('The catalogue must contain a records array.');
    if (raw.records.length>25000) throw new Error('The catalogue exceeds 25,000 records. Use a smaller review batch.');
    const seen=new Set();
    state.records=raw.records.filter(r=>r && r.id != null).map(r=>({...r,id:String(r.id)}));
    const duplicate=state.records.find(r=>seen.has(r.id) || !seen.add(r.id));
    if (duplicate) throw new Error('Duplicate record ID: '+duplicate.id+'. Fix the catalogue before review.');
    state.catalog=raw;state.page=0;state.selected=null;state.pins={};state.views={};
    const families=[...new Set(state.records.map(r=>r.family || 'Unknown'))].sort();
    $('family').innerHTML='<option value="">All families</option>'+families.map(f=>`<option value="${ESC(f)}">${ESC(f)}</option>`).join('');
    const gen=raw.generatedAt ? new Date(raw.generatedAt).toLocaleString('en-GB') : 'unknown';
    $('catalog-status').textContent=`${state.records.length.toLocaleString()} listed records. Catalogue updated ${gen}.`;
    renderSummary();applyFilters();renderCompare();renderInspector();renderCropQueue();
    notice('Review lowest-confidence and structurally inconsistent records first. Model confidence is not calibrated. Accept does not confirm truth labels.');
  }
  function renderSummary() {
    const counts={accuracy:0,diagnostic:0,robustness:0,blocked:0,sealed:0};state.records.forEach(r=>counts[category(r)]++);
    $('summary').innerHTML=Object.entries(CATEGORY_LABELS).map(([key,label])=>`<div class="stat"><strong>${counts[key].toLocaleString()}</strong><span>${ESC(label)}</span></div>`).join('');
    const sources=Array.isArray(state.catalog.sources)?state.catalog.sources:[];
    $('source-summary').innerHTML=`<p>These are catalogue records. Windows, pages and intervals are different units. Repeated base wells are not independent samples.</p>${sources.length?'<table><thead><tr><th>Source</th><th>Available / planned</th><th>State</th></tr></thead><tbody>'+sources.map(s=>`<tr><td>${ESC(s.name||s.id||s.path||'Source')}</td><td>${ESC(s.available ?? s.count ?? 'Unknown')} ${ESC(s.unit||'')} ${s.planned!=null?' / '+ESC(s.planned):''}</td><td>${ESC(s.status||s.notes||'See record details')}</td></tr>`).join('')+'</tbody></table>':'<p>No source totals supplied. Counts above include listed records only.</p>'}`;
    const families=[...new Set(state.records.map(r=>r.family||'Unknown'))].sort();
    $('family-summary').innerHTML=`<p>Counts are listed records, split by source. Bars use separate panels for pages, windows and other units. A group is a source well or parent group, not a rendered copy.</p>${coverageChart(families)}<div class="coverage-scroll"><table><thead><tr><th>Family</th><th>Current</th><th>Synthetic</th><th>Outside</th><th>Known source groups</th><th>Blocked</th></tr></thead><tbody>${families.map(f=>{
      const rows=state.records.filter(r=>(r.family||'Unknown')===f),counts=['current','synthetic','outside'].map(k=>rows.filter(r=>r.sourceKind===k).length),groups=new Set(rows.map(r=>r.baseWell||r.parentGroup||r.groupId).filter(Boolean));
      return `<tr><td>${ESC(f)}</td>${counts.map(n=>`<td>${n}</td>`).join('')}<td class="group-count">${groups.size}${rows.some(r=>!r.baseWell&&!r.parentGroup&&!r.groupId)?'<small>Some groups unknown</small>':''}</td><td>${rows.filter(r=>category(r)==='blocked').length}</td></tr>`;
    }).join('')}</tbody></table></div>`;
  }
  function coverageChart(families) {
    const unitOf=r=>{const u=String(r.unit||'record').toLowerCase();return ({pages:'page',windows:'window',patterns:'pattern',intervals:'interval',records:'record'})[u]||u;};
    const units=[...new Set(state.records.map(unitOf))].sort(),kinds=['current','synthetic','outside'],colors={current:'#2875b4',synthetic:'#b17b2d',outside:'#6d63b5'};
    const required=Array.isArray(state.catalog.requiredFamilies)?state.catalog.requiredFamilies:[];
    const labels=[...new Set([...families,...required])].sort();
    return '<div class="coverage-chart-key">'+kinds.map(k=>`<span style="--bar:${colors[k]}">${ESC(SOURCE_LABELS[k])}</span>`).join('')+'</div><div class="coverage-chart">'+units.map(unit=>{
      const rows=state.records.filter(r=>unitOf(r)===unit),max=Math.max(1,...labels.flatMap(f=>kinds.map(k=>rows.filter(r=>(r.family||'Unknown')===f&&r.sourceKind===k).length)));
      return `<section class="coverage-unit"><h3>${ESC(unit[0].toUpperCase()+unit.slice(1))} counts</h3>${labels.map(f=>{
        const family=rows.filter(r=>(r.family||'Unknown')===f),groups=new Set(family.map(r=>r.baseWell||r.parentGroup||r.groupId).filter(Boolean));
        return `<div class="coverage-family"><div class="coverage-family-title">${ESC(f)}</div><div class="coverage-bars">${kinds.map(k=>{const n=family.filter(r=>r.sourceKind===k).length;return `<div class="coverage-line"><span>${ESC(k==='synthetic'?'Synth':k[0].toUpperCase()+k.slice(1))}</span><div class="coverage-track"><div class="coverage-fill" style="--bar:${colors[k]};width:${100*n/max}%"></div></div><strong>${n}</strong></div>`;}).join('')}</div><div class="coverage-group-note">${groups.size} known group${groups.size===1?'':'s'}${family.some(r=>!r.baseWell&&!r.parentGroup&&!r.groupId)?'<br>Some unknown':''}</div></div>`;
      }).join('')}</section>`;
    }).join('')+'</div>';
  }
  function applyFilters() {
    const search=$('search').value.trim().toLowerCase(),family=$('family').value,source=$('source').value,evidence=$('evidence').value,review=$('review-filter').value,issues=$('issues-only').checked;
    state.filtered=state.records.filter(r=>{
      if(family && (r.family||'Unknown')!==family)return false;
      if(source && r.sourceKind!==source)return false;
      if(evidence && category(r)!==evidence)return false;
      if(review && decision(r)!==review)return false;
      if(issues && !issuesOf(r).length && decision(r)!=='flag')return false;
      if(search && !JSON.stringify(sourceOnly(r)?[r.id,r.family,r.baseWell,r.groupId]:[r.id,r.family,r.baseWell,r.groupId,r.notes,r.truthStatus,issuesOf(r),r.suggestion]).toLowerCase().includes(search))return false;
      return true;
    });
    const order=$('sort').value;
    state.filtered.sort(order==='priority'?(a,b)=>priority(b)-priority(a)||a.id.localeCompare(b.id):order==='family'?(a,b)=>String(a.family||'').localeCompare(String(b.family||''))||a.id.localeCompare(b.id):(a,b)=>a.id.localeCompare(b.id));
    state.page=Math.min(state.page,Math.max(0,Math.ceil(state.filtered.length/PAGE_SIZE)-1));renderCards();
    renderCropQueue();
  }
  function allSuggestions() {
    const top=state.catalog.suggestions || state.catalog.reviewSuggestions || [];
    const fromRecords=state.records.flatMap(r=>(r.suggestions||r.reviewSuggestions||[]).map(s=>({...s,recordId:s.recordId||r.id})));
    const rows=[...(Array.isArray(top)?top:top.records||[]),...fromRecords],seen=new Set();
    return rows.filter(s=>s && s.id && !seen.has(s.id) && seen.add(s.id)).map(s=>({...s,_record:find(String(s.recordId))}));
  }
  function suggestionPriority(s) {
    let c=s.modelConfidence?.value ?? s.modelConfidence ?? s.confidence;
    if(c!=null && Number.isFinite(Number(c))){c=Number(c);if(c>1)c/=100;}else c=1;
    return num(s.reviewPriority)+(s.disagreement?25:0)+(1-c)*20+(s.issues?.length||0)*10;
  }
  function renderCropQueue() {
    const panel=$('crop-review');if(!panel)return;
    const family=$('family').value,source=$('source').value,search=$('search').value.trim().toLowerCase();
    const all=allSuggestions().filter(s=>s._record&&canConfirm(s._record));
    const rows=all.filter(s=>(!family||(s._record.family||'Unknown')===family)&&(!source||s._record.sourceKind===source)&&(!search||JSON.stringify([s.id,s.recordId,s.suggestedLithology,s.uncertaintyReason]).toLowerCase().includes(search))).sort((a,b)=>{
      const doneA=!!state.suggestionReviews[a.id],doneB=!!state.suggestionReviews[b.id];return Number(doneA)-Number(doneB)||suggestionPriority(b)-suggestionPriority(a)||a.id.localeCompare(b.id);
    });
    panel.hidden=!all.length;if(!all.length){panel.replaceChildren();return;}
    state.cropIndex=Math.min(Math.max(0,state.cropIndex),Math.max(0,rows.length-1));
    const reviewed=all.filter(s=>state.suggestionReviews[s.id]).length,s=rows[state.cropIndex];
    if(!s){panel.innerHTML='<h2>Review model suggestions</h2><p>No crop suggestions match these filters.</p>';return;}
    const r=s._record,b=normalizeBox(s.bbox||s.box),confidence=s.modelConfidence?.value??s.modelConfidence??s.confidence,review=state.suggestionReviews[s.id],crop=reviewAsset(r,s.evidenceCrop||s.evidenceCropPath||s.cropImage||s.thumbnail,s.evidenceCropKind||s.thumbnailKind);
    const context=reviewAsset(r,s.contextCropPath||s.contextImage||s.contextCrop,s.contextCropKind),legendPaths=(s.legendContextPaths||s.legendImages||[]).filter(entry=>reviewAsset(r,typeof entry==='string'?entry:entry.path||entry.image,typeof entry==='object'?entry.kind:s.legendContextKind)),contextBox=normalizeBox(s.contextBox);
    const humanTrainEligible=canConfirm(r)&&r.sourceKind==='current' && /^(train|training)$/i.test(r.split||'');
    const contextMarkup=(context?`<details class="crop-context" open><summary>Neighbouring beds and boundaries</summary><img src="${ESC(safePath(context))}" alt="Source context around the model crop" loading="lazy">${contextBox?`<p>Original box [${contextBox.x0}, ${contextBox.y0}, ${contextBox.x1}, ${contextBox.y1}]</p>`:''}</details>`:'')+(legendPaths.length?`<details class="crop-context" open><summary>Source legend swatches</summary><div class="legend-contexts">${legendPaths.map((entry,i)=>{const path=typeof entry==='string'?entry:entry.path||entry.image;return `<figure><img src="${ESC(safePath(path))}" alt="Legend context ${i+1}" loading="lazy"><figcaption>${ESC(typeof entry==='object'?entry.label||entry.legendId||'Legend '+(i+1):'Legend '+(i+1))}</figcaption></figure>`;}).join('')}</div></details>`:'<p class="hint">No legend crop supplied. Use Show page and the source legend before confirmation.</p>');
    const bnd=assetBounds(r),image=asset(r),svg=b&&image?`<svg viewBox="${b.x0} ${b.y0} ${b.x1-b.x0} ${b.y1-b.y0}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Crop from ${ESC(r.id)}"><image href="${ESC(image)}" x="${bnd[0]}" y="${bnd[1]}" width="${bnd[2]-bnd[0]}" height="${bnd[3]-bnd[1]}" preserveAspectRatio="none"/></svg>`:'';
    panel.innerHTML=`<div class="section-bar"><div><h2>Review the least certain crops first</h2><p class="crop-progress">${all.length.toLocaleString()} model suggestions across ${new Set(all.map(s=>s.recordId)).size} records. ${reviewed} reviewed here. These are not new pages or human truth.</p></div><button class="secondary" id="crop-open-page">Show page</button></div><div class="crop-review-grid"><div><div class="crop-evidence">${safePath(crop)?`<img src="${ESC(safePath(crop))}" alt="Model input crop for ${ESC(s.id)}" onerror="this.outerHTML='<p>Crop image unavailable. Use Show page.</p>'">`:svg||'<p>No crop image supplied.</p>'}</div><p class="coord">Original pixels: ${b?`[${b.x0}, ${b.y0}, ${b.x1}, ${b.y1}]`:'Box not supplied'}. ${ESC(s.model||'Model unknown')}.</p>${contextMarkup}<p class="coord">Source image hash: ${ESC(s.sourceImageSha256||s.sourceHashes?.image||'Not supplied')}</p></div><div><h3>${ESC(s.suggestedLithology||s.suggestedLabel||s.label||'No suggested label')}</h3><p>${ESC(r.family||'Unknown family')} · ${ESC(r.id)}</p>${s.rawLegendIds?.length?`<p>Suggested legend IDs: ${ESC(s.rawLegendIds.join(', '))}</p>`:''}${s.boundaryIssue?`<p class="crop-boundary-warning">Boundary issue: ${ESC(typeof s.boundaryIssue==='string'?s.boundaryIssue:JSON.stringify(s.boundaryIssue))}. Keep mixed or uncertain spans marked for review.</p>`:''}<p>${ESC(s.uncertaintyReason||s.reason||'No uncertainty reason supplied.')}</p><p class="hint">${confidence!=null?`Model confidence ${ESC(confidence)}. `:''}Confidence is not calibrated. ${s.disagreement?'Models disagree. ':''}Compare the crop with the source legend before you confirm it.</p><label for="crop-label">Confirmed or corrected label</label><input id="crop-label" class="crop-label" value="${ESC(review?.label||s.suggestedLithology||s.suggestedLabel||s.label||'')}" autocomplete="off"><label for="crop-note">Review note</label><textarea id="crop-note" class="review-note" placeholder="Explain a correction or an uncertain boundary.">${ESC(review?.note||'')}</textarea><div class="review-buttons"><button id="crop-confirm">Confirm label</button><button class="secondary" id="crop-cannot-tell">Cannot tell</button><button class="secondary" id="crop-skip">Next crop</button></div><p class="save-state">${review?'Saved '+ESC(review.status)+'. Source truth is unchanged.':'Confirmation is saved for export. It does not change the dataset files.'}</p></div></div>`;
    $('crop-open-page').onclick=()=>{openRecord(r.id);if(b){viewerState(r).y=b.y0;viewerState(r).window=Math.max(250,b.y1-b.y0+100);renderCompare();}$('comparison').scrollIntoView({block:'nearest'});};
    if(!humanTrainEligible){const p=document.createElement('p');p.className='crop-boundary-warning';p.textContent='Diagnostic review only. This source cannot add labels to the human training pack.';$('crop-confirm').parentElement.before(p);$('crop-confirm').textContent='Confirm diagnostic label';}
    const save=status=>{
      if(!canConfirm(r)){notice('This well is protected. Label confirmation and training export are disabled.',true);return;}
      const label=$('crop-label').value.trim();if(status==='confirmed'&&!label){notice('Enter a label before you confirm this crop.',true);return;}
      state.suggestionReviews[s.id]={status,label:status==='confirmed'?label:null,note:$('crop-note').value,recordId:s.recordId,sourceKind:r.sourceKind,split:r.split,humanTrainEligible,reviewScope:humanTrainEligible?'human-train-candidate':'diagnostic-only',sourceImageSha256:s.sourceImageSha256||s.sourceHashes?.image||null,annotationSha256:s.sourceHashes?.annotation||null,modelInputHash:s.modelInputHash||null,originalSuggestion:s.suggestedLithology||s.suggestedLabel||s.label||null,box:b,humanConfirmed:status==='confirmed',confirmedAt:new Date().toISOString(),meaning:'Human review export. Source truth unchanged until validated import.'};
      storeReviews();state.cropIndex=0;renderCropQueue();
    };
    $('crop-confirm').onclick=()=>save('confirmed');$('crop-cannot-tell').onclick=()=>save('uncertain');$('crop-skip').onclick=()=>{state.cropIndex=(state.cropIndex+1)%rows.length;renderCropQueue();};
  }
  function chips(r) {
    const cat=category(r),review=decision(r),issueCount=issuesOf(r).length;
    return `<div class="chips"><span class="chip ${cat==='blocked'?'bad':cat==='accuracy'?'good':cat==='diagnostic'?'warn':''}">${ESC(CATEGORY_LABELS[cat])}</span><span class="chip">${ESC(r.split||'Split unknown')}</span><span class="chip">${ESC(r.truthStatus||'Truth unknown')}</span>${issueCount?`<span class="chip warn">${issueCount} issue${issueCount===1?'':'s'}</span>`:''}${review!=='unreviewed'?`<span class="chip ${review==='exclude'?'bad':review==='flag'?'warn':'good'}">${ESC(review)}</span>`:''}</div>`;
  }
  function renderCards() {
    const start=state.page*PAGE_SIZE,items=state.filtered.slice(start,start+PAGE_SIZE);
    $('results-status').textContent=`${state.filtered.length.toLocaleString()} match. ${state.records.filter(r=>decision(r)!=='unreviewed').length.toLocaleString()} reviewed in this catalogue.`;
    $('records').innerHTML=items.length?items.map(r=>{
      const image=asset(r,true);
      return `<article class="record ${state.selected===r.id?'selected':''}" data-record="${ESC(r.id)}"><button class="record-top" data-open="${ESC(r.id)}" aria-label="Open ${ESC(r.id)}">${image?`<img class="record-thumb" src="${ESC(image)}" alt="${ESC(r.family||'Log')} preview: ${ESC(r.id)}" loading="lazy" onerror="this.outerHTML='<span class=&quot;thumb-missing&quot;>Image unavailable. See record paths.</span>'">`:`<span class="thumb-missing">${sealed(r)?'Sealed record. Assets are hidden.':'No preview supplied. Open record for paths and issues.'}</span>`}<div class="record-info"><h3>${ESC(r.id)}</h3><p>${ESC(SOURCE_LABELS[r.sourceKind]||r.sourceKind||'Source unknown')} · ${ESC(r.family||'Unknown family')}</p>${chips(r)}<p>${ESC(r.baseWell||r.groupId||'Base well not supplied')}</p></div></button><div class="record-actions"><button class="secondary" data-pin="${ESC(r.id)}">Compare</button><button class="secondary" data-quick-flag="${ESC(r.id)}">Flag</button></div></article>`;
    }).join(''):'<div class="empty-state"><h3>No records match</h3><p>Change a filter or load a catalogue file.</p></div>';
    $('page-status').textContent=`Page ${state.page+1} of ${Math.max(1,Math.ceil(state.filtered.length/PAGE_SIZE))}`;
    $('prev-page').disabled=state.page===0;$('next-page').disabled=start+PAGE_SIZE>=state.filtered.length;
    $('records').querySelectorAll('[data-open]').forEach(el=>el.onclick=()=>openRecord(el.dataset.open));
    $('records').querySelectorAll('.thumb-missing').forEach(el=>{const r=find(el.closest('[data-record]').dataset.record);if(r&&sourceOnly(r)&&!sealed(r))el.textContent='Source preview unavailable. Truth overlays and scorer reports are blocked.';});
    $('records').querySelectorAll('[data-pin]').forEach(el=>el.onclick=()=>pinRecord(el.dataset.pin));
    $('records').querySelectorAll('[data-quick-flag]').forEach(el=>el.onclick=()=>{openRecord(el.dataset.quickFlag);setDecision('flag');});
  }
  function find(id){return state.records.find(r=>r.id===id);}
  function openRecord(id) {
    const r=find(id);if(!r)return;state.selected=id;
    if(!sealed(r)){state.pins[r.sourceKind||'outside']=id;}
    renderCompare();renderInspector();renderCards();
  }
  function pinRecord(id) {
    const r=find(id);if(!r)return;
    if(sealed(r)){notice('This record is sealed. Its assets cannot be compared.',true);return;}
    state.pins[r.sourceKind||'outside']=id;state.selected=id;renderCompare();renderInspector();renderCards();
  }
  function viewerState(r) {
    if(!state.views[r.id]) {
      const bounds=assetBounds(r),boxes=boxesOf(r),lith=boxes.find(b=>/lith|rock|column/i.test(b.label))||boxes[1];
      const region=lith||{x0:bounds[0],y0:bounds[1],x1:bounds[2],y1:bounds[3]};
      state.views[r.id]={mode:lith?'lithology':'page',y:region.y0,window:Math.min(1000,region.y1-region.y0),truth:true,boxes:true};
    }
    return state.views[r.id];
  }
  function viewBox(r) {
    const v=viewerState(r),bounds=assetBounds(r),boxes=boxesOf(r);
    let region={x0:bounds[0],y0:bounds[1],x1:bounds[2],y1:bounds[3]};
    if(v.mode!=='page'&&v.mode!=='overview')region=boxes.find(b=>v.mode==='lithology'?/lith|rock|column/i.test(b.label):new RegExp(v.mode,'i').test(b.label))||boxes[1]||region;
    const clipped={x0:Math.max(bounds[0],region.x0),y0:Math.max(bounds[1],region.y0),x1:Math.min(bounds[2],region.x1),y1:Math.min(bounds[3],region.y1)};
    v.regionUnavailable=clipped.x1<=clipped.x0||clipped.y1<=clipped.y0;
    region=v.regionUnavailable?{x0:bounds[0],y0:bounds[1],x1:bounds[2],y1:bounds[3]}:clipped;
    const width=Math.max(1,region.x1-region.x0),height=Math.max(1,region.y1-region.y0);
    const win=v.mode==='overview'?height:Math.min(v.window,height);
    const y=Math.max(region.y0,Math.min(v.y,region.y1-win));v.y=y;
    return {x:region.x0,y,w:width,h:win,minY:region.y0,maxY:Math.max(region.y0,region.y1-win)};
  }
  function svgMarkup(r) {
    const image=asset(r);if(!image)return `<div class="no-image">${sealed(r)?'Sealed assets are hidden.':sourceOnly(r)?'Source preview unavailable. No fallback to reports or truth overlays is permitted.':'No permitted browser-readable source image supplied.'}</div>`;
    const b=assetBounds(r),v=viewerState(r),view=viewBox(r),colors=['#2e78c7','#d38617','#8c58b5'];
    const mark=(box,color,cls='box-mark')=>`<rect class="${cls}" x="${box.x0}" y="${box.y0}" width="${box.x1-box.x0}" height="${box.y1-box.y0}" stroke="${color}" vector-effect="non-scaling-stroke"/>`;
    const boxes=boxesOf(r),lith=boxes.find(box=>/lith|rock|column/i.test(box.label))||boxes[1]||{x0:b[0],x1:b[2]};
    const intervals=Array.isArray(r.intervals)?r.intervals:Array.isArray(r.truthIntervals)?r.truthIntervals:[];
    const truth=v.truth&&truthVisible(r)?intervals.filter(t=>num(t.bottom_y ?? t.bottomY)>=view.y && num(t.top_y ?? t.topY)<=view.y+view.h).slice(0,800).map(t=>{
      const top=num(t.top_y ?? t.topY),bottom=num(t.bottom_y ?? t.bottomY);
      return `<g><title>${ESC(t.lithology||t.label||'Truth interval')} [${top}, ${bottom})</title><line class="truth-mark" x1="${lith.x0}" x2="${lith.x1}" y1="${top}" y2="${top}" vector-effect="non-scaling-stroke"/></g>`;
    }).join(''):'';
    const rawTruthBoxes=r.truthBoxes||(r.sourceBoxes?(r.boxes||[]).filter(b=>b.humanConfirmed):[]);
    const truthBoxes=v.truth&&truthVisible(r)?rawTruthBoxes.map(normalizeBox).filter(Boolean).filter(t=>t.y1>=view.y&&t.y0<=view.y+view.h).slice(0,800).map(box=>mark(box,'#19a7ad')).join(''):'';
    const saved=state.reviews[r.id]?.boxes||[];
    return `<svg data-viewer="${ESC(r.id)}" viewBox="${view.x} ${view.y} ${view.w} ${view.h}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${ESC(r.id)} image with original-pixel overlays"><image href="${ESC(image)}" x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" preserveAspectRatio="none"/>${v.boxes?boxes.map((box,i)=>`<g><title>${ESC(box.label||'Source box '+(i+1))} [${box.x0}, ${box.y0}, ${box.x1}, ${box.y1}]</title>${mark(box,colors[i%colors.length])}</g>`).join(''):''}${truth}${truthBoxes}${saved.map(box=>mark(box,'#dd2e74','user-box box-mark')).join('')}<rect class="drag-box user-box box-mark" hidden vector-effect="non-scaling-stroke"/></svg>`;
  }
  function renderCompare() {
    const pinned=Object.values(state.pins).map(find).filter(Boolean),families=new Set(pinned.map(r=>r.family));
    $('comparison').innerHTML=['current','synthetic','outside'].map(kind=>{
      const r=find(state.pins[kind]);
      if(!r)return `<article class="compare-pane"><div class="compare-head"><h3>${SOURCE_LABELS[kind]}</h3></div><div class="compare-empty">Select “Compare” on a ${kind} record.</div></article>`;
      const v=viewerState(r),view=viewBox(r),boxes=boxesOf(r),source=asset(r);
      return `<article class="compare-pane"><div class="compare-head"><span class="kind">${SOURCE_LABELS[kind]}</span><h3>${ESC(r.id)}</h3>${chips(r)}</div>${families.size>1?'<p class="group-warning">Different families are pinned. This is not a matched comparison.</p>':''}${v.regionUnavailable?'<p class="group-warning">This preview does not cover the requested region. Showing available pixels instead.</p>':''}<div class="viewer">${svgMarkup(r)}</div><div class="view-tools"><label>Region<select data-region="${ESC(r.id)}"><option value="page" ${v.mode==='page'?'selected':''}>Page window</option><option value="lithology" ${v.mode==='lithology'?'selected':''}>Lithology</option><option value="legend" ${v.mode==='legend'?'selected':''}>Legend</option><option value="depth" ${v.mode==='depth'?'selected':''}>Depth</option><option value="overview" ${v.mode==='overview'?'selected':''}>Overview</option></select></label><label>Rows<select data-window="${ESC(r.id)}">${[250,500,1000,2000,6000].map(n=>`<option ${v.window===n?'selected':''}>${n}</option>`).join('')}</select></label><label><input type="checkbox" data-boxes="${ESC(r.id)}" ${v.boxes?'checked':''}>Boxes</label><label><input type="checkbox" data-truth="${ESC(r.id)}" ${v.truth?'checked':''}>Truth</label><div class="view-range"><input aria-label="Starting row for ${ESC(r.id)}" type="range" min="${view.minY}" max="${view.maxY}" step="1" value="${view.y}" data-row="${ESC(r.id)}"></div></div><p class="coord">Original pixels: x ${Math.round(view.x)}–${Math.round(view.x+view.w)}, y ${Math.round(view.y)}–${Math.round(view.y+view.h)}. Image ${num(r.imageWidth)||'?'} × ${num(r.imageHeight)||'?'}. ${!r.image?'Overview asset only; small beds may not be visible.':''}</p><div class="legend-key">${boxes.slice(0,6).map((box,i)=>`<span style="--key:${['#2e78c7','#d38617','#8c58b5'][i%3]}">${ESC(box.label||'Box '+(i+1))}</span>`).join('')}<span style="--key:#19a7ad">Truth boundaries</span><span style="--key:#dd2e74">Review box</span></div><div class="pane-foot"><button class="secondary" data-inspect="${ESC(r.id)}">Review record</button>${source?`<button class="secondary" data-zoom="${ESC(r.id)}">Open image</button>`:''}<button class="secondary" data-unpin="${kind}">Remove</button></div></article>`;
    }).join('');
    $('comparison').querySelectorAll('[data-inspect]').forEach(el=>el.onclick=()=>{state.selected=el.dataset.inspect;renderInspector();renderCards();$('inspector').scrollIntoView({block:'nearest'});});
    $('comparison').querySelectorAll('[data-unpin]').forEach(el=>el.onclick=()=>{delete state.pins[el.dataset.unpin];renderCompare();});
    $('comparison').querySelectorAll('[data-zoom]').forEach(el=>el.onclick=()=>zoomImage(find(el.dataset.zoom)));
    [['region','mode'],['window','window'],['boxes','boxes'],['truth','truth']].forEach(([data,key])=>{
      $('comparison').querySelectorAll(`[data-${data}]`).forEach(el=>el.onchange=()=>{const r=find(el.dataset[data]),v=viewerState(r);v[key]=el.type==='checkbox'?el.checked:key==='window'?num(el.value):el.value;renderCompare();});
    });
    $('comparison').querySelectorAll('[data-row]').forEach(el=>el.onchange=()=>{viewerState(find(el.dataset.row)).y=num(el.value);renderCompare();});
    $('comparison').querySelectorAll('[data-viewer]').forEach(bindDrawing);
    $('comparison').querySelectorAll('[data-truth]').forEach(el=>{if(sourceOnly(find(el.dataset.truth))){el.checked=false;el.disabled=true;el.title='Truth is hidden for protected source-only review.';}});
  }
  function zoomImage(r) {
    const image=asset(r);if(!image)return;
    let d=$('image-dialog');if(d)d.remove();d=document.createElement('dialog');d.id='image-dialog';d.className='zoom-dialog';
    d.innerHTML=`<button class="secondary">Close image</button><h3>${ESC(r.id)}</h3><p class="muted">Browser-readable asset. Original source may be a separate TIFF.</p><img src="${ESC(image)}" alt="${ESC(r.id)}">`;
    document.body.appendChild(d);d.querySelector('button').onclick=()=>{d.close();d.remove();};d.showModal();
  }
  function bindDrawing(svg) {
    const toPoint=e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());};
    svg.onpointerdown=e=>{
      if(e.button!==0)return;const p=toPoint(e);state.drag={id:svg.dataset.viewer,x:p.x,y:p.y};svg.setPointerCapture(e.pointerId);
      const rect=svg.querySelector('.drag-box');rect.removeAttribute('hidden');rect.setAttribute('x',p.x);rect.setAttribute('y',p.y);rect.setAttribute('width',0);rect.setAttribute('height',0);
    };
    svg.onpointermove=e=>{if(!state.drag||state.drag.id!==svg.dataset.viewer)return;const p=toPoint(e),d=state.drag,rect=svg.querySelector('.drag-box');rect.setAttribute('x',Math.min(p.x,d.x));rect.setAttribute('y',Math.min(p.y,d.y));rect.setAttribute('width',Math.abs(p.x-d.x));rect.setAttribute('height',Math.abs(p.y-d.y));};
    svg.onpointerup=e=>{
      const d=state.drag;if(!d||d.id!==svg.dataset.viewer)return;state.drag=null;
      const p=toPoint(e);if(Math.abs(p.x-d.x)<2||Math.abs(p.y-d.y)<2){svg.querySelector('.drag-box').setAttribute('hidden','');return;}
      state.selected=d.id;const b={x0:Math.round(Math.min(d.x,p.x)),y0:Math.round(Math.min(d.y,p.y)),x1:Math.round(Math.max(d.x,p.x)),y1:Math.round(Math.max(d.y,p.y))};
      const old=state.reviews[d.id]||{};state.reviews[d.id]={...old,decision:'flag',boxes:[...(old.boxes||[]),b],updatedAt:new Date().toISOString()};
      storeReviews();renderCompare();renderInspector();renderCards();notice('Review box saved in original pixels. Add a note to explain the issue.');
    };
    svg.onpointercancel=()=>{state.drag=null;};
  }
  function suggestionText(r) {
    const s=r.suggestion||r.modelReview||r.reviewSuggestion;if(!s && r.modelConfidence==null && !r.disagreement)return '';
    const rawConfidence=r.modelConfidence ?? s?.confidence,confidence=rawConfidence?.value??rawConfidence;const labels=s?.labels||s?.suggestedLabels||s?.label||r.suggestedLabels;
    return `<div class="issue-list"><li><strong>Model suggestion. Human confirmation is still required.</strong>${labels?`<p>Suggested: ${ESC(typeof labels==='string'?labels:JSON.stringify(labels))}</p>`:''}${confidence!=null?`<p>Model confidence: ${ESC(confidence)}. This value is not calibrated.</p>`:''}${r.disagreement||s?.disagreement?'<p>Models disagree on this record.</p>':''}${s?.reason||r.uncertaintyReason?`<p>${ESC(s?.reason||r.uncertaintyReason)}</p>`:''}</li></div>`;
  }
  function fileLinks(r) {
    if(sealed(r))return '<p class="muted">Sealed paths and truth are hidden.</p>';
    if(sourceOnly(r)){
      const image=asset(r),annotation=r.annotation||r.paths?.annotation;
      return '<div class="file-links">'+(image?`<a href="${ESC(image)}" target="_blank" rel="noopener">Source image</a>`:'<span>Source preview unavailable.</span>')+(annotation&&!unsafeReviewPath(annotation)?`<a href="${ESC(safePath(annotation))}" target="_blank" rel="noopener">Public input boxes</a>`:'')+'</div><p class="hint">Source-only review. Truth, scorer reports, model suggestions and training confirmation are hidden.</p>';
    }
    return '<div class="file-links">'+[['Image',asset(r)],['Annotation',r.annotation||r.paths?.annotation],['Truth CSV',truthVisible(r)?r.truth||r.paths?.truth:null],['Ignore CSV',truthVisible(r)?r.ignore||r.paths?.ignore:null],['Manifest',r.showTruthInReview===false?null:r.manifest]].filter(([label,p])=>safePath(p)&&!scorerPath(p)&&(['Truth CSV','Ignore CSV'].includes(label)||!unsafeReviewPath(p))).map(([label,path])=>`<a href="${ESC(safePath(path))}" target="_blank" rel="noopener">${label}</a>`).join('')+'</div>'+(r.showTruthInReview===false?'<p class="hint">Held-out truth is hidden in this review page.</p>':'');
  }
  function readinessMarkup(r) {
    const values=r.metricReadiness||r.metricsReady||r.readiness?.metrics;
    if(!values||typeof values!=='object')return '<p class="statistical-note">Metric readiness is not supplied. Visual acceptance cannot override technical blockers.</p>';
    return '<h3>Metric readiness</h3><div class="metric-readiness">'+Object.entries(values).map(([name,value])=>{
      const val=typeof value==='object'&&value!==null?value:{status:typeof value==='string'?value:value===true?'Ready':value===false?'Blocked':'Unknown',ready:typeof value==='boolean'?value:undefined};
      const yes=val.ready===true||/^ready$|^pass$/i.test(val.status||''),no=val.ready===false||/block|fail|invalid|unavailable/i.test(val.status||'');
      return `<div class="metric ${yes?'yes':no?'no':'unknown'}"><strong>${ESC(name)}</strong><span>${ESC(val.status|| (val.ready===true?'Ready':val.ready===false?'Blocked':'Unknown'))}</span>${val.reason?`<p>${ESC(val.reason)}</p>`:''}</div>`;
    }).join('')+'</div>';
  }
  function renderInspector() {
    const r=find(state.selected),panel=$('inspector');panel.hidden=!r;if(!r){panel.replaceChildren();return;}
    if(sealed(r)){
      panel.innerHTML=`<div class="inspector-head"><h2>${ESC(r.id)}</h2><button class="secondary" id="close-inspector">Close review</button></div><p>This record is sealed. Images, truth, model suggestions and file paths are hidden. Use the test custodian's isolated evaluation process.</p>`;
      $('close-inspector').onclick=()=>{state.selected=null;renderInspector();renderCards();};return;
    }
    if(sourceOnly(r)){
      const saved=state.reviews[r.id]||{};
      panel.innerHTML=`<div class="inspector-head"><h2>${ESC(r.id)}</h2><button class="secondary" id="close-inspector">Close review</button></div><p class="crop-boundary-warning">Protected source-only inspection. Truth labels, scorer reports and model suggestions are hidden. Accept, label confirmation and training actions are disabled.</p><dl class="metadata"><dt>Source</dt><dd>${ESC(SOURCE_LABELS[r.sourceKind]||r.sourceKind)}</dd><dt>Family</dt><dd>${ESC(r.family||'Unknown')}</dd><dt>Base well</dt><dd>${ESC(r.baseWell||r.groupId||'Unknown')}</dd></dl>${fileLinks(r)}<div class="review-buttons"><button class="secondary" data-protected-decision="flag">Flag source issue</button><button class="secondary" data-protected-decision="exclude">Exclude source</button><button class="secondary" data-protected-decision="unreviewed">Clear decision</button></div><label for="review-note">Source issue note</label><textarea id="review-note" class="review-note">${ESC(saved.note||'')}</textarea><p class="hint">Notes and drawn boxes are diagnostic observations only.</p>`;
      $('close-inspector').onclick=()=>{state.selected=null;renderInspector();renderCards();};
      panel.querySelectorAll('[data-protected-decision]').forEach(el=>el.onclick=()=>setDecision(el.dataset.protectedDecision));
      $('review-note').oninput=()=>{state.reviews[r.id]={...(state.reviews[r.id]||{}),note:$('review-note').value,diagnosticOnly:true,updatedAt:new Date().toISOString()};storeReviews();};
      return;
    }
    const review=state.reviews[r.id]||{},boxes=review.boxes||[];
    panel.innerHTML=`<div class="inspector-head"><div><h2>${ESC(r.id)}</h2>${chips(r)}</div><button class="secondary" id="close-inspector">Close review</button></div><div class="detail-grid"><div><dl class="metadata"><dt>Source</dt><dd>${ESC(SOURCE_LABELS[r.sourceKind]||r.sourceKind||'Unknown')}</dd><dt>Family</dt><dd>${ESC(r.family||'Unknown')}</dd><dt>Base well</dt><dd>${ESC(r.baseWell||'Not supplied')}</dd><dt>Group</dt><dd>${ESC(r.groupId||r.parentGroup||'Not supplied')}</dd><dt>Truth status</dt><dd>${ESC(r.truthStatus||'Unknown')}</dd><dt>Truth source</dt><dd>${ESC(typeof r.truthProvenance==='object'?JSON.stringify(r.truthProvenance):r.truthProvenance||'Not supplied')}</dd><dt>Evaluation state</dt><dd>${ESC(CATEGORY_LABELS[category(r)])}</dd>${r.bedCount!=null?`<dt>Beds</dt><dd>${ESC(r.bedCount)}</dd>`:''}${r.medianBedThickness!=null?`<dt>Median thickness</dt><dd>${ESC(r.medianBedThickness)} original rows</dd>`:''}${r.thinBedFraction!=null?`<dt>Thin-bed share</dt><dd>${ESC(r.thinBedFraction)}</dd>`:''}</dl>${r.notes?`<p>${ESC(Array.isArray(r.notes)?r.notes.join(' '):r.notes)}</p>`:''}${readinessMarkup(r)}${fileLinks(r)}${suggestionText(r)}<h3>Recorded issues</h3><ul class="issue-list">${issuesOf(r).length?issuesOf(r).map((i,n)=>`<li class="${ESC(i.severity||'')}"><strong>${ESC(i.code||i.kind||'Issue')}</strong><p>${ESC(i.message||i.explanation||i.description||'No explanation supplied.')}</p>${i.box?`<button class="secondary" data-issue-focus="${n}">Show issue box</button><p>${ESC(JSON.stringify(i.box))}</p>`:''}</li>`).join(''):'<li>No automatic issue is recorded. This does not confirm the data is correct.</li>'}</ul></div><div><h3>Your review</h3><p class="hint">Accept is a triage decision. It does not confirm or replace truth labels. Technical blockers stay in force.</p><div class="review-buttons">${['accept','flag','exclude','unreviewed'].map(d=>`<button class="secondary ${decision(r)===d?'active':''}" data-decision="${d}">${d==='unreviewed'?'Clear decision':d[0].toUpperCase()+d.slice(1)}</button>`).join('')}</div><label for="review-note">Explain the decision</label><textarea id="review-note" class="review-note" placeholder="What differs? Which rows or labels need a check?">${ESC(review.note||'')}</textarea><p class="hint">Draw a box on a comparison image, or enter original pixel coordinates.</p><div class="coord-inputs">${['x0','y0','x1','y1'].map(name=>`<label>${name}<input id="box-${name}" type="number" min="0" step="1"></label>`).join('')}</div><div class="review-buttons"><button class="secondary" id="add-box">Add issue box</button><button class="secondary" id="next-record">Next record</button></div><ul class="issue-list">${boxes.map((b,i)=>`<li>Box ${i+1}: [${b.x0}, ${b.y0}, ${b.x1}, ${b.y1}] <button class="secondary" data-remove-box="${i}">Remove</button></li>`).join('')}</ul><p class="save-state">${review.updatedAt?'Last saved '+ESC(new Date(review.updatedAt).toLocaleString('en-GB')):'No decision saved for this record.'}</p></div></div>`;
    $('close-inspector').onclick=()=>{state.selected=null;renderInspector();renderCards();};
    panel.querySelectorAll('[data-decision]').forEach(el=>el.onclick=()=>setDecision(el.dataset.decision));
    $('review-note').oninput=()=>{const old=state.reviews[r.id]||{};state.reviews[r.id]={...old,note:$('review-note').value,updatedAt:new Date().toISOString()};storeReviews();};
    $('next-record').onclick=()=>stepRecord(1);
    $('add-box').onclick=()=>{
      const b=normalizeBox({x0:$('box-x0').value,y0:$('box-y0').value,x1:$('box-x1').value,y1:$('box-y1').value});
      if(!b){notice('Enter a box with x1 greater than x0 and y1 greater than y0.',true);return;}
      if((num(r.imageWidth)&&b.x1>num(r.imageWidth))||(num(r.imageHeight)&&b.y1>num(r.imageHeight))){notice('The box exceeds the original image bounds.',true);return;}
      state.reviews[r.id]={...(state.reviews[r.id]||{}),decision:'flag',boxes:[...(state.reviews[r.id]?.boxes||[]),b],updatedAt:new Date().toISOString()};storeReviews();renderInspector();renderCompare();renderCards();
    };
    panel.querySelectorAll('[data-remove-box]').forEach(el=>el.onclick=()=>{state.reviews[r.id].boxes.splice(num(el.dataset.removeBox),1);state.reviews[r.id].updatedAt=new Date().toISOString();storeReviews();renderInspector();renderCompare();});
    panel.querySelectorAll('[data-issue-focus]').forEach(el=>el.onclick=()=>{const issue=issuesOf(r)[num(el.dataset.issueFocus)],b=normalizeBox(issue.box);if(b){viewerState(r).y=b.y0;viewerState(r).window=Math.max(250,b.y1-b.y0+50);state.pins[r.sourceKind]=r.id;renderCompare();$('comparison').scrollIntoView({block:'nearest'});}});
  }
  function setDecision(value) {
    if(!state.selected)return;const r=find(state.selected);if(sealed(r)){notice('Sealed records cannot be accepted or changed here.',true);return;}
    if(value==='accept'&&sourceOnly(r)){notice('This well is protected. Acceptance and training confirmation are disabled. You can flag a source issue.',true);return;}
    state.reviews[r.id]={...(state.reviews[r.id]||{}),decision:value,diagnosticOnly:sourceOnly(r),updatedAt:new Date().toISOString(),sourceHash:r.sourceHash||r.imageHash||r.sha256||null};
    storeReviews();renderInspector();renderCompare();applyFilters();
  }
  function stepRecord(direction) {
    if(!state.filtered.length)return;let i=state.filtered.findIndex(r=>r.id===state.selected);i=Math.max(0,Math.min(state.filtered.length-1,i+direction));
    state.page=Math.floor(i/PAGE_SIZE);openRecord(state.filtered[i].id);
  }
  $('load-catalog').onchange=async e=>{try{const raw=JSON.parse(await e.target.files[0].text());loadCatalog(raw);}catch(err){notice(err.message,true);}finally{e.target.value='';}};
  $('load-review').onchange=async e=>{try{const raw=JSON.parse(await e.target.files[0].text());if(!raw.reviews||typeof raw.reviews!=='object'||Array.isArray(raw.reviews))throw new Error('The review file must contain a reviews object.');let count=0;Object.entries(raw.reviews).forEach(([id,v])=>{if(v&&['accept','flag','exclude','unreviewed',undefined].includes(v.decision)){state.reviews[id]=v;count++;}});if(raw.suggestionReviews&&typeof raw.suggestionReviews==='object'&&!Array.isArray(raw.suggestionReviews))Object.entries(raw.suggestionReviews).forEach(([id,v])=>{if(v&&['confirmed','uncertain'].includes(v.status))state.suggestionReviews[id]=v;});storeReviews();applyFilters();renderInspector();renderCompare();notice(`${count} page decisions loaded. Crop reviews are also restored if present. Decisions for absent records are retained.`);}catch(err){notice(err.message,true);}finally{e.target.value='';}};
  function exportSuggestionReviews(){return Object.fromEntries(Object.entries(state.suggestionReviews).map(([id,review])=>{const r=find(review.recordId);return [id,!r||!canConfirm(r)?{...review,status:'blocked-protected',humanConfirmed:false,humanTrainEligible:false,reviewScope:'protected-diagnostic-only',blockedReason:'Current catalogue does not permit label confirmation or training for this source.'}:review];}));}
  function exportPageReviews(){return Object.fromEntries(Object.entries(state.reviews).map(([id,review])=>{const r=find(id);return[id,r&&sourceOnly(r)?{...review,decision:review.decision==='accept'?'unreviewed':review.decision,diagnosticOnly:true,humanTrainEligible:false}:review];}));}
  $('export-review').onclick=()=>download('dataset-review-'+new Date().toISOString().slice(0,10)+'.json',{schemaVersion:1,exportedAt:new Date().toISOString(),catalogGeneratedAt:state.catalog.generatedAt||null,meaning:'Page decisions are triage only. suggestionReviews contains explicit human crop review; source truth stays unchanged until validated import.',reviews:exportPageReviews(),suggestionReviews:exportSuggestionReviews()});
  $('export-selection').onclick=()=>download('dataset-review-selection.json',{schemaVersion:1,exportedAt:new Date().toISOString(),kind:'review-selection-not-harness-manifest',records:state.filtered.filter(r=>!sealed(r)&&decision(r)!=='exclude').map(r=>({id:r.id,sourceKind:r.sourceKind,family:r.family,split:r.split,truthStatus:r.truthStatus,evidenceCategory:category(r),decision:decision(r),manifest:sourceOnly(r)||unsafeReviewPath(r.manifest)?null:r.manifest||null,sourceHash:r.sourceHash||r.imageHash||null,sourceOnly:sourceOnly(r),protectedForTraining:protectedRecord(r),humanTrainEligible:canConfirm(r)&&r.sourceKind==='current'&&/^(train|training)$/i.test(r.split||'')})),note:'Exclusions and sealed records removed. Protected records are source-only diagnostics. Dataset owner must validate and convert this list to the harness format.'});
  ['search','family','source','evidence','review-filter','issues-only','sort'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',()=>{state.page=0;applyFilters();}));
  $('prev-page').onclick=()=>{state.page--;renderCards();};$('next-page').onclick=()=>{state.page++;renderCards();};
  $('clear-pins').onclick=()=>{state.pins={};renderCompare();};
  document.addEventListener('keydown',e=>{if(e.altKey||e.ctrlKey||e.metaKey||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||e.target.isContentEditable)return;const key=e.key.toLowerCase();if(['j','k','f','a','x'].includes(key))e.preventDefault();if(key==='j')stepRecord(1);if(key==='k')stepRecord(-1);if(key==='f')setDecision('flag');if(key==='a')setDecision('accept');if(key==='x')setDecision('exclude');});
  window.DatasetReview={state,loadCatalog,category,priority,normalizeBox,assetBounds,sealed,openRecord,pinRecord,setDecision,stepRecord,allSuggestions,suggestionPriority,sourceOnly,truthVisible,protectedRecord,canConfirm,asset,reviewAsset,unsafeReviewPath,exportSuggestionReviews};
  try { loadCatalog(window.DATASET_CATALOG || {records:[],sources:[]}); } catch(err) { notice(err.message,true); }
})();
