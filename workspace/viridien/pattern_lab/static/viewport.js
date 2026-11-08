/* Source coordinates and screen scale for the document viewer. */
(function(root){
  'use strict';
  function layout(doc,width,height,requestedRows,context){
    const column=doc.boxes.interpreted_lithology_column;
    const depth=doc.boxes.depth_column;
    const left=context&&depth?Math.min(column[0],depth[0]):column[0];
    const right=context&&depth?Math.max(column[2],depth[2]):column[2];
    const bounds=[left,column[1],right,column[3]];
    const scale=Math.min(height/requestedRows,Math.max(1,width-24)/(right-left));
    return {bounds,scale,width:(right-left)*scale,height:(column[3]-column[1])*scale,
      rows:Math.min(column[3]-column[1],height/scale)};
  }
  const api={layout};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.StrataViewport=api;
})(typeof window==='undefined'?globalThis:window);
