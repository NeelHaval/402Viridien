import { useMemo, useState } from 'react';
type Bed={top_y:number;bottom_y:number;lithology:string;cluster_id:number;color:string};
export function PixelColumn({beds,estimated=false}:{beds:Bed[];estimated?:boolean}) {
  const [focus,setFocus]=useState<string|null>(null);
  const data=useMemo(()=>{
    const start=Math.min(...beds.map(b=>b.top_y)),end=Math.max(...beds.map(b=>b.bottom_y));
    const labels=new Map<string,{pixels:number;color:string}>();
    for(const bed of beds){const current=labels.get(bed.lithology)||{pixels:0,color:bed.color};current.pixels+=bed.bottom_y-bed.top_y;labels.set(bed.lithology,current);}
    const total=[...labels.values()].reduce((n,v)=>n+v.pixels,0);
    return {start,end,labels:[...labels.entries()].sort((a,b)=>b[1].pixels-a[1].pixels),total};
  },[beds]);
  if(!beds.length)return null;
  const height=Math.min(1600,Math.max(500,beds.length*5));
  return <section className="pixel-result" aria-label="Extracted layer column">
    <h2>Layers found</h2><p>{estimated?'Depth units unverified':'Depth scale not read'} · {beds.length.toLocaleString()} intervals. The axis shows source-image pixels.</p>
    <div className="pixel-column-layout"><ul aria-label="Lithology legend">{data.labels.map(([label,v])=><li key={label}><button aria-pressed={focus===label} onClick={()=>setFocus(focus===label?null:label)}><span className="pixel-swatch" style={{background:v.color}}/>{label}<small>{(100*v.pixels/data.total).toFixed(1)}% of pixel thickness</small></button></li>)}</ul>
    <svg className="pixel-column" viewBox={`0 0 240 ${height+60}`} role="img" aria-label={`${beds.length} extracted layers on a pixel axis; depth scale not read`}>
      {[0,.25,.5,.75,1].map(t=><g key={t}><text x="66" y={30+t*height} textAnchor="end">{Math.round(data.start+t*(data.end-data.start)).toLocaleString()} px</text><path d={`M70 ${26+t*height}h10`} stroke="currentColor"/></g>)}
      {beds.map((bed,i)=><rect className="pixel-layer" key={i} x="84" width="136" y={26+(bed.top_y-data.start)/(data.end-data.start)*height} height={Math.max(.5,(bed.bottom_y-bed.top_y)/(data.end-data.start)*height)} fill={bed.color} opacity={!focus||focus===bed.lithology?1:.2}><title>{bed.lithology}: {bed.top_y}–{bed.bottom_y} px</title></rect>)}
    </svg></div>
  </section>;
}
