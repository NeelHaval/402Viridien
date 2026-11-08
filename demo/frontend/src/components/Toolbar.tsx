import type {ReactNode} from 'react';
import type {Well} from '../types';
import {Icon} from './Icon';
export function Toolbar({selecting,processing,onSelect,onUpload,library,wells,onJump}: {selecting:boolean;processing:boolean;onSelect:()=>void;onUpload:()=>void;reportOpen:boolean;library:ReactNode;wells:Well[];onJump:(w:Well)=>void}) {
return <nav className="toolbar glass" aria-label="Map tools">
<div className="tool-actions"><button aria-pressed={selecting} onClick={onSelect}><Icon name="select"/>Select area</button><button onClick={onUpload}><Icon name="upload"/>Upload</button>{library}</div>
<label className="well-search"><span className="sr-only">Find a well</span><input type="search" placeholder="Find a well…" list="well-names" onChange={e=>{const w=wells.find(w=>w.name===e.target.value);if(w){onJump(w);e.target.value='';}}}/><datalist id="well-names">{wells.map(w=><option key={w.id} value={w.name}/>)}</datalist></label>
{processing&&<span role="status">Processing…</span>}</nav>}
