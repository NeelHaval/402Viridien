import type { SVGProps } from "react";
export type IconName = 'logo'|'select'|'upload'|'library'|'ready'|'partial'|'processing'|'failed'|'download'|'summarise'|'close'|'plus'|'minus'|'layers';
export const iconPaths: Record<IconName,string> = {
 logo:'M3 7 12 3 21 7 12 11ZM3 12l9 4 9-4M3 17l9 4 9-4',
 select:'M4 9V4h5m6 0h5v5m0 6v5h-5m-6 0H4v-5', upload:'M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6',
 library:'M4 4h16v16H4ZM8 4v16m4-11h5m-5 4h5',ready:'M12 22s7-7 7-13a7 7 0 0 0-14 0c0 6 7 13 7 13Zm-3-13 2 2 4-4',
 partial:'M12 22s7-7 7-13a7 7 0 0 0-14 0c0 6 7 13 7 13ZM8 8h8m-8 4h8',processing:'M12 22s7-7 7-13a7 7 0 0 0-14 0c0 6 7 13 7 13Zm0-17v4l3 2',
 failed:'M12 22s7-7 7-13a7 7 0 0 0-14 0c0 6 7 13 7 13ZM9 6l6 6m0-6-6 6',download:'M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4',
 summarise:'M4 4h16v16H4Zm4 4h8m-8 4h8m-8 4h4',close:'m6 6 12 12M6 18 18 6',plus:'M12 5v14M5 12h14',minus:'M5 12h14',layers:'m3 7 9-4 9 4-9 4Zm0 5 9 4 9-4m-18 5 9 4 9-4'
};
export function Icon({name,...props}:SVGProps<SVGSVGElement>&{name:IconName}){return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={iconPaths[name]}/></svg>}
