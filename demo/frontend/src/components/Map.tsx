import {iconPaths} from "./Icon";
import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { AreaDrawing } from "./AreaDrawing";

import type { Well } from "../types";
maplibregl.setWorkerUrl(workerUrl);

interface Props {
  wells: Well[];
  selected: string[];
  selecting: boolean;
  clearKey: number;
  onCancel: () => void;
  flyTo: Well | null;
  onCluster: (wells: Well[]) => void;
  onWell: (well: Well) => void;
  onPolygon: (coordinates: number[][]) => void;
}
export function Map(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);

  const markers = useRef(new globalThis.Map<string, maplibregl.Marker>());
  const latest = useRef(props);
  latest.current = props;
  const [view,setView] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const instance = new maplibregl.Map({
      container: container.current!,
      style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
      center: [-90, 26.5],
      zoom: 5.2,
    });
    map.current = instance;
    instance.on("moveend",()=>setView(v=>v+1));
    instance.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "bottom-right",
    );
    instance.on("error", () => setError("Map tiles unavailable"));
    instance.on("load", () => {
      setError("");
      setReady(true);
    });
    return () => {

      markers.current.forEach((m) => m.remove());
      markers.current.clear();
      instance.remove();
    };
  }, []);
  useEffect(() => {
    if (!ready || !map.current) return;
    const instance=map.current;
    if(!instance.getSource('well-clusters')) instance.addSource('well-clusters',{type:'geojson',cluster:true,clusterRadius:44,clusterMaxZoom:18,data:{type:'FeatureCollection',features:[]}});
    if(!instance.getLayer('well-cluster-loader')) instance.addLayer({id:'well-cluster-loader',type:'circle',source:'well-clusters',paint:{'circle-radius':0,'circle-opacity':0}});
    const source=instance.getSource('well-clusters') as maplibregl.GeoJSONSource;
    source.setData({type:'FeatureCollection',features:props.wells.filter(w=>w.lon!=null&&w.lat!=null).map(w=>({type:'Feature',properties:{id:w.id},geometry:{type:'Point',coordinates:[w.lon!,w.lat!]}}))});
    const render=()=>{
    markers.current.forEach(m=>m.remove()); markers.current.clear();
    const features=instance.querySourceFeatures('well-clusters');
    const seen=new Set<string>();
    features.forEach(f=>{
      if(f.geometry.type!=='Point')return;
      const p=f.properties!;const key=String(p.cluster_id??p.id);if(seen.has(key))return;seen.add(key);
      const coord=f.geometry.coordinates as [number,number];
      const element=document.createElement('button');element.type='button';element.className='well-marker';
      const well=latest.current.wells.find(w=>w.id===p.id);
      if(p.cluster){element.classList.add('cluster');element.textContent=String(p.point_count);element.setAttribute('aria-label',`${p.point_count} wells, zoom in`);element.onclick=async()=>{const leaves=await source.getClusterLeaves(p.cluster_id,1000,0);const group=leaves.map(f=>latest.current.wells.find(w=>w.id===f.properties?.id)).filter((w):w is Well=>!!w);if(instance.getZoom()>=8||group.every(w=>w.lon===group[0]?.lon&&w.lat===group[0]?.lat)){latest.current.onCluster(group);return;}const bounds=new maplibregl.LngLatBounds();leaves.forEach(f=>{if(f.geometry.type==='Point')bounds.extend(f.geometry.coordinates as [number,number]);});instance.fitBounds(bounds,{padding:{top:110,bottom:80,left:80,right:Math.min(480,instance.getContainer().clientWidth*.4)},maxZoom:Math.min(8,instance.getZoom()+2),duration:matchMedia('(prefers-reduced-motion: reduce)').matches?0:250});};}
      else if(well){element.classList.add(well.status);if(latest.current.selected.includes(well.id))element.classList.add('selected');element.title=well.name;element.setAttribute('aria-label',`${well.name}, ${well.status}`);const state=well.status==='queued'?'processing':well.status;element.innerHTML=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="${iconPaths[state]}"/></svg>`;element.onclick=()=>{const group=latest.current.wells.filter(w=>w.lon===well.lon&&w.lat===well.lat);if(group.length>1)latest.current.onCluster(group);else latest.current.onWell(well);};}
      else return;
      const wrapper=document.createElement('div');wrapper.append(element);markers.current.set(key,new maplibregl.Marker({element:wrapper}).setLngLat(coord).addTo(instance));
    });};
    instance.on('idle',render);render();
    return()=>{instance.off('idle',render);};
  }, [props.wells, props.selected, ready, view]);
  useEffect(() => {
    const well = props.flyTo;
    const instance=map.current;
    if (ready && instance && well?.lon != null && well.lat != null) {
      const point=instance.project([well.lon,well.lat]);
      const panel=document.querySelector('.sidebar')?.getBoundingClientRect();
      const width=instance.getContainer().clientWidth,height=instance.getContainer().clientHeight;
      const phone=width<768;
      const right=panel&&!phone?width-panel.left+24:24;
      const bottom=panel&&phone?height-panel.top+24:40;
      if(point.x>=24&&point.x<=width-right&&point.y>=110&&point.y<=height-bottom)return;
      instance.easeTo({
        center: [well.lon, well.lat],
        zoom: Math.min(instance.getZoom(), 7),
        padding:{top:110,bottom:Math.min(bottom,height*.6),left:24,right:Math.min(right,width*.5)},
        duration: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? 0
          : 250,
      });
    }
  }, [props.flyTo, ready]);
  return (
    <>
      <div className="map" ref={container} />
      {ready && map.current && <AreaDrawing wells={props.wells} map={map.current} active={props.selecting} clearKey={props.clearKey} onFinish={props.onPolygon} onCancel={props.onCancel} />}
      {error && (
        <div className="map-error glass" role="status">
          {error}
        </div>
      )}
    </>
  );
}
