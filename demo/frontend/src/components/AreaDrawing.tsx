import { AreaButton } from "./AreaButton";
import { useEffect, useRef, useState } from "react";
import type { Map } from "maplibre-gl";
import type { Well } from "../types";
import "./AreaDrawing.css";

type Point = [number, number];
// Spherical polygon area, in km². Longitude differences wrap at the date line.
export function areaKm2(points: Point[]) {
  if (points.length < 3) return 0;
  const rad = Math.PI / 180;
  return Math.abs(points.reduce((sum, p, i) => {
    const q = points[(i + 1) % points.length];
    const dx = ((q[0] - p[0] + 540) % 360 - 180) * rad;
    return sum + dx * (2 + Math.sin(p[1] * rad) + Math.sin(q[1] * rad));
  }, 0) * 6371.0088 ** 2 / 2);
}

export function AreaDrawing({ map, wells, active, clearKey, onFinish, onCancel }: {
  map: Map; wells: Well[]; active: boolean; clearKey: number;
  onFinish: (points: number[][]) => void; onCancel: () => void;
}) {
  const [mode, setMode] = useState<"rectangle" | "polygon">("polygon");
  const [points, setPoints] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [, render] = useState(0);
  const gesture = useRef<{ start: Point; index?: number; moved: boolean } | null>(null);
  const saved = useRef<Point[]>([]);
  const svg = useRef<SVGSVGElement>(null);
  const intersects = points.some((a, i) => points.some((c, j) => {
    if (j <= i + 1 || (i === 0 && j === points.length - 1)) return false;
    const b = points[(i + 1) % points.length], d = points[(j + 1) % points.length];
    const cross = (p: Point, q: Point, r: Point) => (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);
    const on = (p: Point, q: Point, r: Point) => Math.abs(cross(p,q,r)) < 1e-10 && r[0] >= Math.min(p[0],q[0]) && r[0] <= Math.max(p[0],q[0]) && r[1] >= Math.min(p[1],q[1]) && r[1] <= Math.max(p[1],q[1]);
    return (cross(a,b,c) * cross(a,b,d) < 0 && cross(c,d,a) * cross(c,d,b) < 0) || on(a,b,c) || on(a,b,d) || on(c,d,a) || on(c,d,b);
  }));
  const valid = points.length >= 3 && !intersects && areaKm2(points) > 0.000001;
  const finish = () => {
    if (!valid) return;
    saved.current = points;
    setCursor(null);
    gesture.current = null;
    onFinish([...points, points[0]]);
  };
  const cancel = () => { setPoints(saved.current); setCursor(null); gesture.current = null; onCancel(); };
  useEffect(() => { setPoints([]); saved.current = []; setCursor(null); }, [clearKey]);
  useEffect(() => {
    if (active) { setPoints([]); setCursor(null); }
  }, [active]);
  useEffect(() => {
    const update = () => render(n => n + 1);
    map.on("move", update); map.on("resize", update);
    return () => { map.off("move", update); map.off("resize", update); };
  }, [map]);
  useEffect(() => {
    if (!active) return;
    const pan = map.dragPan.isEnabled(), zoom = map.doubleClickZoom.isEnabled();
    const touch = map.touchZoomRotate.isEnabled();
    map.dragPan.disable(); map.doubleClickZoom.disable(); map.touchZoomRotate.disable();
    return () => { if (pan) map.dragPan.enable(); if (zoom) map.doubleClickZoom.enable(); if (touch) map.touchZoomRotate.enable(); };
  }, [map, active]);
  useEffect(() => {
    if (!active) return;
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches("input,textarea,select")) return;
      if (e.key === "Escape") { e.preventDefault(); cancel(); }
      if (e.key === "Enter") { e.preventDefault(); finish(); }
      if (e.key === "Backspace") { e.preventDefault(); setPoints(p => p.slice(0, -1)); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  const project = (p: Point) => map.project(p);
  const nearStart = (p: Point) => {
    if (points.length < 3) return false;
    const a = project(points[0]), b = project(p);
    return Math.hypot(a.x - b.x, a.y - b.y) <= (matchMedia("(pointer: coarse)").matches ? 24 : 16);
  };
  const eventPoint = (e: React.PointerEvent): Point => {
    const box = svg.current!.getBoundingClientRect();
    const p = map.unproject([e.clientX - box.left, e.clientY - box.top]);
    return [p.lng, p.lat];
  };
  const preview = active && mode === "polygon" && cursor && points.length ? [...points, nearStart(cursor) ? points[0] : cursor] : points;
  const size = areaKm2(preview);
  const inside = wells.filter(w => {
    if (w.lon == null || w.lat == null || points.length < 3) return false;
    let result = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const a = points[i], b = points[j];
      if ((a[1] > w.lat) !== (b[1] > w.lat) && w.lon < (b[0]-a[0])*(w.lat-a[1])/(b[1]-a[1])+a[0]) result = !result;
    }
    return result;
  }).length;
  return <>
    <svg ref={svg} className={`area-drawing ${active ? "active" : ""}`} aria-label="Area drawing" onContextMenu={e => e.preventDefault()}
      onPointerDown={e => {
        if (e.button !== 0 || !e.isPrimary) return;
        const indexText = (e.target as Element).getAttribute("data-vertex");
        const index = indexText === null ? undefined : Number(indexText);
        if (!active && index === undefined) return;
        e.preventDefault(); e.stopPropagation();
        svg.current!.setPointerCapture(e.pointerId);
        const p = eventPoint(e);
        gesture.current = { start: p, index, moved: false };
        if (active && mode === "rectangle") setPoints([p]);
      }}
      onPointerMove={e => {
        const p = eventPoint(e), g = gesture.current;
        if (active) setCursor(p);
        if (!g) return;
        const a = project(g.start), b = project(p);
        g.moved ||= Math.hypot(a.x - b.x, a.y - b.y) > 3;
        if (active && mode === "rectangle") setPoints([g.start, [p[0], g.start[1]], p, [g.start[0], p[1]]]);
        else if (!active && g.index !== undefined) setPoints(old => old.map((q, i) => i === g.index ? p : q));
      }}
      onPointerUp={e => {
        const g = gesture.current; if (!g) return;
        gesture.current = null;
        if (svg.current!.hasPointerCapture(e.pointerId)) svg.current!.releasePointerCapture(e.pointerId);
        const p = eventPoint(e);
        if (!active) { if (g.moved && valid) { saved.current = points; onFinish([...points, points[0]]); } else if (g.moved) setPoints(saved.current); return; }
        if (mode === "rectangle") return;
        if (nearStart(p)) { finish(); return; }
        if (!g.moved) setPoints(old => {
          const last = old.at(-1);
          if (last) { const a = project(last), b = project(p); if (Math.hypot(a.x-b.x, a.y-b.y) < 4) return old; }
          return [...old, p];
        });
      }}
      onPointerCancel={() => { gesture.current = null; if (active) cancel(); }}
      onDoubleClick={e => { if (active) { e.preventDefault(); finish(); } }}>
      {preview.length > 1 && <polygon className="area-shape" points={preview.map(p => { const q = project(p); return `${q.x},${q.y}`; }).join(" ")} />}
      {points.map((p, i) => { const q = project(p); return <g key={i}>
        <circle className="area-vertex-hit" data-vertex={i} cx={q.x} cy={q.y} r="22" />
        <rect className={`area-vertex ${i === 0 && cursor && nearStart(cursor) ? "snap" : ""}`} x={q.x - 6} y={q.y - 6} width="12" height="12" />
      </g>; })}
    </svg>
    {active ? <>
      <div className="drawing-hint" role="status">Drawing: {mode === "rectangle" ? "drag to select" : "click to add points"} · Esc to cancel</div>
      <aside className="sidebar glass drawing-panel" aria-label="Select area">
        <header><h1>Select area</h1></header>
        <div className="drawing-content">
          <div className="area-modes" role="group" aria-label="Drawing mode">
            <AreaButton aria-pressed={mode === "polygon"} onAction={() => { setMode("polygon"); setPoints([]); }}>Polygon</AreaButton>
            <AreaButton aria-pressed={mode === "rectangle"} onAction={() => { setMode("rectangle"); setPoints([]); }}>Rectangle</AreaButton>
          </div>
          <dl className="drawing-stats" aria-live="polite">
            <div><dt>Area</dt><dd>{size.toLocaleString(undefined, { maximumFractionDigits: 2 })} <small>km²</small></dd></div>
            <div><dt>Vertices</dt><dd>{points.length}</dd></div>
            <div><dt>Wells inside</dt><dd>{inside}</dd></div>
          </dl>
          <p>{mode === "polygon" ? "Click to add a vertex. Finish to view the report." : "Drag to draw a rectangle. Finish to view the report."}</p>
          {intersects && <p role="alert">Edges cross. Undo the last vertex and try again.</p>}
          <AreaButton className="drawing-undo" disabled={!points.length} onAction={() => setPoints(p => p.slice(0, -1))}>Undo last vertex</AreaButton>
        </div>
        <footer className="drawing-footer"><AreaButton onAction={cancel}>Cancel</AreaButton><AreaButton disabled={!valid} onAction={finish}>Finish area</AreaButton></footer>
      </aside>
    </> : points.length > 2 && <div className="area-controls" aria-label="Selection controls"><div className="area-actions"><output>{size.toLocaleString(undefined, { maximumFractionDigits: 2 })} km²</output><span>Drag a vertex to edit</span></div></div>}
  </>;
}
