import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { subscribe, type WellEvent } from "./events";
import type { Panel, Well, WellDetail } from "./types";
const Map = lazy(()=>import('./components/Map').then(m=>({default:m.Map})));
import { Sidebar } from "./components/Sidebar";
import { Toast } from "./components/Toast";
import { Toolbar } from "./components/Toolbar";
import { UploadModal } from "./components/UploadModal";
import {Icon} from "./components/Icon";
import {SourceLibrary} from "./components/SourceLibrary";
const POLL_MS = 2000;
const MAX_BACKOFF_MS = 30000;
const isBusy = (w: Well) => w.status === "queued" || w.status === "processing";
export default function App() {
  const [wells, setWells] = useState<Well[]>([]);
  const [cluster,setCluster] = useState<Well[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [panel, setPanel] = useState<Panel | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [clearKey, setClearKey] = useState(0);
  const [selected, setSelected] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const [flyTo, setFlyTo] = useState<Well | null>(null);
  const [toast, setToast] = useState<{ text: string; network: boolean } | null>(
    null,
  );
  const pending = useRef(new Set<string>());
  const requestId = useRef(0);
  const opened = useRef<string | null>(null);
  const known = useRef<Well[]>([]);
  const detailCache=useRef(new globalThis.Map<string,{data:WellDetail;at:number}>());
  const failures = useRef(0);
  const timer = useRef<number>(undefined);
  const streaming = useRef(false);
  const showWellRef = useRef<(well: Well) => Promise<void>>(async () => {});
  const refresh = useCallback(async () => {
    window.clearTimeout(timer.current);
    try {
      const next = await api.wells();
      failures.current = 0;
      known.current = next;
      setWells(next);
      setLoaded(true);
      setToast((t) => (t?.network ? null : t));
      for (const id of [...pending.current]) {
        const well = next.find((w) => w.id === id);
        if (!well || isBusy(well)) continue;
        pending.current.delete(id);
        if (well.status === "failed")
          setToast({ text: "Processing failed", network: false });
        else { if(well.located) setFlyTo(well); void showWellRef.current(well); }

      }
      setPendingCount(pending.current.size);
    } catch (e) {
      failures.current += 1;
      setToast({ text: (e as Error).message, network: true });
    }
    // The 2 s poll runs only while the event stream is down (or a request failed).
    if (!streaming.current || failures.current > 0)
      timer.current = window.setTimeout(
        () => void refresh(),
        Math.min(POLL_MS * 2 ** failures.current, MAX_BACKOFF_MS),
      );
  }, []);
  const applyEvent = useCallback((event: WellEvent) => {
    detailCache.current.delete(event.id);
    const base: Well = {
      id: event.id,
      name: event.id,
      lat: null,
      lon: null,
      block: null,
      status: "queued",
      scores: null,
      validation_confidence: null,
      located: false,
    };
    const current = known.current.find((w) => w.id === event.id) ?? base;
    let next: Well = current;
    if (event.type === "well_created")
      next = { ...current, name: event.name ?? current.name };
    else if (event.type === "well_located")
      next = {
        ...current,
        lat: event.lat,
        lon: event.lon,
        block: event.block ?? current.block,
        located: true,
      };
    else if (event.type === "well_ready")
      next = {
        ...current,
        status: event.status === "partial" ? "partial" : "ready",
        name: event.name ?? current.name,
        validation_confidence:
          event.validation_confidence ?? current.validation_confidence,
      };
    else if (event.type === "well_scored")
      next = { ...current, scores: event.scores };
    else next = { ...current, status: "failed" };
    known.current = known.current.some((w) => w.id === event.id)
      ? known.current.map((w) => (w.id === event.id ? next : w))
      : [...known.current, next];
    setWells(known.current);
    setPanel(p => p?.mode === "area" ? { ...p, error: "Results changed. Draw the area again to refresh this report.", data: null } : p);
    setLoaded(true);
    if (pending.current.has(event.id)) {
      if (event.type === "well_located") setFlyTo(next);
      else if (event.type === "well_failed") {
        pending.current.delete(event.id);
        setToast({ text: "Processing failed", network: false });
      } else if (event.type === "well_ready") {
        void showWellRef.current(next);
        pending.current.delete(event.id);
        if (!next.located)
          setToast({ text: "Processed, location not found", network: false });
      }
      setPendingCount(pending.current.size);
    }
    if (
      event.id === opened.current &&
      ["well_ready", "well_scored", "well_failed"].includes(event.type)
    )
      void showWellRef.current(next);
  }, []);
  useEffect(() => {
    void refresh();
    const stop = subscribe(applyEvent, (connected) => {
      streaming.current = connected;
      window.clearTimeout(timer.current);
      void refresh(); // resync on every (re)connect; keeps polling only while disconnected
    });
    return () => {
      stop();
      window.clearTimeout(timer.current);
    };
  }, [refresh, applyEvent]);
  const showWell = useCallback(async (well: Well) => {
    setCluster([]);
    const id = ++requestId.current;
    opened.current = well.id;
    const cached=detailCache.current.get(well.id);
    const warm=cached&&Date.now()-cached.at<15000&&cached.data.status===well.status?cached.data:null;
    setPanel({ mode: "well", title: well.name, data: warm });
    setSelected([well.id]);
    try {
      const data = await api.well(well.id);
      if(data.status==='ready'||data.status==='partial'){
        detailCache.current.set(well.id,{data,at:Date.now()});
        if(detailCache.current.size>64)detailCache.current.delete(detailCache.current.keys().next().value!);
      }
      if (id === requestId.current)
        setPanel({ mode: "well", title: data.name, data });
    } catch (e) {
      if (id === requestId.current)
        setPanel({
          mode: "well",
          title: well.name,
          data: null,
          error: (e as Error).message,
        });
    }
  }, []);
  useEffect(() => {
    showWellRef.current = showWell;
  }, [showWell]);
  useEffect(() => {
    const current = wells.find((w) => w.id === opened.current);
    if (
      current &&
      panel?.data &&
      "status" in panel.data &&
      panel.data.status !== current.status
    )
      void showWell(current);
  }, [wells, panel, showWell]);
  const area = useRef<number[][]>([]);
  async function showArea(polygon: number[][]) {
    setCluster([]);
    area.current = polygon;
    setSelecting(false);
    const id = ++requestId.current;
    opened.current = null;
    setPanel({ mode: "area", title: "Area report", data: null });
    try {
      const data = await api.report(polygon);
      if (id === requestId.current) {
        setPanel({ mode: "area", title: `${data.wells.length} wells`, data });
        setSelected(data.wells.map((w) => w.id));
      }
    } catch (e) {
      if (id === requestId.current)
        setPanel({
          mode: "area",
          title: "Area report",
          data: null,
          error: (e as Error).message,
        });
    }
  }
  function close() {
    ++requestId.current;
    opened.current = null;
    setPanel(null);
    setCluster([]);
    setSelected([]);
    setSelecting(false);
    setClearKey((k) => k + 1);
  }
  useEffect(()=>{const escape=(e:KeyboardEvent)=>{if(e.key==='Escape'){close();}};window.addEventListener('keydown',escape);return()=>window.removeEventListener('keydown',escape);},[]);
  const located = wells.filter((w) => w.located);
  return (
    <main>
      <Suspense fallback={<p role="status" className="map-loading">Loading map…</p>}><Map
        onCluster={group=>{setSelecting(false);setCluster(group);setPanel(null);}}
        wells={located}
        selected={selected}
        selecting={selecting}
        clearKey={clearKey}
        flyTo={flyTo}
        onWell={(well) => {
          setSelecting(false);
          void showWell(well);
        }}
        onCancel={() => setSelecting(false)}
        onPolygon={(polygon) => void showArea(polygon)}
      /></Suspense>
      <div className="brand glass">
        <Icon name="logo" className="brand-icon" />
        <span>Strata</span>
      </div>
      {/* Source library is part of the action bar. */}
      <Toolbar
        library={<SourceLibrary onOpen={id=>void showWell({id,name:id} as Well)} onRun={id=>{pending.current.add(id);setPendingCount(pending.current.size);void refresh();void showWell({id,name:id} as Well);}} />}
        wells={wells}
        onJump={well=>{setFlyTo(well);void showWell(well);}}
        selecting={selecting}
        processing={pendingCount > 0}
        reportOpen={panel !== null}
        onSelect={() => {
          const next = !selecting;
          close();
          setSelecting(next);
        }}
        onUpload={() => {close();setUploading(true);}}
      />
      {loaded && located.length === 0 && !toast && (
        <p className="empty-state glass" role="status">
          No wells yet. Upload a scan to put one on the map.
        </p>
      )}
      {cluster.length>0&&<aside className="sidebar glass" aria-label="Wells at this location"><header><h1>{cluster.length} wells here</h1><button className="close" aria-label="Close wells" onClick={()=>setCluster([])}><Icon name="close"/></button></header>{cluster.map(w=><button className="cluster-well" key={w.id} onClick={()=>{setCluster([]);void showWell(w);}}>{w.name}<small>{w.status}</small></button>)}</aside>}
      {panel && (
        <Sidebar
          key={`${panel.mode}-${panel.title}`}
          panel={panel}
          onClose={close}
          onRedraw={() => { close(); setSelecting(true); }}
          onNearest={() => {
            const center = area.current.reduce((a, p) => [a[0] + p[0] / area.current.length, a[1] + p[1] / area.current.length], [0, 0]);
            const nearest = [...located].filter(w => w.lon != null && w.lat != null).sort((a, b) => {
              const distance = (w: Well) => ((w.lon! - center[0]) * Math.cos(center[1] * Math.PI / 180)) ** 2 + (w.lat! - center[1]) ** 2;
              return distance(a) - distance(b);
            })[0];
            if (nearest) { close(); setFlyTo({ ...nearest }); }
          }}
          hasNearest={located.length > 0}
        />
      )}
      {uploading && (
        <UploadModal
          onClose={() => setUploading(false)}
          onUploaded={(id) => {
            pending.current.add(id);
            setPendingCount(pending.current.size);
            void refresh();
            void showWell({ id, name: id } as Well);
          }}
        />
      )}
      {toast && (
        <Toast onRetry={toast.network ? ()=>void refresh() : undefined} message={toast.text} onDismiss={() => setToast(null)} />
      )}
      <div className="credit">
        Made with <span aria-label="love">♥</span> by{" "}
        <strong>402_Payment_Required</strong>
      </div>
    </main>
  );
}
