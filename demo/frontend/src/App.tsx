import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Panel, Well } from "./types";
import { Map } from "./components/Map";
import { Sidebar } from "./components/Sidebar";
import { Toast } from "./components/Toast";
import { Toolbar } from "./components/Toolbar";
import { UploadModal } from "./components/UploadModal";
const POLL_MS = 2000;
const MAX_BACKOFF_MS = 30000;
const isBusy = (w: Well) => w.status === "queued" || w.status === "processing";
export default function App() {
  const [wells, setWells] = useState<Well[]>([]);
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
  const refreshId = useRef(0);
  const opened = useRef<string | null>(null);
  const known = useRef<Well[]>([]);
  const failures = useRef(0);
  const timer = useRef<number>(undefined);
  const refresh = useCallback(async () => {
    const currentRefresh = ++refreshId.current;
    window.clearTimeout(timer.current);
    try {
      const next = await api.wells();
      if (currentRefresh !== refreshId.current) return;
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
        else if (well.located) setFlyTo(well);
        else
          setToast({ text: "Processed, location not found", network: false });
      }
      setPendingCount(pending.current.size);
    } catch (e) {
      if (currentRefresh !== refreshId.current) return;
      failures.current += 1;
      setToast({ text: (e as Error).message, network: true });
    }
    if (pending.current.size > 0 || known.current.some(isBusy) || failures.current > 0)
      timer.current = window.setTimeout(
        () => void refresh(),
        Math.min(POLL_MS * 2 ** failures.current, MAX_BACKOFF_MS),
      );
  }, []);
  useEffect(() => {
    void refresh();
    return () => {
      ++refreshId.current;
      window.clearTimeout(timer.current);
    };
  }, [refresh]);
  const showWell = useCallback(async (well: Well) => {
    const id = ++requestId.current;
    opened.current = well.id;
    setPanel({ mode: "well", title: well.name, data: null });
    setSelected([well.id]);
    try {
      const data = await api.well(well.id);
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
    const current = wells.find((w) => w.id === opened.current);
    if (
      current &&
      panel?.data &&
      "status" in panel.data &&
      panel.data.status !== current.status
    )
      void showWell(current);
  }, [wells, panel, showWell]);
  async function showArea(polygon: number[][]) {
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
    setSelected([]);
    setSelecting(false);
    setClearKey((k) => k + 1);
  }
  const located = wells.filter((w) => w.located);
  return (
    <main>
      <Map
        wells={located}
        selected={selected}
        selecting={selecting}
        clearKey={clearKey}
        flyTo={flyTo}
        onWell={(well) => {
          setSelecting(false);
          void showWell(well);
        }}
        onPolygon={(polygon) => void showArea(polygon)}
      />
      <div className="brand glass">
        <svg
          className="brand-icon"
          viewBox="0 0 32 32"
          aria-hidden="true"
          focusable="false"
        >
          <path d="M7 25 24 8M18 5l9 9M12 8c2.8-2.8 7.2-2.8 10 0l2 2-5 5-2-2c-1.1-1.1-2.9-1.1-4 0l-1 1-5-5 1-1Z" />
        </svg>
        <span>LandMarkr</span>
      </div>
      <Toolbar
        selecting={selecting}
        processing={pendingCount > 0}
        onSelect={() => {
          const next = !selecting;
          close();
          setSelecting(next);
        }}
        onUpload={() => setUploading(true)}
      />
      {loaded && located.length === 0 && !toast && (
        <p className="empty-state glass" role="status">
          No wells yet. Upload a scan to put one on the map.
        </p>
      )}
      {panel && (
        <Sidebar
          key={`${panel.mode}-${panel.title}`}
          panel={panel}
          onClose={close}
        />
      )}
      {uploading && (
        <UploadModal
          onClose={() => setUploading(false)}
          onUploaded={(id) => {
            pending.current.add(id);
            setPendingCount(pending.current.size);
            void refresh();
          }}
        />
      )}
      {toast && (
        <Toast message={toast.text} onDismiss={() => setToast(null)} />
      )}
      <div className="credit">
        Made with <span aria-label="love">♥</span> by{" "}
        <strong>402_Payment_Required</strong>
      </div>
    </main>
  );
}
