import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Panel, Well } from "./types";
import { Map } from "./components/Map";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { UploadModal } from "./components/UploadModal";
export default function App() {
  const [wells, setWells] = useState<Well[]>([]);
  const [panel, setPanel] = useState<Panel | null>(null);
  const [selecting, setSelecting] = useState(false);
  const [clearKey, setClearKey] = useState(0);
  const [selected, setSelected] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [flyTo, setFlyTo] = useState<Well | null>(null);
  const [error, setError] = useState("");
  const pending = useRef(new Set<string>());
  const requestId = useRef(0);
  const opened = useRef<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      const next = await api.wells();
      setWells(next);
      setError("");
      for (const well of next) {
        if (pending.current.has(well.id) && well.status !== "processing") {
          pending.current.delete(well.id);
          if (well.status === "ready") setFlyTo(well);
        }
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  const processing = wells.some((w) => w.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(() => void refresh(), 2000);
    return () => clearInterval(timer);
  }, [processing, refresh]);
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
  return (
    <main>
      <Map
        wells={wells}
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
        onSelect={() => {
          const next = !selecting;
          close();
          setSelecting(next);
        }}
        onUpload={() => setUploading(true)}
      />
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
            setWells((previous) => [
              ...previous,
              {
                id,
                name: "Uploading well",
                lat: 26.5,
                lon: -90,
                block: "Unknown",
                status: "processing",
                scores: null,
              },
            ]);
            void refresh();
          }}
        />
      )}
      {error && (
        <button className="notice glass" onClick={() => void refresh()}>
          {error} · Retry
        </button>
      )}
      <div className="credit">
        Made with <span aria-label="love">♥</span> by{" "}
        <strong>402_Payment_Required</strong>
      </div>
    </main>
  );
}
