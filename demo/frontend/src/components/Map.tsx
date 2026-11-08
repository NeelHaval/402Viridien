import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { TerraDraw, TerraDrawPolygonMode } from "terra-draw";
import { TerraDrawMapLibreGLAdapter } from "terra-draw-maplibre-gl-adapter";
import type { Well } from "../types";
maplibregl.setWorkerUrl(workerUrl);

interface Props {
  wells: Well[];
  selected: string[];
  selecting: boolean;
  clearKey: number;
  flyTo: Well | null;
  onWell: (well: Well) => void;
  onPolygon: (coordinates: number[][]) => void;
}
export function Map(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const draw = useRef<TerraDraw | null>(null);
  const markers = useRef(new globalThis.Map<string, maplibregl.Marker>());
  const latest = useRef(props);
  latest.current = props;
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
    instance.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "bottom-right",
    );
    instance.on("error", () => setError("Map tiles unavailable"));
    instance.on("load", () => {
      setError("");
      const drawing = new TerraDraw({
        adapter: new TerraDrawMapLibreGLAdapter({ map: instance }),
        modes: [
          new TerraDrawPolygonMode({
            styles: {
              fillColor: "#5EEAD4",
              fillOpacity: 0.15,
              outlineColor: "#5EEAD4",
              outlineWidth: 2,
            },
          }),
        ],
      });
      drawing.start();
      draw.current = drawing;
      drawing.on("finish", (id) => {
        const feature = drawing.getSnapshot().find((f) => f.id === id);
        if (feature?.geometry.type === "Polygon") {
          drawing.setMode("static");
          latest.current.onPolygon(feature.geometry.coordinates[0]);
        }
      });
      setReady(true);
    });
    return () => {
      draw.current?.stop();
      markers.current.forEach((m) => m.remove());
      markers.current.clear();
      instance.remove();
    };
  }, []);
  useEffect(() => {
    if (ready) {
      draw.current?.clear();
      draw.current?.setMode(props.selecting ? "polygon" : "static");
    }
  }, [props.selecting, props.clearKey, ready]);
  useEffect(() => {
    if (!ready || !map.current) return;
    const ids = new Set(props.wells.map((w) => w.id));
    markers.current.forEach((marker, id) => {
      if (!ids.has(id)) {
        marker.remove();
        markers.current.delete(id);
      }
    });
    props.wells.forEach((well) => {
      if (
        !well.located ||
        well.lat === null ||
        well.lon === null ||
        !Number.isFinite(well.lat) ||
        !Number.isFinite(well.lon)
      )
        return;
      let marker = markers.current.get(well.id);
      if (!marker) {
        const wrapper = document.createElement("div");
        const element = document.createElement("button");
        element.type = "button";
        element.addEventListener("click", (e) => {
          e.stopPropagation();
          const current = latest.current.wells.find((w) => w.id === well.id);
          if (current) latest.current.onWell(current);
        });
        wrapper.append(element);
        marker = new maplibregl.Marker({ element: wrapper, anchor: "center" })
          .setLngLat([well.lon, well.lat])
          .addTo(map.current!);
        markers.current.set(well.id, marker);
      }
      marker.setLngLat([well.lon, well.lat]);
      const element = marker.getElement().firstElementChild;
      if (!(element instanceof HTMLButtonElement)) return;
      element.className = `well-marker ${well.status === "queued" ? "processing" : well.status} ${props.selected.includes(well.id) ? "selected" : ""}`;
      element.title = well.name;
      element.setAttribute("aria-label", `${well.name}, ${well.status}`);
    });
  }, [props.wells, props.selected, ready]);
  useEffect(() => {
    const well = props.flyTo;
    if (ready && well?.lon != null && well.lat != null)
      map.current?.flyTo({
        center: [well.lon, well.lat],
        zoom: 6,
        duration: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? 0
          : 1200,
      });
  }, [props.flyTo, ready]);
  return (
    <>
      <div className="map" ref={container} />
      {error && (
        <div className="map-error glass" role="status">
          {error}
        </div>
      )}
    </>
  );
}
