import type { Panel, Scores } from "../types";
import { Dial } from "./Dial";
import { LithColumn } from "./LithColumn";
const labels: Record<keyof Scores, string> = {
  drilling: "Drilling ease",
  cost: "Development cost",
  price: "Price safety",
  co2: "CO₂ storage",
};
export function Sidebar({
  panel,
  onClose,
}: {
  panel: Panel;
  onClose: () => void;
}) {
  const data = panel.data;
  const empty = data && "wells" in data && data.wells.length === 0;
  return (
    <aside className="sidebar glass" aria-label="Well report">
      <header>
        <div>
          <span className="eyebrow">
            {panel.mode === "area" ? "AREA REPORT" : "WELL PROFILE"}
          </span>
          <h1>{panel.title}</h1>
        </div>
        <button className="close" onClick={onClose} aria-label="Close report">
          ×
        </button>
      </header>
      {panel.error ? (
        <p role="alert">{panel.error}</p>
      ) : !data ? (
        <div className="loading" aria-label="Loading report">
          <div className="shimmer" />
          <div className="shimmer tall" />
        </div>
      ) : empty ? (
        <p className="empty">No wells in this area</p>
      ) : (
        <>
          {data.scores ? (
            <div className="dials">
              {(Object.keys(labels) as (keyof Scores)[]).map((key) => (
                <Dial key={key} value={data.scores![key]} label={labels[key]} />
              ))}
            </div>
          ) : (
            <p className="empty">
              {"status" in data
                ? data.status === "failed"
                  ? "Processing failed"
                  : "Processing…"
                : ""}
            </p>
          )}
          {data.column.length > 0 && (
            <>
              <div className="column-heading">
                <span>LITHOLOGY</span>
                <span>↓ FT</span>
              </div>
              <LithColumn segments={data.column} area={panel.mode === "area"} />
            </>
          )}
        </>
      )}
    </aside>
  );
}
