import { api } from "../api";
import type { Metric, Panel, Report, WellDetail } from "../types";
import { Dial } from "./Dial";
import { LithColumn } from "./LithColumn";
const labels: Record<Metric, string> = {
  drilling: "Drilling ease",
  cost: "Development cost",
  price: "Price safety",
  co2: "CO₂ storage",
};
const isWell = (data: WellDetail | Report): data is WellDetail =>
  "status" in data;
export function Sidebar({
  panel,
  onClose,
}: {
  panel: Panel;
  onClose: () => void;
}) {
  const data = panel.data;
  const empty = data && !isWell(data) && data.wells.length === 0;
  return (
    <aside className="sidebar glass" aria-label="Well report">
      <header>
        <div>
          <span className="eyebrow">
            {panel.mode === "area" ? "AREA REPORT" : "WELL PROFILE"}
          </span>
          <h1>{panel.title}</h1>
        </div>
        {data && isWell(data) && data.status === "ready" && (
          <a
            className="close download"
            href={api.csvUrl(data.id)}
            download
            aria-label="Download CSV"
            title="Download CSV"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3v12m-5-5 5 5 5-5M4 20h16" />
            </svg>
          </a>
        )}
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
          {isWell(data) && data.status === "failed" ? (
            <p className="empty" role="alert">
              Processing failed: {data.error ?? "unknown error"}
            </p>
          ) : isWell(data) && data.status !== "ready" ? (
            <p className="empty">Processing…</p>
          ) : (
            <div className="dials">
              {(Object.keys(labels) as Metric[]).map((key) => (
                <Dial
                  key={key}
                  value={data.scores ? data.scores[key] : null}
                  label={labels[key]}
                />
              ))}
            </div>
          )}
          {!isWell(data) && data.data_quality.length > 0 && (
            <ul className="pills" aria-label="Data quality">
              {data.data_quality.map((flag) => (
                <li key={flag}>{flag.replace(/_/g, " ")}</li>
              ))}
            </ul>
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
