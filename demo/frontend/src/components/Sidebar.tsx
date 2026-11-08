import {Icon} from "./Icon";
import { AreaButton } from "./AreaButton";
import { api } from "../api";
import type { Metric, Panel, Report, WellDetail } from "../types";
import { Dial } from "./Dial";
import { LithColumn } from "./LithColumn";
import { WellDescription } from "./WellDescription";
import { PixelColumn } from "./PixelColumn";
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
  onClose, onRedraw, onNearest, hasNearest,
}: {
  panel: Panel;
  onClose: () => void;
  onRedraw: () => void; onNearest: () => void; hasNearest: boolean;
}) {
  const data = panel.data;
  const empty = data && !isWell(data) && data.wells.length === 0;
  return (
    <aside className={`sidebar glass${panel.mode === "area" ? " area-report" : ""}`} aria-label={panel.mode === "area" ? "Area report" : "Well report"}>
      <header>
        <div>
          <span className="eyebrow">
            {panel.mode === "area" ? "Area report" : "Well profile"}
          </span>
          <h1>{panel.title}</h1>
        </div>
        {panel.mode === "area" && <nav className="area-report-actions" aria-label="Area actions"><AreaButton onAction={onClose}>Clear</AreaButton><AreaButton onAction={onRedraw}>Redraw</AreaButton></nav>}
        <button className="close" onClick={onClose} aria-label="Close report">
          <Icon name="close"/>
        </button>
      </header>
      {data && <nav className="panel-actions" aria-label="Report actions">{isWell(data) && (data.status==='ready'||data.status==='partial') && <a href={api.csvUrl(data.id)} download aria-label="Download CSV"><Icon name="download"/>Download CSV</a>}<button onClick={e=>{const host=e.currentTarget.closest('aside')?.querySelector('.summary-host');const action=host?.querySelector('button,a') as HTMLElement|null;if(action)action.click();else host?.scrollIntoView({block:'nearest'});}}><Icon name="summarise"/>Summarise</button></nav>}
      <div className="summary-host">{data && (isWell(data) ? <WellDescription key={data.id + data.result?.result_sha256} id={data.id} ready={data.status === "ready" || data.status === "partial"} /> : data.polygon && <WellDescription key={JSON.stringify(data.polygon)} polygon={data.polygon} ready />)}</div>
      {panel.error ? (
        <p role="alert">{panel.error}</p>
      ) : !data ? (
        <div className="loading" aria-label="Loading report">
          <div className="shimmer" />
          <div className="shimmer tall" />
        </div>
      ) : empty ? (
        <div className="area-empty"><p>No wells here yet · zoom out or upload a scan</p><AreaButton disabled={!hasNearest} onAction={onNearest}>Zoom to nearest well</AreaButton>{!hasNearest && <p>Upload a scan to add the first well.</p>}</div>
      ) : (
        <>
          {isWell(data) && data.status === "failed" ? (
            <p className="empty" role="alert">
              Processing failed: {data.error ?? "unknown error"}
            </p>
          ) : isWell(data) && data.status === "partial" ? (<p className="empty">Layers found, depth scale not read.</p>) : isWell(data) && data.status !== "ready" ? (
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
          {isWell(data) && data.result && (
            <>
              <p className="empty">{data.result.engine} · {data.status === "partial" ? "Layer result saved. Verified depth export is unavailable." : "Saved result"}</p>
              {data.result.depth_estimate && <p className="depth-estimate">OCR found a depth estimate: <strong>ft (assumed, OCR)</strong>. The printed unit is unverified. The column and CSV retain pixel coordinates; verified depth export is blocked.</p>}
              {data.pixel_column && data.pixel_column.length>0 && <PixelColumn beds={data.pixel_column} estimated={!!data.result.depth_estimate}/>}
              <nav className="result-exports" aria-label="Result exports">
                <a href={`/api/wells/${data.id}/csv?format=notebook`} download>Notebook CSV</a>
                <a href={`/api/wells/${data.id}/csv?format=slide`} download>Slide CSV</a>
                <a href={`/api/wells/${data.id}/result`} download>Result identity</a>
              </nav>
              <details className="scan-preview"><summary>Annotated source scan</summary><img loading="lazy" decoding="async" className="result-annotation" src={`/api/wells/${data.id}/annotation`} alt={`Annotated pixel column for ${data.name}; ${data.result.engine}`} /></details>
            </>
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


          {!isWell(data) && data.wells.length > 0 && <details className="well-description">
            <summary>Notes for wells in this report</summary>
            {data.wells.slice(0, 5).map((well) => <div key={well.id}>
              <h3>{well.name}</h3><WellDescription id={well.id} ready />
            </div>)}
            {data.wells.length > 5 && <p>Open another well to generate its note.</p>}
          </details>}
        </>
      )}
    </aside>
  );
}
