import { useEffect, useRef, useState } from "react";
import type { Segment } from "../types";
const patternFor = (lithology: string) =>
  lithology.includes("Shale")
    ? "hatch"
    : lithology === "Limestone"
      ? "brick"
      : "dots";
/** One legend entry per lithology: total feet over all segments (from each segment's breakdown), colour of its beds. */
function legendOf(segments: Segment[]) {
  const entries = new Map<string, { feet: number; color: string }>();
  for (const s of segments) {
    const thickness = s.base_ft - s.top_ft;
    for (const b of s.breakdown) {
      const entry = entries.get(b.lithology) ?? {
        feet: 0,
        color: s.lithology === b.lithology ? s.color : "",
      };
      entry.feet += (thickness * b.pct) / 100;
      if (!entry.color && s.lithology === b.lithology) entry.color = s.color;
      entries.set(b.lithology, entry);
    }
  }
  const total = [...entries.values()].reduce((a, e) => a + e.feet, 0) || 1;
  return [...entries.entries()]
    .map(([lithology, e]) => ({
      lithology,
      feet: e.feet,
      share: Math.round((100 * e.feet) / total),
      color: e.color || "#8a8f9c",
    }))
    .sort((a, b) => b.feet - a.feet);
}
export function LithColumn({
  segments,
  area,
}: {
  segments: Segment[];
  area: boolean;
}) {
  const [selected, setSelected] = useState<number | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (
        !host.current?.querySelector(".popover")?.contains(event.target as Node)
      )
        setSelected(null);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);
  const heights = segments.map((s) =>
    Math.max(38, (s.base_ft - s.top_ft) * 0.06),
  );
  const offsets = heights.map(
    (_, i) => 26 + heights.slice(0, i).reduce((a, b) => a + b, 0),
  );
  const height = heights.reduce((a, b) => a + b, 0) + 54;
  const active = selected === null ? null : segments[selected];
  const legend = legendOf(segments);
  return (
    <div className="column-scroll" ref={host}>
      <ul className="lith-legend" aria-label="Lithology legend">
        {legend.map((e) => (
          <li key={e.lithology}>
            <button
              type="button"
              className={focus === e.lithology ? "on" : undefined}
              aria-pressed={focus === e.lithology}
              onClick={() => setFocus(focus === e.lithology ? null : e.lithology)}
            >
              <svg width="22" height="16" aria-hidden="true">
                <rect width="22" height="16" rx="2" fill={e.color} />
                <rect
                  width="22"
                  height="16"
                  rx="2"
                  fill={`url(#${patternFor(e.lithology)})`}
                />
              </svg>
              <span className="name">{e.lithology}</span>
              <span className="share">
                {e.share}% · {Math.round(e.feet).toLocaleString()} ft
              </span>
            </button>
          </li>
        ))}
      </ul>
      <svg
        viewBox={`0 0 350 ${height}`}
        className="lith-column"
        aria-label="Lithology by depth"
      >
        <defs>
          <pattern
            id="hatch"
            width="8"
            height="8"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M-2 2L2-2M0 8L8 0M6 10L10 6"
              stroke="#14232e"
              opacity=".25"
            />
          </pattern>
          <pattern id="dots" width="8" height="8" patternUnits="userSpaceOnUse">
            <circle cx="3" cy="3" r="1" fill="#14232e" opacity=".3" />
          </pattern>
          <pattern
            id="brick"
            width="16"
            height="10"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M0 0H16M0 5H16M8 0V5M0 5V10"
              stroke="#14232e"
              opacity=".25"
            />
          </pattern>
        </defs>
        <text x="2" y="12" className="depth">
          DEPTH / FT
        </text>
        {segments.flatMap((s, i) => {
          const ticks = [];
          for (
            let depth = Math.ceil(s.top_ft / 1000) * 1000;
            depth < s.base_ft;
            depth += 1000
          ) {
            const y =
              offsets[i] +
              (heights[i] * (depth - s.top_ft)) / (s.base_ft - s.top_ft);
            ticks.push(
              <g key={depth}>
                <text x="55" y={y + 4} textAnchor="end" className="depth">
                  {depth.toLocaleString()}
                </text>
                <path d={`M60 ${y}h9`} stroke="#748793" />
              </g>,
            );
          }
          return ticks;
        })}
        {segments.map((s, i) => {
          const y = offsets[i],
            h = heights[i];
          const pattern = patternFor(s.lithology);
          const dim = focus !== null && s.lithology !== focus;
          return (
            <g
              key={i}
              role="button"
              tabIndex={0}
              aria-label={`${s.lithology} ${s.pct} percent, ${Math.round(s.top_ft)} to ${Math.round(s.base_ft)} feet`}
              onClick={() => setSelected(i)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setSelected(i);
                }
              }}
              className="segment"
              opacity={dim ? 0.25 : 1}
            >
              <rect x="76" y={y} width="70" height={h - 1} fill={s.color} />
              <rect
                x="76"
                y={y}
                width="70"
                height={h - 1}
                fill={`url(#${pattern})`}
              />
              <path d={`M147 ${y + h / 2}h18`} stroke="rgba(226,236,242,.35)" />
              <text x="174" y={y + h / 2 + 4}>
                {s.lithology}
              </text>
            </g>
          );
        })}
      </svg>
      {active && (
        <div
          className="popover glass"
          style={{ top: offsets[selected!] }}
          role="dialog"
          aria-label="Bed breakdown"
        >
          <button
            className="close"
            onClick={() => setSelected(null)}
            aria-label="Close bed details"
          >
            ×
          </button>
          <strong>
            {Math.round(active.top_ft).toLocaleString()}–
            {Math.round(active.base_ft).toLocaleString()} ft
          </strong>
          <p>
            {Math.round(active.base_ft - active.top_ft).toLocaleString()} ft
            {area ? ` · ${active.n_wells} wells` : ""}
          </p>
          {active.breakdown.map((b) => (
            <div className="breakdown" key={b.lithology}>
              <span>{b.lithology}</span>
              <span>{b.pct}%</span>
              <i style={{ width: `${b.pct}%` }} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
