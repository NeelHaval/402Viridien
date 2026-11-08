import type { Scores } from "./types";
export type WellEvent =
  | { type: "well_created"; id: string; name?: string; status?: string }
  | {
      type: "well_located";
      id: string;
      lat: number;
      lon: number;
      block?: string | null;
    }
  | {
      type: "well_ready";
      id: string;
      name?: string | null;
      validation_confidence?: number;
    }
  | { type: "well_scored"; id: string; scores: Scores }
  | { type: "well_failed"; id: string; error?: string };
const TYPES = [
  "well_created",
  "well_located",
  "well_ready",
  "well_scored",
  "well_failed",
] as const;
const FIRST_RETRY_MS = 1000;
const MAX_RETRY_MS = 30000;
/** Subscribe to GET /api/events; reconnects with exponential backoff. `onConnection` reports open/closed. */
export function subscribe(
  onEvent: (event: WellEvent) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  let source: EventSource | null = null;
  let retry = FIRST_RETRY_MS;
  let timer: number | undefined;
  let stopped = false;
  const connect = () => {
    if (stopped) return;
    source = new EventSource("/api/events");
    source.onopen = () => {
      retry = FIRST_RETRY_MS;
      onConnection(true);
    };
    for (const type of TYPES)
      source.addEventListener(type, (e) => {
        try {
          onEvent(JSON.parse((e as MessageEvent).data) as WellEvent);
        } catch {
          /* ignore a malformed frame */
        }
      });
    source.onerror = () => {
      source?.close();
      source = null;
      onConnection(false);
      if (stopped) return;
      timer = window.setTimeout(connect, retry);
      retry = Math.min(retry * 2, MAX_RETRY_MS);
    };
  };
  connect();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
    source?.close();
  };
}
