export type Scores = Record<"drilling" | "cost" | "price" | "co2", number>;
export interface Well {
  id: string;
  name: string;
  lat: number | null;
  lon: number | null;
  block: string;
  status: "processing" | "ready" | "failed";
  scores: Scores | null;
}
export interface Segment {
  top_ft: number;
  base_ft: number;
  lithology: string;
  pct: number;
  color: string;
  n_wells: number;
  breakdown: { lithology: string; pct: number }[];
}
export interface WellDetail extends Well {
  total_depth_ft: number;
  column: Segment[];
}
export interface Report {
  wells: Pick<Well, "id" | "name" | "lat" | "lon" | "scores">[];
  scores: Scores | null;
  column: Segment[];
}
export interface UploadResult {
  id: string;
  status: "processing";
}
export interface Panel {
  mode: "well" | "area";
  title: string;
  data: WellDetail | Report | null;
  error?: string;
}
