export type Status = "queued" | "processing" | "ready" | "failed";
export type Metric = "drilling" | "cost" | "price" | "co2";
export type Scores = Record<Metric, number | null>;
export interface Well {
  id: string;
  name: string;
  lat: number | null;
  lon: number | null;
  block: string | null;
  status: Status;
  scores: Scores | null;
  validation_confidence: number | null;
  located: boolean;
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
  total_depth_ft: number | null;
  column: Segment[];
  validation: {
    confidence: number | null;
    components: Record<string, number | null>;
    flags: string[];
  };
  warnings: string[];
  error: string | null;
}
export interface Report {
  wells: (Pick<Well, "id" | "name" | "scores"> & { lat: number; lon: number })[];
  scores: Scores | null;
  column: Segment[];
  data_quality: string[];
}
export interface UploadResult {
  id: string;
  status: Status;
}
export interface Panel {
  mode: "well" | "area";
  title: string;
  data: WellDetail | Report | null;
  error?: string;
}
