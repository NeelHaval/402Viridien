export type Status = "queued" | "processing" | "ready" | "failed" | "partial";
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
  pixel_column?: {top_y:number;bottom_y:number;lithology:string;cluster_id:number;color:string}[];
  result?: { engine: string; source_image_sha256: string; result_sha256: string; depth_status: string; depth_estimate?:{unit:string;unit_association_verified:false;fit:{a:number;b:number;n_points:number;rmse_px:number};calibration_sha256:string} };
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
  polygon?: number[][];
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
