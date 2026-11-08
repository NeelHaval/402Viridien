import type { Report, UploadResult, Well, WellDetail } from "./types";
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}
export const api = {
  wells: () => request<Well[]>("/wells"),
  well: (id: string) => request<WellDetail>(`/wells/${id}`),
  report: (polygon: number[][]) =>
    request<Report>("/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon }),
    }),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadResult>("/upload", { method: "POST", body });
  },
};
