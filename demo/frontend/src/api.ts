import type { Report, UploadResult, Well, WellDetail } from "./types";
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new Error("Cannot reach the server");
  }
  if (!response.ok) {
    if (response.status === 401) window.dispatchEvent(new Event("session-expired"));
    let message = `Request failed (${response.status})`;
    try {
      message = (await response.json()).error.message ?? message;
    } catch {
      /* body was not the standard error shape */
    }
    throw new Error(message);
  }
  return response.json();
}
export const api = {
  wells: () => request<Well[]>("/wells"),
  well: (id: string) => request<WellDetail>(`/wells/${id}`),
  csvUrl: (id: string) => `/api/wells/${id}/csv`,
  report: (polygon: number[][]) =>
    request<Report>("/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon }),
    }).then((report) => ({ ...report, polygon })),
  upload: (file: File, bbox?: File | null) => {
    const body = new FormData();
    body.append("file", file);
    if (bbox) body.append("bbox_json", bbox);
    return request<UploadResult>("/upload", { method: "POST", body });
  },
};
