import { useEffect, useRef, useState } from "react";
import { api } from "../api";
export function UploadModal({
  onClose,
  onUploaded,
}: {
  onClose: () => void;
  onUploaded: (id: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    dialog.current?.focus();
  }, []);
  const [scan, setScan] = useState<File | null>(null);
  const [bbox, setBbox] = useState<File | null>(null);
  const bboxInput = useRef<HTMLInputElement>(null);
  function pick(files: FileList | File[]) {
    for (const file of Array.from(files)) {
      if (file.name.toLowerCase().endsWith(".json")) setBbox(file);
      else setScan(file);
    }
  }
  async function upload() {
    if (!scan || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.upload(scan, bbox);
      onUploaded(result.id);
      onClose();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop" onClick={() => !busy && onClose()}>
      <div
        className="upload-modal glass"
        role="dialog"
        aria-modal="true"
        aria-label="Upload well"
        tabIndex={-1}
        ref={dialog}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Escape" && !busy) onClose();
          if (e.key === "Tab") {
            const buttons = dialog.current!.querySelectorAll<HTMLButtonElement>(
              "button:not(:disabled)",
            );
            const first = buttons[0],
              last = buttons[buttons.length - 1];
            if (e.shiftKey && document.activeElement === first) {
              e.preventDefault();
              last?.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
              e.preventDefault();
              first?.focus();
            }
          }
        }}
      >
        <header>
          <h1>Upload well</h1>
          <button
            className="close"
            onClick={onClose}
            disabled={busy}
            aria-label="Close upload"
          >
            ×
          </button>
        </header>
        <div className="drop-zones">
          {[
            {
              key: "scan",
              title: "SCAN",
              hint: "Drop the mud log image or click to browse",
              file: scan,
              ref: input,
              accept: undefined,
            },
            {
              key: "bbox",
              title: "JSON",
              hint: "Drop the bbox JSON (optional, gives lithology names)",
              file: bbox,
              ref: bboxInput,
              accept: ".json,application/json",
            },
          ].map((z) => (
            <div key={z.key} className="drop-wrap">
              <button
                className={`drop-zone${z.file ? " filled" : ""}`}
                disabled={busy}
                onClick={() => z.ref.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  pick(e.dataTransfer.files);
                }}
              >
                <strong className="drop-title">{z.title}</strong>
                <span className="drop-hint">
                  {busy && z.file ? "Uploading…" : z.file ? z.file.name : z.hint}
                </span>
              </button>
              <input
                hidden
                ref={z.ref}
                type="file"
                accept={z.accept}
                onChange={(e) => e.target.files && pick(e.target.files)}
              />
            </div>
          ))}
        </div>
        <button className="submit" disabled={!scan || busy} onClick={() => void upload()}>
          Upload
        </button>
        {error && <p role="alert">{error}</p>}
      </div>
    </div>
  );
}
