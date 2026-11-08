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
  async function upload(file?: File) {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.upload(file);
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
        <button
          className="drop-zone"
          disabled={busy}
          onClick={() => input.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            void upload(e.dataTransfer.files[0]);
          }}
        >
          <span>↑</span>
          {busy ? "Uploading…" : "Drop a file or click to browse"}
        </button>
        <input
          hidden
          ref={input}
          type="file"
          onChange={(e) => void upload(e.target.files?.[0])}
        />
        {error && <p role="alert">{error}</p>}
      </div>
    </div>
  );
}
