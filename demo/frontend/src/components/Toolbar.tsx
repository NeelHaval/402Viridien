export function Toolbar({
  selecting,
  processing,
  onSelect,
  onUpload,
}: {
  selecting: boolean;
  processing: boolean;
  onSelect: () => void;
  onUpload: () => void;
}) {
  return (
    <nav className="toolbar glass" aria-label="Map tools">
      <button aria-pressed={selecting} onClick={onSelect}>
        <svg viewBox="0 0 24 24">
          <path d="M5 4L20 7L17 20L3 16Z" />
          <circle cx="5" cy="4" r="2" />
          <circle cx="20" cy="7" r="2" />
          <circle cx="17" cy="20" r="2" />
          <circle cx="3" cy="16" r="2" />
        </svg>
        Select area
      </button>
      <span />
      <button onClick={onUpload}>
        <svg viewBox="0 0 24 24">
          <path d="M12 16V3m-5 5l5-5 5 5M4 15v6h16v-6" />
        </svg>
        Upload well
      </button>
      {processing && (
        <em className="chip" role="status">
          Processing
        </em>
      )}
    </nav>
  );
}
