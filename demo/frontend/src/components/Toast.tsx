export function Toast({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div className="notice glass" role="status">
      <span>{message}</span>
      <button onClick={onDismiss} aria-label="Dismiss message">
        ×
      </button>
    </div>
  );
}
