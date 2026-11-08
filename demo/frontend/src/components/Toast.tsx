import {Icon} from "./Icon";
export function Toast({
  message,
  onDismiss,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className="notice glass" role="status">
      <span>{message}</span>
      {onRetry && <button onClick={onRetry}>Retry</button>}
      <button onClick={onDismiss} aria-label="Dismiss message">
        <Icon name="close"/>
      </button>
    </div>
  );
}
