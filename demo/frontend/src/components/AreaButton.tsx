import type { ComponentProps } from "react";

// Touch browsers can suppress the compatibility click after a map drag.
// Activate on touch release; ignore its later compatibility click.
export function AreaButton({ onAction, ...props }: Omit<ComponentProps<"button">, "onClick"> & { onAction: () => void }) {
  return <button {...props} onPointerUp={e => {
    if (e.pointerType === "touch" && !props.disabled) { e.preventDefault(); onAction(); }
  }} onClick={e => {
    if ((e.nativeEvent as PointerEvent).pointerType !== "touch") onAction();
  }} />;
}
