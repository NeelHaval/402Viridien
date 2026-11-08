import { useEffect, useState } from "react";
export function Dial({ value, label }: { value: number; label: string }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const timer = setTimeout(() => setShown(value), 60);
    return () => clearTimeout(timer);
  }, [value]);
  const color = value >= 67 ? "#5EEAD4" : value >= 34 ? "#F4CC82" : "#F3A3B5";
  return (
    <div className="dial">
      <svg
        viewBox="0 0 80 80"
        role="img"
        aria-label={`${label}: ${value} out of 100`}
      >
        <circle cx="40" cy="40" r="33" className="dial-track" />
        <circle
          cx="40"
          cy="40"
          r="33"
          className="dial-value"
          stroke={color}
          strokeDasharray="207.35"
          strokeDashoffset={207.35 * (1 - shown / 100)}
          transform="rotate(-90 40 40)"
        />
        <text x="40" y="46" textAnchor="middle">
          {value}
        </text>
      </svg>
      <span>{label}</span>
    </div>
  );
}
