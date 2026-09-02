"use client";

import type { ReactNode } from "react";

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-neutral-400">{label}</span>
        {hint ? <span className="font-mono text-[10px] text-neutral-600">{hint}</span> : null}
      </div>
      {children}
    </div>
  );
}

export interface Segment<T extends string | number> {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
}

export function SegmentedControl<T extends string | number>({
  segments,
  value,
  onChange,
}: {
  segments: Segment<T>[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex rounded-md border border-neutral-800 bg-neutral-950 p-0.5">
      {segments.map((segment) => {
        const active = segment.value === value;
        return (
          <button
            key={String(segment.value)}
            type="button"
            title={segment.title}
            disabled={segment.disabled}
            onClick={() => onChange(segment.value)}
            className={[
              "flex-1 rounded px-2 py-1.5 text-xs font-medium transition-colors",
              active ? "bg-neutral-200 text-neutral-900" : "text-neutral-400 hover:text-neutral-100",
              segment.disabled ? "cursor-not-allowed opacity-40 hover:text-neutral-400" : "",
            ].join(" ")}
          >
            {segment.label}
          </button>
        );
      })}
    </div>
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      className="h-1 w-full cursor-pointer appearance-none rounded-full bg-neutral-800 accent-neutral-200"
    />
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "good" | "warn"; children: ReactNode }) {
  const tones = {
    neutral: "border-neutral-700 text-neutral-300",
    good: "border-emerald-800 bg-emerald-950/50 text-emerald-300",
    warn: "border-amber-800 bg-amber-950/50 text-amber-300",
  } as const;
  return (
    <span className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide ${tones[tone]}`}>
      {children}
    </span>
  );
}
