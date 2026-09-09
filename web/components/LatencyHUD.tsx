"use client";

import type { DetectorStats } from "@/lib/useDetector";

function ms(value: number): string {
  return value > 0 ? `${value.toFixed(1)}` : "—";
}

function Row({ label, value, unit, hint }: { label: string; value: string; unit?: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-xs text-neutral-500" title={hint}>
        {label}
      </span>
      <span className="font-mono text-sm tabular-nums text-neutral-100">
        {value}
        {unit ? <span className="ml-0.5 text-[10px] text-neutral-500">{unit}</span> : null}
      </span>
    </div>
  );
}

export function LatencyHUD({
  stats,
  renderFps,
  warmupMs,
  loadMs,
}: {
  stats: DetectorStats;
  renderFps: number;
  warmupMs: number;
  loadMs: number;
}) {
  const stageTotal = stats.preprocess + stats.inference + stats.postprocess;
  const share = (value: number) => (stageTotal > 0 ? (value / stageTotal) * 100 : 0);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/80 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Latency</h3>
        <span className="font-mono text-[10px] text-neutral-600">n={stats.samples}</span>
      </div>

      {/* Stage breakdown as a stacked bar: which third of the pipeline to optimise
          is the first question anyone asks, and a bar answers it faster than numbers. */}
      <div className="mb-3 flex h-1.5 overflow-hidden rounded-full bg-neutral-900">
        <div style={{ width: `${share(stats.preprocess)}%` }} className="bg-sky-500" />
        <div style={{ width: `${share(stats.inference)}%` }} className="bg-violet-500" />
        <div style={{ width: `${share(stats.postprocess)}%` }} className="bg-amber-500" />
      </div>

      <div className="divide-y divide-neutral-900">
        <Row label="◆ preprocess" value={ms(stats.preprocess)} unit="ms" hint="Letterbox + RGB->NCHW float32" />
        <Row label="◆ inference" value={ms(stats.inference)} unit="ms" hint="session.run() on the selected execution provider" />
        <Row label="◆ decode + NMS" value={ms(stats.postprocess)} unit="ms" hint="Head decode and class-wise NMS in TypeScript" />
        <Row label="total p50" value={ms(stats.totalP50)} unit="ms" hint="Median over the last 60 frames" />
        <Row label="total p95" value={ms(stats.totalP95)} unit="ms" hint="95th percentile: the stutter you actually notice" />
        <Row label="detect fps" value={stats.detectFps > 0 ? stats.detectFps.toFixed(1) : "—"} hint="Frames the model completed per second" />
        <Row label="render fps" value={renderFps > 0 ? renderFps.toFixed(0) : "—"} hint="Canvas redraws per second — decoupled from detection" />
        <Row label="objects" value={String(stats.detectionCount)} hint="Detections surviving NMS in the last frame" />
        <Row
          label="network input"
          value={stats.inputWidth ? `${stats.inputWidth}x${stats.inputHeight}` : "—"}
          hint="Aspect-matched and stride-aligned, not square: a 16:9 frame in a square tensor spends 44% of the compute on grey padding."
        />
        <Row
          label="frames sent / dropped"
          value={`${stats.submitted} / ${stats.dropped}`}
          hint="Frames handed to the model, and frames skipped because it was still busy. Dropping is correct: the newest frame is the only one worth running."
        />
      </div>

      <div className="mt-2 border-t border-neutral-900 pt-2">
        <Row label="model load" value={ms(loadMs)} unit="ms" hint="Fetch + session creation" />
        <Row label="warm-up" value={ms(warmupMs)} unit="ms" hint="Median of 3 warm-up runs before any measurement was taken" />
      </div>
    </div>
  );
}
