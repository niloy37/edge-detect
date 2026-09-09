"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DetectorStage, type StageSource } from "@/components/DetectorStage";
import { LatencyHUD } from "@/components/LatencyHUD";
import { Badge, Field, SegmentedControl, Slider } from "@/components/ui";
import { createFaceBlurOverlay, createFaceBoxOverlay } from "@/lib/blur";
import { isCrossOriginIsolated } from "@/lib/ep";
import { getModel, getVariant } from "@/lib/models";
import { DEFAULT_DETECT_OPTIONS } from "@/lib/types";
import { useDetector } from "@/lib/useDetector";

const FACE_MODEL_ID = "face-n";
type Mode = "blur" | "boxes";
type SourceChoice = "webcam" | "sample-image" | "upload";

const CAMERA_ACQUIRE_DEADLINE_MS = 9000;

export function FacesClient() {
  const model = getModel(FACE_MODEL_ID);
  const ready = model?.status === "ready";

  if (!model || !ready) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-6">
        <h2 className="text-sm font-semibold text-neutral-200">Model not built yet</h2>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-neutral-400">
          The face weights are produced by the same pipeline as every other model here:{" "}
          <code className="text-neutral-300">prepare_data.py</code> →{" "}
          <code className="text-neutral-300">train.py</code> →{" "}
          <code className="text-neutral-300">export_onnx.py</code> →{" "}
          <code className="text-neutral-300">quantize_int8.py</code>, driven by{" "}
          <code className="text-neutral-300">ml/configs/face.yaml</code>. This page reads the
          generated manifest, so it lights up when the weights land — nothing here is hardcoded.
        </p>
      </div>
    );
  }

  return <FacesDemo />;
}

function FacesDemo() {
  const model = getModel(FACE_MODEL_ID)!;

  const [mode, setMode] = useState<Mode>("blur");
  const [showOutline, setShowOutline] = useState(true);
  const [sourceKind, setSourceKind] = useState<SourceChoice>("webcam");
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const [scoreThreshold, setScoreThreshold] = useState(0.4);
  const [paused, setPaused] = useState(false);
  const [renderFps, setRenderFps] = useState(0);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [isolated, setIsolated] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sourceReadyRef = useRef(false);

  useEffect(() => setIsolated(isCrossOriginIsolated()), []);

  const variant = getVariant(model, "int8") ?? model.variants[0];
  const labels = useMemo(() => model.classes, [model]);

  const {
    ready,
    phase,
    threads,
    workerIsolated,
    error,
    warmupMs,
    loadMs,
    stats,
    beginFrame,
    sendFrame,
    abortFrame,
    detectionsRef,
  } = useDetector({
    modelUrl: variant.file,
    labels,
    ep: "wasm",
    inputSize: 640,
    smoothing: true,
  });

  const options = useMemo(
    () => ({ ...DEFAULT_DETECT_OPTIONS, scoreThreshold, iouThreshold: 0.45 }),
    [scoreThreshold],
  );

  const overlay = useMemo(
    () => (mode === "blur" ? createFaceBlurOverlay({ showOutline }) : createFaceBoxOverlay()),
    [mode, showOutline],
  );

  const sampleImage = model.sample?.image;
  const source: StageSource = useMemo(() => {
    if (sourceKind === "upload") return { kind: "image", url: uploadUrl ?? undefined };
    if (sourceKind === "sample-image") return { kind: "image", url: sampleImage };
    return { kind: "webcam" };
  }, [sourceKind, uploadUrl, sampleImage]);

  const onSourceReady = useCallback(() => {
    sourceReadyRef.current = true;
  }, []);

  useEffect(() => {
    sourceReadyRef.current = false;
    if (!ready || sourceKind !== "webcam" || !sampleImage) return;
    const timer = window.setTimeout(() => {
      if (!sourceReadyRef.current) setSourceKind("sample-image");
    }, CAMERA_ACQUIRE_DEADLINE_MS);
    return () => window.clearTimeout(timer);
  }, [ready, sourceKind, sampleImage]);

  useEffect(() => {
    if (sourceKind === "webcam" && sourceError && sampleImage) setSourceKind("sample-image");
  }, [sourceKind, sourceError, sampleImage]);

  const handleUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });
    setSourceKind("upload");
  }, []);

  const faceCount = stats.detectionCount;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="space-y-3">
        <DetectorStage
          source={source}
          ready={ready}
          paused={paused}
          options={options}
          beginFrame={beginFrame}
          sendFrame={sendFrame}
          abortFrame={abortFrame}
          detectionsRef={detectionsRef}
          onRenderFps={setRenderFps}
          onSourceError={setSourceError}
          onSourceReady={onSourceReady}
          overlay={overlay}
        />

        <div className="flex flex-wrap items-center gap-2">
          <SegmentedControl
            segments={[
              { value: "webcam" as const, label: "Webcam" },
              { value: "sample-image" as const, label: "Sample image", disabled: !sampleImage },
              { value: "upload" as const, label: "Upload" },
            ]}
            value={sourceKind}
            onChange={(value) => {
              if (value === "upload") fileInputRef.current?.click();
              else setSourceKind(value);
            }}
          />
          <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleUpload} />
          <button
            type="button"
            onClick={() => setPaused((p) => !p)}
            className="rounded-md border border-neutral-800 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-900"
          >
            {paused ? "Resume" : "Pause"}
          </button>
        </div>

        {sourceError ? (
          <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            {sourceError}
          </p>
        ) : null}
        {error ? (
          <p className="rounded-md border border-red-900 bg-red-950/40 px-3 py-2 font-mono text-xs text-red-300">
            {error}
          </p>
        ) : null}
      </div>

      <aside className="space-y-4">
        <div className="rounded-lg border border-neutral-800 bg-neutral-950/80 p-3">
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            <Badge tone={ready ? "good" : "warn"}>{ready ? "running" : phase}</Badge>
            <Badge>{ready ? `WASM x${threads}` : "loading"}</Badge>
            <Badge tone={(workerIsolated ?? isolated) ? "good" : "warn"}>
              {(workerIsolated ?? isolated) ? "isolated" : "not isolated"}
            </Badge>
          </div>

          <div className="mb-3 rounded-md border border-neutral-800 bg-neutral-950 p-3 text-center">
            <div className="font-mono text-3xl tabular-nums text-neutral-100">{faceCount}</div>
            <div className="text-[11px] uppercase tracking-wider text-neutral-500">
              {faceCount === 1 ? "face detected" : "faces detected"}
            </div>
          </div>

          <div className="space-y-3">
            <Field label="Mode" hint={mode === "blur" ? "redacting" : "showing"}>
              <SegmentedControl
                segments={[
                  { value: "blur" as const, label: "Blur faces" },
                  { value: "boxes" as const, label: "Show boxes" },
                ]}
                value={mode}
                onChange={setMode}
              />
            </Field>

            {mode === "blur" ? (
              <Field label="Outline blurred regions" hint={showOutline ? "on" : "off"}>
                <SegmentedControl
                  segments={[
                    { value: "on" as const, label: "On" },
                    { value: "off" as const, label: "Off" },
                  ]}
                  value={showOutline ? "on" : "off"}
                  onChange={(value) => setShowOutline(value === "on")}
                />
              </Field>
            ) : null}

            <Field label="Confidence" hint={scoreThreshold.toFixed(2)}>
              <Slider
                value={scoreThreshold}
                min={0.05}
                max={0.9}
                step={0.01}
                onChange={setScoreThreshold}
              />
            </Field>
          </div>
        </div>

        <LatencyHUD stats={stats} renderFps={renderFps} warmupMs={warmupMs} loadMs={loadMs} />

        <p className="rounded-md border border-neutral-800 bg-neutral-950/60 p-3 text-[11px] leading-relaxed text-neutral-500">
          Detection only — this finds <em>where</em> faces are, never <em>whose</em> they are. No
          identity model, no embeddings, no matching. The frames are processed in this tab and never
          uploaded, which is the point: a privacy tool that had to send your camera to a server
          would be arguing against itself.
        </p>
      </aside>
    </div>
  );
}
