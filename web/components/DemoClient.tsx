"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DetectorStage, type StageSource } from "@/components/DetectorStage";
import { LatencyHUD } from "@/components/LatencyHUD";
import { Badge, Field, SegmentedControl, Slider } from "@/components/ui";
import { isCrossOriginIsolated, probeWebGPU } from "@/lib/ep";
import { MODELS, PENDING_MODELS, getVariant, type Precision } from "@/lib/models";
import { DEFAULT_DETECT_OPTIONS, type EPName } from "@/lib/types";
import { useDetector } from "@/lib/useDetector";

type EPChoice = "auto" | EPName;
/** What the visitor picked, which is a finer distinction than the stage's source kind. */
type SourceChoice = "webcam" | "sample-video" | "sample-image" | "upload";
const INPUT_SIZES = [320, 480, 640] as const;

export function DemoClient() {
  const [modelId, setModelId] = useState(MODELS[0]?.id ?? "");
  const [precision, setPrecision] = useState<Precision>("int8");
  const [epChoice, setEpChoice] = useState<EPChoice>("auto");
  const [inputSize, setInputSize] = useState<number>(640);
  const [scoreThreshold, setScoreThreshold] = useState(DEFAULT_DETECT_OPTIONS.scoreThreshold);
  const [iouThreshold, setIouThreshold] = useState(DEFAULT_DETECT_OPTIONS.iouThreshold);
  const [paused, setPaused] = useState(false);
  const [renderFps, setRenderFps] = useState(0);
  const [sourceKind, setSourceKind] = useState<SourceChoice>("webcam");
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [webgpuAvailable, setWebgpuAvailable] = useState<boolean | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const model = useMemo(() => MODELS.find((m) => m.id === modelId) ?? MODELS[0], [modelId]);
  const variant = model ? getVariant(model, precision) : undefined;

  useEffect(() => {
    void probeWebGPU().then((adapter) => setWebgpuAvailable(adapter !== null));
  }, []);

  // "auto" resolves only once the adapter probe lands. Until then no session is
  // created, which avoids building a WASM session and immediately discarding it.
  const ep: EPName | null =
    epChoice === "auto"
      ? webgpuAvailable === null
        ? null
        : webgpuAvailable
          ? "webgpu"
          : "wasm"
      : epChoice;

  const labels = useMemo(() => model?.classes ?? [], [model]);
  const modelUrl = ep && variant ? variant.file : null;

  const {
    ready,
    phase,
    threads,
    error,
    warmupMs,
    loadMs,
    stats,
    submit,
    canSubmit,
    detectionsRef,
  } = useDetector({
    modelUrl,
    labels,
    ep: ep ?? "wasm",
    inputSize,
  });

  const options = useMemo(
    () => ({ scoreThreshold, iouThreshold, maxDetections: DEFAULT_DETECT_OPTIONS.maxDetections }),
    [scoreThreshold, iouThreshold],
  );

  const source: StageSource = useMemo(() => {
    if (sourceKind === "upload") return { kind: "image", url: uploadUrl ?? undefined };
    if (sourceKind === "sample-image") return { kind: "image", url: model?.sample?.image };
    if (sourceKind === "sample-video") return { kind: "sample", url: model?.sample?.video };
    return { kind: "webcam" };
  }, [sourceKind, uploadUrl, model]);

  // A denied camera used to leave an empty stage and a detector with nothing to do.
  // Fall through to the model's sample so the page still demonstrates something.
  const sampleImage = model?.sample?.image;
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

  const [isolated, setIsolated] = useState(true);
  useEffect(() => setIsolated(isCrossOriginIsolated()), []);

  // Reports the threads the worker actually got, not the count we asked for.
  const activeEpLabel =
    ep === "webgpu" ? "WebGPU" : ep === "wasm" ? "WASM x" + (ready ? threads : "?") : "resolving";

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="space-y-3">
        <DetectorStage
          source={source}
          ready={ready}
          paused={paused}
          options={options}
          submit={submit}
          canSubmit={canSubmit}
          detectionsRef={detectionsRef}
          onRenderFps={setRenderFps}
          onSourceError={setSourceError}
        />

        <div className="flex flex-wrap items-center gap-2">
          <SegmentedControl
            segments={[
              { value: "webcam" as const, label: "Webcam" },
              {
                value: "sample-video" as const,
                label: "Sample video",
                disabled: !model?.sample?.video,
                title: model?.sample?.video
                  ? undefined
                  : "No sample clip shipped for this model yet",
              },
              {
                value: "sample-image" as const,
                label: "Sample image",
                disabled: !model?.sample?.image,
              },
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
            {paused ? "Resume detection" : "Pause detection"}
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
            <Badge>{activeEpLabel}</Badge>
            <Badge tone={isolated ? "good" : "warn"}>{isolated ? "isolated" : "1 thread"}</Badge>
          </div>

          <div className="space-y-3">
            <Field label="Model" hint={model ? model.classes.length + " classes" : undefined}>
              <select
                value={modelId}
                onChange={(event) => setModelId(event.target.value)}
                className="w-full rounded-md border border-neutral-800 bg-neutral-950 px-2 py-1.5 text-xs text-neutral-200"
              >
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
                {PENDING_MODELS.map((m) => (
                  <option key={m.id} value={m.id} disabled>
                    {m.name} (training)
                  </option>
                ))}
              </select>
              {model ? (
                <p className="text-[11px] leading-snug text-neutral-500">{model.description}</p>
              ) : null}
            </Field>

            <Field
              label="Precision"
              hint={variant ? (variant.bytes / 1e6).toFixed(1) + " MB" : undefined}
            >
              <SegmentedControl
                segments={(["fp32", "int8"] as const).map((p) => ({
                  value: p,
                  label: p.toUpperCase(),
                  disabled: !model || !getVariant(model, p),
                }))}
                value={precision}
                onChange={setPrecision}
              />
            </Field>

            <Field label="Input size" hint={inputSize + " px"}>
              <SegmentedControl
                segments={INPUT_SIZES.map((size) => ({ value: size, label: String(size) }))}
                value={inputSize}
                onChange={setInputSize}
              />
            </Field>

            <Field label="Backend">
              <SegmentedControl
                segments={[
                  { value: "auto" as const, label: "Auto" },
                  {
                    value: "webgpu" as const,
                    label: "WebGPU",
                    disabled: webgpuAvailable === false,
                    title:
                      webgpuAvailable === false ? "No WebGPU adapter on this device" : undefined,
                  },
                  { value: "wasm" as const, label: "WASM" },
                ]}
                value={epChoice}
                onChange={setEpChoice}
              />
            </Field>

            <Field label="Confidence" hint={scoreThreshold.toFixed(2)}>
              <Slider
                value={scoreThreshold}
                min={0.05}
                max={0.9}
                step={0.01}
                onChange={setScoreThreshold}
              />
            </Field>

            <Field label="NMS IoU" hint={iouThreshold.toFixed(2)}>
              <Slider
                value={iouThreshold}
                min={0.1}
                max={0.9}
                step={0.01}
                onChange={setIouThreshold}
              />
            </Field>
          </div>
        </div>

        <LatencyHUD stats={stats} renderFps={renderFps} warmupMs={warmupMs} loadMs={loadMs} />

        {!isolated ? (
          <p className="rounded-md border border-neutral-800 bg-neutral-950/60 p-3 text-[11px] leading-relaxed text-neutral-500">
            This page is not cross-origin isolated, so SharedArrayBuffer is unavailable and WASM
            runs single-threaded. The deployed build sets COOP/COEP headers to fix this.
          </p>
        ) : null}
      </aside>
    </div>
  );
}
