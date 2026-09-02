/// <reference lib="webworker" />
import * as ort from "onnxruntime-web";

import { decodeYolo } from "@/lib/decode";
import { Preprocessor } from "@/lib/letterbox";
import { nonMaxSuppression } from "@/lib/nms";
import type { WorkerRequest, WorkerResponse } from "@/lib/protocol";
import type { EPName } from "@/lib/types";

// Self-hosted from public/ort (see scripts/sync-ort-assets.mjs) rather than the CDN,
// so the runtime version is pinned by package-lock.json and nothing third-party is
// fetched at run time.
ort.env.wasm.wasmPaths = "/ort/";

const WARMUP_RUNS = 3;

/**
 * Thread count is decided by the caller, not here, and a worker only ever gets one
 * attempt at it. ORT initialises its WASM runtime once per worker: if a threaded
 * start-up hangs, setting numThreads=1 and calling create() again in the same worker
 * reuses the same half-initialised runtime and hangs exactly as before (measured).
 * Recovering therefore means a brand new worker, which only the main thread can do --
 * see the fallback timer in lib/useDetector.ts.
 */

let session: ort.InferenceSession | null = null;
let threadsInUse = 1;
let preprocessor: Preprocessor | null = null;
let labels: string[] = [];
let inputName = "";
let outputName = "";

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(message, transfer);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

async function init(request: Extract<WorkerRequest, { type: "init" }>): Promise<void> {
  labels = request.labels;
  preprocessor = new Preprocessor(request.inputSize);

  // Threads are pointless without SharedArrayBuffer, and asking for them on a page
  // that is not cross-origin isolated throws at session creation.
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  threadsInUse = isolated ? Math.max(1, request.threads) : 1;
  ort.env.wasm.numThreads = threadsInUse;

  post({ type: "status", phase: "creating-session" });
  const loadStart = performance.now();
  session = await ort.InferenceSession.create(request.modelUrl, {
    executionProviders: [request.ep],
    graphOptimizationLevel: "all",
  });
  const loadMs = performance.now() - loadStart;

  inputName = session.inputNames[0];
  outputName = session.outputNames[0];

  // Warm up before reporting anything. The first run pays for shader compilation
  // (WebGPU) or kernel selection and JIT (WASM) and is 10-50x the steady-state cost;
  // publishing it as "inference latency" would be a lie the HUD then repeats.
  post({ type: "status", phase: "warming-up" });
  const size = request.inputSize;
  const dummy = new ort.Tensor("float32", new Float32Array(3 * size * size), [1, 3, size, size]);
  const warmups: number[] = [];
  for (let i = 0; i < WARMUP_RUNS; i++) {
    const start = performance.now();
    await session.run({ [inputName]: dummy });
    warmups.push(performance.now() - start);
  }

  post({
    type: "ready",
    ep: request.ep,
    threads: threadsInUse,
    warmupMs: median(warmups),
    loadMs,
    inputName,
    outputName,
  });
}

async function detect(request: Extract<WorkerRequest, { type: "frame" }>): Promise<void> {
  if (!session || !preprocessor) {
    request.bitmap.close();
    return;
  }

  const t0 = performance.now();
  const { data, transform } = preprocessor.run(request.bitmap);
  request.bitmap.close();
  const size = preprocessor.inputSize;
  const tensor = new ort.Tensor("float32", data, [1, 3, size, size]);
  const t1 = performance.now();

  const output = await session.run({ [inputName]: tensor });
  const t2 = performance.now();

  const raw = output[outputName];
  const [, channels, anchors] = raw.dims as [number, number, number];
  const candidates = decodeYolo(
    raw.data as Float32Array,
    channels,
    anchors,
    transform,
    labels,
    request.options,
  );
  const detections = nonMaxSuppression(candidates, request.options);
  const t3 = performance.now();

  post({
    type: "result",
    seq: request.seq,
    detections,
    inputSize: size,
    timing: {
      preprocess: t1 - t0,
      inference: t2 - t1,
      postprocess: t3 - t2,
      total: t3 - t0,
    },
  });
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  try {
    switch (request.type) {
      case "init":
        await init(request);
        break;
      case "frame":
        await detect(request);
        break;
      case "setInputSize":
        // The graph has dynamic H/W, so only the preprocessor changes. The first run
        // at a new size re-pays shape-dependent setup (a WebGPU shader recompile),
        // which is visible as a one-off spike in the p95 and is expected.
        preprocessor = new Preprocessor(request.inputSize);
        break;
      case "dispose":
        await session?.release();
        session = null;
        break;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    post({ type: "error", message, fatal: request.type === "init" });
  }
};

export type { EPName };
