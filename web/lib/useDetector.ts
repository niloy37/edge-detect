"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { suggestedThreadCount } from "./ep";

/** How long session creation may take before we give up on threads.
 *  Timed from the worker's "creating-session" status, not from worker construction,
 *  so a cold model download does not get counted against it. */
const THREADED_INIT_DEADLINE_MS = 12000;

/** How long WebGPU gets to return its first detection before we fall back to WASM.
 *  Some drivers create and warm up a WebGPU session happily and then never complete
 *  a real inference, which looks to a visitor like a demo that draws no boxes. */
const WEBGPU_FIRST_RESULT_DEADLINE_MS = 8000;
import type { LoadPhase, WorkerRequest, WorkerResponse } from "./protocol";
import { FpsCounter, RollingStats } from "./telemetry";
import type { DetectOptions, Detection, EPName, FrameTiming } from "./types";

export interface DetectorStats {
  preprocess: number;
  inference: number;
  postprocess: number;
  totalP50: number;
  totalP95: number;
  inferenceP50: number;
  detectFps: number;
  samples: number;
  detectionCount: number;
  /** Frames handed to the model, and frames skipped because it was still busy. */
  submitted: number;
  dropped: number;
}

const EMPTY_STATS: DetectorStats = {
  preprocess: 0,
  inference: 0,
  postprocess: 0,
  totalP50: 0,
  totalP95: 0,
  inferenceP50: 0,
  detectFps: 0,
  samples: 0,
  detectionCount: 0,
  submitted: 0,
  dropped: 0,
};

export interface UseDetectorArgs {
  modelUrl: string | null;
  labels: string[];
  ep: EPName;
  inputSize: number;
}

/**
 * Owns the inference worker and the numbers that describe it.
 *
 * Two things here matter more than they look:
 *
 * 1. Exactly one frame is allowed in flight (`busyRef`). Without that backpressure a
 *    30fps camera queues frames onto a 20fps model and latency grows without bound
 *    while the display falls further behind reality. Dropping frames is the correct
 *    behaviour for a live detector -- the newest frame is the only one worth running.
 *
 * 2. Stats accumulate in refs and are published to React on an interval, not per
 *    frame. Re-rendering a component tree 60 times a second to update a millisecond
 *    readout would itself become the bottleneck being measured.
 */
export function useDetector({ modelUrl, labels, ep: requestedEp, inputSize }: UseDetectorArgs) {
  const workerRef = useRef<Worker | null>(null);
  const busyRef = useRef(false);
  const seqRef = useRef(0);
  const detectionsRef = useRef<Detection[]>([]);
  const droppedRef = useRef(0);
  const submittedRef = useRef(0);

  const totalStats = useRef(new RollingStats(60));
  const inferStats = useRef(new RollingStats(60));
  const lastTiming = useRef<FrameTiming>({ preprocess: 0, inference: 0, postprocess: 0, total: 0 });
  const fps = useRef(new FpsCounter());

  const [ready, setReady] = useState(false);
  const [phase, setPhase] = useState<LoadPhase>("idle");
  // Thread cap for the next worker: null means "whatever this device suggests".
  // It is not seeded with suggestedThreadCount() because that reads `navigator`,
  // and a lazy state initialiser still runs during server prerendering. Pinned to 1
  // for the rest of the session if a threaded start-up ever stalls -- see
  // THREADED_INIT_DEADLINE_MS below.
  const [threadCap, setThreadCap] = useState<number | null>(null);
  // Set to the EP we abandoned, so the override applies only while that EP is still
  // the requested one -- picking a different backend by hand clears it implicitly.
  const [fellBackFrom, setFellBackFrom] = useState<EPName | null>(null);
  // What the worker actually got, reported back on ready, so the UI can show the
  // truth rather than what we asked for.
  const [threads, setThreads] = useState(1);
  const fallbackTimer = useRef<number | undefined>(undefined);
  const sawResult = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [warmupMs, setWarmupMs] = useState(0);
  const [loadMs, setLoadMs] = useState(0);
  const [stats, setStats] = useState<DetectorStats>(EMPTY_STATS);

  const ep: EPName = fellBackFrom === requestedEp ? "wasm" : requestedEp;

  // Worker lifecycle: a new model or a new execution provider needs a new session.
  // Input-size changes do not -- the graph has dynamic spatial dims.
  useEffect(() => {
    if (!modelUrl) return;
    setReady(false);
    setPhase("idle");
    setError(null);
    sawResult.current = false;
    totalStats.current.reset();
    inferStats.current.reset();
    fps.current.reset();
    detectionsRef.current = [];
    busyRef.current = false;
    submittedRef.current = 0;
    droppedRef.current = 0;

    // Safe here: effects do not run during prerendering.
    const requestedThreads = threadCap ?? suggestedThreadCount();

    const worker = new Worker(new URL("../workers/detector.worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === "status") {
        setPhase(message.phase);
        if (message.phase === "creating-session" && requestedThreads > 1) {
          // Cross-origin isolation is necessary but not sufficient for threaded WASM:
          // ORT still has to spawn its own thread-pool workers, and in some
          // environments that never resolves and never rejects -- session creation
          // just hangs, with no error to catch. Recovery has to be a fresh worker,
          // because ORT initialises its WASM runtime once per worker and retrying
          // inside the stalled one reuses the same broken runtime.
          window.clearTimeout(fallbackTimer.current);
          fallbackTimer.current = window.setTimeout(() => {
            setPhase("retrying-single-thread");
            setThreadCap(1);
          }, THREADED_INIT_DEADLINE_MS);
        }
      } else if (message.type === "ready") {
        setPhase("ready");
        window.clearTimeout(fallbackTimer.current);
        setThreads(message.threads);
        setWarmupMs(message.warmupMs);
        setLoadMs(message.loadMs);
        setReady(true);
      } else if (message.type === "result") {
        sawResult.current = true;
        busyRef.current = false;
        detectionsRef.current = message.detections;
        lastTiming.current = message.timing;
        totalStats.current.push(message.timing.total);
        inferStats.current.push(message.timing.inference);
        fps.current.tick();
      } else {
        busyRef.current = false;
        // The worker timed out on an inference. If we are on WebGPU there is a
        // working alternative, so switch rather than surfacing a dead demo.
        if (message.code === "run-timeout" && ep === "webgpu") {
          setPhase("idle");
          setFellBackFrom("webgpu");
          return;
        }
        setError(message.message);
        if (message.fatal) setReady(false);
      }
    };
    worker.onerror = (event) => setError(event.message || "worker failed to load");

    const request: WorkerRequest = {
      type: "init",
      modelUrl,
      ep,
      inputSize,
      labels,
      threads: requestedThreads,
    };
    worker.postMessage(request);

    return () => {
      window.clearTimeout(fallbackTimer.current);
      worker.postMessage({ type: "dispose" } satisfies WorkerRequest);
      worker.terminate();
      workerRef.current = null;
    };
    // `inputSize` is intentionally excluded: it is applied via setInputSize below
    // rather than by tearing down and reloading the session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelUrl, ep, labels, threadCap]);

  useEffect(() => {
    if (!ready) return;
    workerRef.current?.postMessage({ type: "setInputSize", inputSize } satisfies WorkerRequest);
    totalStats.current.reset();
    inferStats.current.reset();
    fps.current.reset();
  }, [inputSize, ready]);

  useEffect(() => {
    const id = window.setInterval(() => {
      const timing = lastTiming.current;
      setStats({
        preprocess: timing.preprocess,
        inference: timing.inference,
        postprocess: timing.postprocess,
        totalP50: totalStats.current.p50,
        totalP95: totalStats.current.p95,
        inferenceP50: inferStats.current.p50,
        detectFps: fps.current.fps,
        samples: totalStats.current.count,
        detectionCount: detectionsRef.current.length,
        submitted: submittedRef.current,
        dropped: droppedRef.current,
      });
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  /**
   * True when the worker could accept a frame right now.
   *
   * Callers check this *before* grabbing a frame. The render loop runs at display
   * rate while the model may only manage a few frames a second, so without this the
   * loop allocates and immediately discards hundreds of ImageBitmaps per second --
   * pure churn that competes with the inference it is feeding.
   */
  const canSubmit = useCallback(() => workerRef.current !== null && !busyRef.current, []);

  /** Hands a frame to the worker, or drops it if one is already in flight. */
  const submit = useCallback((bitmap: ImageBitmap, options: DetectOptions) => {
    const worker = workerRef.current;
    if (!worker || busyRef.current) {
      droppedRef.current++;
      bitmap.close();
      return false;
    }
    busyRef.current = true;
    submittedRef.current++;
    const request: WorkerRequest = {
      type: "frame",
      bitmap,
      seq: seqRef.current++,
      options,
    };
    worker.postMessage(request, [bitmap]);
    return true;
  }, [ep]);

  return {
    ready,
    phase,
    threads,
    /** The EP actually in use, which may differ from the one requested. */
    activeEp: ep,
    fellBackFrom,
    error,
    warmupMs,
    loadMs,
    stats,
    submit,
    canSubmit,
    detectionsRef,
  };
}
