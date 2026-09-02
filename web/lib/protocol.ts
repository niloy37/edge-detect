import type { DetectOptions, Detection, EPName, FrameTiming } from "./types";

/** Main thread -> worker. */
export type WorkerRequest =
  | {
      type: "init";
      modelUrl: string;
      ep: EPName;
      inputSize: number;
      labels: string[];
      threads: number;
    }
  | { type: "frame"; bitmap: ImageBitmap; seq: number; options: DetectOptions }
  | { type: "setInputSize"; inputSize: number }
  | { type: "dispose" };

/** Coarse phases of worker start-up, surfaced in the UI so a slow first load
 *  reads as progress rather than as a hang. */
export type LoadPhase =
  | "idle"
  | "creating-session"
  | "retrying-single-thread"
  | "warming-up"
  | "ready";

/** Worker -> main thread. */
export type WorkerResponse =
  | { type: "status"; phase: LoadPhase }
  | {
      type: "ready";
      ep: EPName;
      /** Threads actually in use, which may be fewer than requested (see worker). */
      threads: number;
      /** Median of the warm-up runs: the honest "steady state from cold" number. */
      warmupMs: number;
      loadMs: number;
      inputName: string;
      outputName: string;
    }
  | {
      type: "result";
      seq: number;
      detections: Detection[];
      timing: FrameTiming;
      inputSize: number;
    }
  | { type: "error"; message: string; fatal: boolean };
