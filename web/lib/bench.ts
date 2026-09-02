import benchmark from "@/data/bench.json";

export interface BenchRun {
  provider: string;
  inputSize: number;
  precision: string;
  modelId: string;
  modelBytes: number;
  status: "ok" | "unavailable" | "failed";
  error?: string;
  activeProviders?: string[];
  sessionInitMs?: number;
  p50Ms?: number;
  p95Ms?: number;
  meanMs?: number;
  minMs?: number;
  fps?: number;
  iterations?: number;
}

export interface BenchReport {
  generatedAt: string;
  methodology: {
    batchSize: number;
    warmupIterations: number;
    timedIterations: number;
    note: string;
  };
  environment: Record<string, string | string[] | null>;
  runs: BenchRun[];
}

export const BENCH = benchmark as unknown as BenchReport;

/** "TensorrtExecutionProvider" -> "TensorRT". */
export function providerLabel(provider: string): string {
  const stem = provider.replace("ExecutionProvider", "");
  if (stem === "Tensorrt") return "TensorRT";
  if (stem === "CUDA") return "CUDA";
  if (stem === "CPU") return "CPU";
  return stem;
}

export const okRuns = (runs: BenchRun[]) => runs.filter((r) => r.status === "ok");
