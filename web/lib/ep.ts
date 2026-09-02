import type { EPName } from "./types";

/**
 * Probes for a usable WebGPU adapter.
 *
 * `navigator.gpu` existing is not sufficient — Linux and some mobile Chrome builds
 * expose the object but fail at adapter request, so the only honest test is to ask
 * for one. Result is memoised because requestAdapter is not free.
 */
let webgpuProbe: Promise<GPUAdapterInfo | null> | undefined;

export function probeWebGPU(): Promise<GPUAdapterInfo | null> {
  if (!webgpuProbe) {
    webgpuProbe = (async () => {
      const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
      if (!gpu) return null;
      try {
        const adapter = await gpu.requestAdapter();
        return adapter ? adapter.info ?? ({} as GPUAdapterInfo) : null;
      } catch {
        return null;
      }
    })();
  }
  return webgpuProbe;
}

export async function pickExecutionProvider(preferred?: EPName): Promise<EPName> {
  if (preferred === "wasm") return "wasm";
  const adapter = await probeWebGPU();
  return adapter ? "webgpu" : "wasm";
}

/**
 * True when the page is cross-origin isolated, which is what unlocks SharedArrayBuffer
 * and therefore multi-threaded WASM. Without it ORT silently runs single-threaded and
 * the WASM numbers look ~3x worse than the hardware can actually do — so this is
 * surfaced in the UI rather than left as a hidden footgun.
 */
export function isCrossOriginIsolated(): boolean {
  return typeof crossOriginIsolated !== "undefined" && crossOriginIsolated === true;
}

export function suggestedThreadCount(): number {
  const cores = navigator.hardwareConcurrency || 4;
  // ORT scales poorly past ~4 threads on this workload and oversubscribing starves
  // the render loop, so cap rather than taking every core.
  return Math.max(1, Math.min(4, cores - 1));
}

export function describeDevice(): Record<string, string | number> {
  return {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
    deviceMemoryGB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 0,
    crossOriginIsolated: String(isCrossOriginIsolated()),
  };
}
