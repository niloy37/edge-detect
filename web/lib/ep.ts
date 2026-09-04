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

/**
 * What "Auto" resolves to: WASM.
 *
 * Not the flashier answer, and deliberate. WebGPU is implemented, benchmarked and
 * selectable, but on the development machine (RTX 5070, Chrome, driver 610.88) the
 * WebGPU provider creates a session, warms up on zero tensors in ~1s, reports ready
 * -- and then never returns from the first inference on a real frame. No error, no
 * rejection. A visitor would see a page that loads, says "running", and draws
 * nothing.
 *
 * Multi-threaded WASM is ~95ms per frame at 640px, works on every browser tested,
 * and is verified end to end by /selftest. Defaulting to the backend that reliably
 * produces detections is worth more than defaulting to the one with the better name.
 */
export async function pickExecutionProvider(preferred?: EPName): Promise<EPName> {
  return preferred ?? "wasm";
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
  if (typeof navigator === "undefined") return 1;
  const cores = navigator.hardwareConcurrency || 4;
  // Leave one core for the render loop and the page, and cap at 4: ORT scales poorly
  // past that on this workload and oversubscribing starves the loop feeding it.
  // Note this is whatever the browser chooses to report, which is not necessarily the
  // machine's core count -- Chrome reports 2 in some contexts on a 24-thread CPU. One
  // thread is the correct answer to "2 cores", so a WASM x1 badge is usually the
  // platform being conservative rather than a fallback having fired.
  return Math.max(1, Math.min(4, cores - 1));
}

export function describeDevice(): Record<string, string | number> {
  if (typeof navigator === "undefined") return {};
  return {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
    deviceMemoryGB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 0,
    crossOriginIsolated: String(isCrossOriginIsolated()),
  };
}
