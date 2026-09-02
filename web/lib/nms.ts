import type { Detection, DetectOptions } from "./types";

function iou(a: Detection, b: Detection): number {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w);
  const y1 = Math.min(a.y + a.h, b.y + b.h);
  const overlap = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  if (overlap <= 0) return 0;
  return overlap / (a.w * a.h + b.w * b.h - overlap);
}

/**
 * Greedy class-wise non-maximum suppression.
 *
 * Class-wise, not global: a person standing in front of a car should not have one
 * box suppress the other just because they overlap. Suppressing across classes is
 * a common bug that quietly costs recall in crowded scenes.
 *
 * This deliberately lives in TypeScript rather than being baked into the ONNX graph
 * (`nms=True` at export). Keeping NMS out means the exported graph is pure tensor ops,
 * which stays entirely resident on the WebGPU execution provider; an in-graph
 * NonMaxSuppression node forces a GPU->CPU sync and a fallback partition every frame.
 */
export function nonMaxSuppression(
  candidates: Detection[],
  options: DetectOptions,
): Detection[] {
  const { iouThreshold, maxDetections } = options;
  const byClass = new Map<number, Detection[]>();
  for (const det of candidates) {
    const bucket = byClass.get(det.classId);
    if (bucket) bucket.push(det);
    else byClass.set(det.classId, [det]);
  }

  const kept: Detection[] = [];
  for (const bucket of byClass.values()) {
    bucket.sort((a, b) => b.score - a.score);
    const survivors: Detection[] = [];
    for (const candidate of bucket) {
      let suppressed = false;
      for (const winner of survivors) {
        if (iou(candidate, winner) > iouThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) survivors.push(candidate);
    }
    kept.push(...survivors);
  }

  kept.sort((a, b) => b.score - a.score);
  return kept.length > maxDetections ? kept.slice(0, maxDetections) : kept;
}
