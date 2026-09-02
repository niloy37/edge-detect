import type { Detection, DetectOptions, LetterboxTransform } from "./types";
import { unletterbox } from "./letterbox";

/**
 * Decodes a raw YOLOv8/YOLO11 detection head into candidate boxes.
 *
 * The head emits one tensor shaped [1, 4 + numClasses, numAnchors] — note the
 * *channel-major* layout: all 8400 cx values, then all 8400 cy values, and so on.
 * Anchor `i` of channel `c` therefore lives at `c * numAnchors + i`, which is why
 * this reads as a strided gather rather than a row scan.
 *
 * There is no objectness channel (that is a v5-ism) and no sigmoid to apply:
 * the class scores come out already activated. Boxes are cxcywh in network-input
 * pixels, so they are un-letterboxed back to source pixels here.
 */
export function decodeYolo(
  raw: Float32Array,
  channels: number,
  numAnchors: number,
  transform: LetterboxTransform,
  labels: string[],
  options: DetectOptions,
): Detection[] {
  const numClasses = channels - 4;
  const out: Detection[] = [];
  const { scoreThreshold } = options;

  for (let i = 0; i < numAnchors; i++) {
    // Find the winning class first and bail early — computing box geometry for
    // 8400 anchors when ~20 survive is the single biggest post-processing waste.
    let bestScore = 0;
    let bestClass = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = raw[(4 + c) * numAnchors + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestClass < 0 || bestScore < scoreThreshold) continue;

    const cx = raw[i];
    const cy = raw[numAnchors + i];
    const bw = raw[2 * numAnchors + i];
    const bh = raw[3 * numAnchors + i];

    const [x0, y0] = unletterbox(cx - bw / 2, cy - bh / 2, transform);
    const [x1, y1] = unletterbox(cx + bw / 2, cy + bh / 2, transform);

    // Clamp to the source frame: letterbox padding lets boxes spill outside it.
    const left = Math.max(0, x0);
    const top = Math.max(0, y0);
    const right = Math.min(transform.srcWidth, x1);
    const bottom = Math.min(transform.srcHeight, y1);
    if (right <= left || bottom <= top) continue;

    out.push({
      x: left,
      y: top,
      w: right - left,
      h: bottom - top,
      score: bestScore,
      classId: bestClass,
      label: labels[bestClass] ?? `class_${bestClass}`,
    });
  }
  return out;
}
