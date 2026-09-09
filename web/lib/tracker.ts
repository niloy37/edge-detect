import type { Detection } from "./types";

/**
 * Frame-to-frame box smoothing.
 *
 * Every frame is decoded independently, so a box around a stationary object still
 * shifts a few pixels each time — and at ~4 detections per second that jitter is the
 * most obvious flaw in the demo. It is not an accuracy problem and no better model
 * fixes it; it is the absence of any temporal model at all.
 *
 * Each new detection is matched to the previous frame's boxes by IoU within the same
 * class, and matched coordinates are exponentially smoothed. A track may also coast
 * for one frame when the detector misses it, which removes the single-frame blinking
 * that happens when confidence dips just under the threshold.
 *
 * Deliberately not a Kalman filter: there is no motion model here and no attempt to
 * predict where a box is heading, because a wrong prediction at 4fps looks far worse
 * than a slightly lagging box. This only damps noise on positions the detector
 * actually reported.
 */

export interface SmoothingOptions {
  /** Minimum IoU for a new detection to count as the same object as a track. */
  iouThreshold: number;
  /** EMA weight on the new measurement. 1 disables smoothing entirely. */
  alpha: number;
  /** How many consecutive misses a track survives before it is dropped. */
  maxMissed: number;
}

export const DEFAULT_SMOOTHING: SmoothingOptions = {
  iouThreshold: 0.3,
  // Responsive enough to follow a moving subject at a few frames a second, damped
  // enough that a still subject stops shimmering.
  alpha: 0.55,
  // One frame only. At ~4fps a longer coast leaves a stale box on screen for most of
  // a second, which reads as a bug rather than as smoothing.
  maxMissed: 1,
};

interface Track {
  box: Detection;
  missed: number;
}

function iou(a: Detection, b: Detection): number {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w);
  const y1 = Math.min(a.y + a.h, b.y + b.h);
  const overlap = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  if (overlap <= 0) return 0;
  return overlap / (a.w * a.h + b.w * b.h - overlap);
}

export class BoxSmoother {
  private tracks: Track[] = [];

  constructor(private options: SmoothingOptions = DEFAULT_SMOOTHING) {}

  setOptions(options: SmoothingOptions): void {
    this.options = options;
  }

  reset(): void {
    this.tracks = [];
  }

  update(detections: Detection[]): Detection[] {
    const { iouThreshold, alpha, maxMissed } = this.options;

    // Greedy matching, strongest pair first. Greedy rather than Hungarian because at
    // these object counts the optimal assignment is almost always the greedy one, and
    // this runs on the main thread between paints.
    const pairs: { track: number; detection: number; score: number }[] = [];
    for (let ti = 0; ti < this.tracks.length; ti++) {
      for (let di = 0; di < detections.length; di++) {
        if (detections[di].classId !== this.tracks[ti].box.classId) continue;
        const score = iou(detections[di], this.tracks[ti].box);
        if (score >= iouThreshold) pairs.push({ track: ti, detection: di, score });
      }
    }
    pairs.sort((a, b) => b.score - a.score);

    const trackToDetection = new Map<number, number>();
    const matchedDetections = new Set<number>();
    for (const pair of pairs) {
      if (trackToDetection.has(pair.track) || matchedDetections.has(pair.detection)) continue;
      trackToDetection.set(pair.track, pair.detection);
      matchedDetections.add(pair.detection);
    }

    // Build the next generation in one pass rather than mutating the list being
    // iterated — an earlier version appended new tracks mid-loop and then aged them
    // in the same pass, so a freshly seen object was born already stale.
    const next: Track[] = [];
    const output: Detection[] = [];

    this.tracks.forEach((track, index) => {
      const detectionIndex = trackToDetection.get(index);
      if (detectionIndex === undefined) {
        // Missed this frame: coast on the last known box, or expire.
        const missed = track.missed + 1;
        if (missed <= maxMissed) {
          next.push({ box: track.box, missed });
          output.push(track.box);
        }
        return;
      }
      const measured = detections[detectionIndex];
      const previous = track.box;
      const blended: Detection = {
        ...measured,
        x: previous.x + (measured.x - previous.x) * alpha,
        y: previous.y + (measured.y - previous.y) * alpha,
        w: previous.w + (measured.w - previous.w) * alpha,
        h: previous.h + (measured.h - previous.h) * alpha,
        // The score is smoothed too, so the percentage in the label stops flickering
        // between neighbouring values on a motionless object.
        score: previous.score + (measured.score - previous.score) * alpha,
      };
      next.push({ box: blended, missed: 0 });
      output.push(blended);
    });

    // Unmatched detections are new objects. Shown immediately rather than waiting for
    // confirmation: a hold-down would add visible latency at exactly the moment a
    // viewer waves something at the camera to see whether the demo works.
    detections.forEach((detection, index) => {
      if (matchedDetections.has(index)) return;
      next.push({ box: detection, missed: 0 });
      output.push(detection);
    });

    this.tracks = next;
    output.sort((a, b) => b.score - a.score);
    return output;
  }
}
