import type { StageOverlay } from "./draw";
import type { Detection } from "./types";

/**
 * Obscures detected faces in place.
 *
 * The redaction is done by re-drawing the *original* frame through a canvas blur
 * filter, clipped to each face — not by blurring the canvas we already painted.
 * Blurring a canvas into itself feeds each pass its own output, which smears
 * progressively and leaks the underlying pixels at the edges. Reading from the
 * pristine source every frame keeps the result stable and the boundary clean.
 *
 * Two deliberate choices about how much to hide:
 *
 * - The region is expanded beyond the detector's box. Face detectors bound the face
 *   tightly, and a tight blur leaves hair, jaw and ears legible — enough to recognise
 *   someone. Padding costs nothing and closes that gap.
 * - The blur radius scales with the face, so a face near the camera is obscured as
 *   thoroughly as a distant one. A fixed radius hides small faces and merely softens
 *   large ones.
 */

/** Fraction of the box size added on every side before blurring. */
const PADDING = 0.22;
/** Blur radius as a fraction of the padded face width. */
const RADIUS_RATIO = 0.32;
const MIN_RADIUS = 6;

export interface BlurOptions {
  /** Draw a thin outline so it is visible that a face was found, not just smeared. */
  showOutline: boolean;
}

export function createFaceBlurOverlay(options: BlurOptions): StageOverlay {
  return (ctx, detections, frame) => {
    for (const face of detections) {
      const padX = face.w * PADDING;
      const padY = face.h * PADDING;
      const x = Math.max(0, face.x - padX);
      const y = Math.max(0, face.y - padY);
      const w = Math.min(frame.width - x, face.w + padX * 2);
      const h = Math.min(frame.height - y, face.h + padY * 2);
      if (w <= 0 || h <= 0) continue;

      const radius = Math.max(MIN_RADIUS, w * RADIUS_RATIO);

      ctx.save();
      ctx.beginPath();
      // An ellipse rather than a rectangle: a rectangular smear looks like a bug,
      // an elliptical one reads as an intentional redaction.
      ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
      ctx.clip();
      ctx.filter = `blur(${radius.toFixed(1)}px)`;
      // Redraw the whole frame from the untouched source; the clip confines it.
      ctx.drawImage(frame.source, 0, 0, frame.width, frame.height);
      ctx.restore();

      if (options.showOutline) {
        ctx.save();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
        ctx.lineWidth = Math.max(1.5, w * 0.012);
        ctx.beginPath();
        ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }
  };
}

/** Plain boxes, but tuned for a single-class model: no class colours, no class name. */
export function createFaceBoxOverlay(): StageOverlay {
  return (ctx, detections: Detection[]) => {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgb(56, 189, 248)";
    ctx.font = "600 13px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "bottom";
    for (const face of detections) {
      ctx.strokeRect(face.x, face.y, face.w, face.h);
      const label = `${(face.score * 100).toFixed(0)}%`;
      const width = ctx.measureText(label).width;
      ctx.fillStyle = "rgb(56, 189, 248)";
      ctx.fillRect(face.x - 1, face.y - 19, width + 8, 19);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillText(label, face.x + 3, face.y - 4);
    }
  };
}
