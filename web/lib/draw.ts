import type { Detection } from "./types";

/**
 * Stable per-class colour. Hashing the class id to a hue means a "person" is the
 * same colour on every model and every frame without maintaining an 80-entry palette,
 * and adjacent class ids get far-apart hues via the golden-angle step.
 */
export function classColor(classId: number): string {
  const hue = (classId * 137.508) % 360;
  return `hsl(${hue.toFixed(0)} 85% 60%)`;
}

export function drawDetections(
  ctx: CanvasRenderingContext2D,
  detections: Detection[],
  scale: number,
): void {
  const lineWidth = Math.max(1.5, 2 * scale);
  ctx.lineWidth = lineWidth;
  ctx.textBaseline = "bottom";
  const fontSize = Math.max(11, Math.round(13 * scale));
  ctx.font = `600 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;

  for (const det of detections) {
    const color = classColor(det.classId);
    const x = det.x * scale;
    const y = det.y * scale;
    const w = det.w * scale;
    const h = det.h * scale;

    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
    const padding = 4;
    const textWidth = ctx.measureText(label).width;
    const boxHeight = fontSize + padding * 2;
    // Flip the caption inside the box when the detection touches the top edge,
    // otherwise labels on tall objects get clipped off-canvas.
    const labelY = y - boxHeight < 0 ? y + boxHeight : y;

    ctx.fillStyle = color;
    ctx.fillRect(x - lineWidth / 2, labelY - boxHeight, textWidth + padding * 2, boxHeight);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillText(label, x + padding - lineWidth / 2, labelY - padding);
  }
}
