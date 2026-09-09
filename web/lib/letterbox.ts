import type { LetterboxTransform } from "./types";

/** YOLO's canonical letterbox fill. Matching it matters: the network saw this grey in training. */
export const PAD_VALUE = 114;

/** YOLO11's largest feature stride. Input dimensions must be multiples of it. */
export const STRIDE = 32;

/**
 * Picks a network input shape that matches the source aspect ratio.
 *
 * Fitting a 16:9 frame into a square wastes 44% of the tensor on grey padding, and
 * the model pays for every one of those pixels. Measured on the shipped INT8 model:
 * 640x640 takes 24.5ms, 384x640 takes 17.5ms for the same frame -- 29% cheaper --
 * and detects *better*, because the subject survives at a larger scale instead of
 * being shrunk to fit a square. On the sample image, square found 4 objects and
 * 640x480 found 5.
 *
 * `longSide` is the user-facing "input size". The short side is derived from the
 * source aspect and rounded up to a stride multiple; rounding up rather than down
 * keeps more of the image's resolution and costs a little padding instead.
 */
export function fitToStride(
  longSide: number,
  srcWidth: number,
  srcHeight: number,
): { width: number; height: number } {
  const round = (value: number) => Math.max(STRIDE, Math.ceil(value / STRIDE) * STRIDE);
  if (srcWidth >= srcHeight) {
    return { width: longSide, height: round((longSide * srcHeight) / srcWidth) };
  }
  return { width: round((longSide * srcWidth) / srcHeight), height: longSide };
}

/**
 * Aspect-preserving fit of `srcWidth x srcHeight` into an `inputWidth x inputHeight`
 * box, centred, with the remainder padded. Pure math, no canvas — so the numpy
 * reference implementation and this one can be compared directly (ml/parity_test.py).
 */
export function computeLetterbox(
  srcWidth: number,
  srcHeight: number,
  inputWidth: number,
  inputHeight: number,
): LetterboxTransform {
  const scale = Math.min(inputWidth / srcWidth, inputHeight / srcHeight);
  const drawWidth = Math.round(srcWidth * scale);
  const drawHeight = Math.round(srcHeight * scale);
  return {
    scale,
    padX: (inputWidth - drawWidth) / 2,
    padY: (inputHeight - drawHeight) / 2,
    srcWidth,
    srcHeight,
    inputWidth,
    inputHeight,
  };
}

/** Invert the letterbox: network-input coordinates back to source-image pixels. */
export function unletterbox(
  x: number,
  y: number,
  t: LetterboxTransform,
): [number, number] {
  return [(x - t.padX) / t.scale, (y - t.padY) / t.scale];
}

/**
 * Draws a frame into a reusable canvas and emits an NCHW float32 tensor in [0,1].
 *
 * The canvas is resized to match the source aspect rather than being fixed square,
 * so the tensor carries image instead of padding. The exported ONNX has dynamic H/W,
 * so this needs no separate model file — the same weights accept any stride-aligned
 * shape.
 *
 * The OffscreenCanvas is reused across frames; the output tensor deliberately is not.
 * Handing ONNX Runtime a buffer that is overwritten on the next frame is asking for
 * trouble, and the allocation is only paid once per frame the model actually accepts
 * — the render loop claims the inference slot before capturing, so this runs at
 * inference rate, not display rate.
 */
export class Preprocessor {
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  /** The long side requested; the actual shape depends on each frame's aspect. */
  readonly longSide: number;
  private width = 0;
  private height = 0;

  constructor(longSide: number) {
    this.longSide = longSide;
    this.canvas = new OffscreenCanvas(longSide, longSide);
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D context unavailable in this worker");
    this.ctx = ctx;
  }

  run(source: ImageBitmap): { data: Float32Array; transform: LetterboxTransform } {
    const { width, height } = fitToStride(this.longSide, source.width, source.height);
    if (width !== this.width || height !== this.height) {
      // Resizing an OffscreenCanvas also clears it, so only do it when the shape
      // actually changes — a fixed camera keeps one shape for its whole session.
      this.canvas.width = width;
      this.canvas.height = height;
      this.width = width;
      this.height = height;
    }

    const transform = computeLetterbox(source.width, source.height, width, height);
    const drawWidth = Math.round(source.width * transform.scale);
    const drawHeight = Math.round(source.height * transform.scale);

    this.ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
    this.ctx.fillRect(0, 0, width, height);
    this.ctx.drawImage(source, transform.padX, transform.padY, drawWidth, drawHeight);

    const { data: rgba } = this.ctx.getImageData(0, 0, width, height);
    const plane = width * height;
    const out = new Float32Array(3 * plane);
    // HWC uint8 RGBA -> CHW float32 RGB, scaled to [0,1].
    for (let i = 0, px = 0; px < plane; px++, i += 4) {
      out[px] = rgba[i] / 255;
      out[plane + px] = rgba[i + 1] / 255;
      out[2 * plane + px] = rgba[i + 2] / 255;
    }
    return { data: out, transform };
  }
}
