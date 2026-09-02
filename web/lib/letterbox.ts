import type { LetterboxTransform } from "./types";

/** YOLO's canonical letterbox fill. Matching it matters: the network saw this grey in training. */
export const PAD_VALUE = 114;

/**
 * Aspect-preserving fit of `srcWidth x srcHeight` into a square `inputSize` box,
 * centred, with the remainder padded. Pure math, no canvas — so the Python
 * reference implementation and this one can be compared directly (ml/parity_test.py).
 */
export function computeLetterbox(
  srcWidth: number,
  srcHeight: number,
  inputSize: number,
): LetterboxTransform {
  const scale = Math.min(inputSize / srcWidth, inputSize / srcHeight);
  const drawWidth = Math.round(srcWidth * scale);
  const drawHeight = Math.round(srcHeight * scale);
  return {
    scale,
    padX: (inputSize - drawWidth) / 2,
    padY: (inputSize - drawHeight) / 2,
    srcWidth,
    srcHeight,
    inputSize,
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
 * The OffscreenCanvas is reused across frames; the output tensor deliberately is not.
 * Handing ONNX Runtime a buffer that is overwritten on the next frame is asking for
 * trouble, and the allocation is only paid once per frame the model actually accepts
 * -- the render loop checks `canSubmit()` before grabbing a frame, so this runs at
 * inference rate, not display rate.
 */
export class Preprocessor {
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  readonly inputSize: number;

  constructor(inputSize: number) {
    this.inputSize = inputSize;
    this.canvas = new OffscreenCanvas(inputSize, inputSize);
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D context unavailable in this worker");
    this.ctx = ctx;
  }
  run(source: ImageBitmap): { data: Float32Array; transform: LetterboxTransform } {
    const size = this.inputSize;
    const transform = computeLetterbox(source.width, source.height, size);
    const drawWidth = Math.round(source.width * transform.scale);
    const drawHeight = Math.round(source.height * transform.scale);

    this.ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
    this.ctx.fillRect(0, 0, size, size);
    this.ctx.drawImage(source, transform.padX, transform.padY, drawWidth, drawHeight);

    const { data: rgba } = this.ctx.getImageData(0, 0, size, size);
    const plane = size * size;
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
