/** Which ONNX Runtime execution provider is doing the matmuls. */
export type EPName = "webgpu" | "wasm";

/** A single detection, in *source image* pixel coordinates (not letterboxed space). */
export interface Detection {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  classId: number;
  label: string;
}

/**
 * The affine map from source pixels into the square, padded network input.
 * Kept explicit so post-processing can invert it exactly rather than
 * re-deriving it and drifting by a pixel or two.
 */
export interface LetterboxTransform {
  /** Uniform scale applied to the source image. */
  scale: number;
  /** Left padding in network-input pixels. */
  padX: number;
  /** Top padding in network-input pixels. */
  padY: number;
  srcWidth: number;
  srcHeight: number;
  /** Side length of the square network input. */
  inputSize: number;
}

/** Per-frame stage timings, in milliseconds. */
export interface FrameTiming {
  preprocess: number;
  inference: number;
  postprocess: number;
  total: number;
}

export interface DetectOptions {
  scoreThreshold: number;
  iouThreshold: number;
  maxDetections: number;
}

export const DEFAULT_DETECT_OPTIONS: DetectOptions = {
  scoreThreshold: 0.35,
  iouThreshold: 0.45,
  maxDetections: 100,
};
