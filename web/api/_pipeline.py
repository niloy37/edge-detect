"""Numpy reference implementation of the detector's pre/post-processing.

This is the canonical Python side of the pipeline. It lives under web/api/ rather
than ml/ for a deployment reason: Vercel only uploads the `web` root directory, so
the serverless function must be self-contained. The ml/ scripts reach up into this
file instead of keeping a second copy -- two drifting implementations of letterbox
is precisely the bug class ml/parity_test.py is meant to catch.

It mirrors web/lib/{letterbox,decode,nms}.ts line for line. When you change one,
change the other and re-run the parity test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

# YOLO's canonical letterbox fill. The network saw this grey during training.
PAD_VALUE = 114

# YOLO11's largest feature stride; network input dimensions must be multiples of it.
STRIDE = 32


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    pad_x: float
    pad_y: float
    src_width: int
    src_height: int
    input_width: int
    input_height: int


def fit_to_stride(long_side: int, src_width: int, src_height: int) -> tuple[int, int]:
    """Network input shape matching the source aspect -- mirrors fitToStride in
    web/lib/letterbox.ts. Squaring a 16:9 frame spends 44% of the tensor on padding;
    measured, the aspect-matched shape is ~29% cheaper and detects slightly better."""

    def snap(value: float) -> int:
        return max(STRIDE, math.ceil(value / STRIDE) * STRIDE)

    if src_width >= src_height:
        return long_side, snap(long_side * src_height / src_width)
    return snap(long_side * src_width / src_height), long_side


def compute_letterbox(
    src_width: int, src_height: int, input_width: int, input_height: int
) -> LetterboxTransform:
    scale = min(input_width / src_width, input_height / src_height)
    draw_w = round(src_width * scale)
    draw_h = round(src_height * scale)
    return LetterboxTransform(
        scale=scale,
        pad_x=(input_width - draw_w) / 2,
        pad_y=(input_height - draw_h) / 2,
        src_width=src_width,
        src_height=src_height,
        input_width=input_width,
        input_height=input_height,
    )


def letterbox_image(
    image: Image.Image,
    long_side: int,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Returns an NCHW float32 tensor in [0,1] plus the transform needed to invert it.

    `long_side` is the longer network dimension; the shorter one follows the source
    aspect, so the tensor carries image instead of grey padding.

    `shape` overrides that with an explicit (width, height). Quantization calibration
    needs it: the calibrator stacks the activations from every sample into one array,
    so a batch of differently-shaped inputs -- which aspect-matching produces by
    definition -- fails with an inhomogeneous-shape error before it collects a single
    range."""
    image = image.convert("RGB")
    input_w, input_h = shape if shape is not None else fit_to_stride(long_side, image.width, image.height)
    t = compute_letterbox(image.width, image.height, input_w, input_h)
    draw_w = round(image.width * t.scale)
    draw_h = round(image.height * t.scale)

    # BILINEAR to stay close to what canvas drawImage does in the browser; exact
    # pixel parity across the two resamplers is not achievable, which is why the
    # parity test asserts a tolerance on boxes rather than equality on tensors.
    resized = image.resize((draw_w, draw_h), Image.BILINEAR)
    canvas = Image.new("RGB", (input_w, input_h), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    canvas.paste(resized, (int(round(t.pad_x)), int(round(t.pad_y))))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])  # NCHW
    return tensor, t


def unletterbox(x: np.ndarray, y: np.ndarray, t: LetterboxTransform) -> tuple[np.ndarray, np.ndarray]:
    return (x - t.pad_x) / t.scale, (y - t.pad_y) / t.scale


def decode_yolo(
    raw: np.ndarray,
    t: LetterboxTransform,
    labels: list[str],
    score_threshold: float,
) -> list[dict]:
    """Decode [1, 4+nc, anchors] into xywh boxes in source-image pixels.

    Channel-major layout: anchor `i` of channel `c` is at raw[0, c, i]. No objectness
    channel and no sigmoid -- YOLOv8/11 class scores come out already activated.
    """
    pred = raw[0]  # (4 + nc, anchors)
    boxes = pred[:4]
    scores = pred[4:]

    best_class = scores.argmax(axis=0)
    best_score = scores.max(axis=0)
    keep = best_score >= score_threshold
    if not np.any(keep):
        return []

    cx, cy, bw, bh = boxes[0][keep], boxes[1][keep], boxes[2][keep], boxes[3][keep]
    x0, y0 = unletterbox(cx - bw / 2, cy - bh / 2, t)
    x1, y1 = unletterbox(cx + bw / 2, cy + bh / 2, t)

    x0 = np.clip(x0, 0, t.src_width)
    y0 = np.clip(y0, 0, t.src_height)
    x1 = np.clip(x1, 0, t.src_width)
    y1 = np.clip(y1, 0, t.src_height)

    out: list[dict] = []
    for xa, ya, xb, yb, score, cls in zip(
        x0, y0, x1, y1, best_score[keep], best_class[keep], strict=True
    ):
        if xb <= xa or yb <= ya:
            continue
        out.append(
            {
                "x": float(xa),
                "y": float(ya),
                "w": float(xb - xa),
                "h": float(yb - ya),
                "score": float(score),
                "classId": int(cls),
                "label": labels[int(cls)] if int(cls) < len(labels) else f"class_{int(cls)}",
            }
        )
    return out


def _iou(a: dict, b: dict) -> float:
    x0 = max(a["x"], b["x"])
    y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if overlap <= 0:
        return 0.0
    return overlap / (a["w"] * a["h"] + b["w"] * b["h"] - overlap)


def non_max_suppression(
    candidates: list[dict], iou_threshold: float, max_detections: int
) -> list[dict]:
    """Greedy class-wise NMS -- mirrors web/lib/nms.ts, including the class-wise part."""
    by_class: dict[int, list[dict]] = {}
    for det in candidates:
        by_class.setdefault(det["classId"], []).append(det)

    kept: list[dict] = []
    for bucket in by_class.values():
        bucket.sort(key=lambda d: d["score"], reverse=True)
        survivors: list[dict] = []
        for candidate in bucket:
            if all(_iou(candidate, w) <= iou_threshold for w in survivors):
                survivors.append(candidate)
        kept.extend(survivors)

    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:max_detections]
