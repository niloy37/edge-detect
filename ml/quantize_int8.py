"""Static INT8 post-training quantization of an exported ONNX detector.

Static, not dynamic: dynamic quantization leaves activations in float and buys
almost nothing for a convolutional backbone. Static PTQ needs a calibration pass
over *real* images from the model's own domain -- calibrating a hardhat detector on
COCO photos produces activation ranges that fit the wrong distribution, and the
resulting model benchmarks fast while quietly losing recall. Every config therefore
names its own calibration source.

QDQ format is used because that is what ONNX Runtime Web's WASM backend consumes
efficiently, and it keeps the graph readable in Netron.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web" / "api"))

import manifest as manifest_mod  # noqa: E402
from _pipeline import letterbox_image  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather_calibration_images(cfg: dict, limit: int) -> list[Path]:
    calib = cfg.get("calibration") or {}
    explicit = calib.get("images_dir")
    if explicit:
        root = Path(explicit)
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
    elif calib.get("dataset"):
        from ultralytics.data.utils import check_det_dataset

        info = check_det_dataset(f"{calib['dataset']}.yaml")
        root = Path(info.get("val") or info["train"])
    else:
        raise ValueError("config.calibration needs either 'dataset' or 'images_dir'")

    if root.is_file():
        root = root.parent
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no calibration images under {root}")

    # Deterministic sample: re-running quantization must reproduce the same weights,
    # otherwise the mAP delta reported on the site is not reproducible.
    random.Random(0).shuffle(images)
    selected = images[:limit]
    print(f"[calib] {len(selected)} images from {root}")
    return selected


class LetterboxCalibrationReader:
    """Feeds the quantizer the exact tensors the deployed pipeline will produce."""

    def __init__(self, images: list[Path], input_name: str, input_size: int):
        self.input_name = input_name
        self.input_size = input_size
        self._iter = iter(images)

    def get_next(self):
        from PIL import Image

        path = next(self._iter, None)
        if path is None:
            return None
        with Image.open(path) as im:
            tensor, _ = letterbox_image(im, self.input_size)
        return {self.input_name: tensor.astype(np.float32)}

    def rewind(self):
        raise NotImplementedError


# The detection head's tail is deliberately kept in float. See select_head_tail_nodes.
HEAD_PREFIX = "/model.23/"


def select_head_tail_nodes(onnx_path: Path) -> list[str]:
    """Nodes to keep in float32, and why.

    YOLO11's graph ends with

        Concat_26(Mul_5 -> box xywh in pixels 0..640,
                  Sigmoid -> class scores 0..1) -> output0

    QDQ assigns one scale to a Concat's output. Over the union range [0, 640] that
    scale is ~2.5, so every class score below ~1.25 -- which is all of them --
    quantizes to exactly zero. Measured: fp32 max class score 0.940, INT8 max 0.000,
    while the box channels came through fine. Zero detections, no error raised.

    The fix is to leave the head's *elementwise tail* in float: the Sigmoid, the DFL
    softmax/conv, the box arithmetic and the output concat. Those run on small tensors
    and are not where the time goes. The 88 backbone/neck convolutions -- the actual
    compute -- stay INT8, which is why the speedup survives the fix.
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    excluded = [
        node.name
        for node in graph.node
        if node.name.startswith(HEAD_PREFIX)
        # Keep the head's real convolutions (cv2/cv3) quantized; drop only the
        # 16->1 DFL projection, which is elementwise-cheap and range-sensitive.
        and not (node.op_type == "Conv" and "/dfl/" not in node.name)
    ]
    print(f"[quant] keeping {len(excluded)} head-tail nodes in float32")
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_id = cfg["id"]
    input_size = int(cfg["imgsz"])

    src = manifest_mod.MODEL_DIR / f"{model_id}-fp32.onnx"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- run export_onnx.py first")
    prepped = src.with_name(f"{model_id}-fp32.prep.onnx")
    dst = manifest_mod.MODEL_DIR / f"{model_id}-int8.onnx"

    # Symbolic shape inference + graph cleanup. Skipping the cleanup entirely is the
    # most common cause of "quantize_static succeeded but the model outputs garbage",
    # so it is not optional -- but symbolic shape inference itself cannot resolve the
    # dynamic H/W dims we exported for the resolution toggle and raises "Incomplete
    # symbolic shape inference". Fall back to the optimizer-only path in that case.
    print("[quant] pre-processing graph...")
    try:
        quant_pre_process(str(src), str(prepped), skip_symbolic_shape=False)
    except Exception as exc:
        print(f"[quant] symbolic shape inference unavailable ({exc}); retrying without it")
        quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)

    input_name = ort.InferenceSession(
        str(prepped), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    images = gather_calibration_images(cfg, int((cfg.get("calibration") or {}).get("num_images", 96)))
    reader = LetterboxCalibrationReader(images, input_name, input_size)

    print("[quant] running static PTQ...")
    quantize_static(
        model_input=str(prepped),
        model_output=str(dst),
        calibration_data_reader=reader,
        nodes_to_exclude=select_head_tail_nodes(prepped),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # Per-channel weight scales: detection heads have wildly different dynamic
        # ranges per output channel, and a single per-tensor scale flattens the
        # small-object logits into noise.
        per_channel=True,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    prepped.unlink(missing_ok=True)

    size_mb = dst.stat().st_size / 1e6
    ratio = src.stat().st_size / dst.stat().st_size
    print(f"[quant] {dst.name}  {size_mb:.2f} MB  ({ratio:.2f}x smaller than fp32)")

    session = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    for size in (320, 480, 640):
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        out = session.run(None, {input_name: dummy})[0]
        print(f"[verify] {size}x{size} -> {out.shape}")

    manifest_mod.upsert_variant(model_id, "int8", dst)


if __name__ == "__main__":
    main()
