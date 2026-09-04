"""Export a (possibly fine-tuned) YOLO checkpoint to ONNX, then prove it runs.

Two deliberate choices worth defending in an interview:

* ``nms=False`` -- NMS stays out of the graph. An in-graph NonMaxSuppression node
  forces the WebGPU execution provider to partition the model and sync back to CPU
  every frame; keeping the graph pure tensor ops lets it stay GPU-resident. The
  cost is that we own the decode/NMS implementation (web/lib/{decode,nms}.ts), and
  ml/parity_test.py exists to keep that honest.

* ``dynamic=True`` -- one file serves 320/480/640 inference, so the site can offer a
  real resolution/latency trade-off without shipping three copies of every model.
  Verified below by actually running all three sizes rather than assuming.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

import manifest as manifest_mod

VERIFY_SIZES = (320, 480, 640)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("id", "name", "domain", "imgsz"):
        if key not in cfg:
            raise ValueError(f"{path.name}: missing required key {key!r}")
    return cfg


def resolve_weights(cfg: dict, config_path: Path) -> str:
    """Prefer a fine-tuned checkpoint from ml/runs/ if training has produced one."""
    trained = config_path.parent.parent / "runs" / cfg["id"] / "weights" / "best.pt"
    if trained.exists():
        print(f"[export] using fine-tuned weights {trained}")
        return str(trained)
    print(f"[export] using base weights {cfg['weights']}")
    return cfg["weights"]


def verify(onnx_path: Path, dynamic: bool) -> dict[str, int]:
    """Run the exported graph on the CPU EP at each target size and report shapes."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"[verify] input  {input_name} {session.get_inputs()[0].shape}")
    print(f"[verify] output {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")

    shapes: dict[str, int] = {}
    sizes = VERIFY_SIZES if dynamic else (VERIFY_SIZES[-1],)
    for size in sizes:
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        out = session.run(None, {input_name: dummy})[0]
        print(f"[verify] {size}x{size} -> {out.shape}")
        shapes[str(size)] = int(out.shape[1])
    channels = set(shapes.values())
    if len(channels) != 1:
        raise RuntimeError(f"channel count varies across sizes: {shapes}")
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--static",
        action="store_true",
        help=(
            "export fixed imgsz instead of dynamic spatial dims. Produces a "
            "'<id>-static-fp32.onnx' benchmark variant that is deliberately kept out of "
            "the web manifest: the site wants one file that serves 320/480/640, while "
            "TensorRT's INT8 path requires fully specified shapes."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dynamic = not args.static
    from ultralytics import YOLO

    model = YOLO(resolve_weights(cfg, args.config))
    exported = model.export(
        format="onnx",
        imgsz=int(cfg["imgsz"]),
        opset=int(cfg.get("opset", 17)),
        dynamic=dynamic,
        simplify=True,
        nms=False,
        device="cpu",
    )

    manifest_mod.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['id']}-static-fp32" if args.static else f"{cfg['id']}-fp32"
    target = manifest_mod.MODEL_DIR / f"{stem}.onnx"
    shutil.move(str(exported), target)
    print(f"[export] {target.name}  {target.stat().st_size / 1e6:.2f} MB")

    verify(target, dynamic)

    names = model.names
    classes = [names[i] for i in sorted(names)]
    if args.static:
        print("[export] static variant: not registered in the web manifest")
        return
    manifest_mod.upsert_model(
        cfg,
        classes=classes,
        status="ready",
        dynamicInput=dynamic,
    )
    manifest_mod.upsert_variant(cfg["id"], "fp32", target)
    print(f"[export] done: {len(classes)} classes")


if __name__ == "__main__":
    main()
