"""Measure mAP for the exported FP32 and INT8 models and record the delta.

The delta is the number that matters. "INT8 is 3x smaller and 2x faster" is only
half a claim; without the accuracy it costs, it is not a trade-off, it is a boast.
Both precisions are evaluated on the same split with the same settings so the
comparison means something.

Writes results/accuracy.json and folds the mAP back into the web manifest, so the
site quotes measured values rather than remembered ones.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import manifest as manifest_mod

# Ultralytics auto-installs missing dependencies, and validating an ONNX model makes
# it pip-install the CPU-only `onnxruntime` package -- which overwrites the shared
# onnxruntime.dll belonging to onnxruntime-gpu. The GPU providers then vanish and
# every later "CUDA" benchmark silently measures CPU. Turn the auto-installer off.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "accuracy.json"


def evaluate(model_path: Path, data: str, imgsz: int, split: str) -> dict[str, float]:
    from ultralytics import YOLO

    # Ultralytics runs its own decode + NMS over the raw ONNX output, which is the
    # same head layout web/lib/decode.ts consumes -- so this measures the weights,
    # not our TypeScript. Cross-language agreement is ml/parity_test.py's job.
    metrics = YOLO(str(model_path), task="detect").val(
        data=data,
        imgsz=imgsz,
        split=split,
        batch=1,
        device="cpu",
        plots=False,
        verbose=False,
    )
    box = metrics.box
    return {
        "mAP50-95": round(float(box.map), 5),
        "mAP50": round(float(box.map50), 5),
        "precision": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="val",
        help="dataset split to evaluate; recorded in the output so the claim stays precise",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_id = cfg["id"]
    imgsz = int(cfg["imgsz"])
    evaluation = cfg.get("eval") or {}
    data = evaluation.get("data") or f"{(cfg.get('calibration') or {}).get('dataset', 'coco128')}.yaml"

    report: dict[str, Any] = {}
    if RESULTS_PATH.exists():
        report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    report.setdefault("models", {})

    entry: dict[str, Any] = {
        "dataset": data,
        "split": args.split,
        "imgsz": imgsz,
        # Spelled out because "mAP on COCO" and "mAP on a 128-image COCO subset" are
        # very different claims and only one of them is true here.
        "note": evaluation.get("note", ""),
        "precisions": {},
    }

    for precision in ("fp32", "int8"):
        model_path = manifest_mod.MODEL_DIR / f"{model_id}-{precision}.onnx"
        if not model_path.exists():
            print(f"[eval] skip {model_path.name} (not exported)")
            continue
        print(f"[eval] {model_path.name} on {data} [{args.split}] ...", flush=True)
        scores = evaluate(model_path, data, imgsz, args.split)
        scores["bytes"] = model_path.stat().st_size
        entry["precisions"][precision] = scores
        print(f"[eval] {precision}: mAP50-95={scores['mAP50-95']:.4f} mAP50={scores['mAP50']:.4f}")
        manifest_mod.upsert_variant(model_id, precision, model_path, mAP=scores["mAP50-95"])

    fp32 = entry["precisions"].get("fp32")
    int8 = entry["precisions"].get("int8")
    if fp32 and int8:
        delta = round(int8["mAP50-95"] - fp32["mAP50-95"], 5)
        ratio = round(fp32["bytes"] / int8["bytes"], 3)
        entry["int8Delta"] = {"mAP50-95": delta, "sizeRatio": ratio}
        print(f"[eval] INT8 costs {abs(delta):.4f} mAP50-95 and buys {ratio:.2f}x smaller weights")

    report["models"][model_id] = entry
    report["generatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {RESULTS_PATH}")

    web_copy = Path(__file__).resolve().parent.parent / "web" / "data" / "accuracy.json"
    web_copy.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {web_copy}")


if __name__ == "__main__":
    main()
