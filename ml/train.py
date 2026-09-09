"""Fine-tune a detector from a dataset config.

One script for every domain: the config names the data and the hyperparameters, and
the rest of the pipeline (export -> quantize -> eval -> benchmark) picks the result
up from ml/runs/<id>/weights/best.pt without knowing what was trained.

That is the actual claim this repository makes -- not that one model works, but that
retargeting it is a config change.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

# Ultralytics auto-installs missing dependencies, and in this venv that has already
# clobbered onnxruntime-gpu with the CPU build once, silently turning every later
# "CUDA" benchmark into a CPU measurement. Never let it write to the environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

ML_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="0", help="CUDA index, or 'cpu'")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = cfg.get("train") or {}
    dataset_yaml = ML_ROOT / "datasets" / cfg["id"] / "dataset.yaml"
    if not dataset_yaml.exists():
        raise SystemExit(f"{dataset_yaml} missing -- run prepare_data.py first")

    from ultralytics import YOLO

    model = YOLO(cfg["weights"])
    model.train(
        data=str(dataset_yaml),
        epochs=int(train_cfg.get("epochs", 40)),
        imgsz=int(cfg["imgsz"]),
        batch=int(train_cfg.get("batch", 32)),
        patience=int(train_cfg.get("patience", 10)),
        device=args.device,
        project=str(ML_ROOT / "runs"),
        name=cfg["id"],
        exist_ok=True,
        # Pinned so a re-run reproduces the weights the published numbers describe.
        seed=0,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    best = ML_ROOT / "runs" / cfg["id"] / "weights" / "best.pt"
    print(f"[train] best weights: {best}  exists={best.exists()}")


if __name__ == "__main__":
    main()
