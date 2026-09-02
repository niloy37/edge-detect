"""Read/modify web/data/models.json.

The web app treats this manifest as the source of truth for what models exist,
how big they are and what they score. Every number in it is written by a script
that measured it, never by hand -- a portfolio that quotes stale mAP next to
freshly re-exported weights is worse than one that quotes none.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "web" / "data" / "models.json"
MODEL_DIR = REPO_ROOT / "web" / "public" / "models"


def load() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"generatedAt": None, "models": []}


def save(manifest: dict[str, Any]) -> None:
    manifest["generatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["models"].sort(key=lambda m: m.get("order", 99))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")


def upsert_model(cfg: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Insert or update the entry for cfg['id'], preserving unrelated fields."""
    manifest = load()
    entry = next((m for m in manifest["models"] if m["id"] == cfg["id"]), None)
    if entry is None:
        entry = {"id": cfg["id"], "variants": []}
        manifest["models"].append(entry)

    entry.setdefault("variants", [])
    entry.update(
        {
            "name": cfg["name"],
            "domain": cfg["domain"],
            "description": " ".join(str(cfg.get("description", "")).split()),
            "inputSize": int(cfg["imgsz"]),
            "order": int(cfg.get("order", 99)),
        }
    )
    if cfg.get("dataset"):
        entry["dataset"] = cfg["dataset"]
    if cfg.get("sample") and cfg["sample"].get("video"):
        entry["sample"] = cfg["sample"]
    entry.update(fields)
    save(manifest)
    return entry


def upsert_variant(model_id: str, precision: str, path: Path, mAP: float | None = None) -> None:
    """Record one exported .onnx file, measuring its size rather than trusting a guess."""
    manifest = load()
    entry = next((m for m in manifest["models"] if m["id"] == model_id), None)
    if entry is None:
        raise KeyError(f"model {model_id!r} not in manifest; run export first")

    variants = [v for v in entry.get("variants", []) if v["precision"] != precision]
    previous = next((v for v in entry.get("variants", []) if v["precision"] == precision), {})
    variants.append(
        {
            "precision": precision,
            "file": f"/models/{path.name}",
            "bytes": path.stat().st_size,
            "mAP": mAP if mAP is not None else previous.get("mAP"),
        }
    )
    variants.sort(key=lambda v: v["precision"])
    entry["variants"] = variants
    save(manifest)
