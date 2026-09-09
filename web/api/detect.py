"""Serverless single-image detection, running the same INT8 weights on CPU.

This endpoint exists to make an argument measurable, not because it is how you serve
video. It reports whether the request paid a cold start and how long each stage took,
so the /benchmarks and /how-it-works pages can contrast it against on-device
inference with real numbers instead of an assertion.

Deployed as a Vercel Python function. Kept to onnxruntime + numpy + Pillow: adding
opencv-python alone would push the bundle past the 250MB unzipped ceiling.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Vercel imports this entrypoint with the working directory at /var/task, so the
# directory holding this file is not on sys.path and a plain sibling import of
# _pipeline raises ModuleNotFoundError at cold start -- even though the file sits
# right next to it. It works locally only because you tend to run from inside api/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from _pipeline import decode_yolo, letterbox_image, non_max_suppression  # noqa: E402

MODEL_ID = "coco80-n"
PRECISION = "int8"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_INPUT_SIZE = 640

# Module-level state survives between invocations on a warm container. The first
# request in a container pays model load; later ones do not, and the difference is
# the whole point of measuring this.
_session: ort.InferenceSession | None = None
_labels: list[str] = []
_input_name = ""
_load_ms = 0.0
_served = 0
_model_source = "unknown"


def _find_asset(*relative: str) -> Path | None:
    """Locate a bundled file, tolerating the several roots Vercel may run us from."""
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, Path.cwd()):
        candidate = base.joinpath(*relative)
        if candidate.exists():
            return candidate
    return None


# Every ONNX file is a protobuf whose first field is ir_version, so it starts 0x08.
# Cheap sanity check with a specific purpose -- see _fetch_to_tmp.
_ONNX_MAGIC = b""


def _fetch_to_tmp(url: str, name: str) -> Path:
    """Last-resort fallback: pull the model from the deployment's own static assets.

    Validated rather than trusted. Deployment URLs sit behind Vercel's access
    protection, so an unauthenticated fetch returns an HTML login page with HTTP 200
    -- which this used to write to /tmp as a .onnx and hand to the runtime, producing
    "Protobuf parsing failed" a layer away from the actual cause. Fail here, with the
    reason, instead of there.
    """
    target = Path("/tmp") / name
    if target.exists() and target.stat().st_size > 0:
        return target
    with urllib.request.urlopen(url, timeout=20) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        payload = response.read()
    if "html" in content_type or not payload.startswith(_ONNX_MAGIC):
        raise RuntimeError(
            f"{url} did not return an ONNX model (content-type={content_type!r}, "
            f"{len(payload)} bytes) -- the model was not bundled and the static asset "
            f"is not publicly readable"
        )
    target.write_bytes(payload)
    return target


def _load() -> None:
    global _session, _labels, _input_name, _load_ms, _model_source
    if _session is not None:
        return

    start = time.perf_counter()

    manifest_path = _find_asset("data", "models.json")
    if manifest_path is None:
        raise RuntimeError("models.json was not bundled with the function")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(m for m in manifest["models"] if m["id"] == MODEL_ID)
    _labels = entry["classes"]

    filename = f"{MODEL_ID}-{PRECISION}.onnx"
    model_path = _find_asset("public", "models", filename)
    _model_source = "bundled"
    if model_path is None:
        _model_source = "fetched"
        host = os.environ.get("VERCEL_URL")
        if not host:
            raise RuntimeError(f"{filename} not bundled and VERCEL_URL unset")
        model_path = _fetch_to_tmp(f"https://{host}/models/{filename}", filename)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # One thread. A serverless container is billed on wall-clock and typically gets
    # fractional CPU; spawning a thread pool mostly buys contention.
    options.intra_op_num_threads = 1

    _session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    _input_name = _session.get_inputs()[0].name
    _load_ms = (time.perf_counter() - start) * 1000


def run_detection(image_bytes: bytes, score_threshold: float, iou_threshold: float, size: int) -> dict:
    global _served
    import io

    from PIL import Image

    cold = _session is None
    _load()
    assert _session is not None

    t0 = time.perf_counter()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        tensor, transform = letterbox_image(image, size)
    t1 = time.perf_counter()

    raw = _session.run(None, {_input_name: tensor.astype(np.float32)})[0]
    t2 = time.perf_counter()

    detections = non_max_suppression(
        decode_yolo(raw, transform, _labels, score_threshold), iou_threshold, 100
    )
    t3 = time.perf_counter()
    _served += 1

    return {
        "detections": detections,
        "model": {"id": MODEL_ID, "precision": PRECISION, "inputSize": size},
        "timing": {
            "coldStart": cold,
            "modelLoadMs": round(_load_ms, 2) if cold else 0.0,
            "preprocessMs": round((t1 - t0) * 1000, 2),
            "inferenceMs": round((t2 - t1) * 1000, 2),
            "postprocessMs": round((t3 - t2) * 1000, 2),
            "serverTotalMs": round((t3 - t0) * 1000, 2),
            "requestsServedByThisContainer": _served,
            "modelSource": _model_source,
        },
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
        self._send(
            200,
            {
                "status": "ready",
                "model": f"{MODEL_ID}-{PRECISION}",
                "usage": "POST raw image bytes; optional ?conf=&iou=&size= query parameters",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._send(400, {"error": "empty body; POST raw image bytes"})
                return
            if length > MAX_UPLOAD_BYTES:
                self._send(413, {"error": f"image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB"})
                return

            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)

            def number(key: str, fallback: float) -> float:
                try:
                    return float(query[key][0])
                except (KeyError, IndexError, ValueError):
                    return fallback

            size = int(number("size", DEFAULT_INPUT_SIZE))
            if size not in (320, 480, 640):
                self._send(400, {"error": "size must be one of 320, 480, 640"})
                return

            result = run_detection(
                self.rfile.read(length),
                score_threshold=number("conf", 0.35),
                iou_threshold=number("iou", 0.45),
                size=size,
            )
            self._send(200, result)
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *_args) -> None:
        """Silence the default stderr access log; Vercel captures its own."""
