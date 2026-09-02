"""Benchmark one ONNX detector across every execution provider this machine offers.

Ground rules, because a latency number without them is decoration:

* batch size 1 -- this is a real-time single-stream detector, not a throughput farm
* 50 warm-up iterations discarded before timing (CUDA/TensorRT need the first
  launches to build kernels and autotune; TensorRT may build an engine entirely)
* 300 timed iterations, reported as p50 and p95 rather than a mean
* the same .onnx file for every provider, so the only variable is the runtime
* providers that fail to initialise are recorded as failures with their error text,
  never silently dropped -- "TensorRT was unavailable because X" is a real result

Writes ml/results/bench.json, which the site's /benchmarks page renders directly.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Importing torch first loads the CUDA 12.8 runtime DLLs it ships into the process,
# which is what lets onnxruntime-gpu's CUDA provider find them on Windows without a
# separate system-wide CUDA toolkit install.
try:
    import torch  # noqa: F401

    _TORCH_OK = True
except Exception:  # pragma: no cover - torch is optional for CPU-only runs
    _TORCH_OK = False

import onnxruntime as ort  # noqa: E402

import manifest as manifest_mod  # noqa: E402

WARMUP_ITERS = 50
TIMED_ITERS = 300
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "bench.json"
ENGINE_CACHE = Path(__file__).resolve().parent / ".trt_cache"


def provider_options(provider: str, input_name: str, size: int) -> dict[str, Any]:
    if provider == "TensorrtExecutionProvider":
        shape = f"{input_name}:1x3x{size}x{size}"
        ENGINE_CACHE.mkdir(exist_ok=True)
        return {
            "trt_fp16_enable": True,
            # Engine builds are expensive (minutes). Cache them so a re-run measures
            # inference rather than compilation.
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(ENGINE_CACHE),
            # The ONNX has dynamic H/W for the site's resolution toggle; TensorRT wants
            # a concrete optimisation profile, so pin one per benchmarked size.
            "trt_profile_min_shapes": shape,
            "trt_profile_opt_shapes": shape,
            "trt_profile_max_shapes": shape,
        }
    if provider == "CUDAExecutionProvider":
        return {"device_id": 0}
    return {}


def bench_one(model_path: Path, provider: str, size: int) -> dict[str, Any]:
    result: dict[str, Any] = {"provider": provider, "inputSize": size}
    try:
        probe = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = probe.get_inputs()[0].name
        del probe

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        build_start = time.perf_counter()
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[(provider, provider_options(provider, input_name, size))],
        )
        session_ms = (time.perf_counter() - build_start) * 1000

        # ORT falls back to CPU without raising if a provider cannot be created.
        # Reporting that fallback as if it were the requested provider is how fake
        # "TensorRT" numbers end up on portfolio sites.
        active = session.get_providers()
        if provider not in active:
            result["status"] = "unavailable"
            result["error"] = f"provider not active; ORT used {active}"
            return result

        rng = np.random.default_rng(0)
        dummy = rng.random((1, 3, size, size), dtype=np.float32)
        feed = {input_name: dummy}

        for _ in range(WARMUP_ITERS):
            session.run(None, feed)

        samples: list[float] = []
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            session.run(None, feed)
            samples.append((time.perf_counter() - start) * 1000)

        samples.sort()
        result.update(
            {
                "status": "ok",
                "activeProviders": active,
                "sessionInitMs": round(session_ms, 1),
                "p50Ms": round(statistics.median(samples), 3),
                "p95Ms": round(samples[int(0.95 * len(samples)) - 1], 3),
                "meanMs": round(statistics.fmean(samples), 3),
                "minMs": round(samples[0], 3),
                "fps": round(1000 / statistics.median(samples), 1),
                "iterations": TIMED_ITERS,
            }
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:400]
    return result


def gpu_info() -> dict[str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        name, driver = (part.strip() for part in out.stdout.strip().split(","))
        return {"gpu": name, "driver": driver}
    except Exception:
        return {"gpu": "unknown", "driver": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="coco80-n")
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 640])
    parser.add_argument("--precisions", nargs="+", default=["fp32", "int8"])
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"],
    )
    args = parser.parse_args()

    print(f"[bench] ORT {ort.__version__}  available={ort.get_available_providers()}")
    runs: list[dict[str, Any]] = []
    for precision in args.precisions:
        model_path = manifest_mod.MODEL_DIR / f"{args.model_id}-{precision}.onnx"
        if not model_path.exists():
            print(f"[bench] skip {model_path.name} (not exported)")
            continue
        for provider in args.providers:
            for size in args.sizes:
                label = f"{precision:>4} {provider.replace('ExecutionProvider',''):>8} @{size}"
                print(f"[bench] {label} ...", flush=True)
                run = bench_one(model_path, provider, size)
                run["precision"] = precision
                run["modelId"] = args.model_id
                run["modelBytes"] = model_path.stat().st_size
                runs.append(run)
                if run["status"] == "ok":
                    print(f"[bench] {label}  p50={run['p50Ms']}ms  p95={run['p95Ms']}ms  {run['fps']}fps")
                else:
                    print(f"[bench] {label}  {run['status'].upper()}: {run.get('error','')}")

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methodology": {
            "batchSize": 1,
            "warmupIterations": WARMUP_ITERS,
            "timedIterations": TIMED_ITERS,
            "note": (
                "Same ONNX file across all providers. Warm-up discarded. Latency is "
                "session.run() only and excludes letterbox preprocessing and NMS, both "
                "of which the browser HUD reports separately."
            ),
        },
        "environment": {
            "onnxruntime": ort.__version__,
            "availableProviders": ort.get_available_providers(),
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "torch": getattr(__import__("torch"), "__version__", None) if _TORCH_OK else None,
            **gpu_info(),
        },
        "runs": runs,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] wrote {RESULTS_PATH}")

    web_copy = Path(__file__).resolve().parent.parent / "web" / "data" / "bench.json"
    web_copy.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] wrote {web_copy}")


if __name__ == "__main__":
    main()
