#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
PY=".venv/Scripts/python.exe"
"$PY" -m pip install --upgrade pip -q
echo "=== installing torch cu128 (large) ==="
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "=== installing ml requirements ==="
"$PY" -m pip install -r requirements.txt
echo "=== versions ==="
"$PY" - <<'PYEOF'
import torch, onnxruntime as ort, ultralytics
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("ort", ort.__version__)
print("providers", ort.get_available_providers())
print("ultralytics", ultralytics.__version__)
PYEOF
echo "=== BOOTSTRAP DONE ==="
