# ml/ — train → export → quantize → evaluate → benchmark

One parameterized pipeline, retargeted per dataset via `configs/<name>.yaml`.

## Setup (Windows, RTX 5070 / sm_120)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CUDA 12.8 wheels are required: earlier builds have no sm_120 kernels and fall back to
"no kernel image is available for execution on the device".
