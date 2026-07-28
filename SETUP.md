# Setup: GPU stack (CUDA / cuDNN) & PostgreSQL

## Database (PostgreSQL)

Enter/exit events (`EventLog` in [human_detection.py](human_detection.py)) are
stored in PostgreSQL instead of a local file, so the detector (writer) and
`webapp.py` dashboard (reader) can hold independent connections without any
file-locking contention.

1. Install PostgreSQL locally (or point at an existing server) and create a
   database and user, e.g.:
   ```sql
   CREATE DATABASE human_detection;
   CREATE USER postgres WITH PASSWORD 'your-password-here';
   GRANT ALL PRIVILEGES ON DATABASE human_detection TO postgres;
   ```
2. Set connection settings via `config.yaml` / `HD_*` env vars (see
   `config.yaml.example` and [config.py](config.py)):
   - `DB_HOST` (default `localhost`)
   - `DB_PORT` (default `5433`)
   - `DB_NAME` (default `human_detection`)
   - `DB_USER` (default `postgres`)
3. Set the password in `.env` (never in `config.yaml`, so it stays out of
   version control): copy `.env.example` to `.env` and fill in
   `DB_PASSWORD`.
4. The `events` table is created automatically on first run
   (`CREATE TABLE IF NOT EXISTS`) — no manual migration needed.
5. The Python driver is `psycopg2-binary`, already pinned in
   [requirements.txt](requirements.txt); no separate libpq install is
   required on Windows since the binary wheel bundles it.

## GPU stack (CUDA / cuDNN)

This project runs YOLOv8 + InsightFace on GPU via PyTorch and
`onnxruntime-gpu`. No system-wide CUDA Toolkit install is required — the
CUDA/cuDNN runtime is provided entirely by pip packages, and
[human_detection.py](human_detection.py) wires up the DLL search path at
import time (see the `os.add_dll_directory` block near the top of the file).

## Why this needs documenting

PyTorch's `+cu126` wheel bundles its own private copy of the CUDA/cuDNN
runtime, but that copy is only visible to PyTorch itself. `onnxruntime-gpu`
(used for InsightFace's ONNX models) looks for CUDA/cuDNN DLLs on the
Windows DLL search path / `PATH`, and can't see PyTorch's private copy.

To satisfy that without a full CUDA Toolkit install, `requirements.txt`
pulls in the NVIDIA redistributable pip packages (`nvidia-cudnn-cu12`,
`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-cuda-nvrtc-cu12`,
`nvidia-cufft-cu12`) as transitive dependencies of the `+cu126` torch wheel.
[human_detection.py](human_detection.py) then explicitly adds each package's
`nvidia\<pkg>\bin` directory (under the venv's site-packages) to the DLL
search path before `onnxruntime`/`insightface` are imported, so
`onnxruntime-gpu`'s `CUDAExecutionProvider` can find them.

If any of the following pieces drift out of sync, this breaks.

## Required stack (pinned)

| Component | Version | Where it's pinned |
|---|---|---|
| CUDA runtime (via pip) | 12.6 (`cu126`) | `--extra-index-url .../whl/cu126` in [requirements.txt](requirements.txt) |
| cuDNN | 9.x (whatever `cu126` torch 2.13 pulls in) | transitive dep of `torch==2.13.0+cu126` |
| PyTorch | 2.13.0+cu126 | [requirements.txt](requirements.txt) |
| torchvision | 0.28.0+cu126 | [requirements.txt](requirements.txt) |
| onnxruntime-gpu | 1.21.1 | [requirements.txt](requirements.txt) |
| NVIDIA driver | must support CUDA 12.6 (R560+ recommended) | installed separately, see below |
| Python | 3.11 or 3.12 recommended | matches available `cu126` wheels |

**Install order matters:** `torch`/`torchvision` must be installed from the
`cu126` index *before* (or in the same `pip install -r requirements.txt`
call as) `onnxruntime-gpu`, so pip resolves the matching `nvidia-*-cu12`
redistributable packages into site-packages first. Installing
`onnxruntime-gpu` alone does **not** pull those in.

## One-time host prerequisites

1. An NVIDIA GPU with a driver new enough for CUDA 12.6 (check with
   `nvidia-smi` — driver version should be ≥ 560.xx).
2. No separate CUDA Toolkit or cuDNN install needed — pip provides the
   runtime DLLs. Do not rely on a machine-wide CUDA install; if one exists
   it's irrelevant to this project (and a mismatched one on `PATH` could in
   theory shadow the pip-provided DLLs, though `add_dll_directory` is
   ordered to prefer the pip copies).

## Install steps

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Verifying the GPU path works

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then run the app and check the startup log line
`[INFO] Running inference on: GPU (<device name>)`. If it falls back to
CPU, or `onnxruntime` logs a `CUDAExecutionProvider` load failure, re-check
the install order above and confirm the driver supports CUDA 12.6.

## Upgrading the stack later

If you bump `torch`/`onnxruntime-gpu` to a new CUDA line (e.g. cu128), the
package names under `site-packages/nvidia/` change to match — the DLL-path
loop in [human_detection.py](human_detection.py) discovers whatever
`nvidia\<pkg>\bin` directories exist, so no code change is needed there.
Just update the `--extra-index-url` and version pins in
[requirements.txt](requirements.txt) together, then reinstall into a clean
venv to avoid stale DLLs from the old CUDA line lingering in site-packages.
