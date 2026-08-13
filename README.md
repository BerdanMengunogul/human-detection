# human-detection

A camera-based human detection and tracking pipeline with re-identification (ReID), zone-based
enter/leave logic, PostgreSQL event logging, ntfy alerting, and a live web dashboard. Runs on a
single GPU box, reading an RTSP stream and processing it in real time.

## How it works

Each frame goes through a fixed pipeline (orchestrated in `pipeline.py`):

1. **Detect + track** — [YOLOv8](https://github.com/ultralytics/ultralytics) (`model.track(...,
   persist=True)`) detects people and assigns them a persistent track ID via the built-in
   BoT-SORT/ByteTrack tracker (configured in `tracker_reid.yaml`). Low-confidence or too-small
   boxes are discarded (drawn in red for tuning visibility).
2. **Identify** — `identity.py` maintains a `PersonGallery` of body-appearance and face
   embeddings. When the tracker hands a track a *new* ID (e.g. after someone leaves and re-enters
   frame), `identify_many()` compares its embeddings against the gallery via cosine similarity —
   a confident face match wins outright (survives clothing/headwear changes), otherwise it falls
   back to body-appearance ReID, or mints a new `Person-N` id. Face embeddings come from
   InsightFace (`FaceAnalysis`); body embeddings from a YOLO-ReID ONNX model
   (`models/yolo26m-reid.onnx`).
3. **Zones** — `zones.py` loads polygon door-zones from `data/door_zones.json` (drawn/edited from
   the dashboard) and does point-in-polygon checks to detect when a person's foot-point enters or
   leaves a zone. "Ignore zones" can also be defined to suppress detections in dead areas (mirrors,
   posters, etc.).
4. **Log + alert** — Zone enter/leave and detection events are written to PostgreSQL
   (`events.py`, `EventLog`, table `events`), optionally pushed to ntfy (`notify.py`) for
   zone-alert push notifications, and optionally forwarded to a companion dashboard backend
   (`nestjs_ingest.py`, see below).
5. **Draw + serve** — annotated frames are JPEG-encoded and exposed to the local web dashboard
   (`webapp.py`, FastAPI) as an MJPEG stream / snapshot poll, alongside a JSON API for occupancy,
   live tracks, zone status, and people management.

`DetectionState` in `pipeline.py` is the thread-safe handle the web server reads live
frames/stats from — the detection loop runs in a background thread independent of any dashboard
client.

## Project structure

```
human_detection.py      Entry point: starts the capture/detection loop and (optionally) the web server
pipeline.py              Per-frame orchestration: detect -> track -> identify -> zones -> draw -> encode; DetectionState
identity.py               ReID: appearance + face embedding gallery, person_id assignment, name mapping (people.json)
zones.py                  Polygon door-zone loading, hot-reload, point-in-polygon / enter-leave geometry
events.py                 PostgreSQL-backed ENTER/EXIT event log (EventLog)
notify.py                 ntfy.sh push notifications for zone alerts
nestjs_ingest.py          Optional fire-and-forget push of detection/zone events to a companion NestJS dashboard
stream.py                 RTSP capture: LatestFrameReader, open_stream (always-latest-frame reader, reconnect handling)
config.py                 Layered config loader: defaults -> config.yaml -> HD_* env vars (+ .env for secrets)
webapp.py                 FastAPI dashboard: video feed, occupancy/zone/people APIs, HTTP Basic auth
dataset_collector.py      Saves crops for building/retraining the ReID dataset
templates/, static/       Dashboard frontend (Jinja2 + vanilla JS/CSS)
data/door_zones.json      Dashboard-drawn zone polygons (id, name, task, points)
tracker_reid.yaml         Ultralytics tracker configuration (BoT-SORT/ByteTrack tuning)
scripts/                  One-off/maintenance scripts: seed_gallery, train_reid, calibrate_reid_thresholds,
                           relabel_tool, inspect_dataset_mixing, widen_ignore_zone
tools/                    analyze_log.py, run_test.py — log analysis / test running helpers
tests/                    pytest unit tests for zone enter/exit/status logic and track-id bookkeeping
docs/SETUP.md             Detailed setup: dashboard auth, PostgreSQL, GPU/CUDA DLL wiring, pinned versions
docs/IMPROVEMENT_PLAN.md  Design notes / planned work
config.yaml.example       Template for config.yaml (gitignored; non-secret local config)
.env.example              Template for .env (gitignored; secrets — DB password, RTSP credentials)
```

Gitignored / runtime-generated, not tracked: `.env`, `config.yaml`, `*.pt`, `*.onnx` (model
weights), `gallery.npz` (ReID embedding gallery), `people.json` (id-to-name mapping),
`door_zones.version`, `*.db*`, `dataset/`, `dataset_named/`, `contact_sheets/`, log/output files.

## Setup

See [`docs/SETUP.md`](docs/SETUP.md) for full details. Summary:

1. **Python deps** — `pip install -r requirements.txt`. Pinned for a matched GPU stack: PyTorch
   2.13.0+cu126, torchvision 0.28.0+cu126, onnxruntime-gpu 1.21.1, requires CUDA driver ≥560.xx.
   No system CUDA Toolkit install needed — `identity.py` wires the DLL search path at import time
   from the pip-installed NVIDIA redistributable packages (`nvidia-cudnn-cu12` etc.), since
   onnxruntime's CUDA execution provider needs a PATH-discoverable CUDA/cuDNN runtime that
   PyTorch's private bundled copy doesn't provide.
2. **Config** — copy `config.yaml.example` → `config.yaml` for non-secret settings (model paths,
   thresholds, dashboard integration toggles, etc.), and `.env.example` → `.env` for secrets
   (`CAMERA_RTSP_URL`, `DB_PASSWORD`). `config.py` layers: hardcoded defaults → `config.yaml` →
   `HD_*` env vars (highest precedence). `.env` is loaded via `python-dotenv` and is never
   read from `config.yaml`, so credentials can't accidentally land in a committed file.
3. **Database** — PostgreSQL; set `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER` via config/env and
   `DB_PASSWORD` via `.env`. The `events` table is auto-created on first connect.
4. **Run** — `python human_detection.py` starts capture + detection (+ the FastAPI dashboard, if
   enabled). Dashboard auth is HTTP Basic (`DASHBOARD_USER`/`DASHBOARD_PASSWORD`), enforced once
   bound to a non-localhost address. **This is LAN-only** — there's no TLS, session auth, or
   rate-limiting, so don't expose it to the open internet without adding those.

## Testing

`pytest tests/` covers zone enter/exit/status transitions and track-id bookkeeping in
`pipeline.py`. `tools/run_test.py` and `TEST_SCENARIOS.md` cover manual/scripted end-to-end test
scenarios against a live or recorded camera feed. `tools/analyze_log.py` parses pipeline log
output for offline analysis.

## NestJS dashboard integration (optional)

This pipeline can optionally push detection and zone enter/leave events to a companion
NestJS + Next.js dashboard, [`people-counter-nestjs`](../people-counter-nestjs), living in a
sibling repo. Events are sent fire-and-forget from a background thread (see
`nestjs_ingest.py`), so a slow or unreachable dashboard never blocks the detection loop. Failed
sends are queued to a local backlog file (`nestjs_ingest_backlog.jsonl`, capped at 5000 entries)
and retried in order — preserving event ordering — the next time an event is sent.

Configure it in `config.yaml`:

```yaml
NESTJS_INGEST_URL: ""        # e.g. "http://localhost:3001". Empty/unset disables the integration entirely.
NESTJS_API_KEY: ""           # must match the dashboard API's API_KEY env var
NESTJS_CAMERA_ID: "front-door"  # identifier reported for this camera
```

- `NESTJS_INGEST_URL` — base URL of the dashboard API. Leaving it empty (the default) disables
  the integration entirely; no requests are made.
- `NESTJS_API_KEY` — sent as the `X-Api-Key` header on every request; must match the
  dashboard's configured API key.
- `NESTJS_CAMERA_ID` — the camera identifier included in each event payload, so the dashboard
  can distinguish between multiple cameras.
