"""Fire-and-forget detection/zone event ingest for a separate NestJS
dashboard service. Set NESTJS_INGEST_URL in config.yaml/env to enable;
leaving it empty disables ingest entirely."""

import json
import threading
import urllib.error
import urllib.request

import config as _config

_cfg = _config.load()

NESTJS_INGEST_URL = _cfg.NESTJS_INGEST_URL.rstrip("/")
NESTJS_API_KEY = _cfg.NESTJS_API_KEY
NESTJS_CAMERA_ID = _cfg.NESTJS_CAMERA_ID


def enabled():
    return bool(NESTJS_INGEST_URL)


def send_detection(person_id, timestamp, zone_id=None, zone_event=None):
    """Fire a detection/zone event at the NestJS ingest endpoint. Sends in
    a background thread so a slow/unreachable NestJS server can never
    block the caller (the main detection loop)."""
    if not enabled():
        return
    payload = {
        "personId": str(person_id),
        "cameraId": NESTJS_CAMERA_ID,
        "timestamp": timestamp,
    }
    if zone_id is not None:
        payload["zoneId"] = zone_id
        payload["zoneEvent"] = zone_event
    threading.Thread(target=_post, args=(payload,), daemon=True).start()


def _post(payload):
    url = f"{NESTJS_INGEST_URL}/ingest/detections"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Api-Key": NESTJS_API_KEY}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[NESTJS] Failed to send detection event: {exc}")
