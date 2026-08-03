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


if enabled():
    print(f"[NESTJS] Dashboard ingest enabled -> {NESTJS_INGEST_URL} "
          f"(cameraId={NESTJS_CAMERA_ID})")
else:
    print("[NESTJS] NESTJS_INGEST_URL is not set - dashboard ingest is DISABLED. "
          "No detection/zone events will reach the dashboard. Set "
          "NESTJS_INGEST_URL, NESTJS_API_KEY, NESTJS_CAMERA_ID in config.yaml/env "
          "to enable.")


def send_detection(person_id, timestamp, zone_id=None, zone_event=None):
    """Fire a detection/zone event at the NestJS ingest endpoint. Sends in
    a background thread so a slow/unreachable NestJS server can never
    block the caller (the main detection loop)."""
    if not enabled():
        return
    payload = {
        "personId": person_id,
        "cameraId": NESTJS_CAMERA_ID,
        "timestamp": timestamp,
    }
    if zone_id is not None:
        payload["zoneId"] = zone_id
        payload["zoneEvent"] = zone_event
    threading.Thread(target=_post, args=(payload,), daemon=True).start()


def sync_zones(web_zones):
    """Push door-zone id/name to NestJS so zone-based alerts have a Zone row
    to join against. Fires in a background thread; a slow/unreachable NestJS
    server can never block the caller. Only id/name are synced -- `task`
    (which drives alerting) is dashboard-managed in NestJS and must not be
    overwritten here."""
    if not enabled():
        return
    payload = {
        "zones": [{"id": z["id"], "name": z["name"]} for z in web_zones],
    }
    threading.Thread(target=_post, args=(payload, "/ingest/zones"), daemon=True).start()


def _post(payload, path="/ingest/detection"):
    url = f"{NESTJS_INGEST_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Api-Key": NESTJS_API_KEY}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[NESTJS] Failed to send request to {path}: {exc}")
