"""Fire-and-forget detection/zone event ingest for a separate NestJS
dashboard service. Set NESTJS_INGEST_URL in config.yaml/env to enable;
leaving it empty disables ingest entirely."""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import config as _config

_cfg = _config.load()

NESTJS_INGEST_URL = _cfg.NESTJS_INGEST_URL.rstrip("/")
NESTJS_API_KEY = _cfg.NESTJS_API_KEY
NESTJS_CAMERA_ID = _cfg.NESTJS_CAMERA_ID

_BACKLOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nestjs_ingest_backlog.jsonl")
_BACKLOG_MAX_LINES = 5000
_backlog_lock = threading.Lock()
_FLUSH_RETRY_INTERVAL_SEC = 2
_last_flush_attempt = 0.0


def enabled():
    return bool(NESTJS_INGEST_URL)


print("=" * 70)
if enabled():
    print(f"[NESTJS] STARTUP: Dashboard ingest ENABLED -> {NESTJS_INGEST_URL} "
          f"(cameraId={NESTJS_CAMERA_ID})")
else:
    print(f"[NESTJS] STARTUP: Dashboard ingest DISABLED "
          f"(NESTJS_INGEST_URL is not set; cameraId={NESTJS_CAMERA_ID!r}). "
          "No detection/zone events will reach the dashboard. Set "
          "NESTJS_INGEST_URL, NESTJS_API_KEY, NESTJS_CAMERA_ID in config.yaml/env "
          "to enable.")
print("=" * 70)


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


def _send(payload, path):
    url = f"{NESTJS_INGEST_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Api-Key": NESTJS_API_KEY}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    urllib.request.urlopen(req, timeout=5).close()


def _enqueue_backlog(payload, path):
    with _backlog_lock:
        try:
            with open(_BACKLOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"path": path, "payload": payload}) + "\n")
        except OSError as exc:
            print(f"[NESTJS] Failed to write backlog entry: {exc}")


def _flush_backlog():
    """Retry backlogged entries in order; stop at the first failure so
    ordering is preserved and nothing is dropped. Caps how many lines are
    kept if the backlog file grows past _BACKLOG_MAX_LINES."""
    global _last_flush_attempt
    with _backlog_lock:
        now = time.monotonic()
        if now - _last_flush_attempt < _FLUSH_RETRY_INTERVAL_SEC:
            return
        _last_flush_attempt = now
        if not os.path.exists(_BACKLOG_PATH):
            return
        try:
            with open(_BACKLOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            print(f"[NESTJS] Failed to read backlog: {exc}")
            return
        if not lines:
            return
        if len(lines) > _BACKLOG_MAX_LINES:
            lines = lines[-_BACKLOG_MAX_LINES:]

        remaining = list(lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                remaining = remaining[1:]
                continue
            try:
                entry = json.loads(line)
                _send(entry["payload"], entry["path"])
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                print(f"[NESTJS] Backlog flush stopped at entry {i}: {exc}")
                break
            remaining = remaining[1:]

        if remaining:
            with open(_BACKLOG_PATH, "w", encoding="utf-8") as f:
                f.writelines(remaining)
        else:
            try:
                os.remove(_BACKLOG_PATH)
            except OSError:
                pass


def _post(payload, path="/ingest/detection"):
    _flush_backlog()
    try:
        _send(payload, path)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[NESTJS] Failed to send request to {path}, queuing to backlog: {exc}")
        _enqueue_backlog(payload, path)
