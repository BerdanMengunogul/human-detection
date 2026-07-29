"""Push notifications for zone alerts, delivered via ntfy.sh (or a
self-hosted ntfy server). Set NTFY_TOPIC in config.yaml/env to enable;
leaving it empty disables notifications entirely."""

import threading
import urllib.error
import urllib.request

import config as _config

_cfg = _config.load()

NTFY_SERVER = _cfg.NTFY_SERVER.rstrip("/")
NTFY_TOPIC = _cfg.NTFY_TOPIC


def enabled():
    return bool(NTFY_TOPIC)


def notify_zone_alert(zone_name, task):
    """Fire a push notification for a zone alert transitioning to active.
    Sends in a background thread so a slow/unreachable ntfy server can
    never block the caller (e.g. the SSE loop)."""
    if not enabled():
        return
    title = "Presence detected" if task == "alert_presence" else "Zone entry"
    body = f"{zone_name}: {title.lower()}"
    threading.Thread(target=_post, args=(title, body), daemon=True).start()


def _post(title, body):
    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Title": title, "Priority": "high", "Tags": "warning"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[NTFY] Failed to send notification: {exc}")
