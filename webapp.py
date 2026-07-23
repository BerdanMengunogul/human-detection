"""FastAPI dashboard for human_detection.py.

Runs the existing detection loop (unmodified logic - YOLO/BoT-SORT/ReID/
EventLog/zone-gating) on a background thread, and serves:

  GET /            tabbed HTML dashboard (Live / Occupancy / History)
  GET /video_feed   MJPEG stream of the latest annotated frame
  GET /api/occupancy  current human count + who is currently present
  GET /api/events     recent enter/exit history from tracking.db

Meant to be started via `python human_detection.py --web`, which bootstraps
uvicorn bound to a LAN-reachable interface (0.0.0.0 by default).
"""

import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import human_detection as hd

app = FastAPI(title="Human Detection Dashboard")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

_security = HTTPBasic(auto_error=False)

# Auth is only enforced once the dashboard is bound to something other than
# localhost (LAN/public hosting); serve_dashboard() flips this at startup.
_auth_required = False


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)):
    """Gates every route behind a single shared username/password, set via
    DASHBOARD_USER / DASHBOARD_PASSWORD env vars (required once the
    dashboard is reachable outside localhost, e.g. LAN or a tunnel)."""
    if not _auth_required:
        return "local"

    expected_user = os.environ.get("DASHBOARD_USER")
    expected_pass = os.environ.get("DASHBOARD_PASSWORD")
    if not expected_user or not expected_pass:
        raise RuntimeError(
            "DASHBOARD_USER and DASHBOARD_PASSWORD must be set before starting the dashboard."
        )
    if credentials is None or not (
        secrets.compare_digest(credentials.username, expected_user)
        and secrets.compare_digest(credentials.password, expected_pass)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

state = hd.DetectionState()
_detection_thread = None
_thread_lock = threading.Lock()
_show_window = False


def _start_detection():
    global _detection_thread
    with _thread_lock:
        if _detection_thread is not None and _detection_thread.is_alive():
            return False
        state.reset_stop()
        _detection_thread = threading.Thread(
            target=hd.run_detection,
            kwargs={"state": state, "show_window": _show_window},
            daemon=True,
        )
        _detection_thread.start()
        return True


def _stop_detection():
    with _thread_lock:
        if _detection_thread is None or not _detection_thread.is_alive():
            return False
        state.stop()
        _detection_thread.join(timeout=10)
        return True


def _events_db_readonly():
    """Read-only connection to tracking.db so the dashboard never
    contends with the detection loop's own writes (WAL-friendly)."""
    conn = sqlite3.connect(f"file:{hd.DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _load_zones_file():
    if os.path.exists(hd.ZONES_PATH):
        with open(hd.ZONES_PATH) as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("zones", [])
    data.setdefault("web_zones", [])
    return data


def _save_zones_file(data):
    with open(hd.ZONES_PATH, "w") as f:
        json.dump(data, f, indent=2)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, _user: str = Depends(require_auth)):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed(_user: str = Depends(require_auth)):
    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpeg = state.latest_jpeg()
            if jpeg is not None:
                yield boundary + jpeg + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/occupancy")
def api_occupancy(_user: str = Depends(require_auth)):
    snap = state.snapshot()
    conn = _events_db_readonly()
    try:
        rows = conn.execute(
            """
            SELECT person_id, event_type, timestamp
            FROM events e
            WHERE timestamp = (
                SELECT MAX(timestamp) FROM events WHERE person_id = e.person_id
            )
            ORDER BY person_id
            """
        ).fetchall()
    finally:
        conn.close()

    live_present = snap["present_person_ids"]
    names = hd.PEOPLE_STORE.all()
    people = [
        {
            "person_id": row["person_id"],
            "name": names.get(row["person_id"]),
            # Live tracker state wins over DB history: a person can be
            # visibly in-frame before a zone-gated ENTER ever gets written.
            "status": "IN" if row["person_id"] in live_present else "OUT",
            "since": row["timestamp"],
        }
        for row in rows
    ]
    known_ids = {row["person_id"] for row in rows}
    for person_id in live_present - known_ids:
        people.append({"person_id": person_id, "name": names.get(person_id), "status": "IN", "since": None})
    people.sort(key=lambda p: p["person_id"])
    unique_count = len(known_ids | live_present)
    return JSONResponse(
        {
            "count": snap["count"],
            "fps": round(snap["fps"], 1),
            "detector_running": snap["running"],
            "unique_count": unique_count,
            "people": people,
        }
    )


@app.post("/api/start")
def api_start(_user: str = Depends(require_auth)):
    started = _start_detection()
    return JSONResponse({"ok": True, "started": started, "running": state.snapshot()["running"]})


@app.post("/api/stop")
def api_stop(_user: str = Depends(require_auth)):
    stopped = _stop_detection()
    return JSONResponse({"ok": True, "stopped": stopped, "running": state.snapshot()["running"]})


@app.post("/api/reset")
def api_reset(_user: str = Depends(require_auth)):
    """Clears the ReID gallery, live tracking state, and event history so
    unique-person counting starts over from Person-1. Handled on the
    detection thread (which owns the DB connection and in-memory gallery)
    to avoid racing its writes."""
    state.request_reset()
    return JSONResponse({"ok": True})


@app.get("/api/events")
def api_events(limit: int = 50, _user: str = Depends(require_auth)):
    limit = max(1, min(limit, 500))
    conn = _events_db_readonly()
    try:
        rows = conn.execute(
            "SELECT id, person_id, event_type, timestamp FROM events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    names = hd.PEOPLE_STORE.all()
    return JSONResponse(
        {
            "events": [
                {
                    "id": row["id"],
                    "person_id": row["person_id"],
                    "name": names.get(row["person_id"]),
                    "event_type": row["event_type"],
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]
        }
    )


@app.get("/api/live-tracks")
def api_live_tracks(_user: str = Depends(require_auth)):
    """Live person bounding boxes (original-frame coords) plus assigned
    names, for the People tab overlay. Reuses /api/zone-snapshot's image
    and /api/zone-snapshot/meta's dims for coordinate scaling."""
    boxes = state.live_boxes()
    names = hd.PEOPLE_STORE.all()
    return JSONResponse(
        {
            "people": [
                {
                    "person_id": person_id,
                    "box": list(box),
                    "name": names.get(person_id),
                }
                for person_id, box in boxes.items()
            ]
        }
    )


@app.get("/api/people")
def api_people(_user: str = Depends(require_auth)):
    names = hd.PEOPLE_STORE.all()
    return JSONResponse(
        {"people": [{"person_id": pid, "name": name} for pid, name in sorted(names.items())]}
    )


@app.post("/api/people/{person_id}")
async def api_set_person_name(person_id: int, request: Request, _user: str = Depends(require_auth)):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    hd.PEOPLE_STORE.set(person_id, name)
    return JSONResponse({"person_id": person_id, "name": name})


@app.delete("/api/people/{person_id}")
def api_delete_person_name(person_id: int, _user: str = Depends(require_auth)):
    hd.PEOPLE_STORE.delete(person_id)
    return JSONResponse({"ok": True})


@app.get("/api/box-color")
def api_get_box_color(_user: str = Depends(require_auth)):
    b, g, r = hd.BOX_COLOR
    return JSONResponse({"color": "#{:02x}{:02x}{:02x}".format(r, g, b)})


@app.post("/api/box-color")
async def api_set_box_color(request: Request, _user: str = Depends(require_auth)):
    body = await request.json()
    color = (body.get("color") or "").strip().lstrip("#")
    if len(color) != 6 or any(c not in "0123456789abcdefABCDEF" for c in color):
        raise HTTPException(status_code=400, detail="color must be a #rrggbb hex string")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    hd.BOX_COLOR = (b, g, r)
    return JSONResponse({"ok": True, "color": "#" + color})


@app.get("/api/zone-snapshot")
def api_zone_snapshot(_user: str = Depends(require_auth)):
    jpeg = state.latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/zone-snapshot/meta")
def api_zone_snapshot_meta(_user: str = Depends(require_auth)):
    return JSONResponse(state.dims())


@app.get("/api/zones")
def api_zones(_user: str = Depends(require_auth)):
    return JSONResponse(_load_zones_file()["web_zones"])


VALID_ZONE_TASKS = {"none", "alert_entry", "alert_presence"}


@app.post("/api/zones")
async def api_create_zone(request: Request, _user: str = Depends(require_auth)):
    body = await request.json()
    task = body.get("task", "none")
    if task not in VALID_ZONE_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}")
    zone = {
        "id": secrets.token_hex(4),
        "name": body["name"],
        "type": body["type"],
        "points": body["points"],
        "task": task,
        "created_at": datetime.utcnow().isoformat(),
    }
    data = _load_zones_file()
    data["web_zones"].append(zone)
    _save_zones_file(data)
    return JSONResponse(zone)


@app.patch("/api/zones/{zone_id}")
async def api_update_zone(zone_id: str, request: Request, _user: str = Depends(require_auth)):
    body = await request.json()
    data = _load_zones_file()
    zone = next((z for z in data["web_zones"] if z["id"] == zone_id), None)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")

    if "task" in body:
        if body["task"] not in VALID_ZONE_TASKS:
            raise HTTPException(status_code=400, detail=f"Invalid task: {body['task']}")
        zone["task"] = body["task"]
    if "name" in body:
        zone["name"] = body["name"]
    if "type" in body:
        zone["type"] = body["type"]

    _save_zones_file(data)
    return JSONResponse(zone)


@app.delete("/api/zones/{zone_id}")
def api_delete_zone(zone_id: str, _user: str = Depends(require_auth)):
    data = _load_zones_file()
    data["web_zones"] = [z for z in data["web_zones"] if z["id"] != zone_id]
    _save_zones_file(data)
    return JSONResponse({"ok": True})


@app.get("/api/zone-status")
def api_zone_status(_user: str = Depends(require_auth)):
    data = _load_zones_file()
    zone_status = state.zone_status()
    result = []
    for zone in data["web_zones"]:
        task = zone.get("task", "none")
        info = zone_status.get(zone["id"], {"occupants": set(), "entered": False})
        occupants = info.get("occupants", set())
        entered = info.get("entered", False)
        alert = False
        if task == "alert_presence":
            alert = len(occupants) > 0
        elif task == "alert_entry":
            alert = entered
        result.append(
            {
                "id": zone["id"],
                "name": zone["name"],
                "task": task,
                "occupants": sorted(occupants),
                "alert": alert,
            }
        )
    return JSONResponse({"zones": result})


def serve_dashboard(host="0.0.0.0", port=8000, show_window=False):
    import uvicorn

    global _auth_required
    _auth_required = host not in ("127.0.0.1", "localhost", "::1")
    if _auth_required and (
        not os.environ.get("DASHBOARD_USER") or not os.environ.get("DASHBOARD_PASSWORD")
    ):
        raise SystemExit(
            "Set DASHBOARD_USER and DASHBOARD_PASSWORD env vars before starting the dashboard "
            "(required when binding to a non-localhost host)."
        )

    global _show_window
    _show_window = show_window
    _start_detection()

    print(f"[WEB] Dashboard starting on http://{host}:{port} (LAN-reachable if host is 0.0.0.0)")
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        _stop_detection()
