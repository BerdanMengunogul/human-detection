# Improvement Plan — human-detection

Derived from code review of `human_detection.py` (~1,300 lines) and `webapp.py`/`static/app.js`. Verified against current source on 2026-07-23:
- `point_in_any_zone`/`point_in_polygon` zone math *is* computed (`zone_status`, `entered`) but never consulted before `event_log.record(..., "enter"/"exit", ...)` — confirmed dead for gating purposes.
- `RTSP_URL` at `human_detection.py:60` contains a live camera password in plaintext.
- `requirements.txt` lists only 6 packages; code imports torch/numpy/insightface/onnxruntime (unpinned/missing).

Work is grouped into phases so each phase ships something testable rather than one big rewrite.

---

## Phase 0 — Secrets & hygiene (do first, low risk, ~1 day)(DONE)

1.[x] **Rotate the RTSP camera password** (it's currently sitting in source + probably in `server.log`/`.output` files). Confirm with user before rotating, since it's a physical device credential.
2.[x] Move `RTSP_URL` to an environment variable / `.env` (e.g. `python-dotenv`), read via `os environ["CAMERA_RTSP_URL"]`. Fail fast with a clear error if unset — don't fall back to a hardcoded default.
3.[x] Add `.gitignore`: `__pycache__/`, `*.pyc`, `*.db`, `*.db-shm`, `*.db-wal`, `*.output`, `*.log`, `*.pt`, `*.onnx`, `.env`.
4.[x] Delete or move out of the repo: `live_test*.output`, `live_debug_run.output`, `live_verify_run.output`, `web_run.output`, `server.log`, `all_events.txt`. Check first whether any of these still hold info the user wants (git history isn't initialized here — repo isn't even a git repo yet — so deletion is final; confirm before removing).
5.[x] Grep all `.output`/`.log` files for the RTSP password string before deleting, in case rotation needs to also scrub distributed copies.

**Exit criteria:** no secret in any tracked file; `git init` + first commit only after this is done, so the password never enters history.

---

## Phase 1 — Fix zone gating (P0 correctness)(DONE)

Goal: ENTER/EXIT events only fire when a track's foot-point crosses into/out of a configured door zone; tracks that wander mid-room without crossing a zone don't generate events.

1.[x] Decide the model: keep **web polygon zones** (`web_zones`/door_zones.json) as the one system; the CLI rectangle `zones` param is legacy — confirm with user, then delete the rectangle path entirely (item 3 below).
2.[x] In the main loop, `zone_status[zone_id]["entered"]` is already computed per-frame from `occupants_now - occupants_prev`. Thread this into the enter/exit decision:
   - On ENTER: only call `event_log.record(person_id, "enter", ...)` when the person's foot-point actually transitioned from outside all zones to inside a designated "door" zone (or however zones are typed — check `wz["type"]` in `door_zones.json`).
   - On EXIT: symmetric — only fire when the last known position was inside a door zone before the track disappeared, or use zone-exit transition instead of pure track-disappearance.
   - If **no zones are configured**, fall back to current appear/disappear behavior (documented explicitly, not implicit) so the system still works for a bare install.
   - Implemented via foot-point ever-in-zone (`pending_track_zone_hit`) for ENTER and last-known in-zone (`person_in_zone`) for EXIT; all `web_zones` gate (no `type` filter yet). `zone_status["entered"]` still drives dashboard alerts.
3.[x] Delete `point_in_any_zone` if it's genuinely superseded by `point_in_polygon` + `web_zones`, or wire it in if it was meant to gate something else — check its one call site before removing. (Replaced by `_foot_point_in_any_zone`.)
4.[x] Remove the CLI rectangle `zones` argument/parsing path once confirmed unused, to eliminate the "two systems" ambiguity per item 3 of the review.

**Test plan:** walk through a configured door zone → 1 enter event. Walk in circles inside the room without crossing a zone boundary → 0 events. Remove all zones → old appear/disappear behavior returns.

---

## Phase 2 — Hot-reload zones

1.[x] `web_zones = load_web_zones()` now loads once at startup via `WebZonesStore.__init__` and reloads only when `read_zones_version()` changes. **Event-driven** chosen: `webapp.py`'s `_save_zones_file()` calls `hd.bump_zones_version()` on every create/update/delete, writing a counter file (`ZONES_VERSION_PATH`); the detection loop calls `web_zones_store.maybe_reload()` each frame (`human_detection.py:948`), which is a cheap int-file read/compare, not a JSON re-parse.
2.[x] `zone_occupants_prev` is preserved across reload via `merge_zone_occupants` (`human_detection.py:342`) — keeps occupancy sets for zone ids that still exist, only adds/removes for changed ids.

**Test plan:** create a zone in the dashboard while the detector is running; confirm it starts generating events within the poll interval without restarting the process.

---

## Phase 3 — Dependencies & config (DONE)

1.[x] `requirements.txt` regenerated with actual pinned versions (torch==2.13.0+cu126, torchvision, numpy, insightface, onnxruntime-gpu==1.21.1, ultralytics, opencv-python, fastapi, uvicorn, jinja2, python-multipart, python-dotenv, pyyaml), GPU (cu126) variant confirmed via `--extra-index-url`.
2.[x] Single config source added: `config.py` reads env vars + optional `config.yaml` (see `config.yaml.example`), covering model paths, thresholds (`EXIT_GRACE_SECONDS`, `IDENTIFY_DELAY_SECONDS`, `IDENTIFY_MIN_CANDIDATES`, etc.), `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`, `GALLERY_PATH`; `CAMERA_RTSP_URL` and `DB_PASSWORD` stay env-only per Phase 0. Defaults preserve current behavior.
3.[x] `SETUP.md` documents the CUDA/cuDNN stack (pinned versions, install order, DLL-path hack rationale) instead of containerizing.
4.[x] **Switched `EventLog` from SQLite to PostgreSQL** (`human_detection.py`): `psycopg2` connection using `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD`, same `events` table schema via `CREATE TABLE IF NOT EXISTS`. `webapp.py` opens its own separate read-only connection (`_events_db_readonly`) rather than sharing the detector's connection — this also satisfies Phase 5 item 1 (SQLite locking) by construction, since Postgres handles concurrent writer/reader access natively. See `requirements.txt` (`psycopg2-binary`) and `.env.example` (`DB_PASSWORD`).

---

## Phase 4 — Performance-Done

Each of these is independent; do in any order based on measured impact:

[x]1. **Pose model**: make the second YOLO pose pass optional (env/config flag) or run every K-th frame, since it only feeds the skeleton overlay, not tracking/ReID.
[x]2. **Pending-ReID frame copies**: replace `frame.copy()` per candidate with storing a small crop (bounding box region) or capping to a few best frames per track instead of full-frame copies.
[x]3. **Gallery save debounce**: `gallery.save()` currently writes to disk on every identify/top-up. Batch: save at most once every N seconds or on a background thread/timer, plus on clean shutdown.
[x]4. **Skip JPEG encode with no viewers**: track active MJPEG subscriber count in `webapp.py`; skip `cv2.imencode` in the capture loop when it's zero.

---

## Phase 5 — Reliability & concurrency

1. ~~**SQLite locking**~~ — moot: `EventLog` now runs on PostgreSQL (see Phase 3 item 4) with the detector and dashboard each holding their own connection, so there's no shared-file-handle contention to work around.
[x]2. **RTSP reconnect**: wrap `cv2.VideoCapture` read loop — on read failure, attempt reopen with exponential backoff (e.g. 1s, 2s, 5s, 10s cap) instead of sleeping forever. Log reconnect attempts.
[x]3. **EXIT hold fix**: item 15 — one ambiguous pending track blocking all exits. Change the hold to be scoped per-person or per-zone rather than global (`if pending_tracks:` currently blocks everyone). At minimum, cap the hold duration so it can't block indefinitely.
[x]4. **ReID merge cap**: when a merge candidate would attach to a person_id that already has another *live* track, refuse the merge (log it as a rejected candidate) instead of only logging "MULTIPLE live tracks" as a warning.

---

## Phase 6 — Architecture split (do after behavior stabilizes, so you're not refactoring a moving target)

Split `human_detection.py` into modules once phases 1–5 land and are verified, so the split is mechanical rather than another source of bugs:
- `stream.py` — RTSP capture, reconnect loop
- `identity.py` — gallery, ReID, face embedding, person_id assignment
- `zones.py` — polygon/zone loading, hot-reload, point-in-zone
- `events.py` — EventLog/SQLite, enter/exit decision logic
- `pipeline.py` — main per-frame orchestration (detect → track → identify → zones → draw → encode)
- keep `webapp.py` as the FastAPI layer importing from the above

Also address stable identity keys (item 6): decide whether names should bind to a persistent face/body embedding key rather than the session `person_id`, so a Reset doesn't orphan names. This is a design decision — flag to user before implementing, since it changes gallery schema.

---

## Phase 7 — Security & frontend

1. `static/app.js`: replace `innerHTML` usage for person names with `textContent` (item 22) — quick, do independent of everything else.
2. Zone API (`webapp.py`): validate incoming zone JSON — cap point count, require numeric coordinates within frame bounds, reject oversized payloads.
3. Keep HTTP Basic for LAN-only use but document that it's not meant for internet exposure; if the dashboard will ever be exposed beyond LAN, swap for session/token auth first.
4. Consider SSE for occupancy/alerts to replace 1s polling — nice-to-have, not urgent given MJPEG already dominates bandwidth.

---

## Phase 8 — Product/UX (defer until above is stable)

1. Real alert delivery: sound, webhook, or push — ask user which integration they actually want (e.g. ntfy, Discord webhook, email) before building.
2. Clarify occupancy metrics: separate "currently in room" (live track count) from "all-time unique persons" (DB history) in the UI, since item 25 notes they're currently conflated.
3. Reconcile the People-naming vs Occupancy "Saved persons" duplication and add a confirmation step before destructive Reset actions.

---

## Suggested execution order

Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8.

Phases 0–2 are the correctness/security-critical ones and should happen before any further feature work (including the dashboard work already in progress per prior context). Phases 3–5 can interleave with dashboard work. Phase 6 (module split) is best done as its own dedicated pass once the system's behavior is trusted, not bundled with behavior changes.
