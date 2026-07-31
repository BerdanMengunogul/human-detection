# Vercel + Railway deployment design

Date: 2026-07-31

## Problem

`human-detection` currently runs entirely on one local Windows machine: the
GPU detector (`human_detection.py`/`pipeline.py`), a PostgreSQL database, and
the dashboard (`webapp.py`, a FastAPI app serving `templates/index.html` +
`static/app.js`) all live together, reachable only on the LAN via HTTP Basic
auth.

The goal is to make the dashboard (occupancy, history, zones, people) reachable
from the internet, while the GPU-bound detection loop stays on the local
machine (no GPU/camera support on Vercel or Railway).

## Non-goals

- Moving detection/inference off the local GPU machine.
- Exposing the live MJPEG video feed or live per-frame bounding boxes to the
  internet — these stay LAN-only.
- Remote start/stop of the detector process.
- Rewriting the local dashboard's own UI (it keeps using
  `templates/index.html` / `static/app.js` as-is).

## Architecture

Three deployment targets:

1. **Local machine (unchanged role, extended):** runs the GPU detector and a
   local `webapp.py` instance in `DEPLOYMENT_MODE=local`. Keeps HTTP Basic
   auth, `/video_feed`, `/api/live-tracks`, `/api/start`, `/api/stop`, and all
   existing local Zones/People functionality. Additionally, the detector gains
   a background task that uploads a periodic snapshot to Railway (see Data
   flow).
2. **Railway:** hosts (a) a Postgres database (Railway's built-in plugin) and
   (b) the same `webapp.py` codebase running in `DEPLOYMENT_MODE=cloud` — a
   persistent FastAPI service exposing occupancy/history/zones (incl.
   create/edit)/people/reset/auth/snapshot routes only. No detector thread, no
   GPU, no camera access.
3. **Vercel:** hosts a new Next.js frontend (rewrite of the dashboard UI),
   calling the Railway API over HTTPS via an absolute base URL
   (`NEXT_PUBLIC_API_BASE`). No live-video tab.

Both `webapp.py` deployments (local and Railway) read/write the same Postgres
event log; in the cloud case that's Railway's managed Postgres, and the local
detector's `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` config is
pointed at Railway's Postgres instead of the local one. This requires no code
changes — only config/env changes — per the existing layered config in
`config.py` (defaults → `config.yaml` → `HD_*` env vars, with
`DB_PASSWORD`/`CAMERA_RTSP_URL` staying env-only via `.env`, never in
`config.yaml`).

`DEPLOYMENT_MODE` (`local` | `cloud`, default `local`) gates which routes
`webapp.py` mounts and which auth scheme is active, so there is one codebase,
not a fork.

## Data flow

- **Detector → Postgres:** unchanged (writes enter/exit events).
- **Local `webapp.py` → Postgres:** unchanged (reads for local dashboard).
- **Detector → Railway (new):** a background task in the detector process
  runs on a fixed interval (~60s) and does `POST /api/zone-snapshot` to the
  Railway API with the latest camera frame as JPEG bytes. This is independent
  of whether anyone is viewing the cloud Zones tab.
- **Railway `webapp.py` (cloud mode) → Postgres:** serves occupancy, history,
  zones (including create/edit, validated against the most recent uploaded
  snapshot's dimensions), people-names, and reset to the Vercel frontend. It
  also stores/serves the latest uploaded snapshot via
  `POST /api/zone-snapshot` (from the detector) and
  `GET /api/zone-snapshot` + `GET /api/zone-snapshot/meta` (to the Vercel
  Zones tab) — the same contract shape the local dashboard already uses for
  its own zone-snapshot endpoints, so the Next.js Zones UI can reuse the same
  draw/edit interaction model against a cached rather than live frame.
- **Live video and live per-frame boxes stay local-only:** `/video_feed` and
  `/api/live-tracks` continue to be served directly by the local `webapp.py`
  instance and are not proxied through Railway or exposed on Vercel. Anyone
  wanting live video must be on the LAN.

## Auth

The cloud dashboard is internet-facing, so HTTP Basic (fine for LAN-only, per
`SETUP.md`) is replaced for cloud mode:

- New `POST /api/login` on the cloud API checks `DASHBOARD_USER` /
  `DASHBOARD_PASSWORD` (same env-based credentials as today) and issues a
  short-lived signed session (httpOnly, secure cookie; `SameSite=None` since
  Vercel and Railway are different origins).
- Login attempts are rate-limited (fixed-window counter, e.g. per-IP) to
  address the brute-force gap `SETUP.md` calls out for Basic auth.
- All cloud-mode `/api/*` routes require a valid session; missing/expired
  session returns 401.
- Local mode is unchanged: HTTP Basic, LAN trust model, as documented in
  `SETUP.md`.
- TLS is automatic on both Vercel and Railway by default, satisfying
  `SETUP.md`'s TLS requirement without extra work.

This satisfies all three of `SETUP.md`'s pre-internet-exposure requirements:
TLS, session/token auth, and login rate-limiting.

## Frontend (Vercel)

- New Next.js app in a new top-level `frontend/` directory.
- Pages/components for Occupancy, History, Zones (with create/edit against
  the cached snapshot), and People — mirroring the local dashboard's
  equivalent tabs minus Live video.
- A small API client wraps `fetch`, prefixes all calls with
  `NEXT_PUBLIC_API_BASE`, and sends `credentials: "include"` so the session
  cookie is attached.
- Live-ish updates (occupancy/zone-status) reuse the existing SSE endpoint
  (`/api/stream`) pointed at the absolute Railway URL — `EventSource` works
  cross-origin, so no redesign needed there, just a login screen gating
  access before the dashboard loads.

## Error handling

- **Detector offline:** Railway's Postgres simply stops receiving new events;
  the cloud dashboard shows the last known occupancy state. This is correct,
  expected behavior — no special-casing needed.
- **No snapshot uploaded yet** (fresh deploy, or detector offline long enough
  that the snapshot is very stale): `GET /api/zone-snapshot` returns 404 on
  Railway; the Vercel Zones tab shows a "no snapshot available yet" state
  instead of a broken image/canvas.
- **Cloud-mode routes that don't exist in cloud mode** (`/video_feed`,
  `/api/live-tracks`, `/api/start`, `/api/stop`): simply not mounted when
  `DEPLOYMENT_MODE=cloud`, so they 404 rather than requiring runtime checks
  scattered through handlers.

## Testing

- Extend `tests/` with: a cloud-mode route-mounting test (asserting
  start/stop/video/live-tracks are absent/404 under
  `DEPLOYMENT_MODE=cloud`), login/session/rate-limit tests, and a
  snapshot-upload round-trip test (`POST` then `GET` returns the same image
  and correct metadata).
- Manual smoke test before considering this done: deploy to a Railway staging
  service + a Vercel preview, point a real local detector at the Railway
  Postgres instance, and confirm occupancy/history/zones/people/login/reset
  all work end-to-end, and that the snapshot upload loop populates the cloud
  Zones tab within one upload interval.
