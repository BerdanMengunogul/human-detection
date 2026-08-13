# People Counter Dashboard (NestJS + Prisma + Next.js) — Design

## Purpose

Build a standalone web dashboard, backed by NestJS/Prisma/PostgreSQL/Socket.io with a Next.js frontend, that visualizes real-time and historical detection data from the existing Python human-detection pipeline. Built for a job-application portfolio, to demonstrate a full-stack real-time system on top of already-working computer-vision infrastructure.

The dashboard is a separate repo (`people-counter-nestjs`, sibling to `human-detection`) fed by the Python pipeline over HTTP, mirroring the existing fire-and-forget POST pattern already used by `notify.py` for ntfy alerts. The Python side is the source of truth for detections; the new system only ingests, stores, and visualizes.

## Architecture

```
[Python pipeline: pipeline.py / webapp.py]
        |  POST /api/ingest/detection  (fire-and-forget, X-Api-Key header)
        v
[NestJS backend]
  - Prisma -> PostgreSQL (Camera, Zone, Person, DetectionEvent)
  - In-memory occupancy state (Map<zoneId, Set<personId>>)
  - Socket.io gateway (broadcasts live updates)
        |
        v
[Next.js dashboard]
  - REST calls for history/config
  - Socket.io client for live occupancy/detection/alert feed
```

The Python pipeline keeps its existing local behavior (local SQLite/logging, ntfy alerts via `notify.py`) unchanged; it additionally POSTs detection/zone-transition events to the NestJS ingest endpoint. NestJS is purely additive — if it's down, the Python pipeline is unaffected (fire-and-forget, short timeout, errors swallowed on the Python side, same as the existing ntfy call).

Occupancy state (who is currently in which zone) is *not* persisted as its own table — it's derived in-memory in NestJS from the stream of enter/leave events and broadcast live. Only discrete events (`DetectionEvent` rows) are persisted to Postgres, since that's what's needed for history; current occupancy is always reconstructable from recent events plus the live in-memory map.

## Data Model (Prisma Schema)

```prisma
model Camera {
  id        String   @id            // matches existing CAMERA_RTSP_URL-derived identifier
  name      String
  createdAt DateTime @default(now())
}

model Zone {
  id   String @id                   // raw short-hex id from door_zones.json, reused directly
  name String
  task ZoneTask @default(NONE)
}

model Person {
  id        Int      @id            // matches Python's PersonGallery-assigned integer person_id directly
  name      String?
  firstSeen DateTime @default(now())
  events    DetectionEvent[]
}

model DetectionEvent {
  id         Int      @id @default(autoincrement())
  personId   Int
  person     Person   @relation(fields: [personId], references: [id])
  cameraId   String
  camera     Camera   @relation(fields: [cameraId], references: [id])
  zoneId     String?
  zone       Zone?    @relation(fields: [zoneId], references: [id])
  zoneEvent  ZoneEvent?
  timestamp  DateTime
}

enum ZoneTask {
  ALERT_ENTRY
  ALERT_PRESENCE
  ALERT_LEAVE
  IGNORE
  NONE
}

enum ZoneEvent {
  ENTER
  LEAVE
}
```

Design notes:
- `Person.id` and `Zone.id` reuse the Python pipeline's existing integer person IDs and string zone IDs directly — no translation/mapping layer needed, since both systems read the same identifiers.
- `DetectionEvent.zoneId`/`zoneEvent` are optional: most detections are plain (no zone transition that frame).
- One `Camera` row is seeded at setup time, matching the single existing `CAMERA_RTSP_URL` env var. Multi-camera support is out of scope (YAGNI — the Python pipeline is single-camera today).
- No standalone `Occupancy` table. Current occupancy is in-memory only in the NestJS process and broadcast via Socket.io; it is reconstructable from recent `DetectionEvent` rows if the process restarts.

## API Endpoints

REST for CRUD/history/config, Socket.io for live push. Base path `/api`.

```
GET   /api/events?limit=50&cursor=...&personId=&zoneId=   # paginated detection history
GET   /api/people                                          # list known people (id, name, last seen)
GET   /api/people/:id                                       # person detail + recent events
PATCH /api/people/:id                                       # rename a person
GET   /api/zones                                             # list zones + current task config
PATCH /api/zones/:id                                          # update task (none/alert_entry/alert_presence/alert_leave/ignore)
GET   /api/occupancy                                          # current occupancy snapshot (in-memory, not DB-backed)
POST  /api/ingest/detection                                   # called by Python pipeline
```

`POST /api/ingest/detection` is the single write path from Python (fire-and-forget, notify.py-style). Everything else is read/admin for the dashboard. All endpoints require a shared-secret header (`X-Api-Key`) since the service is LAN-exposed.

### Ingest payload

```json
{
  "personId": 3,
  "cameraId": "front-door",
  "timestamp": "2026-08-03T10:15:00Z",
  "zoneId": "a1b2c3",
  "zoneEvent": "enter"
}
```

`zoneId`/`zoneEvent` are omitted for plain detections (person seen, no zone boundary crossed that frame). `zoneEvent` is `"enter"` or `"leave"` only — these are edge-triggered, discrete occurrences worth their own row.

`alert_presence` is **not** sent as a `zoneEvent` on every frame someone stands in a zone (it's a level/state, not a discrete event, and would otherwise flood `DetectionEvent` with redundant rows). Instead, presence is derived server-side from the in-memory occupancy map (`occupancy.get(zoneId).size > 0`), which is already kept up to date from `enter`/`leave` events.

## WebSocket Events (Socket.io)

```
"occupancy:update"  -> { zoneId, occupants: number[], task, alert: boolean }
  // sent whenever a zone's occupancy set changes (on enter/leave)

"detection:new"     -> { personId, cameraId, timestamp, zoneId? }
  // sent on every ingested detection, for a live feed view

"zone:alert"        -> { zoneId, zoneName, task, alert: true }
  // sent only on false -> true alert-state transitions
```

`zone:alert`'s edge-detection mirrors the existing pattern in `webapp.py`'s `_last_zone_alert` dict (alert fires once per transition into alert state, not every frame it remains true) — ported into NestJS as in-memory server state instead of a Python module-level dict.

## Error Handling

- **Ingest endpoint**: validate body (personId/cameraId/timestamp required, zoneEvent enum-checked if present) — 400 on bad payloads, 401 on missing/bad `X-Api-Key`. Since Python's caller discards the response (fire-and-forget, same as `notify.py`), errors are only logged server-side for debugging; no retry logic on either end.
- **Unknown zone/camera references**: if `zoneId`/`cameraId` doesn't match a seeded row, log a warning and store the detection without that reference rather than rejecting the whole request — a bad reference shouldn't blind the dashboard to a real detection.
- **Database unavailable**: wrap Prisma writes in try/catch; on failure return 503 (ignored by the Python caller) and log. No in-memory queue or buffering — YAGNI for a single-user LAN portfolio app.
- **Socket.io broadcast failures**: best-effort; a disconnected dashboard client just misses the event and re-syncs via `GET /api/occupancy` on reconnect. No event replay/backlog.
- **Zone task updates (`PATCH /api/zones/:id`)**: validate `task` against the `ZoneTask` enum, 400 if invalid.

Nothing here needs retries, queues, or circuit breakers — consistent with the existing Python pipeline's own fire-and-forget philosophy toward notifications.

## Testing

- **Unit tests**: occupancy service — enter/leave updates the in-memory map correctly; presence/alert derivation (`occupancy.size > 0`); edge-detection for `zone:alert` (fires only on false->true, not every frame it stays true). This mirrors a bug class already encountered in the Python pipeline (the EXIT_GRACE_FRAMES/IDENTIFY_DELAY_FRAMES race), so it gets direct, deliberate coverage.
- **Integration tests**: against a test Postgres/Prisma test-db, hit `POST /api/ingest/detection` with plain/enter/leave/missing-reference/bad-api-key payloads and assert resulting DB rows and emitted Socket.io events.
- **E2E/manual**: point the real Python pipeline's ingest POST at a local NestJS instance and confirm the Next.js dashboard updates live. This is the realistic smoke test for a single-user LAN portfolio app.
- **Out of scope**: load testing, contract testing, mutation testing.

## Out of Scope

- Multi-camera support.
- Historical occupancy persistence (only discrete events are stored; occupancy is always live/derived).
- Authentication beyond a single shared API key for the ingest endpoint (no user accounts/roles — LAN-only, single operator).
- Retry/replay/durability guarantees for ingest traffic.
