# People Counter Dashboard (NestJS + Prisma + Next.js) — Implementation Plan

**Sub-skill:** Use `subagent-driven-development` to execute this plan (multiple independently-testable tasks across two repos; each task is small enough for a fresh subagent with no prior context).

**Spec:** `docs/superpowers/specs/2026-08-03-people-counter-nestjs-design.md` (human-detection repo)

## Goal

Ship a real-time people-counter dashboard: a NestJS/Prisma/PostgreSQL/Socket.io backend that ingests detection events over HTTP from the existing Python pipeline, plus a Next.js frontend that visualizes live occupancy and history. Additionally, add a new `alert_leave` zone task to the Python pipeline (mirroring the existing `alert_entry`/`entered` pattern with a new `left` edge signal) and wire the Python side to POST events into the new backend.

## Architecture

```
[Python pipeline: pipeline.py / webapp.py]
        |  POST /api/ingest/detection  (fire-and-forget, X-Api-Key header)
        v
[NestJS backend]  (new repo: people-counter-nestjs)
  - Prisma -> PostgreSQL (Camera, Zone, Person, DetectionEvent)
  - In-memory occupancy state (Map<zoneId, Set<personId>>)
  - Socket.io gateway (broadcasts live updates)
        |
        v
[Next.js dashboard]  (same repo, apps/web)
  - REST calls for history/config
  - Socket.io client for live occupancy/detection/alert feed
```

Two repos are touched:
- `people-counter-nestjs` (new, currently an empty git repo with no commits) — the whole backend + frontend.
- `human-detection` (this repo) — small, additive changes: new `alert_leave` zone task, new ingest POST calls, new config entries.

## Tech Stack

- **Backend**: NestJS 10, Prisma 5, PostgreSQL, Socket.io (via `@nestjs/websockets` + `@nestjs/platform-socket.io`), Jest (unit + e2e).
- **Frontend**: Next.js 14 (App Router), TypeScript, `socket.io-client`, plain CSS (no UI framework — portfolio-scoped, YAGNI).
- **Monorepo layout**: npm workspaces, two packages — `apps/api` (NestJS) and `apps/web` (Next.js) — inside `people-counter-nestjs`.
- **Python side**: stdlib only (`threading`, `urllib.request`, `json`), matching `notify.py`'s existing dependency-free pattern. No new pip packages.

## Global Constraints

- No placeholders, no TODOs, no stubbed logic — every task ends in working, tested code.
- Fire-and-forget from Python: the ingest POST must never block or crash the detection loop. Same pattern as `notify.py` (`_post` in `notify.py:1-30`): background daemon thread, short timeout, swallow `URLError`/`TimeoutError`, log and continue.
- NestJS is purely additive — if it's down, Python's existing local behavior (SQLite/Postgres event log, ntfy alerts) is unaffected.
- `Person.id` and `Zone.id` reuse the Python pipeline's existing identifiers directly (integer person IDs, string zone hex IDs) — no translation layer.
- All NestJS endpoints require a shared-secret `X-Api-Key` header (LAN-exposed service, single operator, no user accounts).
- Every task follows red-green: write a failing test, verify it fails, implement, verify it passes, commit.

## File Structure (people-counter-nestjs, new repo)

```
people-counter-nestjs/
  package.json                 # npm workspaces root
  apps/
    api/
      package.json
      prisma/
        schema.prisma
      src/
        main.ts
        app.module.ts
        prisma/
          prisma.service.ts
          prisma.module.ts
        auth/
          api-key.guard.ts
        ingest/
          ingest.module.ts
          ingest.controller.ts
          ingest.service.ts
          dto/ingest-detection.dto.ts
        occupancy/
          occupancy.module.ts
          occupancy.service.ts
          occupancy.gateway.ts
        events/
          events.module.ts
          events.controller.ts
          events.service.ts
        people/
          people.module.ts
          people.controller.ts
          people.service.ts
        zones/
          zones.module.ts
          zones.controller.ts
          zones.service.ts
      test/
        ingest.e2e-spec.ts
    web/
      package.json
      app/
        layout.tsx
        page.tsx                 # occupancy overview
        people/page.tsx
        events/page.tsx
      lib/
        socket.ts
        api.ts
```

---

## Task 1: Scaffold the monorepo and Prisma schema

**Files:**
- Create: `people-counter-nestjs/package.json`, `people-counter-nestjs/apps/api/package.json`, `people-counter-nestjs/apps/api/prisma/schema.prisma`, `people-counter-nestjs/apps/api/src/main.ts`, `people-counter-nestjs/apps/api/src/app.module.ts`, `people-counter-nestjs/apps/api/src/prisma/prisma.service.ts`, `people-counter-nestjs/apps/api/src/prisma/prisma.module.ts`, `people-counter-nestjs/apps/api/.env.example`, `people-counter-nestjs/.gitignore`
- Test: `people-counter-nestjs/apps/api/test/app.e2e-spec.ts`

**Interfaces:**
- Produces: a running NestJS app on `PORT` (default 3001) with `GET /api/health` returning `{ status: "ok" }`, and a Prisma client generated from `schema.prisma` connected to Postgres via `DATABASE_URL`.

**Steps:**
1. `npm init -y` at repo root; edit `package.json` to declare `"workspaces": ["apps/api", "apps/web"]` and add root scripts (`dev:api`, `dev:web`).
2. `cd apps/api && npx @nestjs/cli new . --skip-git --package-manager npm` (or manually scaffold `main.ts`/`app.module.ts` if the CLI prompts interactively — write them directly per Nest's standard minimal structure: `main.ts` bootstraps `AppModule` with global prefix `api`; `app.module.ts` imports `PrismaModule`).
3. `npm install prisma @prisma/client --workspace=apps/api`; `npx prisma init --datasource-provider postgresql` inside `apps/api`.
4. Write `schema.prisma` exactly per the design spec (`Camera`, `Zone`, `Person`, `DetectionEvent` models; `ZoneTask` enum with `ALERT_ENTRY/ALERT_PRESENCE/ALERT_LEAVE/IGNORE/NONE`; `ZoneEvent` enum with `ENTER/LEAVE`), datasource `env("DATABASE_URL")`.
5. Write `PrismaService` (extends `PrismaClient`, implements `OnModuleInit`/`OnModuleDestroy`, calls `this.$connect()`/`this.$disconnect()`) and `PrismaModule` (`@Global()`, exports `PrismaService`).
6. Write `.env.example` with `DATABASE_URL="postgresql://postgres:password@localhost:5432/people_counter"`, `API_KEY="changeme"`, `PORT=3001`. Add `.env`, `node_modules`, `dist` to `.gitignore`.
7. Add a trivial `GET /api/health` route directly in `AppController` (create if the Nest CLI didn't scaffold one) returning `{ status: "ok" }`.
8. Write `test/app.e2e-spec.ts`: boots the Nest app via `Test.createTestingModule`, calls `GET /api/health` via `supertest`, asserts `200` and `{ status: "ok" }`. Run it — it should fail (no app yet) before step 2-7, pass after.
9. `npx prisma migrate dev --name init` against a local Postgres (create the `people_counter` DB first) to verify the schema applies cleanly.
10. `git add -A && git commit -m "Scaffold NestJS API with Prisma schema"`.

---

## Task 2: API key auth guard

**Files:**
- Create: `apps/api/src/auth/api-key.guard.ts`, `apps/api/src/auth/api-key.guard.spec.ts`
- Modify: `apps/api/src/app.module.ts` (register guard globally via `APP_GUARD`)

**Interfaces:**
- Consumes: `process.env.API_KEY`, request header `X-Api-Key`.
- Produces: a global `CanActivate` guard that throws `UnauthorizedException` (401) when the header is missing or doesn't match `API_KEY`.

**Steps:**
1. Write `api-key.guard.spec.ts`: unit-test `ApiKeyGuard.canActivate` with a mocked `ExecutionContext` — case A (correct header) returns `true`; case B (missing header) throws `UnauthorizedException`; case C (wrong header) throws `UnauthorizedException`. Run — fails (no guard file yet).
2. Implement `ApiKeyGuard` reading `process.env.API_KEY` and comparing against `request.headers['x-api-key']`.
3. Run the spec — passes.
4. Register `{ provide: APP_GUARD, useClass: ApiKeyGuard }` in `app.module.ts` providers.
5. Update `test/app.e2e-spec.ts`'s health check to send the correct `X-Api-Key` header (read from `process.env.API_KEY` set in test env) — re-run full e2e suite, confirm still passes.
6. Commit: `git commit -m "Add global X-Api-Key auth guard"`.

---

## Task 3: Occupancy service (in-memory state + edge detection)

**Files:**
- Create: `apps/api/src/occupancy/occupancy.service.ts`, `apps/api/src/occupancy/occupancy.service.spec.ts`, `apps/api/src/occupancy/occupancy.module.ts`

**Interfaces:**
- Produces: `OccupancyService` with:
  - `handleEnter(zoneId: string, personId: number): { occupants: number[], alertEdge: boolean }`
  - `handleLeave(zoneId: string, personId: number): { occupants: number[], alertEdge: boolean }`
  - `getSnapshot(): Record<string, number[]>` — current `zoneId -> personId[]` map.
  - `getZoneOccupants(zoneId: string): number[]`
  - Internally tracks `Map<string, Set<number>>` for occupants and `Map<string, boolean>` for last-alert-state per zone (mirrors `webapp.py`'s `_last_zone_alert` dict, `webapp.py:432`), keyed off the zone's configured task fetched from `ZonesService` (Task 6) — but for this task, `alertEdge` is computed purely from occupancy transition (non-empty-set false→true) so the service has no dependency on task config yet; task-aware alert semantics (`alert_entry` vs `alert_presence` vs `alert_leave`) are composed in the ingest service (Task 4), which is the one that knows the event type and zone task.

**Steps:**
1. Write `occupancy.service.spec.ts` covering:
   - `handleEnter` adds personId to the zone's set; `getZoneOccupants` reflects it.
   - `handleLeave` removes personId; if the set becomes empty, `getZoneOccupants` returns `[]`.
   - `handleLeave` on a personId not present is a no-op (no throw).
   - Two entries then one leave still leaves the zone occupied (tests presence isn't cleared prematurely — this mirrors the exact bug class already hit in the Python pipeline's EXIT_GRACE_FRAMES/IDENTIFY_DELAY_FRAMES race).
   - `getSnapshot` returns all zones' current occupants keyed by zoneId.
2. Run spec — fails (no service).
3. Implement `OccupancyService` with the two `Map`s described above, plain synchronous methods (no async needed — pure in-memory).
4. Run spec — passes.
5. Write `OccupancyModule` exporting `OccupancyService`.
6. Commit: `git commit -m "Add in-memory occupancy service with unit tests"`.

---

## Task 4: Ingest endpoint (DTO, validation, service, controller)

**Files:**
- Create: `apps/api/src/ingest/dto/ingest-detection.dto.ts`, `apps/api/src/ingest/ingest.service.ts`, `apps/api/src/ingest/ingest.service.spec.ts`, `apps/api/src/ingest/ingest.controller.ts`, `apps/api/src/ingest/ingest.module.ts`
- Modify: `apps/api/src/app.module.ts` (import `IngestModule`, `OccupancyModule`)
- Test: `apps/api/test/ingest.e2e-spec.ts`

**Interfaces:**
- Consumes: `POST /api/ingest/detection` body `{ personId: number, cameraId: string, timestamp: string, zoneId?: string, zoneEvent?: "enter"|"leave" }` (per spec's Ingest payload section).
- Produces:
  - Writes a `DetectionEvent` row via Prisma (upserting the `Person` row by id if it doesn't exist — `prisma.person.upsert({ where: { id: personId }, create: { id: personId }, update: {} })` — since Python's person IDs are assigned independently of NestJS).
  - On `zoneEvent: "enter"`: calls `occupancyService.handleEnter`, looks up the zone's `task` via Prisma, and if `task === "ALERT_ENTRY"` and this is a false→true occupancy transition, emits `zone:alert`; always emits `occupancy:update`.
  - On `zoneEvent: "leave"`: calls `occupancyService.handleLeave`, and if `task === "ALERT_LEAVE"` and the zone just became empty (true→false transition on this specific edge), emits `zone:alert`; always emits `occupancy:update`.
  - For `task === "ALERT_PRESENCE"`: presence is a level, not an edge — after any enter/leave, if `occupants.length > 0` and previous state was empty (or vice versa), emit `zone:alert` (reuses the same last-alert-state Map from Task 3, keyed separately per task type is unnecessary — one `alertEdge` boolean per zone is sufficient since only one task applies to a zone at a time, per the `Zone.task` field being singular).
  - Always emits `detection:new` for every ingested row (plain or zone-tagged).
  - Unknown `zoneId`/`cameraId` (no matching row): log a warning, persist the `DetectionEvent` with that FK column left `null` rather than rejecting (per spec's Error Handling section).
  - 400 on missing `personId`/`cameraId`/`timestamp` or invalid `zoneEvent` enum value (via `class-validator` decorators on the DTO + a global `ValidationPipe`).
  - 503 (with server-side log) if the Prisma write throws (DB unavailable).

**Steps:**
1. Write `ingest-detection.dto.ts` with `class-validator` decorators: `@IsInt() personId`, `@IsString() cameraId`, `@IsISO8601() timestamp`, `@IsOptional() @IsString() zoneId`, `@IsOptional() @IsIn(['enter','leave']) zoneEvent`.
2. Enable a global `ValidationPipe({ whitelist: true, forbidNonWhitelisted: true })` in `main.ts`.
3. Write `ingest.service.spec.ts` with a mocked `PrismaService` and mocked `OccupancyService`/gateway-emit function, covering:
   - Plain detection (no zoneId/zoneEvent): creates `DetectionEvent` with null zone fields, no occupancy call, emits `detection:new` only.
   - `zoneEvent: "enter"` on a zone with `task = ALERT_ENTRY`, first entrant: occupancy transitions empty→non-empty, emits `occupancy:update` and `zone:alert`.
   - Same zone, second entrant (already occupied): occupancy stays non-empty→non-empty, emits `occupancy:update` only (no duplicate `zone:alert`) — this directly covers the edge-detection bug class called out in the spec's Testing section.
   - `zoneEvent: "leave"` on a zone with `task = ALERT_LEAVE`, last occupant leaving: emits `zone:alert`.
   - Unknown `zoneId`: logs a warning (spy on `Logger.warn`), still creates the `DetectionEvent` row with `zoneId: null`.
   - Prisma write throws: service throws a `ServiceUnavailableException` (mapped to 503 by Nest).
4. Run spec — fails (no service).
5. Implement `IngestService.ingest(dto)` per the behaviors above.
6. Run spec — passes.
7. Write `IngestController` with `@Post('ingest/detection')`, `@UseGuards` inherited globally from Task 2, delegating to `IngestService`.
8. Write `IngestModule` importing `PrismaModule`, `OccupancyModule`, providing `IngestService`/`IngestController`.
9. Write `test/ingest.e2e-spec.ts` against a real test Postgres (`DATABASE_URL` pointed at a `people_counter_test` DB, migrated via `prisma migrate deploy` in a `beforeAll`): POST plain/enter/leave/missing-field/bad-api-key payloads, assert HTTP status codes and resulting `DetectionEvent` rows via `prisma.detectionEvent.findMany`.
10. Run e2e — passes.
11. Commit: `git commit -m "Add detection ingest endpoint with occupancy/alert side effects"`.

---

## Task 5: Socket.io gateway

**Files:**
- Create: `apps/api/src/occupancy/occupancy.gateway.ts`, `apps/api/src/occupancy/occupancy.gateway.spec.ts`
- Modify: `apps/api/src/occupancy/occupancy.module.ts` (provide gateway), `apps/api/src/ingest/ingest.service.ts` (inject gateway instead of a bare emit function stub from Task 4 — replace that placeholder wiring with the real gateway)

**Interfaces:**
- Produces: `OccupancyGateway` (`@WebSocketGateway({ namespace: '/', cors: true })`) exposing `emitOccupancyUpdate(zoneId, occupants, task, alert)`, `emitDetectionNew(payload)`, `emitZoneAlert(zoneId, zoneName, task)` — each calling `this.server.emit(eventName, payload)` per the three event shapes in the spec's WebSocket Events section.
- On client connect, no auth handshake is required beyond the existing LAN-only assumption (spec explicitly scopes auth to the ingest endpoint's `X-Api-Key`; Socket.io connections are read-only broadcast, out of scope for auth per spec's "Out of Scope" list).

**Steps:**
1. Write `occupancy.gateway.spec.ts`: instantiate the gateway directly, stub `gateway.server = { emit: jest.fn() }`, call each `emit*` method, assert `server.emit` was called with the exact event name (`"occupancy:update"`, `"detection:new"`, `"zone:alert"`) and payload shape.
2. Run — fails (no gateway).
3. Implement `OccupancyGateway`.
4. Run — passes.
5. Wire `IngestService` to call `occupancyGateway.emit*` methods instead of any placeholder from Task 4 (Task 4's spec mocks the gateway calls directly — this step makes it the real class via DI).
6. Re-run `ingest.service.spec.ts` and `ingest.e2e-spec.ts` — still passing (mocks/DI unaffected).
7. Commit: `git commit -m "Add Socket.io gateway for occupancy/detection/alert broadcasts"`.

---

## Task 6: Zones, People, Events REST modules

**Files:**
- Create: `apps/api/src/zones/zones.module.ts`, `zones.controller.ts`, `zones.service.ts`, `zones.service.spec.ts`
- Create: `apps/api/src/people/people.module.ts`, `people.controller.ts`, `people.service.ts`, `people.service.spec.ts`
- Create: `apps/api/src/events/events.module.ts`, `events.controller.ts`, `events.service.ts`, `events.service.spec.ts`
- Modify: `apps/api/src/app.module.ts` (import all three)
- Test: `apps/api/test/zones.e2e-spec.ts`, `apps/api/test/people.e2e-spec.ts`, `apps/api/test/events.e2e-spec.ts`

**Interfaces:**
- `GET /api/zones` — list all `Zone` rows.
- `PATCH /api/zones/:id` — body `{ task: ZoneTask }`, validated against the enum (400 if invalid per spec), updates and returns the row.
- `GET /api/occupancy` — returns `OccupancyService.getSnapshot()` joined with zone names/tasks from Prisma (in-memory snapshot, not a DB query per spec — "not DB-backed").
- `GET /api/people` — list `Person` rows with a `lastSeen` computed from their most recent `DetectionEvent.timestamp` (Prisma `orderBy`/`take: 1` per person, or a single grouped query).
- `GET /api/people/:id` — person + last 50 `DetectionEvent` rows, 404 if not found.
- `PATCH /api/people/:id` — body `{ name: string }`, updates and returns the row, 404 if not found.
- `GET /api/events?limit=&cursor=&personId=&zoneId=` — cursor-paginated `DetectionEvent` list (Prisma cursor pagination on `id`), optional filters.

**Steps:**
1. Write `zones.service.spec.ts` (mocked Prisma): `findAll` returns rows; `updateTask` with an invalid enum value throws `BadRequestException`; with a valid value calls `prisma.zone.update` and returns the row; `updateTask` on a missing id throws `NotFoundException`. Run — fails, implement `ZonesService`, run — passes.
2. Write `ZonesController` (`GET /zones`, `PATCH /zones/:id`) + `ZonesModule`. Add `occupancy` endpoint here too: `GET /zones/occupancy` was considered but spec fixes the path as `GET /api/occupancy` (top-level, not nested under zones) — so instead create this route directly on `ZonesController` via a second controller path decorator `@Controller('occupancy')` in the same module, or a small dedicated `OccupancyController` in `occupancy.module.ts` calling `OccupancyService.getSnapshot()` plus a Prisma zone lookup for names/tasks. Choose the latter (keeps `OccupancyModule` self-contained) — create `apps/api/src/occupancy/occupancy.controller.ts` + spec, same red-green cycle.
3. Write `people.service.spec.ts` (mocked Prisma): `findAll`, `findOne` (404 case), `update` (404 case, success case). Run — fails, implement `PeopleService`, run — passes. Write `PeopleController` + `PeopleModule`.
4. Write `events.service.spec.ts` (mocked Prisma): cursor pagination shape, filter-by-personId, filter-by-zoneId. Run — fails, implement `EventsService`, run — passes. Write `EventsController` + `EventsModule`.
5. Write the three e2e spec files against the test Postgres DB, seeding rows via Prisma in `beforeEach`, hitting each real HTTP route, asserting response shapes and status codes (including the `PATCH /zones/:id` 400-on-invalid-task and 404-on-missing-id cases).
6. Run full e2e suite — passes.
7. Commit: `git commit -m "Add zones/people/events/occupancy REST endpoints"`.

---

## Task 7: Next.js dashboard scaffold + occupancy overview page

**Files:**
- Create: `apps/web/package.json`, `apps/web/app/layout.tsx`, `apps/web/app/page.tsx`, `apps/web/lib/socket.ts`, `apps/web/lib/api.ts`, `apps/web/.env.local.example`

**Interfaces:**
- Consumes: `NEXT_PUBLIC_API_BASE_URL` (e.g. `http://localhost:3001/api`), `NEXT_PUBLIC_API_KEY` for REST calls, `NEXT_PUBLIC_WS_URL` for the Socket.io client.
- Produces: `lib/api.ts` — a thin `fetch` wrapper attaching `X-Api-Key`; `lib/socket.ts` — a singleton `socket.io-client` instance; `app/page.tsx` — server-fetches `GET /api/occupancy` for initial render, then subscribes client-side to `occupancy:update` and `zone:alert` to update live.

**Steps:**
1. `npx create-next-app@latest apps/web --typescript --app --no-tailwind --no-eslint --src-dir=false --import-alias "@/*"` (non-interactive flags to avoid prompts).
2. `npm install socket.io-client --workspace=apps/web`.
3. Write `.env.local.example` with the three vars above; add `.env.local` to `.gitignore`.
4. Write `lib/api.ts`: exported `apiFetch(path, init?)` prefixing `NEXT_PUBLIC_API_BASE_URL` and injecting the `X-Api-Key` header.
5. Write `lib/socket.ts`: exported `getSocket()` returning a module-level singleton `io(NEXT_PUBLIC_WS_URL)`.
6. Write `app/page.tsx` as a client component (`"use client"`): on mount, `apiFetch('/occupancy')` for initial state into `useState`; `useEffect` subscribing `getSocket().on('occupancy:update', ...)` and `.on('zone:alert', ...)` to update state; render a simple table of zone name / occupant count / alert flag.
7. Manual smoke test: `npm run dev` in `apps/web` (with `apps/api` also running against the local Postgres from Task 1), confirm the page loads and shows zones from a seeded DB. This is inherently a UI change, so it must be verified in a browser per project convention — do not mark this task done from type-checking alone.
8. Commit: `git commit -m "Scaffold Next.js dashboard with live occupancy overview"`.

---

## Task 8: People and Events pages

**Files:**
- Create: `apps/web/app/people/page.tsx`, `apps/web/app/events/page.tsx`

**Interfaces:**
- `app/people/page.tsx`: fetches `GET /api/people`, renders a list with name/lastSeen, each linking to a detail view (`GET /api/people/:id`) inline (client-side fetch on click, or a nested `[id]/page.tsx` — use a nested dynamic route `app/people/[id]/page.tsx` for a clean URL-addressable detail page) with an editable name field calling `PATCH /api/people/:id`.
- `app/events/page.tsx`: fetches `GET /api/events` with cursor pagination, subscribes to `detection:new` via the socket to prepend new rows live.

**Steps:**
1. Write `app/people/page.tsx` (client component): fetch + render list, link to `/people/[id]`.
2. Write `app/people/[id]/page.tsx`: fetch person detail + recent events; a controlled `<input>` + save button calling `apiFetch('/people/:id', { method: 'PATCH', body })`.
3. Write `app/events/page.tsx`: fetch first page via `GET /api/events?limit=50`; "load more" button advancing the `cursor` param; socket subscription to `detection:new` prepending rows (dedupe by event id against a `Set` in state to avoid double-counting a row that arrives both via live push and a subsequent page fetch).
4. Manual smoke test in browser: seed a few `DetectionEvent`/`Person` rows, confirm listing, detail view, rename, and live event feed all work.
5. Commit: `git commit -m "Add people and events dashboard pages"`.

---

## Task 9: Python — `alert_leave` zone task in `pipeline.py`

**Files:**
- Modify: `pipeline.py` (State getter/setter at lines 256-298, zone_status computation at lines 773-782)
- Test: `test_pipeline_zone_status.py` (new, if no existing test harness covers `pipeline.py`'s zone logic — check for one first; if `zones.py` or `pipeline.py` has no existing unit tests, create a minimal one scoped to just this new computation, following the project's existing test conventions if any exist under a `tests/` dir)

**Interfaces:**
- `State.update(..., zone_status=...)` and `State.zone_status()` currently pass/return `{"occupants": set, "entered": bool}` per zone id (`pipeline.py:281-284`, `293-297`). Add a third key `"left": bool`.
- The per-frame computation block (`pipeline.py:773-782`) currently computes `entered = bool(occupants_now - occupants_prev)`. Add `left = bool(occupants_prev - occupants_now)` alongside it.

**Steps:**
1. Check for an existing test directory/convention (`Glob` for `test_*.py` or `tests/` in the repo root). If none exists, create `test_pipeline_zone_status.py` at the repo root importing the specific pure computation — since it's inline in a large loop, factor the zone_status computation into a small pure function first: extract lines 773-779 into a module-level function `_compute_zone_status(web_zones, zone_occupants_now, zone_occupants_prev)` returning the per-zone dict, so it's unit-testable without running the whole pipeline loop.
2. Write the test: given `occupants_prev={'z1': {1,2}}`, `occupants_now={'z1': {2,3}}`, assert `entered=True` (3 arrived), `left=True` (1 left) for zone `z1` simultaneously — this is the key new case proving `entered`/`left` aren't mutually exclusive. Also cover: no change (both False), zone newly occupied (entered=True, left=False), zone newly emptied (entered=False, left=True).
3. Run — fails (function doesn't exist yet / doesn't have `left` key).
4. Implement `_compute_zone_status` with the `left` key added, and call it from the main loop in place of the current inline block.
5. Run — passes.
6. Update `State.update`'s zone_status setter (`pipeline.py:282-285`) and `State.zone_status()` getter (`pipeline.py:293-297`) to carry the `"left"` key through the copy (currently only copies `"occupants"` and `"entered"` — add `"left": info["left"]` in both dict comprehensions).
7. Manual verification: run the pipeline against the live camera (or a recorded clip if available), create a zone with a temporary debug print of `zone_status`, walk into and out of it, confirm `left` flips true on exit and false again next frame — since this is a stateful, hardware-adjacent change, don't rely on the unit test alone.
8. Commit: `git commit -m "Add left signal to zone_status for alert_leave support"`.

---

## Task 10: Python — `alert_leave` in `webapp.py` + config entries for NestJS ingest

**Files:**
- Modify: `webapp.py` (`VALID_ZONE_TASKS` at line ~396, `_build_zone_status_payload()` at lines ~424-455)
- Modify: `config.py` (`_SCHEMA` list)
- Modify: `static/app.js` (`TASK_LABELS` at lines ~414-417, the reset-to-`"none"` line at 402)
- Modify: `templates/index.html` (`<select id="zone-task-select">` options at lines 117-121)
- Test: none new (this task is config/glue + two small conditionals; covered by the existing manual zone-task-switching flow used for `alert_entry`/`alert_presence`, exercised in Task 12's E2E step)

**Interfaces:**
- `VALID_ZONE_TASKS` gains `"alert_leave"`.
- `_build_zone_status_payload()` gains `elif task == "alert_leave": alert = info.get("left", False)` alongside the existing `alert_presence`/`alert_entry` branches (`webapp.py:432-436`).
- `config.py`'s `_SCHEMA` gains two new entries following the exact `(name, type, default)` tuple pattern used for `NTFY_SERVER`/`NTFY_TOPIC` (`config.py:113-115`): `("NESTJS_INGEST_URL", str, "")` and `("NESTJS_API_KEY", str, "")`. Both default to empty string (disables the feature, same convention as `NTFY_TOPIC`), loaded the standard layered way (defaults → config.yaml → `HD_*` env vars) since neither is a live credential requiring `.env`-only handling like `DB_PASSWORD`/`CAMERA_RTSP_URL` — an ingest URL isn't secret, and the API key is a shared LAN secret at the same sensitivity tier as `NTFY_TOPIC`, not a database password.

**Steps:**
1. Edit `config.py`: add the two `_SCHEMA` tuples immediately after the `NTFY_*` entries (line 115), with a comment mirroring the existing ntfy comment style: `# NestJS dashboard ingest (empty URL disables).`
2. Edit `webapp.py`: add `"alert_leave"` to `VALID_ZONE_TASKS`.
3. Edit `webapp.py`'s `_build_zone_status_payload()`: add the `elif task == "alert_leave": alert = info.get("left", False)` branch. Confirm `info` already carries `"left"` after Task 9's change to `State.zone_status()`.
4. Edit `static/app.js`: add `alert_leave: "Alert on leave"` to `TASK_LABELS`.
5. Edit `templates/index.html`: add `<option value="alert_leave">Alert on leave</option>` to the `#zone-task-select` dropdown.
6. Manual verification: start `webapp.py`, open the Zones tab in a browser, confirm the new "Alert on leave" option appears in both the create-zone dropdown and the per-row edit dropdown, select it on a zone, confirm `PATCH /api/zones/:id` succeeds and the zone's task persists across a page reload.
7. Commit: `git commit -m "Add alert_leave zone task to webapp API and UI"`.

---

## Task 11: Python — ingest POST to NestJS

**Files:**
- Create: `nestjs_ingest.py` (new module, sibling to `notify.py`, same fire-and-forget pattern)
- Modify: `pipeline.py` (call sites at line 640 for "enter", line 771 for "exit"/"leave", and the zone_status block for zone-tagged leave events)

**Interfaces:**
- `nestjs_ingest.py` exposes `enabled() -> bool` (true iff `NESTJS_INGEST_URL` is non-empty) and `send_detection(person_id, camera_id, timestamp, zone_id=None, zone_event=None)`, which spawns a daemon thread posting JSON to `f"{NESTJS_INGEST_URL}/ingest/detection"` with header `X-Api-Key: NESTJS_API_KEY`, 5s timeout, swallowing `URLError`/`TimeoutError` and printing `[NESTJS] Failed to send detection: {exc}` — this is a direct structural copy of `notify.py`'s `notify_zone_alert`/`_post` pair (`notify.py:1-30`), substituting the payload/URL/header.
- `camera_id`: since the design spec seeds a single `Camera` row matching `CAMERA_RTSP_URL`, use a fixed camera id string. Read it the same way `stream.py` does (`os.environ.get("CAMERA_RTSP_URL")`) — but the raw RTSP URL is not a clean identifier (contains credentials/IP). Instead add one more `config.py` entry, `("NESTJS_CAMERA_ID", str, "front-door")`, giving pipeline.py a stable id to send that matches whatever the NestJS side seeds — this decision is made here rather than deferred, since a raw RTSP URL must never be sent over the wire as an identifier.

**Steps:**
1. Write `nestjs_ingest.py`, structurally mirroring `notify.py`: module-level `_cfg = config.load()`, `NESTJS_INGEST_URL = _cfg.NESTJS_INGEST_URL.rstrip("/")`, `NESTJS_API_KEY = _cfg.NESTJS_API_KEY`, `NESTJS_CAMERA_ID = _cfg.NESTJS_CAMERA_ID`, `enabled()`, `send_detection(...)` spawning `threading.Thread(target=_post, args=(payload,), daemon=True).start()`, and `_post(payload)` doing the `urllib.request.Request` POST with `Content-Type: application/json` and `X-Api-Key` headers.
2. Add the `NESTJS_CAMERA_ID` entry to `config.py`'s `_SCHEMA` (from Task 10's edit point, extend it here).
3. In `pipeline.py`, add `import nestjs_ingest` near the existing `from events import EventLog` import.
4. At line 640 (`event_log.record(new_person_id, "enter", track_id=track_id)`), add immediately after: `if nestjs_ingest.enabled(): nestjs_ingest.send_detection(new_person_id, nestjs_ingest.NESTJS_CAMERA_ID, datetime.now().isoformat(), zone_event="enter")` — but note this "enter" here is the identity-resolution enter (a person first appearing), not zone-specific; per the spec, `zoneEvent` is about zone boundary crossings, not general presence. Re-check against the spec: the ingest payload's `zoneEvent` requires a `zoneId` too. This call site (person-level enter) has no zone context, so it must be sent as a **plain detection** (no `zoneId`/`zoneEvent`) here — only the zone_status block (which knows `entered`/`left` per specific zone) should ever set `zoneEvent`.
5. Correct step 4's design: at line 640's call site, call `nestjs_ingest.send_detection(new_person_id, nestjs_ingest.NESTJS_CAMERA_ID, datetime.now().isoformat())` (plain, no zone args) — this covers "a person was detected/identified this frame."
6. At line 771 (`event_log.record(person_id, "exit", track_id=missing_track_id)`), similarly add a plain `nestjs_ingest.send_detection(person_id, nestjs_ingest.NESTJS_CAMERA_ID, datetime.now().isoformat())` call — this is a person-level disappearance, still not zone-scoped.
7. For actual zone-scoped `enter`/`leave` events: in the zone_status computation block (post-Task-9, now calling `_compute_zone_status`), after computing each zone's `entered`/`left` for this frame, iterate zones where `entered` or `left` is true and call `nestjs_ingest.send_detection(person_id, camera_id, timestamp, zone_id=zone_id, zone_event="enter"|"leave")` — but this block operates on zone-level occupant sets, not a single `person_id` in scope. Resolve this by iterating the *diff* of occupant sets: for a zone with `entered=True`, the new arrivals are `occupants_now - occupants_prev`; send one `zone_event="enter"` call per arriving person_id. For `left=True`, the departures are `occupants_prev - occupants_now`; send one `zone_event="leave"` call per departing person_id. Implement this directly inside `_compute_zone_status`'s caller (not inside the pure function itself, to keep that function side-effect-free and unit-testable per Task 9) — add a small loop right after the `_compute_zone_status` call in the main loop.
8. Add `NESTJS_INGEST_URL`/`NESTJS_API_KEY`/`NESTJS_CAMERA_ID` to `config.yaml.example` with empty/placeholder values and a comment, matching how `NTFY_TOPIC` is documented there (check `config.yaml.example`'s existing ntfy section first and follow its exact format).
9. Manual E2E verification (per spec's Testing section): point `NESTJS_INGEST_URL` at a locally running NestJS instance (from Tasks 1-6), run the Python pipeline against the live camera, confirm rows appear in `DetectionEvent` via `GET /api/events`, confirm the Next.js dashboard (Task 7-8) updates live on enter/leave.
10. Commit: `git commit -m "POST detection and zone enter/leave events to NestJS ingest endpoint"`.

---

## Task 12: End-to-end smoke test and README

**Files:**
- Create: `people-counter-nestjs/README.md`
- Modify: `human-detection/README.md` (add a short section pointing to the sibling repo and the new config entries)

**Steps:**
1. Write `people-counter-nestjs/README.md`: setup instructions (Postgres, `npm install`, `prisma migrate deploy`, env vars for both `apps/api` and `apps/web`, `npm run dev` for each), and a note that it's fed by the `human-detection` Python pipeline's `nestjs_ingest.py`.
2. Add a short section to `human-detection/README.md` documenting `NESTJS_INGEST_URL`/`NESTJS_API_KEY`/`NESTJS_CAMERA_ID` config entries and linking to the sibling repo.
3. Full manual E2E pass: fresh Postgres DBs for both the human-detection event log and the new NestJS API, both services running, Python pipeline live against the camera, walk through a monitored zone configured as `alert_leave`, confirm: (a) local ntfy/webapp behavior unaffected, (b) NestJS receives enter/leave rows, (c) dashboard's occupancy page and zone alert update live, (d) people/events pages show the same activity.
4. Commit both README changes: `git commit -m "Add setup docs for the NestJS dashboard integration"` (in the respective repos).

---

## Self-Review Notes

- **Placeholder scan**: no TBD/TODO markers; Task 11 steps 4-5 explicitly work through and correct an initial design mistake inline (plain vs. zone-scoped detection calls) rather than leaving it ambiguous — resolved to: person-level enter/exit → plain detections; zone entered/left diffs → per-person zoneEvent calls.
- **Type consistency**: `Person.id`/`DetectionEvent.personId` are `Int` throughout (matches Python's integer person IDs); `Zone.id`/`DetectionEvent.zoneId` are `String` throughout (matches Python's hex zone IDs); `ZoneTask`/`ZoneEvent` enums are used consistently between Prisma schema, DTO validation, and the frontend's `TASK_LABELS`-equivalent (occupancy page renders the enum values returned by `GET /api/zones`).
- **Spec coverage**: every REST endpoint, WebSocket event, and error-handling rule from the design spec has a corresponding task/step. The `alert_leave` Python-side addition (spec section not present verbatim but implied by the `ZoneTask.ALERT_LEAVE` enum and `zoneEvent: "leave"` payload) is fully covered by Tasks 9-11.
- **Scope check**: 12 tasks, each independently testable and committable; sized appropriately for `subagent-driven-development` (each task is a self-contained unit of work with clear file lists, interfaces, and a red-green test cycle).
