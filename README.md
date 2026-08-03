# human-detection

A camera-based human detection and tracking pipeline with re-identification, zone-based
enter/leave logic, and ntfy alerting.

## NestJS dashboard integration (optional)

This pipeline can optionally push detection and zone enter/leave events to a companion
NestJS + Next.js dashboard, [`people-counter-nestjs`](../people-counter-nestjs), living in a
sibling repo. Events are sent fire-and-forget from a background thread (see
`nestjs_ingest.py`), so a slow or unreachable dashboard never blocks the detection loop.

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
