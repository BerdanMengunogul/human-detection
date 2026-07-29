"""PostgreSQL-backed ENTER/EXIT event log."""

from datetime import datetime

import psycopg2

import config as _config

_cfg = _config.load()

DB_HOST = _cfg.DB_HOST
DB_PORT = _cfg.DB_PORT
DB_NAME = _cfg.DB_NAME
DB_USER = _cfg.DB_USER
DB_PASSWORD = _cfg.DB_PASSWORD


class EventLog:
    """Records ENTER/EXIT events for each Person-N to a PostgreSQL database,
    so an admin dashboard can later show who was present and when without
    needing to run or parse this detection script itself."""

    def __init__(self, host, port, dbname, user, password):
        self._conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
        )
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('enter', 'exit')),
                    timestamp TEXT NOT NULL
                )
                """
            )

    def record(self, person_id, event_type, track_id=None):
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (person_id, event_type, timestamp) VALUES (%s, %s, %s)",
                (person_id, event_type, timestamp),
            )
        print(f"[EVENT] Person-{person_id} {event_type.upper()} at {timestamp} (track={track_id})")

    def reset(self):
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM events")

    def close(self):
        self._conn.close()
