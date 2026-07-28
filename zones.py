"""Polygon door-zone loading, hot-reload, and point-in-zone geometry."""

import json
import os

import config as _config

_cfg = _config.load()

ZONES_PATH = _cfg.ZONES_PATH
ZONES_VERSION_PATH = _cfg.ZONES_VERSION_PATH


def read_zones_version():
    """Cheap counter written by the webapp whenever door_zones.json changes."""
    if not os.path.isfile(ZONES_VERSION_PATH):
        return 0
    try:
        with open(ZONES_VERSION_PATH, "r") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return 0


def bump_zones_version():
    """Increment the zones version counter (called from webapp zone-save endpoints)."""
    version = read_zones_version() + 1
    with open(ZONES_VERSION_PATH, "w") as f:
        f.write(str(version))
    return version


def load_web_zones(zones_path=ZONES_PATH):
    """Load dashboard-drawn polygon zones as a list of
    {id, name, task, points: [(x, y), ...]} dicts, in original-frame pixel
    coordinates. Returns an empty list if the file doesn't exist yet."""
    if not os.path.isfile(zones_path):
        return []
    with open(zones_path, "r") as f:
        data = json.load(f)
    zones = []
    for zone in data.get("web_zones", []):
        zones.append({
            "id": zone["id"],
            "name": zone.get("name", ""),
            "task": zone.get("task", "none"),
            "points": [(p[0], p[1]) for p in zone.get("points", [])],
        })
    return zones


def merge_zone_occupants(prev, web_zones):
    """Keep occupancy sets for zones that still exist after a hot-reload."""
    new_ids = {z["id"] for z in web_zones}
    merged = {zone_id: prev[zone_id] for zone_id in new_ids if zone_id in prev}
    for zone in web_zones:
        merged.setdefault(zone["id"], set())
    return merged


class WebZonesStore:
    """Loads web zones once and reloads only when the version counter changes."""

    def __init__(self, zones_path=ZONES_PATH):
        self._zones_path = zones_path
        self._version = read_zones_version()
        self._zones = load_web_zones(zones_path)

    @property
    def zones(self):
        return self._zones

    def maybe_reload(self):
        current = read_zones_version()
        if current == self._version:
            return False
        self._version = current
        self._zones = load_web_zones(self._zones_path)
        return True


def point_in_polygon(x, y, points):
    """Standard ray-casting point-in-polygon test. `points` is a list of
    (x, y) tuples in the same coordinate space as (x, y)."""
    inside = False
    n = len(points)
    if n < 3:
        return False
    x1, y1 = points[0]
    for i in range(1, n + 1):
        x2, y2 = points[i % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def _foot_point_in_any_zone(x1, y1, x2, y2, zones):
    """True if the bottom-center foot-point of box (x1,y1,x2,y2) falls
    inside any of the given web zones."""
    foot_x = (x1 + x2) / 2
    foot_y = y2
    return any(point_in_polygon(foot_x, foot_y, wz["points"]) for wz in zones)
