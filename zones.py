"""Polygon door-zone loading, hot-reload, and point-in-zone geometry."""

import json
import os

import config as _config

_cfg = _config.load()

ZONES_PATH = _cfg.ZONES_PATH
ZONES_VERSION_PATH = _cfg.ZONES_VERSION_PATH
IGNORE_ZONE_MIN_OVERLAP = _cfg.IGNORE_ZONE_MIN_OVERLAP


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


def _box_overlaps_zone(x1, y1, x2, y2, zone, min_fraction):
    """True if at least `min_fraction` of box (x1,y1,x2,y2) lies inside the
    zone polygon, approximated by sampling a grid of points across the box."""
    steps = 5
    inside = 0
    for i in range(steps):
        px = x1 + (x2 - x1) * (i + 0.5) / steps
        for j in range(steps):
            py = y1 + (y2 - y1) * (j + 0.5) / steps
            if point_in_polygon(px, py, zone["points"]):
                inside += 1
    return inside / (steps * steps) >= min_fraction


def box_in_any_ignore_zone(x1, y1, x2, y2, zones,
                           min_fraction=IGNORE_ZONE_MIN_OVERLAP):
    """True if box (x1,y1,x2,y2) is mostly inside any ignore zone.

    Unlike `_foot_point_in_any_zone`, this tests the whole box. An ignore zone
    marks a region where detections aren't real people at all (a TV, a poster,
    a mirror), so a figure on a screen must be discarded regardless of where
    its feet land -- a waist-up figure on a monitor has its foot point in the
    middle of the screen, or above the zone entirely, and a foot-point test
    lets it through. Door zones keep the foot-point test, where "which side of
    the threshold is this person standing on" is exactly the right question."""
    return any(
        _box_overlaps_zone(x1, y1, x2, y2, wz, min_fraction) for wz in zones
    )


def split_ignore_zones(web_zones):
    """Splits loaded web zones into (door_zones, ignore_zones). Zones tasked
    "ignore" are excluded from door/occupancy logic entirely and used only
    to blank out detections before tracking sees them."""
    ignore_zones = [z for z in web_zones if z.get("task") == "ignore"]
    door_zones = [z for z in web_zones if z.get("task") != "ignore"]
    return door_zones, ignore_zones
