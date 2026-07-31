"""One-off: expand the 'monitor' ignore-zone polygon in door_zones.json
outward from its centroid by a fixed margin. YOLO's detected box for a
person appearing on a TV/monitor screen can extend past the physical
bezel, so a polygon drawn exactly on the bezel doesn't catch those
detections. Bumps the zones version so a running pipeline.py hot-reloads
the change without a restart.

Usage:
    python widen_ignore_zone.py [--margin 0.2] [--zone-id 721332fc]
"""

import argparse
import json

from zones import ZONES_PATH, bump_zones_version


def expand_polygon(points, margin):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [[cx + (x - cx) * (1 + margin), cy + (y - cy) * (1 + margin)] for x, y in points]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=0.2, help="Fractional outward expansion (default: 0.2)")
    parser.add_argument("--zone-id", default=None, help="Zone id to expand (default: all task=ignore zones)")
    args = parser.parse_args()

    with open(ZONES_PATH, "r") as f:
        data = json.load(f)

    expanded = []
    for zone in data.get("web_zones", []):
        if zone.get("task") != "ignore":
            continue
        if args.zone_id and zone["id"] != args.zone_id:
            continue
        zone["points"] = expand_polygon(zone["points"], args.margin)
        expanded.append(zone["id"])

    if not expanded:
        raise SystemExit("No matching ignore zone found - nothing changed")

    with open(ZONES_PATH, "w") as f:
        json.dump(data, f, indent=2)
    version = bump_zones_version()

    print(f"[INFO] Expanded zone(s) {expanded} by {args.margin:.0%}, zones_version={version}")


if __name__ == "__main__":
    main()
