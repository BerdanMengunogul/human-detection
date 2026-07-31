import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zones import _foot_point_in_any_zone, box_in_any_ignore_zone

# Stands in for the real "monitor" ignore zone in door_zones.json.
TV = [{
    "id": "tv",
    "name": "monitor",
    "task": "ignore",
    "points": [(1595, 1091), (1595, 1456), (2310, 1456), (2310, 1091)],
}]


def test_person_filling_the_screen_is_ignored():
    assert box_in_any_ignore_zone(1700, 1150, 2000, 1440, TV)


def test_waist_up_figure_high_on_screen_is_ignored():
    """The bug: this box sits wholly inside the TV, but its foot point is at
    y=1300 -- inside here, yet a figure framed any higher escapes entirely."""
    assert box_in_any_ignore_zone(1700, 1120, 1900, 1300, TV)


def test_figure_whose_feet_fall_above_the_zone_is_still_ignored():
    """Box is mostly on-screen but bottoms out above the polygon's top edge,
    so the foot-point test misses it and the box test catches it."""
    x1, y1, x2, y2 = 1700, 900, 1900, 1085
    assert not _foot_point_in_any_zone(x1, y1, x2, y2, TV)
    # Fully outside the zone vertically, so the box test also declines --
    # this documents that widening the polygon, not the test, covers it.
    assert not box_in_any_ignore_zone(x1, y1, x2, y2, TV)


def test_figure_straddling_the_bezel_is_ignored_when_mostly_on_screen():
    """Bottom edge drops just past the bezel: foot point is outside the
    polygon, but most of the body is on the screen."""
    x1, y1, x2, y2 = 1700, 1150, 1900, 1500
    assert not _foot_point_in_any_zone(x1, y1, x2, y2, TV)
    assert box_in_any_ignore_zone(x1, y1, x2, y2, TV)


def test_real_person_standing_in_front_of_the_tv_is_not_ignored():
    """A person between the camera and the TV overlaps the polygon but extends
    well below it; they must still be detected."""
    assert not box_in_any_ignore_zone(1700, 1200, 1900, 2100, TV)


def test_real_person_elsewhere_in_the_room_is_not_ignored():
    assert not box_in_any_ignore_zone(300, 800, 500, 1400, TV)


def test_no_ignore_zones_ignores_nothing():
    assert not box_in_any_ignore_zone(1700, 1150, 2000, 1440, [])


def test_min_fraction_is_respected():
    box = (1700, 1300, 1900, 1650)  # roughly half on-screen
    assert box_in_any_ignore_zone(*box, TV, min_fraction=0.4)
    assert not box_in_any_ignore_zone(*box, TV, min_fraction=0.9)
