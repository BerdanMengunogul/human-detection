import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import _compute_zone_status


WEB_ZONES = [{"id": "z1"}]


def test_simultaneous_enter_and_leave():
    """Person 3 arrives while person 1 leaves in the same frame -- entered
    and left must not be treated as mutually exclusive."""
    status = _compute_zone_status(WEB_ZONES, {"z1": {2, 3}}, {"z1": {1, 2}})
    assert status["z1"]["entered"] is True
    assert status["z1"]["left"] is True


def test_no_change():
    status = _compute_zone_status(WEB_ZONES, {"z1": {1, 2}}, {"z1": {1, 2}})
    assert status["z1"]["entered"] is False
    assert status["z1"]["left"] is False


def test_zone_newly_occupied():
    status = _compute_zone_status(WEB_ZONES, {"z1": {1}}, {"z1": set()})
    assert status["z1"]["entered"] is True
    assert status["z1"]["left"] is False


def test_zone_newly_emptied():
    status = _compute_zone_status(WEB_ZONES, {"z1": set()}, {"z1": {1}})
    assert status["z1"]["entered"] is False
    assert status["z1"]["left"] is True
