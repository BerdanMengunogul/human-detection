import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import should_exit


def test_person_who_walked_off_camera_mid_room_still_exits():
    """The bug: door zones are narrow strips near the frame edges, so a
    person who disappears from the middle of the room (the common case for
    walking out of camera view) never has a last tracked frame inside a door
    polygon. EXIT must not depend on that."""
    present = {"Person-1"}
    assert should_exit(still_present=False, person_id="Person-1", present_person_ids=present)


def test_person_still_tracked_on_another_track_id_does_not_exit():
    present = {"Person-1"}
    assert not should_exit(still_present=True, person_id="Person-1", present_person_ids=present)


def test_person_not_currently_present_does_not_exit_again():
    present = set()
    assert not should_exit(still_present=False, person_id="Person-1", present_person_ids=present)
