import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import should_enter


def test_lone_person_who_never_crossed_a_door_zone_still_enters():
    """The bug: door zones are narrow strips near the frame edges, so a lone
    person who stays in the middle of the room never has a foot-point inside
    one. Without this, they never enter present_person_ids and therefore
    never appear as a clickable box in the People tab -- unnameable."""
    present = set()
    assert should_enter(person_id="Person-1", present_person_ids=present)


def test_person_already_present_does_not_enter_again():
    present = {"Person-1"}
    assert not should_enter(person_id="Person-1", present_person_ids=present)
