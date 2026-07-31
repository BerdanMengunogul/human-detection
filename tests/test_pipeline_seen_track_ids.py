import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import compute_seen_track_ids


class _FakeTrackIds:
    """Stands in for the ultralytics boxes.id tensor - only .tolist() is used."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def test_no_ignored_tracks_returns_all_seen_ids():
    track_ids = _FakeTrackIds([1.0, 2.0, 3.0])
    result = compute_seen_track_ids(track_ids, ignored_track_ids=set())
    assert result == {1, 2, 3}


def test_ignored_track_id_excluded_from_seen_set():
    track_ids = _FakeTrackIds([1.0, 2.0, 3.0])
    result = compute_seen_track_ids(track_ids, ignored_track_ids={2})
    assert result == {1, 3}


def test_all_tracks_ignored_returns_empty_set():
    track_ids = _FakeTrackIds([5.0, 6.0])
    result = compute_seen_track_ids(track_ids, ignored_track_ids={5, 6})
    assert result == set()


def test_none_track_ids_returns_empty_set():
    result = compute_seen_track_ids(None, ignored_track_ids={1})
    assert result == set()


def test_ignored_track_id_not_present_in_track_ids_is_harmless():
    track_ids = _FakeTrackIds([1.0])
    result = compute_seen_track_ids(track_ids, ignored_track_ids={99})
    assert result == {1}
