# Ignore-Zone Ghost-Occupant Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Critical bug where a tracked person who walks into an "ignore" zone is never marked EXIT and never leaves `present_person_ids`/zone occupancy, because `seen_track_ids` is rebuilt from raw tracker output instead of from tracks actually processed this frame.

**Architecture:** Extract the `seen_track_ids` computation out of `run_detection`'s per-frame loop into a small, pure, independently-testable function `compute_seen_track_ids(track_ids, ignored_track_ids)` in `pipeline.py`. Collect a `ignored_track_ids_this_frame` set at the exact point detections are discarded for being inside an ignore zone (pipeline.py:466-467), then call the new function when building `seen_track_ids` (pipeline.py:607-609) instead of building it inline. No changes are needed to the missing-track/EXIT_GRACE_SECONDS block (pipeline.py:610-691) — once an ignore-zoned track_id is correctly excluded from `seen_track_ids`, it flows through that existing logic unmodified.

**Tech Stack:** Python 3, pytest (new dev dependency for this repo — no test infra exists yet), ultralytics YOLO track results (mocked in tests via a minimal stand-in object, not real inference).

## Global Constraints

- This repo has no existing `tests/` directory or pytest config — this plan introduces both, scoped minimally to this fix.
- Do not modify `identity.py`, `zones.py`, `webapp.py`, `config.py`, or any file outside `pipeline.py` and the new test file.
- Do not touch the Important/Minor findings from the prior code review (`.gitignore` cleanup, `widen_ignore_zone.py` atomic writes, `dashboard_pid.txt`/`contact_sheets/` cleanup) — those are out of scope for this plan.
- Preserve all existing behavior for tracks not in an ignore zone — the fix must be a no-op when `ignore_zones` is empty or no detection falls inside one.
- Keep the extracted function pure (no I/O, no reliance on globals) so it can be unit tested without mocking the model, camera stream, gallery, or event log.

---

### Task 1: Extract and fix `seen_track_ids` computation

**Files:**
- Modify: `pipeline.py:464-467` (collect ignored track_ids during the loop)
- Modify: `pipeline.py:607-609` (replace inline rebuild with call to new function)
- Modify: `pipeline.py` (add new top-level function, placed near `_foot_point_in_any_zone`/other module-level helpers — e.g. directly above `run_detection`, around line 313)
- Create: `tests/test_pipeline_seen_track_ids.py`

**Interfaces:**
- Produces: `compute_seen_track_ids(track_ids, ignored_track_ids)` — pure function.
  - `track_ids`: an object with a `.tolist()` method returning a list of raw track id values (matches the real `boxes.id` tensor's interface used at pipeline.py:608-609), or `None`.
  - `ignored_track_ids`: a `set[int]` of track_ids that were skipped this frame due to being inside an ignore zone.
  - Returns: `set[int]` — the track ids considered "seen" this frame, i.e. all ids from `track_ids.tolist()` cast to `int`, minus `ignored_track_ids`. Returns `set()` if `track_ids` is `None`.
- Consumes (inside `run_detection`): a new local variable `ignored_track_ids_this_frame = set()`, initialized at the top of the per-frame loop body (alongside `human_count = 0` / `zone_occupants_now = {}` at pipeline.py:432-433), populated with `track_id` at the ignore-zone `continue` (pipeline.py:466-467).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_seen_track_ids.py`:

```python
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
    # This is the ghost-occupant bug: track 2 is still reported by the
    # tracker (it's physically visible, just inside an ignore zone), but
    # must NOT count as "seen" so the missing-track/EXIT logic can fire
    # for it.
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
    # A track_id that was ignored on a previous frame but isn't in this
    # frame's raw output at all (tracker dropped it) must not raise.
    track_ids = _FakeTrackIds([1.0])
    result = compute_seen_track_ids(track_ids, ignored_track_ids={99})
    assert result == {1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_seen_track_ids.py -v`
Expected: FAIL/ERROR with `ImportError: cannot import name 'compute_seen_track_ids' from 'pipeline'`

- [ ] **Step 3: Add the pure function to `pipeline.py`**

Add this function near the other module-level helpers, directly above the `run_detection` definition (currently at pipeline.py:315):

```python
def compute_seen_track_ids(track_ids, ignored_track_ids):
    """Returns the set of track_ids the tracker reported this frame that were
    actually processed (i.e. not discarded for being inside an ignore zone).
    A track_id skipped via the ignore-zone check must not count as "seen" -
    otherwise it never enters the missing-track/EXIT_GRACE_SECONDS logic and
    a person who walks into an ignore zone is never marked EXIT."""
    if track_ids is None:
        return set()
    return {int(tid) for tid in track_ids.tolist()} - ignored_track_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_seen_track_ids.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Wire the function into the per-frame loop**

In `pipeline.py`, at line 432-433 (start of per-frame bookkeeping, right before `boxes = results.boxes`), add the new tracking set:

```python
            human_count = 0
            zone_occupants_now = {}  # zone_id -> person_ids inside this frame
            ignored_track_ids_this_frame = set()  # track_ids skipped this frame for being inside an ignore zone
            boxes = results.boxes
            track_ids = boxes.id
```

At line 464-467, record the track_id before the `continue`:

```python
                track_id = int(track_ids[i])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if ignore_zones and _foot_point_in_any_zone(x1, y1, x2, y2, ignore_zones):
                    ignored_track_ids_this_frame.add(track_id)
                    continue
```

At line 607-609, replace the inline rebuild with a call to the new function:

```python
            seen_track_ids = compute_seen_track_ids(track_ids, ignored_track_ids_this_frame)
```

(This replaces both the `seen_track_ids = set()` line and the `if track_ids is not None: ...` block — 3 lines become 1.)

- [ ] **Step 6: Run the full test file again to confirm nothing broke**

Run: `python -m pytest tests/test_pipeline_seen_track_ids.py -v`
Expected: PASS (5 passed) — this step is a sanity check that editing `pipeline.py` didn't introduce a syntax error or import-time failure (e.g. from missing model files), since `tests/test_pipeline_seen_track_ids.py` imports the whole `pipeline` module.

If import fails due to heavy dependencies (e.g. model loading) executing at module import time, note the failure and inspect `pipeline.py`'s top-level (non-function) statements to confirm — module-level code should only be constant/config assignments (as seen at lines 1-84), not model loading (which happens inside `run_detection` at lines 321-324). If it does fail for this reason, that's a pre-existing structural issue outside this plan's scope; report it rather than restructuring the module.

- [ ] **Step 7: Commit**

```bash
git add pipeline.py tests/test_pipeline_seen_track_ids.py
git commit -m "fix: exclude ignore-zoned tracks from seen_track_ids so EXIT fires correctly"
```

---

## Manual Verification (not automated - requires live camera/stream)

After merging, confirm against the real pipeline:
1. Configure an "ignore" zone covering an area the camera can see.
2. Have a recognized person walk into the door zone (ENTER fires), then into the ignore zone, and stay there past `EXIT_GRACE_SECONDS` (default 1.0s, check `config.yaml`/`HD_EXIT_GRACE_SECONDS`).
3. Confirm an `exit` event is recorded (check `event_log` / the dashboard's occupancy view) instead of the person remaining "present" indefinitely.
4. Confirm a person who is tracked normally (never entering any ignore zone) still gets ENTER/EXIT events as before — this task must be a no-op for that path.
