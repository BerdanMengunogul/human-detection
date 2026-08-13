# Identity-assignment test protocol

Purpose: get real evidence on why Person-N assignment isn't 100% reliable,
before changing any matching/threshold code. Three symptoms are in scope:

- **A. Duplicate ID** — same person leaves and comes back, gets a new Person-N.
- **B. Flicker/split** — one continuous visit briefly splits into >1 Person-N.
- **C. Merge** — two different people end up sharing one Person-N.

## How to run a test

Each run is one one launch of the pipeline, capturing every line it prints
(the `[IDENTITY]`, `[IDENTITY-DEBUG]`, and `[EVENT]` lines) to a timestamped
log file, plus a plain-English note of what you actually did and when.

```
python tools\run_test.py <scenario-name>
```

This starts `human_detection.py` (local window, so you can see what the
tracker/gallery is doing live) and tees all stdout to
`logs\<scenario-name>_<timestamp>.log`. Press `q` in the video window (or
Ctrl+C in the terminal) to stop — that ends the capture.

Immediately after, jot down what happened in plain terms — e.g. "walked out
at ~0:07, came back at ~0:12, was recognized as same person" — either as a
note back to me or into `logs\<scenario-name>_<timestamp>.notes.txt`. The
log alone tells us what the *system* concluded; your note tells us what was
*actually true*, which is what makes the log analyzable as pass/fail.

Then run:

```
python tools\analyze_log.py logs\<scenario-name>_<timestamp>.log
```

This prints a timeline of every identity decision (mints, reuses,
rejections, high-confidence overrides, enter/exit) plus flags anything that
looks like flicker or a rejected re-match. Paste that output (or just point
me at the log file) and I'll read it.

## Scenarios

Run these one at a time, in a normal-traffic environment (i.e. don't stage
multiple people unless the scenario calls for it). Repeat each 2-3 times if
the first run doesn't reproduce anything interesting — intermittent bugs
need repetition.

### A1 — Short absence (targets: duplicate ID)
Walk into frame, stand for ~5s so you get identified as Person-N, walk
fully out of frame for **5 seconds**, walk back in. Expected: same Person-N.
This is well within `EXIT_GRACE_SECONDS` (1.0s) + gallery re-match, so any
new ID here is a strong signal.

### A2 — Longer absence (targets: duplicate ID)
Same as A1 but stay out of frame for **30-60 seconds**. This is past all
grace windows and relies purely on the gallery's body/face re-match —
tests whether the appearance gallery itself is the weak link vs. the
grace-period race.

### A3 — Absence with a pose/lighting change (targets: duplicate ID)
Like A1, but change something about your appearance/pose on return — turn
around so your back is to the camera first, or put on/take off a jacket.
Tests whether re-match fails specifically when the embedding shifts.

### B1 — Walk-through occlusion (targets: flicker/split)
Walk across the frame passing behind a fixed obstruction (doorway edge,
furniture) that fully hides you for ~1 second mid-walk, in one continuous
visit. Expected: one Person-N the whole time, no EXIT/ENTER pair.

### B2 — Stand still with partial occlusion (targets: flicker/split)
Stand in frame for 30+ seconds, periodically turning or partially blocking
yourself with an arm/object, without ever leaving frame. Expected: one
Person-N throughout.

### B3 — Sudden pose change (targets: flicker/split)
Stand in frame, then quickly crouch/bend down out of the tracker's
expected pose and back up. Tests whether BoT-SORT's own tracker (not the
gallery) drops and re-mints the track_id, which is a different bug
surface than gallery re-match.

### C1 — Two similar-looking people cross paths (targets: merge)
Two people (ideally similar in clothing color/build) walk toward each
other, cross paths closely (near-contact), and continue to opposite
sides. Expected: two distinct Person-N throughout, no merge.

### C2 — Two people, one leaves as the other enters (targets: merge)
Person 1 is in frame; Person 2 enters as Person 1 exits, ideally
overlapping briefly at the doorway. Expected: two distinct Person-N — this
specifically stresses the "high-confidence bypass" path, since Person 1's
track_id may still be lingering in `track_to_person` during EXIT grace
when Person 2's new track tries to identify.

### C3 — Same person, twice, back to back (control for C1/C2)
One person walks through twice in a row (exits, re-enters within a few
seconds) while nobody else is present. This is actually an A1 repeat but
serves as a control: confirms the harness and your notes correctly
capture a clean single-person case before trusting the two-person results.

## What "pass" means

- A-series: exactly one Person-N number across the whole log for the one
  real person present.
- B-series: exactly one Person-N number, and no EXIT event followed by an
  ENTER within the same continuous visit.
- C-series: exactly two distinct Person-N numbers, each with its own
  clean ENTER/EXIT, and no `[IDENTITY] Rejected ... already has another
  live track` line pointing at the wrong person, no incorrect
  high-confidence "trusting it as a tracker split" absorption between the
  two different people.

Run `analyze_log.py` against each log — it applies these checks
automatically where it can infer them from the log alone (distinct
Person-N count, EXIT-then-ENTER-same-id patterns, rejection/absorption
events) and leaves a `NEEDS REVIEW` marker where it needs your note to
judge (e.g. was a second Person-N *supposed* to appear).
