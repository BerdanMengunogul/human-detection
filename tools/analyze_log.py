"""Parse a captured run_test.py log and report an identity-decision timeline
plus automatic flags for the three known symptoms:

  A. duplicate ID   - a Person-N EXITs then a *different* Person-N ENTERs
                       shortly after, when a note says it was the same person.
  B. flicker/split  - EXIT immediately followed by ENTER of the same or a
                       new Person-N within a short window (no real absence).
  C. merge          - a REJECTED match line, or a high-confidence
                       "trusting it as a tracker split" absorption line,
                       naming a person who (per the enter/exit timeline) had
                       another live, distinct visit in progress.

This only reads plain stdout text captured by run_test.py - never touches
config, .env, or the database, so no credentials pass through it.

Usage:
    python tools\\analyze_log.py logs\\A1_20260804_120000.log
"""

import re
import sys
from pathlib import Path

_TS_RE = re.compile(r"^\[t\+\s*([\d.]+)s\]\s?(.*)$")

_EVENT_RE = re.compile(r"^\[EVENT\] Person-(\d+) (ENTER|EXIT) at (\S+) \(track=(\S+)\)")
_MINTED_RE = re.compile(r"^\[IDENTITY\] Person-(\d+) minted NEW \(best body_sim=([-\d.]+), face_sim=(\S+)\)")
_REUSED_RE = re.compile(r"^\[IDENTITY\] Person-(\d+) reused via (FACE|BODY) match \(sim=([-\d.]+)\)")
_REJECTED_RE = re.compile(
    r"^\[IDENTITY\] Rejected (FACE|BODY) match to Person-(\d+) \(sim=([-\d.]+)\): "
    r"already has another live track - minting new identity instead\."
)
_ABSORBED_RE = re.compile(
    r"^\[IDENTITY\] Person-(\d+) has another live track but (FACE|BODY) match is "
    r"high-confidence.*trusting it as a tracker split and absorbing the stale track\."
)
_BORDERLINE_RE = re.compile(
    r"^\[IDENTITY\] Borderline BODY match to Person-(\d+) \(sim=([-\d.]+)\) accepted via corroboration"
)

FLICKER_WINDOW_SECONDS = 3.0


def parse_log(path):
    events = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _TS_RE.match(raw_line)
        if not m:
            continue
        t = float(m.group(1))
        line = m.group(2)

        if m2 := _EVENT_RE.match(line):
            pid, etype, ts, track = m2.groups()
            events.append({"t": t, "kind": "event", "person_id": int(pid), "event_type": etype, "track": track})
        elif m2 := _MINTED_RE.match(line):
            pid, body_sim, face_sim = m2.groups()
            events.append({"t": t, "kind": "minted", "person_id": int(pid), "body_sim": body_sim, "face_sim": face_sim})
        elif m2 := _REUSED_RE.match(line):
            pid, via, sim = m2.groups()
            events.append({"t": t, "kind": "reused", "person_id": int(pid), "via": via, "sim": float(sim)})
        elif m2 := _REJECTED_RE.match(line):
            via, pid, sim = m2.groups()
            events.append({"t": t, "kind": "rejected", "person_id": int(pid), "via": via, "sim": float(sim)})
        elif m2 := _ABSORBED_RE.match(line):
            pid, via = m2.groups()
            events.append({"t": t, "kind": "absorbed", "person_id": int(pid), "via": via})
        elif m2 := _BORDERLINE_RE.match(line):
            pid, sim = m2.groups()
            events.append({"t": t, "kind": "borderline", "person_id": int(pid), "sim": float(sim)})
    return events


def render_timeline(events):
    lines = ["=== Timeline ==="]
    for e in events:
        t = f"t+{e['t']:7.2f}s"
        k = e["kind"]
        if k == "event":
            lines.append(f"{t}  {e['event_type']:5s} Person-{e['person_id']} (track={e['track']})")
        elif k == "minted":
            lines.append(f"{t}  MINTED  Person-{e['person_id']} new identity (body_sim={e['body_sim']}, face_sim={e['face_sim']})")
        elif k == "reused":
            lines.append(f"{t}  REUSE   Person-{e['person_id']} via {e['via']} (sim={e['sim']:.3f})")
        elif k == "rejected":
            lines.append(f"{t}  REJECT  candidate Person-{e['person_id']} via {e['via']} (sim={e['sim']:.3f}) -> new ID minted instead")
        elif k == "absorbed":
            lines.append(f"{t}  ABSORB  Person-{e['person_id']} stale track absorbed via high-confidence {e['via']} match")
        elif k == "borderline":
            lines.append(f"{t}  BORDER  Person-{e['person_id']} borderline BODY match (sim={e['sim']:.3f}) accepted via corroboration")
    return "\n".join(lines)


def analyze(events):
    flags = []
    enter_events = [e for e in events if e["kind"] == "event" and e["event_type"] == "ENTER"]
    exit_events = [e for e in events if e["kind"] == "event" and e["event_type"] == "EXIT"]
    distinct_persons = sorted({e["person_id"] for e in events if e["kind"] == "event"})

    flags.append(f"Distinct Person-N seen in ENTER/EXIT events: {distinct_persons} ({len(distinct_persons)} total)")

    # Flicker/split: an EXIT followed shortly after by an ENTER (same or
    # different person) with no other ENTER of that same person to explain
    # a genuine gap - i.e. rapid EXIT->ENTER churn.
    all_ee = sorted(enter_events + exit_events, key=lambda e: e["t"])
    for i in range(len(all_ee) - 1):
        a, b = all_ee[i], all_ee[i + 1]
        if a["event_type"] == "EXIT" and b["event_type"] == "ENTER":
            gap = b["t"] - a["t"]
            if gap <= FLICKER_WINDOW_SECONDS:
                same = "SAME" if a["person_id"] == b["person_id"] else "DIFFERENT"
                flags.append(
                    f"[FLICKER?] EXIT Person-{a['person_id']} at t+{a['t']:.2f}s -> "
                    f"ENTER Person-{b['person_id']} at t+{b['t']:.2f}s (gap={gap:.2f}s, {same} person_id) "
                    f"- check your notes: was this a real ~{gap:.0f}s absence, or one continuous visit?"
                )

    # Rejections: candidate matched an existing person but was refused
    # because that person had another live track, so a *new* Person-N was
    # minted instead. This is the direct mechanism behind "duplicate ID
    # after absence" and "ID flicker/split" - always worth surfacing.
    for e in events:
        if e["kind"] == "rejected":
            flags.append(
                f"[REJECT->NEW-ID] t+{e['t']:.2f}s: candidate matched Person-{e['person_id']} "
                f"(via {e['via']}, sim={e['sim']:.3f}) but was rejected as still-live, so a NEW "
                f"Person-N was minted. If your notes say this was the same person returning, "
                f"this rejection is the bug."
            )

    # Absorptions: the high-confidence bypass fired. Not inherently bad
    # (it's the intended self-heal), but worth flagging for the merge
    # symptom - if two *different* people were both present, an absorption
    # here would be a merge.
    for e in events:
        if e["kind"] == "absorbed":
            flags.append(
                f"[ABSORB] t+{e['t']:.2f}s: Person-{e['person_id']}'s stale track was absorbed via "
                f"high-confidence {e['via']} match (tracker-split self-heal). If your notes say two "
                f"DIFFERENT people were present at this point, this is a MERGE bug."
            )

    # Borderline corroboration accepts are a softer signal, still worth a look.
    for e in events:
        if e["kind"] == "borderline":
            flags.append(
                f"[BORDERLINE] t+{e['t']:.2f}s: Person-{e['person_id']} accepted on a borderline "
                f"body-sim={e['sim']:.3f} match via corroboration - lower-confidence than a normal reuse."
            )

    if len(enter_events) != len(exit_events):
        flags.append(
            f"[UNBALANCED] {len(enter_events)} ENTER vs {len(exit_events)} EXIT events - "
            f"someone was likely still in frame when the log ended (expected if you stopped mid-visit)."
        )

    return flags


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]
    events = parse_log(log_path)
    if not events:
        print(f"No [IDENTITY]/[EVENT] lines found in {log_path}. Was the log captured correctly?")
        sys.exit(1)

    print(render_timeline(events))
    print("\n=== Flags ===")
    for flag in analyze(events):
        print(f"- {flag}")
    print(
        "\n(NEEDS REVIEW markers are implicit: any [FLICKER?], [REJECT->NEW-ID], or [ABSORB] "
        "flag above needs your notes on what actually happened to judge pass/fail.)"
    )


if __name__ == "__main__":
    main()
