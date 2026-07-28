"""
Phase 1: Human detection with bounding boxes from a Hikvision RTSP stream.

Connects to the camera, runs YOLOv8 on each frame, draws green boxes
around detected people, prints the per-frame human count, and shows
the annotated video with an FPS overlay. Press 'q' to quit.

A background thread continuously reads frames from the RTSP socket and
keeps only the latest one. This prevents the multi-second latency that
builds up when frame grabbing can't keep pace with YOLO inference.

Each person is assigned a persistent "Person-N" ID for the life of the
running session. BoT-SORT (with its own built-in ReID) keeps IDs stable
frame-to-frame, including brief occlusions. On top of that, a session-long
appearance gallery re-matches people by embedding similarity whenever the
tracker mints a *new* track ID, so someone who fully leaves the frame and
comes back later (beyond the tracker's own track_buffer window) is still
recognized as the same Person-N instead of getting a new number.
"""

import argparse
import json
import os
import psycopg2
import sys
import threading
import time
from collections import deque
from datetime import datetime

# onnxruntime-gpu's CUDA execution provider needs a system-PATH-discoverable
# CUDA 12.x + cuDNN 9.x runtime; PyTorch bundles its own copy privately, so
# it doesn't help onnxruntime. These pip-installed NVIDIA redistributable
# packages provide that runtime; this must run before onnxruntime/insightface
# import for the DLL search path change to take effect.
import site

for _site_dir in site.getsitepackages():
    _nvidia_root = os.path.join(_site_dir, "nvidia")
    if os.path.isdir(_nvidia_root):
        for _pkg in ("cudnn", "cuda_runtime", "cublas", "cuda_nvrtc", "cufft"):
            _dll_dir = os.path.join(_nvidia_root, _pkg, "bin")
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)
                os.environ["PATH"] = _dll_dir + os.pathsep + os.environ["PATH"]
        break

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from ultralytics.trackers.utils.reid import ReID

import config as _config

load_dotenv()

DEVICE = 0 if torch.cuda.is_available() else "cpu"

_cfg = _config.load()

# Main stream (101) is full resolution (2560x1440). GPU inference is fast
# enough (~8ms/frame) that it's no longer the bottleneck, so we use the
# higher-quality stream for better detection accuracy and a sharper display.
RTSP_URL = os.environ.get("CAMERA_RTSP_URL")
if not RTSP_URL:
    sys.exit(
        "CAMERA_RTSP_URL is not set. Copy .env.example to .env and fill in "
        "your camera's RTSP URL, or export CAMERA_RTSP_URL in your shell."
    )
# "s" model is more accurate than "n" and still runs well under GPU inference.
MODEL_NAME = _cfg.MODEL_NAME
# Dedicated ReID checkpoint (not the detector backbone) for the session-long
# gallery: detector features aren't discriminative enough for appearance
# matching (same person scored as low as 0.64 cosine sim in testing), so a
# purpose-built person-ReID model is used instead. Auto-downloaded by
# Ultralytics on first use.
REID_MODEL_NAME = _cfg.REID_MODEL_NAME
TRACKER_CONFIG = _cfg.TRACKER_CONFIG
PERSON_CLASS_ID = _cfg.PERSON_CLASS_ID
CONF_THRESHOLD = _cfg.CONF_THRESHOLD
BOX_COLOR = (0, 255, 0)  # green, BGR
DISPLAY_MAX_WIDTH = _cfg.DISPLAY_MAX_WIDTH
DISPLAY_MAX_HEIGHT = _cfg.DISPLAY_MAX_HEIGHT

# Separate pose-estimation model run alongside the detector/tracker, purely
# for the skeleton overlay - does not feed into tracking, ReID, or identity.
POSE_MODEL_NAME = _cfg.POSE_MODEL_NAME
POSE_ENABLED = _cfg.POSE_ENABLED
POSE_EVERY_N_FRAMES = max(1, _cfg.POSE_EVERY_N_FRAMES)
POSE_CONF_THRESHOLD = _cfg.POSE_CONF_THRESHOLD
POSE_KEYPOINT_CONF_THRESHOLD = _cfg.POSE_KEYPOINT_CONF_THRESHOLD  # per-keypoint conf below this is skipped (occluded/off-frame joint)
POSE_COLOR = (0, 200, 255)  # orange, BGR - distinct from the green tracking box
# IoU above which a pose-model detection is considered to belong to the same
# person as a tracked detector box, for overlaying that person's skeleton.
POSE_IOU_MATCH_THRESHOLD = _cfg.POSE_IOU_MATCH_THRESHOLD
# COCO-17 keypoint indices: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders,
# 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles.
POSE_SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders, arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (0, 5), (0, 6),  # head to shoulders
]

# Filters out false-positive detections on tiny specks of background clutter
# that happen to clear the confidence threshold. A ratio/shape filter was
# tried too, but this camera's close-up framing means legitimate face/
# upper-body-only views are often square or wider than tall, so a shape
# filter rejects real people; CONF_THRESHOLD was lowered to 0.5 to reduce
# flicker on borderline/occluded people, so this size floor now carries more
# of the false-positive rejection burden.
MIN_BOX_AREA_FRACTION = _cfg.MIN_BOX_AREA_FRACTION  # reject tiny specks relative to frame area

# Number of recent frames' counts to smooth over for the displayed number,
# so a single flickery frame doesn't make the count spike up or down.
COUNT_SMOOTHING_WINDOW = _cfg.COUNT_SMOOTHING_WINDOW

# EMA weight given to each new raw box when smoothing a track's drawn
# bounding box. Lower = smoother but laggier; 1.0 disables smoothing.
# YOLO's per-frame box coordinates wobble a few pixels even for a person
# standing still, which reads as a vibrating border - blending toward the
# new box instead of snapping to it removes that jitter.
BOX_SMOOTHING_ALPHA = _cfg.BOX_SMOOTHING_ALPHA

# A track missing from the tracker's output for up to this many *consecutive*
# frames still has its last-known (smoothed) box drawn instead of vanishing,
# so a brief missed detection (motion blur, partial occlusion, a confidence
# dip below CONF_THRESHOLD) reads as a steady box instead of a blink.
# Deliberately short and frame-based (unlike EXIT_GRACE_SECONDS): this only
# covers single-frame flicker, not the longer wall-clock window that decides
# ENTER/EXIT events.
BOX_COAST_FRAMES = _cfg.BOX_COAST_FRAMES

# Cosine similarity above which a newly-seen track is considered the same
# person as an earlier one in the session-long gallery (rather than someone
# new). Lower catches more re-entries but risks merging two different people.
REID_MATCH_THRESHOLD = _cfg.REID_MATCH_THRESHOLD

# ArcFace embeddings (from InsightFace's buffalo_s recognition model) have a
# much sharper same-person/different-person cosine-similarity separation than
# body-appearance ReID, so this threshold is higher and more conservative.
# Checked first when a face is visible: a confident face match overrides
# body appearance, since it survives clothing/headwear changes that defeat
# the body-ReID gallery. Falls back to body appearance when no face is
# detected (person facing away, too far, out of frame at head height, etc).
FACE_MATCH_THRESHOLD = _cfg.FACE_MATCH_THRESHOLD
FACE_MAX_SAMPLES_PER_PERSON = _cfg.FACE_MAX_SAMPLES_PER_PERSON
# Faces need to be reasonably large in the frame to produce a reliable
# embedding; SCRFD will happily detect tiny/blurry faces that then embed
# unreliably. Rejects detections below this box height in pixels.
FACE_MIN_BOX_HEIGHT = _cfg.FACE_MIN_BOX_HEIGHT

# Seconds between top-up embeddings for a track that's already been
# identified. All of a track's initial candidates come from a narrow window
# right as it enters frame, so they tend to share the same distance/pose -
# e.g. always a seated desk closeup, never a standing full-body shot. Without
# this, a person's gallery bank only ever reflects how they looked in that
# first window, so a later re-entry with a different framing (closer/farther,
# sitting/standing) has nothing similar to match against. Periodically adding
# a fresh good-quality sample from later in the same track's life fills out
# that coverage over time.
#
# Expressed in wall-clock seconds (not frame count) so the behavior doesn't
# silently change if the pipeline's FPS changes - see EXIT_GRACE_SECONDS
# below for why frame counts are the wrong unit here.
REID_TOPUP_INTERVAL_SECONDS = _cfg.REID_TOPUP_INTERVAL_SECONDS

# Seconds to wait after a track first appears before computing its ReID
# embedding. A track's very first frame is often a bad crop (motion blur,
# head down, mid-turn from just entering frame), which produces an
# unreliable embedding and can split one person into multiple Person-IDs.
# Waiting a bit and picking the largest-area crop from that window avoids
# committing to an embedding from a bad frame.
#
# Expressed in wall-clock seconds - see EXIT_GRACE_SECONDS below.
IDENTIFY_DELAY_SECONDS = _cfg.IDENTIFY_DELAY_SECONDS
# A track needs at least this many candidate frames (regardless of how much
# time that took) before it's identified, so a stalled/near-zero-FPS run
# still waits for a few real samples instead of identifying off just one.
IDENTIFY_MIN_CANDIDATES = _cfg.IDENTIFY_MIN_CANDIDATES

# PostgreSQL connection settings for the database that records ENTER/EXIT
# events for the admin dashboard. DB_PASSWORD comes from .env, never from
# config.yaml.
DB_HOST = _cfg.DB_HOST
DB_PORT = _cfg.DB_PORT
DB_NAME = _cfg.DB_NAME
DB_USER = _cfg.DB_USER
DB_PASSWORD = _cfg.DB_PASSWORD

# Set to True to print rolling average per-stage timings (track/reid/face/
# encode) every PROFILE_REPORT_INTERVAL frames. Diagnostic only - leave off
# for normal runs since the extra dict bookkeeping and prints have a small
# cost of their own.
PROFILE = os.environ.get("HD_PROFILE", "0") == "1"
PROFILE_REPORT_INTERVAL = _cfg.PROFILE_REPORT_INTERVAL


class StageTimer:
    """Accumulates elapsed time per named stage across frames and prints a
    rolling average every `report_interval` frames, then resets."""

    def __init__(self, report_interval=PROFILE_REPORT_INTERVAL):
        self._totals = {}
        self._counts = {}
        self._report_interval = report_interval
        self._frame_count = 0

    def add(self, name, elapsed):
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def end_frame(self):
        self._frame_count += 1
        if self._frame_count < self._report_interval:
            return
        parts = []
        for name, total in self._totals.items():
            count = self._counts.get(name, 1)
            parts.append(f"{name}={1000 * total / count:.1f}ms(avg,n={count})")
        print(f"[PROFILE] over {self._frame_count} frames: " + " | ".join(parts))
        self._totals.clear()
        self._counts.clear()
        self._frame_count = 0

# A track_id must be absent from the tracker's output for this many
# wall-clock seconds before its Person-N is considered to have left. This is
# meant to match the tracker's own track_buffer window, so a brief occlusion
# that BoT-SORT itself would have bridged doesn't produce a spurious
# EXIT/ENTER pair for the same continuous visit.
#
# Deliberately measured in seconds, not frames: track_buffer (tracker_reid.yaml)
# and this used to both be frame counts (30) on the assumption they'd always
# mean the same wall-clock window. When a later perf fix (removing an inert
# GMC step) roughly tripled FPS, both windows silently shrank to a third of
# their intended duration, causing tracks to get dropped/respawned as new
# Person-N identities during brief occlusions that used to be bridged fine.
# Keeping this one in seconds means it stays correct regardless of FPS;
# tracker_reid.yaml's track_buffer is still frame-based (an ultralytics
# constant we don't control) and is set generously to match at target FPS.
EXIT_GRACE_SECONDS = _cfg.EXIT_GRACE_SECONDS

# Path to the JSON file storing the room's door zones, as polygons drawn from
# the web dashboard (in original, un-scaled frame coordinate space). A
# track's foot-point must fall inside one of these polygons for its
# disappearance to be logged as an EXIT (and, symmetrically, a
# newly-identified track must have its foot-point inside one to be logged as
# an ENTER) - a person who is simply occluded or drops below the confidence
# threshold mid-room, then picked back up, never crosses a door zone, so no
# spurious EXIT/ENTER pair is produced for them. This is the room's actual
# physical constraint (it has a fixed number of doors), used to reject
# tracking noise that time-based grace frames alone can't distinguish from a
# real departure. If no zones are configured, gating is skipped entirely and
# ENTER/EXIT fall back to plain appear/disappear behavior.
ZONES_PATH = _cfg.ZONES_PATH
ZONES_VERSION_PATH = _cfg.ZONES_VERSION_PATH
PEOPLE_PATH = _cfg.PEOPLE_PATH
GALLERY_PATH = _cfg.GALLERY_PATH
GALLERY_SAVE_INTERVAL_SECONDS = _cfg.GALLERY_SAVE_INTERVAL_SECONDS


class PeopleStore:
    """Thread-safe JSON-persisted mapping of person_id -> assigned name.

    person_ids are only stable within a session (Reset wipes the ReID
    gallery and restarts numbering from 1), so entries are keyed by the
    session-local person_id and are expected to be reassigned after Reset.
    """

    def __init__(self, path=PEOPLE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._names = self._load()

    def _load(self):
        if not os.path.isfile(self._path):
            return {}
        with open(self._path, "r") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("names", {}).items()}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump({"names": {str(k): v for k, v in self._names.items()}}, f, indent=2)

    def all(self):
        with self._lock:
            return dict(self._names)

    def get(self, person_id):
        with self._lock:
            return self._names.get(person_id)

    def set(self, person_id, name):
        with self._lock:
            self._names[person_id] = name
            self._save()

    def delete(self, person_id):
        with self._lock:
            self._names.pop(person_id, None)
            self._save()

    def clear(self):
        with self._lock:
            self._names = {}
            self._save()


PEOPLE_STORE = PeopleStore()


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


def _smoothed_box(track_id, raw_box, track_smoothed_box):
    """EMA-blend `raw_box` into the track's previously smoothed box, seeding
    with the raw box on first sight so a track doesn't visibly slide into
    place from (0, 0)."""
    prev = track_smoothed_box.get(track_id)
    if prev is None:
        smoothed = raw_box
    else:
        alpha = BOX_SMOOTHING_ALPHA
        smoothed = tuple(
            alpha * new + (1 - alpha) * old for new, old in zip(raw_box, prev)
        )
    track_smoothed_box[track_id] = smoothed
    return smoothed


class EventLog:
    """Records ENTER/EXIT events for each Person-N to a PostgreSQL database,
    so an admin dashboard can later show who was present and when without
    needing to run or parse this detection script itself."""

    def __init__(self, host, port, dbname, user, password):
        self._conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
        )
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('enter', 'exit')),
                    timestamp TEXT NOT NULL
                )
                """
            )

    def record(self, person_id, event_type, track_id=None):
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (person_id, event_type, timestamp) VALUES (%s, %s, %s)",
                (person_id, event_type, timestamp),
            )
        print(f"[EVENT] Person-{person_id} {event_type.upper()} at {timestamp} (track={track_id})")

    def reset(self):
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM events")

    def close(self):
        self._conn.close()


class LatestFrameReader:
    """Reads frames from a VideoCapture on a background thread and
    always exposes only the most recently read frame, discarding any
    older ones that the consumer didn't get to in time."""

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                time.sleep(0.1)

    def read(self):
        with self._lock:
            return self._ok, self._frame

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2)


class PersonGallery:
    """Session-long store of appearance embeddings, one bank per Person-N seen so far.

    BoT-SORT's own ReID only re-matches a lost track while it's still within
    track_buffer frames of being dropped. Once a track is gone for good and a
    person re-enters later, the tracker hands out a brand-new track ID. This
    gallery catches that case: whenever a new track ID shows up, its embedding
    is compared against everyone seen so far this session, and if it's a good
    enough match, the old Person-N is reused instead of minting a new one.

    Each person keeps a small bank of recent embeddings (rather than a single
    blended average) so a distinctive pose - e.g. head down - doesn't get
    diluted away or drift the average toward a pose that no longer matches
    normal sightings. A new embedding only needs to match the *best* of a
    person's stored samples, not the average of all of them.

    Body-appearance embeddings are defeated by a clothing/headwear change
    between a walk-out and a walk-back-in, since they're the only signal.
    A second, parallel bank of ArcFace face embeddings is kept per person as
    a fallback-resistant signal: a confident face match is checked first and
    wins outright (it survives outfit changes), and body appearance is only
    consulted when no face is available or no face matches confidently.

    Both banks are persisted to disk (see `save`/`load`) so a person named
    via the dashboard is still recognized as the same Person-N after the
    detection process restarts, instead of every restart wiping identities
    back to session-local numbering.
    """

    MAX_SAMPLES_PER_PERSON = _cfg.REID_MAX_SAMPLES_PER_PERSON

    def __init__(self, path=GALLERY_PATH):
        self._path = path
        self._embeddings = {}  # person_id -> deque of body embeddings (L2-normalized)
        self._face_embeddings = {}  # person_id -> deque of face embeddings (L2-normalized)
        self._next_id = 1
        self._lock = threading.Lock()
        self._dirty = False
        self._last_save_time = 0.0
        self.load()

    def reset(self):
        with self._lock:
            self._embeddings.clear()
            self._face_embeddings.clear()
            self._next_id = 1
            self._dirty = False
            if os.path.isfile(self._path):
                os.remove(self._path)

    def load(self):
        """Restore embedding banks and `_next_id` from disk, if a previous
        session left a gallery file. Missing/unreadable files just start
        empty rather than erroring, since a fresh install has none yet."""
        if not os.path.isfile(self._path):
            return
        try:
            data = np.load(self._path, allow_pickle=False)
        except Exception as exc:
            print(f"[GALLERY] Failed to load {self._path}: {exc}")
            return
        max_id = 0
        for key in data.files:
            kind, _, person_id_str = key.partition("_")
            person_id = int(person_id_str)
            max_id = max(max_id, person_id)
            bank_dict = self._embeddings if kind == "body" else self._face_embeddings
            max_len = (
                self.MAX_SAMPLES_PER_PERSON if kind == "body" else FACE_MAX_SAMPLES_PER_PERSON
            )
            bank_dict[person_id] = deque(data[key], maxlen=max_len)
        self._next_id = max_id + 1
        print(
            f"[GALLERY] Loaded {len(self._face_embeddings)} face and "
            f"{len(self._embeddings)} body identities from {self._path}"
        )

    def _write(self):
        """Persist both embedding banks to a single .npz, one array per
        person per bank (key format 'body_<id>' / 'face_<id>')."""
        arrays = {}
        for person_id, bank in self._embeddings.items():
            if bank:
                arrays[f"body_{person_id}"] = np.stack(bank)
        for person_id, bank in self._face_embeddings.items():
            if bank:
                arrays[f"face_{person_id}"] = np.stack(bank)
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp_path, self._path)
        self._dirty = False
        self._last_save_time = time.time()

    def save(self):
        """Mark the gallery dirty and write to disk immediately if at least
        GALLERY_SAVE_INTERVAL_SECONDS have passed since the last write.
        Debounces the previous every-call save, which under steady
        identify/top-up traffic was doing a full disk write per event."""
        self._dirty = True
        if time.time() - self._last_save_time >= GALLERY_SAVE_INTERVAL_SECONDS:
            self._write()

    def flush(self):
        """Force a write if there are unsaved changes - called on shutdown
        so the debounce in `save()` never loses the tail of a session."""
        with self._lock:
            if self._dirty:
                self._write()

    def _best_match(self, embedding, bank_dict):
        best_id, best_sim = None, -1.0
        for person_id, bank in bank_dict.items():
            for gallery_emb in bank:
                sim = float(np.dot(embedding, gallery_emb))
                if sim > best_sim:
                    best_id, best_sim = person_id, sim
        return best_id, best_sim

    def _new_id(self):
        person_id = self._next_id
        self._next_id += 1
        return person_id

    def identify_many(self, embeddings, face_embedding=None):
        """Return the Person-N id for a track, given several body-appearance
        candidate embeddings (matches if any is a close enough match, rather
        than betting everything on one frame) and an optional face embedding.

        A confident face match wins outright over body appearance, since it
        survives clothing/headwear changes that defeat body-ReID. Otherwise
        falls back to the body-appearance match, or mints a new Person-N."""
        normalized = []
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm >= 1e-12:
                normalized.append(embedding / norm)

        face_norm = None
        if face_embedding is not None:
            fn = np.linalg.norm(face_embedding)
            if fn >= 1e-12:
                face_norm = face_embedding / fn

        face_id, face_sim = (None, -1.0)
        if face_norm is not None:
            face_id, face_sim = self._best_match(face_norm, self._face_embeddings)

        if face_id is not None and face_sim >= FACE_MATCH_THRESHOLD:
            person_id = face_id
            with self._lock:
                self._face_embeddings[person_id].append(face_norm)
                for embedding in normalized:
                    self._embeddings.setdefault(
                        person_id, deque(maxlen=self.MAX_SAMPLES_PER_PERSON)
                    ).append(embedding)
                self.save()
            print(f"[IDENTITY] Person-{person_id} reused via FACE match (sim={face_sim:.2f})")
            return person_id

        body_id, body_sim = (None, -1.0)
        for embedding in normalized:
            cand_id, cand_sim = self._best_match(embedding, self._embeddings)
            if cand_sim > body_sim:
                body_id, body_sim = cand_id, cand_sim

        if body_id is not None and body_sim >= REID_MATCH_THRESHOLD:
            person_id = body_id
            print(f"[IDENTITY] Person-{person_id} reused via BODY match (sim={body_sim:.2f}, face_sim={face_sim:.2f})")
        elif not normalized and face_norm is None:
            person_id = self._new_id()
            print(f"[IDENTITY] Person-{person_id} minted NEW (no embeddings)")
            return person_id
        else:
            person_id = self._new_id()
            print(f"[IDENTITY] Person-{person_id} minted NEW (best body_sim={body_sim:.2f}, face_sim={face_sim:.2f})")

        with self._lock:
            for embedding in normalized:
                self._embeddings.setdefault(
                    person_id, deque(maxlen=self.MAX_SAMPLES_PER_PERSON)
                ).append(embedding)
            if face_norm is not None:
                self._face_embeddings.setdefault(
                    person_id, deque(maxlen=FACE_MAX_SAMPLES_PER_PERSON)
                ).append(face_norm)
            self.save()
        return person_id

    def add_sample(self, person_id, embedding=None, face_embedding=None):
        """Add another embedding to a person's bank(s) without re-matching -
        used to top up an already-identified track's gallery coverage with
        samples from later in its life (different distance/pose than the
        crops captured when the track first appeared)."""
        with self._lock:
            if embedding is not None:
                norm = np.linalg.norm(embedding)
                if norm >= 1e-12:
                    self._embeddings.setdefault(
                        person_id, deque(maxlen=self.MAX_SAMPLES_PER_PERSON)
                    ).append(embedding / norm)
            if face_embedding is not None:
                norm = np.linalg.norm(face_embedding)
                if norm >= 1e-12:
                    self._face_embeddings.setdefault(
                        person_id, deque(maxlen=FACE_MAX_SAMPLES_PER_PERSON)
                    ).append(face_embedding / norm)
            self.save()


def load_model():
    try:
        model = YOLO(MODEL_NAME)
        model.to(DEVICE)
        print(f"[INFO] Running inference on: {'GPU (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 0 else 'CPU'}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model '{MODEL_NAME}': {e}")
        sys.exit(1)


def load_pose_model():
    if not POSE_ENABLED:
        return None
    try:
        model = YOLO(POSE_MODEL_NAME)
        model.to(DEVICE)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO pose model '{POSE_MODEL_NAME}': {e}")
        sys.exit(1)


def draw_skeleton(frame, keypoints_xy, keypoints_conf, offset_x=0, offset_y=0):
    """Draws COCO-17 skeleton edges and joints for one person's keypoints.
    keypoints_xy/keypoints_conf are relative to (offset_x, offset_y)."""
    def pt(idx):
        if keypoints_conf[idx] < POSE_KEYPOINT_CONF_THRESHOLD:
            return None
        x, y = keypoints_xy[idx]
        return (int(x + offset_x), int(y + offset_y))

    for a, b in POSE_SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, POSE_COLOR, 2)

    for idx in range(len(keypoints_conf)):
        p = pt(idx)
        if p is not None:
            cv2.circle(frame, p, 3, POSE_COLOR, -1)


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_reid_encoder():
    try:
        return ReID(REID_MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"[ERROR] Failed to load ReID encoder: {e}")
        sys.exit(1)


def load_face_analyzer():
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == 0 else ["CPUExecutionProvider"]
        analyzer = FaceAnalysis(name="buffalo_s", providers=providers)
        analyzer.prepare(ctx_id=0 if DEVICE == 0 else -1, det_size=(640, 640))
        return analyzer
    except Exception as e:
        print(f"[ERROR] Failed to load InsightFace analyzer: {e}")
        sys.exit(1)


def best_face_embedding(analyzer, frame, xyxy):
    """Detect faces within a person crop and return the largest reliable
    face's ArcFace embedding, or None if no face clears FACE_MIN_BOX_HEIGHT."""
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    faces = analyzer.get(crop)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.bbox[3] - f.bbox[1])
    if (best.bbox[3] - best.bbox[1]) < FACE_MIN_BOX_HEIGHT:
        return None
    return best.normed_embedding


def open_stream():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[ERROR] Could not connect to RTSP stream: {RTSP_URL}")
        print("Check that the IP, port, credentials, and channel are correct,")
        print("and that the camera is reachable on the network.")
        sys.exit(1)
    return cap


class DetectionState:
    """Shared, thread-safe handle to the latest annotated frame and the
    latest smoothed human count, so a web server running on another thread
    can read live values without touching the detection loop's internals."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._count = 0
        self._fps = 0.0
        self._running = False
        self._stop_requested = False
        self._reset_requested = False
        self._present_person_ids = set()
        self._orig_w = 0
        self._orig_h = 0
        self._scale = 1.0
        self._zone_status = {}
        self._live_boxes = {}
        self._viewer_count = 0

    def add_viewer(self):
        with self._lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self._lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    def has_viewers(self):
        with self._lock:
            return self._viewer_count > 0

    def update(self, frame, count, fps, present_person_ids=None,
               orig_w=None, orig_h=None, scale=None, zone_status=None,
               live_boxes=None):
        # Skip the JPEG encode (a meaningful chunk of per-frame cost) when
        # nobody is watching /video_feed or requesting a zone snapshot.
        if self.has_viewers():
            ok, buf = cv2.imencode(".jpg", frame)
            jpeg = buf.tobytes() if ok else None
        else:
            jpeg = None
        with self._lock:
            if jpeg is not None:
                self._jpeg = jpeg
            self._count = count
            self._fps = fps
            if present_person_ids is not None:
                self._present_person_ids = set(present_person_ids)
            if orig_w is not None:
                self._orig_w = orig_w
            if orig_h is not None:
                self._orig_h = orig_h
            if scale is not None:
                self._scale = scale
            if zone_status is not None:
                self._zone_status = {
                    zone_id: {"occupants": set(info["occupants"]), "entered": info["entered"]}
                    for zone_id, info in zone_status.items()
                }
            if live_boxes is not None:
                self._live_boxes = dict(live_boxes)

    def live_boxes(self):
        with self._lock:
            return dict(self._live_boxes)

    def zone_status(self):
        with self._lock:
            return {
                zone_id: {"occupants": set(info["occupants"]), "entered": info["entered"]}
                for zone_id, info in self._zone_status.items()
            }

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def dims(self):
        with self._lock:
            return {
                "orig_w": self._orig_w,
                "orig_h": self._orig_h,
                "scale": self._scale,
            }

    def snapshot(self):
        with self._lock:
            return {
                "count": self._count,
                "fps": self._fps,
                "running": self._running,
                "present_person_ids": set(self._present_person_ids),
            }

    def mark_running(self, running):
        with self._lock:
            self._running = running

    def stop(self):
        self._stop_requested = True

    def reset_stop(self):
        """Clears the stop flag so a fresh detection thread can run again
        after a previous stop() call."""
        self._stop_requested = False

    @property
    def stop_requested(self):
        return self._stop_requested

    def request_reset(self):
        self._reset_requested = True

    def consume_reset_request(self):
        """Returns True (once) if a reset was requested, clearing the flag."""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False


def run_detection(state=None, show_window=True):
    """Runs the detection loop. If `state` (a DetectionState) is given, every
    annotated frame is published there for a web server to stream; if
    `show_window` is True, a local cv2 window is also shown (both can be
    used at once). Loop exits on 'q' (when a window is shown), Ctrl+C, or
    state.stop() being called."""
    model = load_model()
    pose_model = load_pose_model()
    reid_encoder = load_reid_encoder()
    face_analyzer = load_face_analyzer()
    gallery = PersonGallery()
    event_log = EventLog(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    track_to_person = {}  # BoT-SORT track_id -> Person-N id (persists until track drops)
    pending_tracks = {}  # track_id -> list of (score, bbox crop, local xywh) candidates
    pending_track_first_seen = {}  # track_id -> time.time() when first added to pending_tracks
    pending_track_zone_hit = {}  # track_id -> was its foot-point ever inside a web_zone while pending
    track_last_topup_time = {}  # track_id -> time.time() of last gallery top-up
    track_missing_since = {}  # track_id -> time.time() when it first went missing from tracker output
    present_person_ids = set()  # Person-N ids currently considered "in" (ENTER logged, no EXIT yet)
    person_in_zone = {}  # Person-N id -> was its last-seen foot-point inside any web_zone
    track_last_box = {}  # track_id -> last-seen (x1, y1, x2, y2)
    track_smoothed_box = {}  # track_id -> EMA-smoothed (x1, y1, x2, y2) used for drawing
    track_coast_frames = {}  # track_id -> consecutive frames drawn from a stale box while missing
    people_store = PEOPLE_STORE
    stage_timer = StageTimer() if PROFILE else None

    pose_frame_counter = 0
    cached_pose_boxes = []
    cached_pose_keypoints_xy = []
    cached_pose_keypoints_conf = []

    web_zones_store = WebZonesStore()
    web_zones = web_zones_store.zones
    if web_zones:
        print(f"[ZONES] Loaded {len(web_zones)} door zone(s) from {ZONES_PATH}")
    else:
        print("[ZONES] No web zones configured - ENTER/EXIT events will not be zone-gated "
              "(falling back to appear/disappear behavior). Draw zones from the web dashboard.")
    zone_occupants_prev = {z["id"]: set() for z in web_zones}  # zone_id -> person_ids inside last frame

    cap = open_stream()
    reader = LatestFrameReader(cap)

    window_name = "Human Detection - Press 'q' to quit"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    recent_counts = deque(maxlen=COUNT_SMOOTHING_WINDOW)

    if state is not None:
        state.mark_running(True)

    try:
        while True:
            if state is not None and state.stop_requested:
                print("Stop requested. Exiting.")
                break
            if state is not None and state.consume_reset_request():
                print("[RESET] Clearing gallery, tracking state, and event history.")
                gallery.reset()
                event_log.reset()
                track_to_person.clear()
                pending_tracks.clear()
                pending_track_first_seen.clear()
                track_last_topup_time.clear()
                track_missing_since.clear()
                present_person_ids.clear()
                track_last_box.clear()
                track_smoothed_box.clear()
                track_coast_frames.clear()
                people_store.clear()
                for zone_id in zone_occupants_prev:
                    zone_occupants_prev[zone_id] = set()
            if web_zones_store.maybe_reload():
                web_zones = web_zones_store.zones
                zone_occupants_prev = merge_zone_occupants(zone_occupants_prev, web_zones)
                print(f"[ZONES] Reloaded {len(web_zones)} door zone(s) from {ZONES_PATH}")
            ret, frame = reader.read()
            if not ret or frame is None:
                print("[WARNING] Waiting for frame from stream...")
                time.sleep(0.1)
                continue

            if stage_timer is not None:
                _t0 = time.time()
            results = model.track(
                frame, persist=True, tracker=TRACKER_CONFIG, device=DEVICE, verbose=False,
            )[0]
            if stage_timer is not None:
                stage_timer.add("track", time.time() - _t0)

            if stage_timer is not None:
                _t0 = time.time()
            if pose_model is not None and pose_frame_counter % POSE_EVERY_N_FRAMES == 0:
                pose_results = pose_model(frame, conf=POSE_CONF_THRESHOLD, device=DEVICE, verbose=False)[0]
                cached_pose_boxes = (
                    pose_results.boxes.xyxy.tolist() if pose_results.boxes is not None else []
                )
                cached_pose_keypoints_xy = (
                    pose_results.keypoints.xy.tolist() if pose_results.keypoints is not None else []
                )
                cached_pose_keypoints_conf = (
                    pose_results.keypoints.conf.tolist()
                    if pose_results.keypoints is not None and pose_results.keypoints.conf is not None
                    else []
                )
            pose_frame_counter += 1
            pose_boxes = cached_pose_boxes
            pose_keypoints_xy = cached_pose_keypoints_xy
            pose_keypoints_conf = cached_pose_keypoints_conf
            if stage_timer is not None:
                stage_timer.add("pose", time.time() - _t0)

            human_count = 0
            zone_occupants_now = {}  # zone_id -> person_ids inside this frame
            boxes = results.boxes
            track_ids = boxes.id
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != PERSON_CLASS_ID:
                    continue

                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                box_w, box_h = bx2 - bx1, by2 - by1
                frame_area = frame.shape[0] * frame.shape[1]
                below_conf = conf < CONF_THRESHOLD
                below_area = (box_w * box_h) < MIN_BOX_AREA_FRACTION * frame_area

                if below_conf or below_area:
                    # Debug overlay: show what got filtered out and why, so
                    # borderline people (occluded/off-angle/small) can be seen
                    # scoring just under the cutoffs instead of vanishing silently.
                    reason = "conf" if below_conf else "area"
                    cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 1)
                    cv2.putText(
                        frame, f"{conf:.2f} <{reason}>", (int(bx1), max(int(by1) - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                    )
                    continue

                if track_ids is None:
                    continue

                track_id = int(track_ids[i])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_last_box[track_id] = (x1, y1, x2, y2)
                track_coast_frames[track_id] = 0
                sx1, sy1, sx2, sy2 = _smoothed_box(
                    track_id, (x1, y1, x2, y2), track_smoothed_box
                )

                if track_id not in track_to_person:
                    box_w, box_h = x2 - x1, y2 - y1
                    frame_h, frame_w = frame.shape[:2]
                    # Raw box area rewards close-up crops that are actually
                    # clipped by the frame edge (e.g. just a chin/ear when
                    # someone is right next to the camera), which produces a
                    # near-useless embedding. Penalize crops touching any
                    # frame edge and crops that are wider than tall (usually
                    # a partial view at close range), so a smaller but fully-
                    # visible, person-shaped crop wins instead.
                    touches_edge = x1 <= 1 or y1 <= 1 or x2 >= frame_w - 1 or y2 >= frame_h - 1
                    aspect_ok = box_h >= box_w
                    quality = (0 if touches_edge else 2) + (1 if aspect_ok else 0)
                    box_area = box_w * box_h
                    score = (quality, box_area)
                    cx1, cy1 = max(int(x1), 0), max(int(y1), 0)
                    cx2, cy2 = min(int(x2), frame_w), min(int(y2), frame_h)
                    crop = frame[cy1:cy2, cx1:cx2].copy()
                    crop_h, crop_w = crop.shape[:2]
                    xywh = np.array([[crop_w / 2, crop_h / 2, crop_w, crop_h]])
                    candidates = pending_tracks.setdefault(track_id, [])
                    candidates.append((score, crop, xywh))
                    first_seen = pending_track_first_seen.setdefault(track_id, time.time())
                    in_zone_now = _foot_point_in_any_zone(x1, y1, x2, y2, web_zones)
                    if in_zone_now:
                        pending_track_zone_hit[track_id] = True

                    elapsed = time.time() - first_seen
                    if len(candidates) < IDENTIFY_MIN_CANDIDATES or elapsed < IDENTIFY_DELAY_SECONDS:
                        human_count += 1
                        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), BOX_COLOR, 2)
                        continue

                    # A single "best" frame can still be an unusual pose (e.g.
                    # head tucked all the way down) that embeds farther from
                    # a person's other sightings than their embeddings are
                    # from each other. Taking the top few candidates and
                    # keeping every resulting embedding - matching against
                    # whichever one scores highest - smooths over a single
                    # outlier pose instead of betting everything on one crop.
                    top_candidates = sorted(candidates, key=lambda c: c[0], reverse=True)[:3]
                    del pending_tracks[track_id]
                    embeddings = []
                    face_embedding = None
                    for score, cand_crop, cand_xywh in top_candidates:
                        if stage_timer is not None:
                            _t0 = time.time()
                        emb = reid_encoder(cand_crop, cand_xywh)[0]
                        if stage_timer is not None:
                            stage_timer.add("reid", time.time() - _t0)
                        if emb is not None:
                            embeddings.append(emb)
                        if face_embedding is None:
                            cx, cy, cw, ch = cand_xywh[0]
                            cand_xyxy = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
                            if stage_timer is not None:
                                _t0 = time.time()
                            face_embedding = best_face_embedding(face_analyzer, cand_crop, cand_xyxy)
                            if stage_timer is not None:
                                stage_timer.add("face", time.time() - _t0)
                    if not embeddings and face_embedding is None:
                        continue
                    track_to_person[track_id] = gallery.identify_many(embeddings, face_embedding)
                    track_last_topup_time[track_id] = time.time()
                    pending_track_first_seen.pop(track_id, None)
                    new_person_id = track_to_person[track_id]
                    other_live_tracks = [
                        tid for tid, pid in track_to_person.items()
                        if pid == new_person_id and tid != track_id
                    ]
                    if other_live_tracks:
                        print(f"[IDENTITY] Person-{new_person_id} now mapped to MULTIPLE live tracks: "
                              f"{sorted(other_live_tracks + [track_id])} - possible ReID false-positive merge.")
                    now_in_zone = _foot_point_in_any_zone(x1, y1, x2, y2, web_zones)
                    ever_in_zone = pending_track_zone_hit.pop(track_id, False) or now_in_zone
                    if new_person_id not in present_person_ids and (not web_zones or ever_in_zone):
                        present_person_ids.add(new_person_id)
                        event_log.record(new_person_id, "enter", track_id=track_id)
                else:
                    last_topup = track_last_topup_time.setdefault(track_id, time.time())
                    if time.time() - last_topup >= REID_TOPUP_INTERVAL_SECONDS:
                        frame_h, frame_w = frame.shape[:2]
                        touches_edge = x1 <= 1 or y1 <= 1 or x2 >= frame_w - 1 or y2 >= frame_h - 1
                        aspect_ok = (y2 - y1) >= (x2 - x1)
                        if not touches_edge and aspect_ok:
                            xywh = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]])
                            if stage_timer is not None:
                                _t0 = time.time()
                            emb = reid_encoder(frame, xywh)[0]
                            if stage_timer is not None:
                                stage_timer.add("reid_topup", time.time() - _t0)
                                _t0 = time.time()
                            face_embedding = best_face_embedding(face_analyzer, frame, (x1, y1, x2, y2))
                            if stage_timer is not None:
                                stage_timer.add("face_topup", time.time() - _t0)
                            if emb is not None or face_embedding is not None:
                                gallery.add_sample(track_to_person[track_id], emb, face_embedding)
                        track_last_topup_time[track_id] = time.time()
                person_id = track_to_person[track_id]
                track_last_box[track_id] = (x1, y1, x2, y2)
                if web_zones:
                    for wz in web_zones:
                        if point_in_polygon((x1 + x2) / 2, y2, wz["points"]):
                            zone_occupants_now.setdefault(wz["id"], set()).add(person_id)
                person_in_zone[person_id] = _foot_point_in_any_zone(x1, y1, x2, y2, web_zones)

                human_count += 1
                x1, y1, x2, y2 = map(int, (sx1, sy1, sx2, sy2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                assigned_name = people_store.get(person_id)
                label = assigned_name if assigned_name else f"Person-{person_id}"
                cv2.putText(
                    frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, BOX_COLOR, 2,
                )

                best_iou, best_pose_idx = 0.0, -1
                for pi, pbox in enumerate(pose_boxes):
                    iou = _box_iou((x1, y1, x2, y2), pbox)
                    if iou > best_iou:
                        best_iou, best_pose_idx = iou, pi
                if best_pose_idx >= 0 and best_iou > POSE_IOU_MATCH_THRESHOLD:
                    draw_skeleton(
                        frame,
                        pose_keypoints_xy[best_pose_idx],
                        pose_keypoints_conf[best_pose_idx],
                    )

            seen_track_ids = set()
            if track_ids is not None:
                seen_track_ids = {int(tid) for tid in track_ids.tolist()}
            for seen_track_id in seen_track_ids:
                track_missing_since.pop(seen_track_id, None)
            known_track_ids = (
                set(track_missing_since) | set(track_to_person) | set(pending_tracks)
            )
            for missing_track_id in known_track_ids - seen_track_ids:
                coast = track_coast_frames.get(missing_track_id, 0)
                stale_box = track_smoothed_box.get(missing_track_id)
                if stale_box is not None and coast < BOX_COAST_FRAMES:
                    track_coast_frames[missing_track_id] = coast + 1
                    cx1, cy1, cx2, cy2 = map(int, stale_box)
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), BOX_COLOR, 2)
                    coast_person_id = track_to_person.get(missing_track_id)
                    if coast_person_id is not None:
                        human_count += 1
                        coast_name = people_store.get(coast_person_id)
                        label = coast_name if coast_name else f"Person-{coast_person_id}"
                        cv2.putText(
                            frame, label, (cx1, max(cy1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, BOX_COLOR, 2,
                        )

                first_missing = track_missing_since.setdefault(missing_track_id, time.time())
                if time.time() - first_missing < EXIT_GRACE_SECONDS:
                    continue
                # A track that never accumulated enough candidate frames to
                # reach IDENTIFY_DELAY_SECONDS/IDENTIFY_MIN_CANDIDATES before
                # the tracker dropped it (e.g. a brief false detection) would
                # otherwise sit in pending_tracks forever, permanently
                # blocking the check below. It never became a real Person-N,
                # so just discard it.
                pending_tracks.pop(missing_track_id, None)
                pending_track_first_seen.pop(missing_track_id, None)
                pending_track_zone_hit.pop(missing_track_id, None)
                if pending_tracks:
                    # One or more *other* tracks haven't finished their
                    # identify delay yet. If the tracker split this same
                    # person into a new track_id right as this one dropped,
                    # that new track will resolve to the same Person-N a few
                    # frames from now via the gallery - but only if we don't
                    # tear down this track's bookkeeping first. Hold the EXIT
                    # decision until pending tracks resolve, instead of firing
                    # a spurious EXIT/ENTER pair for one continuous visit.
                    continue
                track_missing_since.pop(missing_track_id, None)
                person_id = track_to_person.pop(missing_track_id, None)
                track_last_topup_time.pop(missing_track_id, None)
                last_box = track_last_box.pop(missing_track_id, None)
                track_smoothed_box.pop(missing_track_id, None)
                track_coast_frames.pop(missing_track_id, None)
                if person_id is None:
                    continue
                # Only tracks the tracker actually reported *this frame* count
                # as "still here" - a track_id sitting elsewhere in
                # track_to_person may itself be mid-EXIT-grace (or belong to a
                # later, genuinely separate visit by the same recognized
                # person), and must not suppress this EXIT.
                still_present = any(
                    track_to_person.get(other_tid) == person_id
                    for other_tid in seen_track_ids
                    if other_tid != missing_track_id
                )
                was_in_zone = person_in_zone.pop(person_id, False)
                if (
                    not still_present
                    and person_id in present_person_ids
                    and (not web_zones or was_in_zone)
                ):
                    present_person_ids.discard(person_id)
                    event_log.record(person_id, "exit", track_id=missing_track_id)

            zone_status = {}
            for wz in web_zones:
                zone_id = wz["id"]
                occupants_now = zone_occupants_now.get(zone_id, set())
                occupants_prev = zone_occupants_prev.get(zone_id, set())
                entered = bool(occupants_now - occupants_prev)
                zone_status[zone_id] = {"occupants": occupants_now, "entered": entered}
            zone_occupants_prev = {
                wz["id"]: zone_occupants_now.get(wz["id"], set()) for wz in web_zones
            }

            recent_counts.append(human_count)
            smoothed_count = max(set(recent_counts), key=recent_counts.count)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            print(f"Humans detected: {smoothed_count} (raw: {human_count}) | FPS: {fps:.1f}")

            cv2.putText(
                frame, f"FPS: {fps:.1f}  Humans: {smoothed_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
            )

            h, w = frame.shape[:2]
            scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
            display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            if state is not None:
                if stage_timer is not None:
                    _t0 = time.time()
                live_boxes = {
                    track_to_person[tid]: box
                    for tid, box in track_last_box.items()
                    if tid in track_to_person and track_to_person[tid] in present_person_ids
                }
                state.update(display_frame, smoothed_count, fps, present_person_ids,
                             orig_w=w, orig_h=h, scale=scale, zone_status=zone_status,
                             live_boxes=live_boxes)
                if stage_timer is not None:
                    stage_timer.add("state_update(imencode)", time.time() - _t0)

            if show_window:
                cv2.imshow(window_name, display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit key pressed. Exiting.")
                    break

            if stage_timer is not None:
                stage_timer.end_frame()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        reader.stop()
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        event_log.close()
        gallery.flush()
        if state is not None:
            state.mark_running(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--web", action="store_true",
        help="Serve the detection feed and stats over a LAN-accessible web dashboard "
             "(FastAPI + uvicorn) instead of a local cv2 window.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host/interface to bind the web dashboard to (default: 127.0.0.1, localhost-only; "
             "use 0.0.0.0 for LAN access).",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the web dashboard (default: 8000).",
    )
    parser.add_argument(
        "--show-window", action="store_true",
        help="When used with --web, also show the local cv2 window alongside the web dashboard.",
    )
    args = parser.parse_args()
    if args.web:
        from webapp import serve_dashboard
        serve_dashboard(host=args.host, port=args.port, show_window=args.show_window)
    else:
        run_detection(state=None, show_window=True)
