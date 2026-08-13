"""Main per-frame orchestration: detect -> track -> identify -> zones ->
draw -> encode. Also owns DetectionState, the thread-safe handle a web
server reads live frames/stats from."""

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config as _config
from identity import (
    PEOPLE_STORE,
    PersonGallery,
    load_reid_encoder,
    load_face_analyzer,
    best_face_embedding,
)
from zones import (
    bump_zones_version,
    merge_zone_occupants,
    WebZonesStore,
    point_in_polygon,
    _foot_point_in_any_zone,
    box_in_any_ignore_zone,
    split_ignore_zones,
)
from stream import LatestFrameReader, open_stream
from dataset_collector import DatasetCollector
from events import EventLog, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import nestjs_ingest

_cfg = _config.load()

DEVICE = 0 if torch.cuda.is_available() else "cpu"

INFER_IMGSZ = _cfg.INFER_IMGSZ
# fp16 is a real speedup on the GPU but emulated (slower) on CPU.
INFER_HALF = _cfg.INFER_HALF and DEVICE == 0
TARGET_FPS = _cfg.TARGET_FPS
JPEG_QUALITY = _cfg.JPEG_QUALITY
# How long a /api/zone-snapshot request keeps the encode loop "warm" for,
# so a poller (Zones/People tab) gets fresh frames like an MJPEG viewer does.
SNAPSHOT_VIEWER_TIMEOUT = 5.0

# Extra kwargs shared by the detect and pose calls. Built once so the
# per-frame call sites stay readable and the two models can't drift apart.
# Ultralytics 8.4 replaced the `half` flag with `quantize`, where 16 means fp16;
# passing `half` still works but emits a deprecation warning on every single
# call, which would flood the log at ~20 calls a second.
_INFER_KWARGS = {}
if INFER_HALF:
    _INFER_KWARGS["quantize"] = 16
if INFER_IMGSZ:
    _INFER_KWARGS["imgsz"] = INFER_IMGSZ

MODEL_NAME = _cfg.MODEL_NAME
TRACKER_CONFIG = _cfg.TRACKER_CONFIG
PERSON_CLASS_ID = _cfg.PERSON_CLASS_ID
CONF_THRESHOLD = _cfg.CONF_THRESHOLD
BOX_COLOR = (0, 255, 0)  # green, BGR
DISPLAY_MAX_WIDTH = _cfg.DISPLAY_MAX_WIDTH
DISPLAY_MAX_HEIGHT = _cfg.DISPLAY_MAX_HEIGHT

POSE_MODEL_NAME = _cfg.POSE_MODEL_NAME
POSE_ENABLED = _cfg.POSE_ENABLED
POSE_EVERY_N_FRAMES = max(1, _cfg.POSE_EVERY_N_FRAMES)
POSE_CONF_THRESHOLD = _cfg.POSE_CONF_THRESHOLD
POSE_KEYPOINT_CONF_THRESHOLD = _cfg.POSE_KEYPOINT_CONF_THRESHOLD
POSE_COLOR = (0, 200, 255)  # orange, BGR
POSE_IOU_MATCH_THRESHOLD = _cfg.POSE_IOU_MATCH_THRESHOLD
POSE_SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders, arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (0, 5), (0, 6),  # head to shoulders
]

MIN_BOX_AREA_FRACTION = _cfg.MIN_BOX_AREA_FRACTION
COUNT_SMOOTHING_WINDOW = _cfg.COUNT_SMOOTHING_WINDOW
BOX_SMOOTHING_ALPHA = _cfg.BOX_SMOOTHING_ALPHA
BOX_COAST_FRAMES = _cfg.BOX_COAST_FRAMES

REID_MATCH_THRESHOLD = _cfg.REID_MATCH_THRESHOLD
FACE_MATCH_THRESHOLD = _cfg.FACE_MATCH_THRESHOLD
FACE_MAX_SAMPLES_PER_PERSON = _cfg.FACE_MAX_SAMPLES_PER_PERSON
FACE_MIN_BOX_HEIGHT = _cfg.FACE_MIN_BOX_HEIGHT

REID_TOPUP_INTERVAL_SECONDS = _cfg.REID_TOPUP_INTERVAL_SECONDS
IDENTIFY_DELAY_SECONDS = _cfg.IDENTIFY_DELAY_SECONDS
IDENTIFY_MIN_CANDIDATES = _cfg.IDENTIFY_MIN_CANDIDATES

ZONES_PATH = _cfg.ZONES_PATH

EXIT_GRACE_SECONDS = _cfg.EXIT_GRACE_SECONDS

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
        self._last_snapshot_request = 0.0

    def add_viewer(self):
        with self._lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self._lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    def request_snapshot(self):
        """Called by /api/zone-snapshot so the encode loop knows a poller
        (Zones/People tab) wants fresh frames, same as an MJPEG viewer."""
        with self._lock:
            self._last_snapshot_request = time.time()

    def has_viewers(self):
        with self._lock:
            recent_snapshot_request = (
                time.time() - self._last_snapshot_request < SNAPSHOT_VIEWER_TIMEOUT
            )
            return self._viewer_count > 0 or recent_snapshot_request

    def update(self, frame, count, fps, present_person_ids=None,
               orig_w=None, orig_h=None, scale=None, zone_status=None,
               live_boxes=None):
        # Skip the JPEG encode (a meaningful chunk of per-frame cost) when
        # nobody is watching /video_feed or requesting a zone snapshot.
        if self.has_viewers():
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
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
                    zone_id: {
                        "occupants": set(info["occupants"]),
                        "entered": info["entered"],
                        "left": info["left"],
                    }
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
                zone_id: {
                    "occupants": set(info["occupants"]),
                    "entered": info["entered"],
                    "left": info["left"],
                }
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


def should_enter(person_id, present_person_ids):
    """True if a person who was just identified should be marked ENTER (added
    to present_person_ids, which is what makes them show up as a clickable,
    nameable box in the People tab via live_boxes). Door-zone crossing is
    deliberately not a factor: door zones are narrow strips near the frame
    edges, so a lone person who stays in the middle of the room never has a
    foot-point inside one, and gating ENTER on that leaves them with no box
    to click at all -- unable to ever be named. Mirrors the should_exit fix,
    which removed the same zone gate from the other side of presence."""
    return person_id not in present_person_ids


def should_exit(still_present, person_id, present_person_ids):
    """True if a person whose track just went missing (past EXIT_GRACE_SECONDS)
    should be marked EXIT. Door-zone position at the moment tracking was lost
    is deliberately not a factor: someone who exits by walking out of camera
    view (the common case) is rarely standing in a door polygon on their last
    tracked frame, since door zones are narrow strips near the frame edges.
    Gating EXIT on that would leave them stuck "present" forever with no
    recovery path. ENTER no longer gates on door zones either (see
    should_enter above) -- both directions of presence are now zone-free."""
    return not still_present and person_id in present_person_ids


def compute_seen_track_ids(track_ids, ignored_track_ids):
    """Returns the set of track_ids the tracker reported this frame that were
    actually processed (i.e. not discarded for being inside an ignore zone).
    A track_id skipped via the ignore-zone check must not count as "seen" -
    otherwise it never enters the missing-track/EXIT_GRACE_SECONDS logic and
    a person who walks into an ignore zone is never marked EXIT."""
    if track_ids is None:
        return set()
    return {int(tid) for tid in track_ids.tolist()} - ignored_track_ids


def _compute_zone_status(web_zones, zone_occupants_now, zone_occupants_prev):
    """Pure, per-zone occupancy diff with no side effects and no reference to
    individual person_ids (it only ever looks at zone-level occupant sets),
    which keeps it unit-testable in isolation. Returns, per zone id:
    {"occupants": set, "entered": bool, "left": bool}. "entered" is True if
    anyone newly arrived in the zone this frame; "left" is True if anyone who
    was in the zone last frame is no longer there this frame. The two are not
    mutually exclusive -- one person can leave while another enters the same
    zone in the same frame."""
    zone_status = {}
    for wz in web_zones:
        zone_id = wz["id"]
        occupants_now = zone_occupants_now.get(zone_id, set())
        occupants_prev = zone_occupants_prev.get(zone_id, set())
        entered = bool(occupants_now - occupants_prev)
        left = bool(occupants_prev - occupants_now)
        zone_status[zone_id] = {
            "occupants": occupants_now,
            "entered": entered,
            "left": left,
        }
    return zone_status


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
    dataset_collector = DatasetCollector(_cfg)
    track_to_person = {}  # BoT-SORT track_id -> Person-N id (persists until track drops)
    pending_tracks = {}  # track_id -> list of (score, bbox crop, local xywh) candidates
    pending_track_first_seen = {}  # track_id -> time.time() when first added to pending_tracks
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
    web_zones, ignore_zones = split_ignore_zones(web_zones_store.zones)
    nestjs_ingest.sync_zones(web_zones)
    if web_zones:
        print(f"[ZONES] Loaded {len(web_zones)} door zone(s) from {ZONES_PATH}")
    else:
        print("[ZONES] No web zones configured - ENTER/EXIT events will not be zone-gated "
              "(falling back to appear/disappear behavior). Draw zones from the web dashboard.")
    if ignore_zones:
        print(f"[ZONES] Loaded {len(ignore_zones)} ignore zone(s) - detections inside will be discarded")
    zone_occupants_prev = {z["id"]: set() for z in web_zones}  # zone_id -> person_ids inside last frame

    cap = open_stream()
    reader = LatestFrameReader(cap, reopen_fn=lambda: open_stream(exit_on_failure=False))

    window_name = "Human Detection - Press 'q' to quit"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    recent_counts = deque(maxlen=COUNT_SMOOTHING_WINDOW)
    last_frame_seq = 0
    # Reported FPS is smoothed over a short window. The raw 1/dt of a single
    # frame swings wildly with how many people are in view, which makes a
    # steady pipeline look erratic on the dashboard.
    recent_frame_times = deque(maxlen=15)
    min_frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.0
    next_frame_deadline = time.time()

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
                web_zones, ignore_zones = split_ignore_zones(web_zones_store.zones)
                zone_occupants_prev = merge_zone_occupants(zone_occupants_prev, web_zones)
                nestjs_ingest.sync_zones(web_zones)
                print(f"[ZONES] Reloaded {len(web_zones)} door zone(s), "
                      f"{len(ignore_zones)} ignore zone(s) from {ZONES_PATH}")
            # Pace the loop to TARGET_FPS. read_latest already blocks until the
            # camera has a genuinely new frame, so this only bites when the
            # source runs faster than we want to process; sleeping the
            # remainder here (rather than after the read) means the frame we
            # then pick up is the freshest one, not one aged by the wait.
            if min_frame_interval:
                slack = next_frame_deadline - time.time()
                if slack > 0:
                    time.sleep(slack)
                next_frame_deadline = max(
                    time.time(), next_frame_deadline + min_frame_interval
                )

            ret, frame, last_frame_seq = reader.read_latest(last_frame_seq)
            if not ret or frame is None:
                print("[WARNING] Waiting for frame from stream...")
                time.sleep(0.1)
                continue

            if stage_timer is not None:
                _t0 = time.time()
            results = model.track(
                frame, persist=True, tracker=TRACKER_CONFIG, device=DEVICE, verbose=False,
                conf=CONF_THRESHOLD, **_INFER_KWARGS,
            )[0]
            if stage_timer is not None:
                stage_timer.add("track", time.time() - _t0)

            if stage_timer is not None:
                _t0 = time.time()
            if pose_model is not None and pose_frame_counter % POSE_EVERY_N_FRAMES == 0:
                pose_results = pose_model(
                    frame, conf=POSE_CONF_THRESHOLD, device=DEVICE, verbose=False,
                    **_INFER_KWARGS,
                )[0]
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
            ignored_track_ids_this_frame = set()  # track_ids skipped this frame for being inside an ignore zone
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
                if ignore_zones and box_in_any_ignore_zone(x1, y1, x2, y2, ignore_zones):
                    ignored_track_ids_this_frame.add(track_id)
                    continue
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
                    def _has_other_live_track(candidate_person_id):
                        return any(
                            pid == candidate_person_id and tid != track_id
                            for tid, pid in track_to_person.items()
                        )

                    track_to_person[track_id] = gallery.identify_many(
                        embeddings, face_embedding, reject_id=_has_other_live_track
                    )
                    dataset_collector.save(track_to_person[track_id], top_candidates[0][1])
                    track_last_topup_time[track_id] = time.time()
                    pending_track_first_seen.pop(track_id, None)
                    new_person_id = track_to_person[track_id]
                    if should_enter(new_person_id, present_person_ids):
                        present_person_ids.add(new_person_id)
                        event_log.record(new_person_id, "enter", track_id=track_id)
                        nestjs_ingest.send_detection(new_person_id, datetime.now(timezone.utc).isoformat())
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
                                cx1, cy1 = max(int(x1), 0), max(int(y1), 0)
                                cx2, cy2 = min(int(x2), frame_w), min(int(y2), frame_h)
                                dataset_collector.save(track_to_person[track_id], frame[cy1:cy2, cx1:cx2])
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

            seen_track_ids = compute_seen_track_ids(track_ids, ignored_track_ids_this_frame)
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
                now = time.time()
                still_resolving = any(
                    now - first_seen < IDENTIFY_DELAY_SECONDS + EXIT_GRACE_SECONDS
                    for first_seen in pending_track_first_seen.values()
                )
                if still_resolving:
                    # One or more *other* tracks haven't finished their
                    # identify delay yet. If the tracker split this same
                    # person into a new track_id right as this one dropped,
                    # that new track will resolve to the same Person-N a few
                    # frames from now via the gallery - but only if we don't
                    # tear down this track's bookkeeping first. Hold the EXIT
                    # decision until pending tracks resolve, instead of firing
                    # a spurious EXIT/ENTER pair for one continuous visit.
                    #
                    # Capped by IDENTIFY_DELAY_SECONDS + EXIT_GRACE_SECONDS: a
                    # pending track can only ever need that long to resolve
                    # into a Person-N (see the elapsed check above). Past
                    # that, it's stuck (e.g. lost before reaching
                    # IDENTIFY_MIN_CANDIDATES) and must not be allowed to
                    # block every other person's EXIT indefinitely.
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
                person_in_zone.pop(person_id, None)
                if should_exit(still_present, person_id, present_person_ids):
                    present_person_ids.discard(person_id)
                    event_log.record(person_id, "exit", track_id=missing_track_id)
                    nestjs_ingest.send_detection(person_id, datetime.now(timezone.utc).isoformat())

            zone_status = _compute_zone_status(
                web_zones, zone_occupants_now, zone_occupants_prev
            )
            for wz in web_zones:
                zone_id = wz["id"]
                occupants_now = zone_occupants_now.get(zone_id, set())
                occupants_prev = zone_occupants_prev.get(zone_id, set())
                now_iso = datetime.now(timezone.utc).isoformat()
                for arrived_person_id in occupants_now - occupants_prev:
                    nestjs_ingest.send_detection(
                        arrived_person_id, now_iso, zone_id=zone_id, zone_event="enter"
                    )
                for left_person_id in occupants_prev - occupants_now:
                    nestjs_ingest.send_detection(
                        left_person_id, now_iso, zone_id=zone_id, zone_event="leave"
                    )
            zone_occupants_prev = {
                wz["id"]: zone_occupants_now.get(wz["id"], set()) for wz in web_zones
            }

            recent_counts.append(human_count)
            smoothed_count = max(set(recent_counts), key=recent_counts.count)

            now = time.time()
            if now > prev_time:
                recent_frame_times.append(now - prev_time)
            prev_time = now
            # Mean over the window, not 1/dt of the newest frame: a single slow
            # frame (someone walks in, identification fires) otherwise makes the
            # readout jump even though throughput is steady.
            fps = (
                len(recent_frame_times) / sum(recent_frame_times)
                if recent_frame_times else 0.0
            )

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
        if show_window:
            cv2.destroyAllWindows()
        gallery.flush()
        event_log.close()
        if state is not None:
            state.mark_running(False)
