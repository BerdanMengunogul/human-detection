"""Re-identification: appearance/face gallery, person_id assignment, and
the persistent person_id -> name mapping."""

import json
import os
import site
import threading
import time
from collections import deque

# onnxruntime-gpu's CUDA execution provider needs a system-PATH-discoverable
# CUDA 12.x + cuDNN 9.x runtime; PyTorch bundles its own copy privately, so
# it doesn't help onnxruntime. These pip-installed NVIDIA redistributable
# packages provide that runtime; this must run before onnxruntime/insightface
# import for the DLL search path change to take effect.
for _site_dir in site.getsitepackages():
    _nvidia_root = os.path.join(_site_dir, "nvidia")
    if os.path.isdir(_nvidia_root):
        for _pkg in ("cudnn", "cuda_runtime", "cublas", "cuda_nvrtc", "cufft"):
            _dll_dir = os.path.join(_nvidia_root, _pkg, "bin")
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)
                os.environ["PATH"] = _dll_dir + os.pathsep + os.environ["PATH"]
        break

import numpy as np
import torch
from insightface.app import FaceAnalysis
from ultralytics.trackers.utils.reid import ReID

import config as _config

DEVICE = 0 if torch.cuda.is_available() else "cpu"

_cfg = _config.load()

PEOPLE_PATH = _cfg.PEOPLE_PATH
GALLERY_PATH = _cfg.GALLERY_PATH
GALLERY_SAVE_INTERVAL_SECONDS = _cfg.GALLERY_SAVE_INTERVAL_SECONDS
REID_MODEL_NAME = _cfg.REID_MODEL_NAME
REID_MATCH_THRESHOLD = _cfg.REID_MATCH_THRESHOLD
REID_BORDERLINE_THRESHOLD = _cfg.REID_BORDERLINE_THRESHOLD
REID_CORROBORATION_THRESHOLD = _cfg.REID_CORROBORATION_THRESHOLD
REID_CORROBORATION_MIN_SAMPLES = _cfg.REID_CORROBORATION_MIN_SAMPLES
REID_HIGH_CONF_THRESHOLD = _cfg.REID_HIGH_CONF_THRESHOLD
REID_TOPK_MATCH = _cfg.REID_TOPK_MATCH
REID_MAX_SAMPLES_PER_PERSON = _cfg.REID_MAX_SAMPLES_PER_PERSON
FACE_MATCH_THRESHOLD = _cfg.FACE_MATCH_THRESHOLD
FACE_HIGH_CONF_THRESHOLD = _cfg.FACE_HIGH_CONF_THRESHOLD
FACE_MAX_SAMPLES_PER_PERSON = _cfg.FACE_MAX_SAMPLES_PER_PERSON
FACE_MIN_BOX_HEIGHT = _cfg.FACE_MIN_BOX_HEIGHT


class PeopleStore:
    """Thread-safe, disk-persisted person_id -> name mapping, so names
    given via the dashboard survive a detection-process restart."""

    def __init__(self, path=PEOPLE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._names = self._load()

    def _load(self):
        if not os.path.isfile(self._path):
            return {}
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.get("names", {}).items()}
        except Exception as exc:
            print(f"[PEOPLE] Failed to load {self._path}: {exc}")
            return {}

    def _save(self):
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"names": {str(k): v for k, v in self._names.items()}}, f)
        os.replace(tmp_path, self._path)

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

    def _best_match(self, embedding, bank_dict, top_k=1):
        """Compare `embedding` against every person's bank and return the
        id with the highest score, where a person's score is the mean of
        their top-`top_k` per-sample similarities (falling back to fewer
        samples for banks smaller than top_k), plus that same winner's
        single-best per-sample similarity. top_k=1 reproduces plain
        single-best-sample matching (in which case both returned scores
        are identical).

        Averaging the top few samples (instead of just the single closest
        one) protects against a person being permanently unmatchable after
        one atypical/noisy early sighting, since a fresh bank of only 1-2
        samples has no averaging to smooth that out - the "cold-bank"
        fragmentation case this is meant to fix. The single-best score is
        kept alongside it for callers deciding whether a match is strong
        enough to override a same-person conflict: a below-threshold mean
        can still hide a genuinely high-confidence best-sample match, e.g.
        one unusual pose among the candidate crops pulling the average
        down."""
        best_id, best_sim, best_top_sim = None, -1.0, -1.0
        for person_id, bank in bank_dict.items():
            sims = sorted((float(np.dot(embedding, e)) for e in bank), reverse=True)
            if not sims:
                continue
            sim = sum(sims[:top_k]) / min(top_k, len(sims))
            if sim > best_sim:
                best_id, best_sim, best_top_sim = person_id, sim, sims[0]
        return best_id, best_sim, best_top_sim

    def _corroboration_count(self, embedding, bank, threshold):
        return sum(1 for e in bank if float(np.dot(embedding, e)) >= threshold)

    def _new_id(self):
        person_id = self._next_id
        self._next_id += 1
        return person_id

    def identify_many(self, embeddings, face_embedding=None, reject_id=None):
        """Return the Person-N id for a track, given several body-appearance
        candidate embeddings (matches if any is a close enough match, rather
        than betting everything on one frame) and an optional face embedding.

        A confident face match wins outright over body appearance, since it
        survives clothing/headwear changes that defeat body-ReID. Otherwise
        falls back to the body-appearance match, or mints a new Person-N.

        `reject_id`, if given, is a callable(person_id) -> bool. When the
        best-matching person_id is rejected (e.g. because it already has
        another live track - a ReID false-positive merge), a new Person-N is
        minted instead of attaching to it, and none of this track's samples
        are added to the rejected person's bank. The rejection is bypassed
        for a high-confidence match (face_sim >= FACE_MATCH_THRESHOLD, or
        body_sim >= REID_HIGH_CONF_THRESHOLD): a match that strong is taken
        to mean the tracker split one person into two tracks, not that a
        second person appeared, so the live track it conflicts with is
        trusted to be stale rather than blocking the merge."""
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

        face_id, face_sim, face_top_sim = (None, -1.0, -1.0)
        if face_norm is not None:
            face_id, face_sim, face_top_sim = self._best_match(face_norm, self._face_embeddings)
            face_all = {
                pid: round(max((float(np.dot(face_norm, e)) for e in bank), default=-1.0), 3)
                for pid, bank in self._face_embeddings.items()
            }
            print(f"[IDENTITY-DEBUG] FACE scores vs gallery: {face_all} "
                  f"(threshold={FACE_MATCH_THRESHOLD:.2f}, best=Person-{face_id} sim={face_sim:.3f})")

        face_rejected = False
        if face_id is not None and face_sim >= FACE_MATCH_THRESHOLD and reject_id is not None and reject_id(face_id):
            if face_sim >= FACE_HIGH_CONF_THRESHOLD:
                print(f"[IDENTITY] Person-{face_id} has another live track but FACE match is "
                      f"high-confidence (sim={face_sim:.2f} >= {FACE_HIGH_CONF_THRESHOLD:.2f}) - "
                      f"trusting it as a tracker split and absorbing the stale track.")
            else:
                face_rejected = True
                print(f"[IDENTITY] Rejected FACE match to Person-{face_id} (sim={face_sim:.2f}): "
                      f"already has another live track - minting new identity instead.")

        if face_id is not None and face_sim >= FACE_MATCH_THRESHOLD and not face_rejected:
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

        body_id, body_sim, body_top_sim, body_embedding = (None, -1.0, -1.0, None)
        for embedding in normalized:
            cand_id, cand_sim, cand_top_sim = self._best_match(
                embedding, self._embeddings, top_k=REID_TOPK_MATCH
            )
            if cand_sim > body_sim:
                body_id, body_sim, body_top_sim, body_embedding = cand_id, cand_sim, cand_top_sim, embedding

        if normalized:
            body_all = {
                pid: round(
                    max(
                        (
                            sum(sorted((float(np.dot(e, s)) for s in bank), reverse=True)[:REID_TOPK_MATCH])
                            / min(REID_TOPK_MATCH, len(bank))
                            for e in normalized
                        ),
                        default=-1.0,
                    ),
                    3,
                )
                for pid, bank in self._embeddings.items() if bank
            }
            print(f"[IDENTITY-DEBUG] BODY scores vs gallery: {body_all} "
                  f"(match_thr={REID_MATCH_THRESHOLD:.2f}, borderline_thr={REID_BORDERLINE_THRESHOLD:.2f}, "
                  f"best=Person-{body_id} sim={body_sim:.3f})")

        body_match_ok = body_id is not None and body_sim >= REID_MATCH_THRESHOLD
        if body_id is not None and REID_BORDERLINE_THRESHOLD <= body_sim < REID_MATCH_THRESHOLD:
            corroborated = self._corroboration_count(
                body_embedding, self._embeddings[body_id], REID_CORROBORATION_THRESHOLD
            ) >= REID_CORROBORATION_MIN_SAMPLES
            if corroborated:
                body_match_ok = True
                print(f"[IDENTITY] Borderline BODY match to Person-{body_id} (sim={body_sim:.2f}) "
                      f"accepted via corroboration")

        body_rejected = False
        if (body_id is not None and body_match_ok and reject_id is not None
                and reject_id(body_id) and body_top_sim < REID_HIGH_CONF_THRESHOLD):
            body_rejected = True
            print(f"[IDENTITY] Rejected BODY match to Person-{body_id} (sim={body_sim:.2f}): "
                  f"already has another live track - minting new identity instead.")
        elif (body_id is not None and body_match_ok and reject_id is not None
                and reject_id(body_id)):
            print(f"[IDENTITY] Person-{body_id} has another live track but BODY match is "
                  f"high-confidence (sim={body_sim:.2f} >= {REID_HIGH_CONF_THRESHOLD:.2f}) - "
                  f"trusting it as a tracker split and absorbing the stale track.")

        if body_id is not None and body_match_ok and not body_rejected:
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


def load_reid_encoder():
    try:
        return ReID(REID_MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"[ERROR] Failed to load ReID encoder: {e}")
        raise SystemExit(1)


def load_face_analyzer():
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == 0 else ["CPUExecutionProvider"]
        analyzer = FaceAnalysis(name="buffalo_s", providers=providers)
        analyzer.prepare(ctx_id=0 if DEVICE == 0 else -1, det_size=(640, 640))
        return analyzer
    except Exception as e:
        print(f"[ERROR] Failed to load InsightFace analyzer: {e}")
        raise SystemExit(1)


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
