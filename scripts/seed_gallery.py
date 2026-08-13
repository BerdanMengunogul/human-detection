"""One-off: reset gallery.npz and re-seed it with clean per-person embedding
banks built from the relabeled dataset/<Name>/ crops, using the current
REID_MODEL_NAME encoder and InsightFace for faces. Assigns real names in
people.json so pipeline.py identifies people by name from the first frame
of the next run, instead of accumulating a new person_id every session.

Identities are discovered from the named (non-numeric) folders under
dataset/, so adding a person needs no code change:

    1. mkdir dataset/<TheirName>
    2. Put crops of just that person in it -- 60+ is comfortable, and some
       should show the face clearly, since faces carry recognition.
    3. python seed_gallery.py
    4. Restart the detector (it loads gallery.npz at startup).

Existing person_ids are read back from people.json and preserved, so a new
name is appended as the next id rather than renumbering everyone.

Usage:
    python seed_gallery.py
"""

import os
from collections import deque

import cv2
import numpy as np

import config as _config
from identity import (
    FACE_MAX_SAMPLES_PER_PERSON,
    FACE_MIN_BOX_HEIGHT,
    REID_MAX_SAMPLES_PER_PERSON,
    load_face_analyzer,
    load_reid_encoder,
)

DATASET_PATH = _config.load().DATASET_PATH
GALLERY_PATH = _config.load().GALLERY_PATH
PEOPLE_PATH = _config.load().PEOPLE_PATH


def body_embedding(encoder, img):
    h, w = img.shape[:2]
    xywh = np.array([w / 2, h / 2, w, h], dtype=np.float32)
    emb = encoder(img, xywh[None, :])[0]
    return emb / np.linalg.norm(emb)


def face_embedding(analyzer, img):
    faces = analyzer.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.bbox[3] - f.bbox[1])
    if (best.bbox[3] - best.bbox[1]) < FACE_MIN_BOX_HEIGHT:
        return None
    return best.normed_embedding


def discover_people():
    """Every dataset/<Name>/ folder whose name isn't purely numeric, in a
    stable order. Numeric folders are the old unlabeled tracker output and
    hold mixed identities, so they must not become gallery entries.

    Adding a person is therefore just: create dataset/<TheirName>/, drop in
    crops of them, and re-run this script.
    """
    if not os.path.isdir(DATASET_PATH):
        return []
    return sorted(
        name for name in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, name)) and not name.isdigit()
    )


def assign_ids(folders):
    """Map folder -> person_id, preserving the ids already in people.json so a
    new person is appended rather than renumbering everyone. Reusing an id for
    a different person would silently relabel their history in the events DB."""
    existing = {}
    if os.path.isfile(PEOPLE_PATH):
        import json
        try:
            with open(PEOPLE_PATH) as f:
                existing = {v: int(k) for k, v in json.load(f).get("names", {}).items()}
        except (ValueError, OSError):
            existing = {}

    ids = {f: existing[f] for f in folders if f in existing}
    next_id = max(ids.values(), default=0) + 1
    for folder in folders:
        if folder not in ids:
            ids[folder] = next_id
            next_id += 1
    return ids


def main():
    folders = discover_people()
    if not folders:
        print(f"[FAIL] No named folders in {DATASET_PATH}/. Create dataset/<Name>/ first.")
        return
    people = assign_ids(folders)
    print(f"[INFO] Seeding {len(people)} identities: "
          + ", ".join(f"{n}=id{i}" for n, i in sorted(people.items(), key=lambda kv: kv[1])))

    print("[INFO] Loading ReID encoder and face analyzer...")
    encoder = load_reid_encoder()
    analyzer = load_face_analyzer()

    body_banks = {}
    face_banks = {}
    names = {}

    for folder, person_id in sorted(people.items(), key=lambda kv: kv[1]):
        person_dir = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(person_dir):
            print(f"[WARN] Skipping {folder}: no such dataset folder")
            continue

        files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            print(f"[WARN] Skipping {folder}: no crops")
            continue

        body_bank = deque(maxlen=REID_MAX_SAMPLES_PER_PERSON)
        face_bank = deque(maxlen=FACE_MAX_SAMPLES_PER_PERSON)

        # Sample evenly across the folder so the bank isn't just the first N
        # frames of one pose/angle.
        step = max(1, len(files) // (REID_MAX_SAMPLES_PER_PERSON * 3))
        candidates = files[::step]

        for fname in candidates:
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            if len(body_bank) < REID_MAX_SAMPLES_PER_PERSON:
                body_bank.append(body_embedding(encoder, img))
            if len(face_bank) < FACE_MAX_SAMPLES_PER_PERSON:
                fe = face_embedding(analyzer, img)
                if fe is not None:
                    face_bank.append(fe)
            if len(body_bank) >= REID_MAX_SAMPLES_PER_PERSON and len(face_bank) >= FACE_MAX_SAMPLES_PER_PERSON:
                break

        body_banks[person_id] = body_bank
        if face_bank:
            face_banks[person_id] = face_bank
        names[str(person_id)] = folder
        print(f"[INFO] Person-{person_id} ({folder}): {len(body_bank)} body, {len(face_bank)} face samples "
              f"from {len(files)} crops")

    arrays = {}
    for person_id, bank in body_banks.items():
        if bank:
            arrays[f"body_{person_id}"] = np.stack(bank)
    for person_id, bank in face_banks.items():
        if bank:
            arrays[f"face_{person_id}"] = np.stack(bank)

    tmp_path = GALLERY_PATH + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, GALLERY_PATH)
    print(f"[INFO] Wrote {GALLERY_PATH} with {len(body_banks)} identities")

    import json
    with open(PEOPLE_PATH, "w") as f:
        json.dump({"names": names}, f, indent=2)
    print(f"[INFO] Wrote {PEOPLE_PATH}: {names}")


if __name__ == "__main__":
    main()
